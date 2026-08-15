"""用禁忌搜索（TS）批量求解柔性作业车间调度问题。

本模块复用 ILS 的解编码、MWKR-EET 初解和主动调度解码器。禁忌表记录
近期动作的反向动作；若禁忌候选能改善全局最好值，则通过特赦准则放行。
直接运行会求解 data 中的 50 个基准实例，并输出 performance_ts_mwkr_eet.csv。
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

try:
    from . import ILS
except ImportError:
    import ILS


PROJECT_DIRECTORY = Path(__file__).resolve().parent.parent
DocuProcess = ILS.DocuProcess


def _sample_neighbor_with_move(solution, jobs, rng):
    """生成邻居，同时返回用于检查禁忌和登记反向动作的属性。"""
    sequence = list(solution.sequence)
    machines = list(solution.machines)
    flexible = [
        (flat_index, choices)
        for flat_index, choices in enumerate(choices for job in jobs for choices in job)
        if len(choices) > 1
    ]

    if flexible and rng.random() < 0.35:
        flat_index, choices = rng.choice(flexible)
        old_machine = machines[flat_index]
        alternatives = [machine for machine, _ in choices if machine != old_machine]
        new_machine = rng.choice(alternatives)
        machines[flat_index] = new_machine
        move_key = ("machine", flat_index, new_machine)
        reverse_key = ("machine", flat_index, old_machine)
    else:
        first, second = rng.sample(range(len(sequence)), 2)
        while sequence[first] == sequence[second]:
            first, second = rng.sample(range(len(sequence)), 2)
        first, second = sorted((first, second))
        sequence[first], sequence[second] = sequence[second], sequence[first]
        move_key = ("swap", first, second)
        reverse_key = move_key

    neighbor = ILS.Solution(tuple(sequence), tuple(machines))
    return neighbor, move_key, reverse_key


def solve(data, iterations=200, candidate_size=100, tabu_tenure=7, seed=42):
    """运行禁忌搜索，返回历史最好调度和可复现实验信息。"""
    if iterations < 0:
        raise ValueError("TS 迭代次数不能小于 0")
    if candidate_size <= 0:
        raise ValueError("候选邻域规模必须大于 0")
    if tabu_tenure <= 0:
        raise ValueError("禁忌期限必须大于 0")

    n_jobs, n_machines, jobs = DocuProcess.parse(data)
    rng = random.Random(seed)
    initial = ILS.build_initial_solution(jobs, n_machines)
    initial_makespan = ILS.decode(initial, jobs, n_machines, validate=True)[0]
    current = initial
    current_value = initial_makespan
    best = initial
    best_value = initial_makespan
    tabu_until = {}
    aspiration_uses = 0
    history = []

    for iteration in range(1, iterations + 1):
        # 到期值小于当前迭代编号时，该动作已经解除禁忌。
        tabu_until = {
            key: expiry for key, expiry in tabu_until.items() if expiry >= iteration
        }
        candidates = []
        seen = {current}

        for _ in range(candidate_size):
            neighbor, move_key, reverse_key = _sample_neighbor_with_move(
                current, jobs, rng
            )
            if neighbor in seen:
                continue
            seen.add(neighbor)
            value = ILS.decode(neighbor, jobs, n_machines)[0]
            is_tabu = tabu_until.get(move_key, 0) >= iteration
            aspiration = is_tabu and value < best_value
            if not is_tabu or aspiration:
                candidates.append((value, neighbor, reverse_key, aspiration))

        if not candidates:
            # 极小候选集可能恰好全部被禁忌；清空短期记忆后继续搜索。
            tabu_until.clear()
            neighbor, _, reverse_key = _sample_neighbor_with_move(current, jobs, rng)
            value = ILS.decode(neighbor, jobs, n_machines)[0]
            candidates.append((value, neighbor, reverse_key, False))

        candidate_value, candidate, reverse_key, used_aspiration = min(
            candidates, key=lambda item: item[0]
        )
        current = candidate
        current_value = candidate_value
        tabu_until[reverse_key] = iteration + tabu_tenure
        if used_aspiration:
            aspiration_uses += 1
        if current_value < best_value:
            best = current
            best_value = current_value

        history.append(
            {
                "iteration": iteration,
                "current_makespan": current_value,
                "best_makespan": best_value,
                "tabu_size": len(tabu_until),
                "aspiration": used_aspiration,
            }
        )

    makespan, machine_intervals, job_end = ILS.decode(
        best, jobs, n_machines, validate=True
    )
    return {
        "n_jobs": n_jobs,
        "n_machines": n_machines,
        "seed": seed,
        "iterations": iterations,
        "candidate_size": candidate_size,
        "tabu_tenure": tabu_tenure,
        "aspiration_uses": aspiration_uses,
        "initial_makespan": initial_makespan,
        "makespan": makespan,
        "improvement": initial_makespan - makespan,
        "sequence": list(best.sequence),
        "machine_assignment": list(best.machines),
        "machine_intervals": machine_intervals,
        "job_end": job_end,
        "history": history,
    }


def _empty_performance_row(dataset, instance):
    return {
        "dataset": dataset,
        "instance": instance,
        "bsk_instance": "",
        "size": "",
        "lower_bound": "",
        "makespan": "",
        "gap_percent": "",
        "makespan_gap": "",
        "status": "ok",
        "message": "",
    }


def run_batch(
    data_directory,
    output_path,
    iterations=200,
    candidate_size=100,
    tabu_tenure=7,
    seed=42,
):
    """批量运行 TS，匹配 bks.json 下界并保存 performance CSV。"""
    with (data_directory / "bks.json").open("r", encoding="utf-8-sig") as file:
        benchmark_records = DocuProcess.find_benchmark_records(json.load(file))

    performance = []
    instance_files = DocuProcess.get_instance_files(data_directory)
    total = len(instance_files)
    for index, (dataset, instance, instance_path) in enumerate(instance_files, start=1):
        row = _empty_performance_row(dataset, instance)
        try:
            result = solve(
                instance_path.read_text(encoding="utf-8-sig"),
                iterations=iterations,
                candidate_size=candidate_size,
                tabu_tenure=tabu_tenure,
                seed=seed,
            )
            lower_bound, bks_instance = DocuProcess.get_lower_bound(
                benchmark_records, dataset, instance
            )
            makespan = result["makespan"]
            gap = (makespan - lower_bound) / lower_bound * 100
            row.update(
                {
                    "bsk_instance": bks_instance,
                    "size": f"{result['n_jobs']} x {result['n_machines']}",
                    "lower_bound": lower_bound,
                    "makespan": makespan,
                    "gap_percent": f"{gap:.2f}",
                    "makespan_gap": f"{makespan}({gap:.2f}%)",
                }
            )
            print(f"[{index}/{total}] {dataset}/{instance}: {row['makespan_gap']}")
        except Exception as error:
            row["status"] = "error"
            row["message"] = f"{type(error).__name__}: {error}"
            print(f"[{index}/{total}] {dataset}/{instance}: ERROR - {error}")
        performance.append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    DocuProcess.save_performance(performance, output_path)
    return performance


def main():
    parser = argparse.ArgumentParser(description="使用 TS 批量求解 FJSP 实例")
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--candidate-size", type=int, default=100)
    parser.add_argument("--tabu-tenure", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "performance_ts_mwkr_eet.csv",
    )
    args = parser.parse_args()

    try:
        performance = run_batch(
            PROJECT_DIRECTORY / "data",
            args.output,
            iterations=args.iterations,
            candidate_size=args.candidate_size,
            tabu_tenure=args.tabu_tenure,
            seed=args.seed,
        )
    except ValueError as error:
        parser.error(str(error))

    successful = sum(row["status"] == "ok" for row in performance)
    print(f"\n完成：{successful}/{len(performance)} 个实例求解成功")
    print(f"performance 表格：{args.output.resolve()}")


if __name__ == "__main__":
    main()
