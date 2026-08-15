"""用模拟退火（SA）批量求解柔性作业车间调度问题。

本模块复用 ILS 的解编码、MWKR-EET 初解、邻域和主动调度解码器。SA 在
高温阶段以较高概率接受较差解，随后通过几何降温逐步加强对优质解的偏好。
直接运行会求解 data 中的 50 个基准实例，并输出 performance_sa_mwkr_eet.csv。
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

try:
    from . import ILS
except ImportError:
    import ILS


PROJECT_DIRECTORY = Path(__file__).resolve().parent.parent
DocuProcess = ILS.DocuProcess


def solve(
    data,
    initial_temperature=100.0,
    minimum_temperature=1.0,
    cooling_rate=0.90,
    iterations_per_temperature=100,
    seed=42,
):
    """运行模拟退火，返回历史最好调度和可复现实验信息。"""
    if initial_temperature <= 0:
        raise ValueError("初始温度必须大于 0")
    if minimum_temperature <= 0:
        raise ValueError("最低温度必须大于 0")
    if minimum_temperature >= initial_temperature:
        raise ValueError("最低温度必须小于初始温度")
    if not 0 < cooling_rate < 1:
        raise ValueError("冷却率必须在 0 和 1 之间")
    if iterations_per_temperature < 0:
        raise ValueError("每个温度的迭代次数不能小于 0")

    n_jobs, n_machines, jobs = DocuProcess.parse(data)
    rng = random.Random(seed)
    initial = ILS.build_initial_solution(jobs, n_machines)
    initial_makespan = ILS.decode(initial, jobs, n_machines, validate=True)[0]
    current = initial
    current_value = initial_makespan
    best = initial
    best_value = initial_makespan
    temperature = float(initial_temperature)
    temperature_step = 0
    total_iterations = 0
    accepted_worse_total = 0
    history = []

    while temperature > minimum_temperature:
        accepted = 0
        accepted_worse = 0
        improvements = 0

        for _ in range(iterations_per_temperature):
            candidate = ILS._sample_neighbor(current, jobs, rng)
            candidate_value = ILS.decode(candidate, jobs, n_machines)[0]
            delta = candidate_value - current_value

            if delta <= 0 or rng.random() < math.exp(-delta / temperature):
                current = candidate
                current_value = candidate_value
                accepted += 1
                if delta > 0:
                    accepted_worse += 1
                    accepted_worse_total += 1

            if candidate_value < best_value:
                best = candidate
                best_value = candidate_value
                improvements += 1

            total_iterations += 1

        history.append(
            {
                "temperature_step": temperature_step,
                "temperature": temperature,
                "current_makespan": current_value,
                "best_makespan": best_value,
                "accepted": accepted,
                "accepted_worse": accepted_worse,
                "improvements": improvements,
            }
        )
        temperature *= cooling_rate
        temperature_step += 1

    makespan, machine_intervals, job_end = ILS.decode(
        best, jobs, n_machines, validate=True
    )
    return {
        "n_jobs": n_jobs,
        "n_machines": n_machines,
        "seed": seed,
        "initial_temperature": initial_temperature,
        "minimum_temperature": minimum_temperature,
        "cooling_rate": cooling_rate,
        "iterations_per_temperature": iterations_per_temperature,
        "temperature_steps": temperature_step,
        "total_iterations": total_iterations,
        "accepted_worse": accepted_worse_total,
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
    initial_temperature=100.0,
    minimum_temperature=1.0,
    cooling_rate=0.90,
    iterations_per_temperature=100,
    seed=42,
):
    """批量运行 SA，匹配 bks.json 下界并保存 performance CSV。"""
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
                initial_temperature=initial_temperature,
                minimum_temperature=minimum_temperature,
                cooling_rate=cooling_rate,
                iterations_per_temperature=iterations_per_temperature,
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
    parser = argparse.ArgumentParser(description="使用 SA 批量求解 FJSP 实例")
    parser.add_argument("--initial-temperature", type=float, default=100.0)
    parser.add_argument("--minimum-temperature", type=float, default=1.0)
    parser.add_argument("--cooling-rate", type=float, default=0.90)
    parser.add_argument("--iterations-per-temperature", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "performance_sa_mwkr_eet.csv",
    )
    args = parser.parse_args()

    try:
        performance = run_batch(
            PROJECT_DIRECTORY / "data",
            args.output,
            initial_temperature=args.initial_temperature,
            minimum_temperature=args.minimum_temperature,
            cooling_rate=args.cooling_rate,
            iterations_per_temperature=args.iterations_per_temperature,
            seed=args.seed,
        )
    except ValueError as error:
        parser.error(str(error))

    successful = sum(row["status"] == "ok" for row in performance)
    print(f"\n完成：{successful}/{len(performance)} 个实例求解成功")
    print(f"performance 表格：{args.output.resolve()}")


if __name__ == "__main__":
    main()
