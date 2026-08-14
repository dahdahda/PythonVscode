"""用迭代局部搜索（ILS）批量求解柔性作业车间实例。

解由两部分组成：工件序列决定工序的调度优先次序，机器向量决定每道
工序使用哪台候选机器。工件编号第 k 次出现代表该工件的第 k 道工序，
因此交换序列中的工件编号不会破坏工艺顺序。解码器使用最早可行空隙
生成主动调度，并检查工序优先约束和机器互斥约束。

直接运行本文件会批量求解 ``data`` 中的 vdata 和 Brandimarte 文本实例，
从 ``bks.json`` 读取对应下界，并输出 ``Metahe/performance_ils.csv``。
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_DIRECTORY = Path(__file__).resolve().parent.parent
HEURISTICS_DIRECTORY = PROJECT_DIRECTORY / "Heuristics"

if str(HEURISTICS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(HEURISTICS_DIRECTORY))

import DocuProcess  # noqa: E402
import MWKR_EET  # noqa: E402


@dataclass(frozen=True)
class Solution:
    """ILS 的不可变编码。machines 按 (工件, 工序) 的展平顺序存储。"""

    sequence: tuple[int, ...]
    machines: tuple[int, ...]


def _operation_offsets(jobs):
    """
    计算每个工件的工序在展平机器向量中的起始索引。
    假设三个工件的工序数量分别为：[3, 2, 4]
    则返回 [0, 3, 5]，表示工件 0 的工序在索引 0-2，工件 1 的工序在索引 3-4，
    工件 2 的工序在索引 5-8。
    工件 job 的工序 operation 对应的展平位置为：offsets[job] + operation
    """
    offsets = []
    total = 0
    for job in jobs:
        offsets.append(total)
        total += len(job)
    return offsets


def _earliest_gap(intervals, ready, duration):
    start = ready
    for old_start, old_end, *_ in intervals:
        if start + duration <= old_start:
            return start
        if start < old_end:
            start = old_end
    return start


def decode(solution, jobs, n_machines, validate=False):
    """把编码解码为主动调度，返回 makespan、机器时间表和工件结束时间。
    根据solution将调度结果计算处理->解码
    """
    offsets = _operation_offsets(jobs)
    expected_operations = sum(len(job) for job in jobs)

    if len(solution.sequence) != expected_operations:
        raise ValueError("工件序列长度与工序总数不一致")
    if len(solution.machines) != expected_operations:
        raise ValueError("机器向量长度与工序总数不一致")

    next_operation = [0] * len(jobs)
    job_end = [0] * len(jobs)
    machine_intervals = [[] for _ in range(n_machines)]
    records = []

    for job in solution.sequence:
        if not 0 <= job < len(jobs):
            raise ValueError(f"无效工件编号：{job}")
        operation = next_operation[job]
        if operation >= len(jobs[job]):
            raise ValueError(f"工件 {job} 在序列中出现次数过多")

        machine = solution.machines[offsets[job] + operation]
        processing_by_machine = dict(jobs[job][operation])

        if machine not in processing_by_machine:
            raise ValueError(f"机器 {machine} 不能加工工件 {job} 的工序 {operation}")

        processing_time = processing_by_machine[machine]
        start = _earliest_gap(machine_intervals[machine], job_end[job], processing_time)
        end = start + processing_time
        record = (start, end, job, operation, machine, processing_time)
        machine_intervals[machine].append(record)
        machine_intervals[machine].sort()
        records.append(record)
        job_end[job] = end
        next_operation[job] += 1

    if any(next_operation[job] != len(jobs[job]) for job in range(len(jobs))):
        raise ValueError("工件序列遗漏了工序")
    if validate:
        _validate_schedule(jobs, records, machine_intervals)

    return max(job_end, default=0), machine_intervals, job_end


def _validate_schedule(jobs, records, machine_intervals):
    if len(records) != sum(len(job) for job in jobs):
        raise AssertionError("并非所有工序都恰好调度一次")
    for job in range(len(jobs)):
        job_records = sorted(
            (record for record in records if record[2] == job),
            key=lambda record: record[3],
        )
        if any(
            job_records[index][0] < job_records[index - 1][1]
            for index in range(1, len(job_records))
        ):
            raise AssertionError(f"工件 {job} 的工序优先约束被破坏")

    for machine, intervals in enumerate(machine_intervals):
        if any(
            intervals[index][0] < intervals[index - 1][1]
            for index in range(1, len(intervals))
        ):
            raise AssertionError(f"机器 {machine} 上存在重叠工序")


def build_initial_solution(jobs, n_machines):
    """使用项目已有的 MWKR-EET 规则构造确定性的 ILS 初解。"""
    priority = MWKR_EET.build_mwkr_priority(jobs)
    machine_intervals = [[] for _ in range(n_machines)]
    job_end = [0] * len(jobs)
    offsets = _operation_offsets(jobs)
    machines = [0] * sum(len(job) for job in jobs)

    for job, operation in priority:
        machine, processing_time, start, end = MWKR_EET.select_machine_eet(
            jobs[job][operation], machine_intervals, job_end[job]
        )
        record = (start, end, job, operation, machine, processing_time)
        machine_intervals[machine].append(record)
        machine_intervals[machine].sort()
        job_end[job] = end
        machines[offsets[job] + operation] = machine

    return Solution(
        sequence=tuple(job for job, _ in priority),
        machines=tuple(machines),
    )


def _sample_neighbor(solution, jobs, rng):
    """随机生成一个交换邻域或机器重分配邻域中的相邻解。
    35% 的概率选择机器重分配邻域，否则选择交换邻域。"""
    sequence = list(solution.sequence)
    machines = list(solution.machines)
    flexible = [
        (flat_index, choices)
        for flat_index, choices in enumerate(choices for job in jobs for choices in job)
        if len(choices) > 1
    ]

    if flexible and rng.random() < 0.35:
        flat_index, choices = rng.choice(flexible)
        alternatives = [
            machine for machine, _ in choices if machine != machines[flat_index]
        ]
        machines[flat_index] = rng.choice(alternatives)
    else:
        first, second = rng.sample(range(len(sequence)), 2)
        while sequence[first] == sequence[second]:
            first, second = rng.sample(range(len(sequence)), 2)
        sequence[first], sequence[second] = sequence[second], sequence[first]
    return Solution(tuple(sequence), tuple(machines))


def local_search(solution, jobs, n_machines, rng, trials=350, rounds=12):
    """随机最陡下降：每轮抽样邻域，采用其中使 makespan 最小的改进解。
    trials 为每轮抽样邻域的次数，rounds 为最多进行12轮成功的改进尝试。可以增加patient参数，
    连续若干轮没有改进再提前终止，而不是一轮没有改进就终止。
    """
    current = solution
    current_value = decode(current, jobs, n_machines)[0]
    for _ in range(rounds):
        best_neighbor = current
        best_value = current_value
        seen = {current}
        for _ in range(trials):
            neighbor = _sample_neighbor(current, jobs, rng)
            if neighbor in seen:
                continue
            seen.add(neighbor)
            value = decode(neighbor, jobs, n_machines)[0]
            if value < best_value:
                best_neighbor, best_value = neighbor, value
        if best_value >= current_value:
            break
        current, current_value = best_neighbor, best_value
    return current, current_value


def perturb(solution, jobs, rng, strength=3):
    """连续执行若干随机邻域动作，使搜索跳出当前局部最优。"""
    perturbed = solution
    for _ in range(max(1, strength)):
        perturbed = _sample_neighbor(perturbed, jobs, rng)
    return perturbed


def solve(data, iterations=80, seed=42, local_trials=350, local_rounds=12):
    """运行 ILS，返回最佳调度及可复现实验信息。"""
    n_jobs, n_machines, jobs = DocuProcess.parse(data)
    rng = random.Random(seed)
    initial = build_initial_solution(jobs, n_machines)
    initial_makespan = decode(initial, jobs, n_machines, validate=True)[0]
    current, current_value = local_search(
        initial, jobs, n_machines, rng, local_trials, local_rounds
    )
    best, best_value = current, current_value
    history = [
        {
            "iteration": 0,
            "current_makespan": current_value,
            "best_makespan": best_value,
        }
    ]

    for iteration in range(1, iterations + 1):
        candidate = perturb(current, jobs, rng, strength=2 + iteration % 3)
        candidate, candidate_value = local_search(
            candidate, jobs, n_machines, rng, local_trials, local_rounds
        )

        # 接受准则：不劣于当前解则接受；每 10 轮从全局最好解重新出发，
        # 防止在质量较差的搜索区域持续漂移。
        """
        current：控制下一轮从哪里继续搜索，可以暂时变差
        best：记录整个搜索过程中找到的最好解，绝不变差
        """
        current, current_value = candidate, candidate_value
        if candidate_value < best_value:
            best, best_value = candidate, candidate_value
        if iteration % 10 == 0:
            current, current_value = best, best_value
        history.append(
            {
                "iteration": iteration,
                "current_makespan": current_value,
                "best_makespan": best_value,
            }
        )

    makespan, machine_intervals, job_end = decode(best, jobs, n_machines, validate=True)
    return {
        "n_jobs": n_jobs,
        "n_machines": n_machines,
        "seed": seed,
        "iterations": iterations,
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
    """创建一条固定字段的空性能记录，失败时也能保留实例信息。"""
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
    iterations=80,
    seed=42,
    local_trials=350,
    local_rounds=12,
):
    """批量求解全部基准实例，并把 makespan 与下界差距保存为 CSV。"""
    bks_path = data_directory / "bks.json"
    with bks_path.open("r", encoding="utf-8-sig") as file:
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
                seed=seed,
                local_trials=local_trials,
                local_rounds=local_rounds,
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
            print(f"[{index}/{total}] {dataset}/{instance}: " f"{row['makespan_gap']}")
        except Exception as error:
            row["status"] = "error"
            row["message"] = f"{type(error).__name__}: {error}"
            print(f"[{index}/{total}] {dataset}/{instance}: ERROR - {error}")
        performance.append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    DocuProcess.save_performance(performance, output_path)
    return performance


def main():
    parser = argparse.ArgumentParser(description="使用 ILS 批量求解 FJSP 实例")
    parser.add_argument("--iterations", type=int, default=80, help="ILS 迭代次数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument(
        "--local-trials",
        type=int,
        default=350,
        help="每轮局部搜索抽样的邻居数",
    )
    parser.add_argument(
        "--local-rounds",
        type=int,
        default=12,
        help="每次局部搜索的最大改进轮数",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "performance_ils_mwkr_eet.csv",
        help="performance CSV 输出路径",
    )
    args = parser.parse_args()

    if args.iterations < 0:
        parser.error("--iterations 不能小于 0")
    if args.local_trials < 0:
        parser.error("--local-trials 不能小于 0")
    if args.local_rounds < 0:
        parser.error("--local-rounds 不能小于 0")

    performance = run_batch(
        PROJECT_DIRECTORY / "data",
        args.output,
        iterations=args.iterations,
        seed=args.seed,
        local_trials=args.local_trials,
        local_rounds=args.local_rounds,
    )
    successful = sum(row["status"] == "ok" for row in performance)
    print(f"\n完成：{successful}/{len(performance)} 个实例求解成功")
    print(f"performance 表格：{args.output.resolve()}")


if __name__ == "__main__":
    main()
