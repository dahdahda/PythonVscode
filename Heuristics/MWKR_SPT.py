"""MWKR-SPT 柔性作业车间调度算法。

调度规则：
1. MWKR（Most Work Remaining）：剩余估计加工量最大的工件优先。
2. SPT（Shortest Processing Time）：为当前工序选择加工时间最短的机器。

由于一道柔性工序可能有多台候选机器，本模块使用该工序的最短候选加工
时间估计其工作量。工件的剩余工作量等于尚未调度工序估计工作量之和。

输入：由 ``DocuProcess.parse`` 支持的 FJSP 文本实例。
输出：单实例调度结果字典；批量运行时输出 performance_mwkr_spt.csv。
"""

import json
from pathlib import Path

import DocuProcess


def earliest_gap(intervals, ready, duration):
    """返回工序在一台机器上的最早可行开始时间。

    参数：
        intervals (list[tuple]): 机器已有调度记录，单条格式为
            ``(start, end, job, operation, machine, processing_time)``。
        ready (int | float): 工件就绪时间，即前一道工序的结束时间。
        duration (int | float): 当前工序的加工时间。

    返回：
        int | float: 不早于 ``ready`` 且不与已有任务重叠的最早开始时间。
    """
    start = ready
    for old_start, old_end, *_ in sorted(intervals):
        if start + duration <= old_start:
            return start
        if start < old_end:
            start = old_end
    return start


def operation_work(choices):
    """计算一道柔性工序用于 MWKR 排序的估计工作量。

    参数：
        choices (list[tuple[int, int]]): ``(机器编号, 加工时间)`` 候选列表。

    返回：
        int | float: 所有候选机器中的最短加工时间。
    """
    if not choices:
        raise ValueError("工序没有可用的候选机器")
    return min(processing_time for _, processing_time in choices)


def build_mwkr_priority(jobs):
    """生成满足工件工序顺序的 MWKR 优先序列。

    参数：
        jobs (list[list[list[tuple]]]): ``DocuProcess.parse`` 生成的工件数据。

    返回：
        list[tuple[int, int]]: ``(工件编号, 工序编号)`` 优先序列。

    规则：
        每次选择剩余估计加工量最大的工件；同分时工件编号较小者优先。
        选中一道工序后，从该工件的剩余工作量中扣除该工序的估计工作量。
    """
    n_jobs = len(jobs)
    next_op = [0] * n_jobs
    operation_works = [
        [operation_work(choices) for choices in job] for job in jobs
    ]
    remaining_work = [sum(works) for works in operation_works]
    priority = []

    while any(next_op[job] < len(jobs[job]) for job in range(n_jobs)):
        job = min(
            (
                job
                for job in range(n_jobs)
                if next_op[job] < len(jobs[job])
            ),
            key=lambda job: (-remaining_work[job], job),
        )
        operation = next_op[job]
        priority.append((job, operation))
        remaining_work[job] -= operation_works[job][operation]
        next_op[job] += 1

    return priority


def solve(data):
    """使用 MWKR-SPT 求解一个 FJSP 实例。

    参数：
        data (str): 完整 FJSP 文本实例。

    返回：
        dict: 包含以下字段：
            ``n_jobs``：工件数；``n_machines``：机器数；
            ``makespan``：最大完工时间；
            ``priority``：MWKR 工序优先序列；
            ``machine_intervals``：各机器调度记录；
            ``job_end``：各工件最终完工时间。
    """
    n_jobs, n_machines, jobs = DocuProcess.parse(data)
    priority = build_mwkr_priority(jobs)

    machine_intervals = [[] for _ in range(n_machines)]
    job_end = [0] * n_jobs
    records = []

    for job, operation in priority:
        # SPT：加工时间最短者优先；同分时机器编号较小者优先。
        machine, processing_time = min(
            jobs[job][operation], key=lambda choice: (choice[1], choice[0])
        )
        start = earliest_gap(
            machine_intervals[machine], job_end[job], processing_time
        )
        end = start + processing_time
        record = (
            start,
            end,
            job,
            operation,
            machine,
            processing_time,
        )
        machine_intervals[machine].append(record)
        machine_intervals[machine].sort()
        job_end[job] = end
        records.append(record)

    _validate_schedule(jobs, records, machine_intervals)
    return {
        "n_jobs": n_jobs,
        "n_machines": n_machines,
        "makespan": max(job_end, default=0),
        "priority": priority,
        "machine_intervals": machine_intervals,
        "job_end": job_end,
    }


def _validate_schedule(jobs, records, machine_intervals):
    """检查工序完整性、工件先后约束和机器不重叠约束。"""
    assert len(records) == sum(len(job) for job in jobs)

    for job in range(len(jobs)):
        job_records = sorted(
            (record for record in records if record[2] == job),
            key=lambda record: record[3],
        )
        assert all(
            job_records[index][0] >= job_records[index - 1][1]
            for index in range(1, len(job_records))
        )

    for machine_records in machine_intervals:
        assert all(
            machine_records[index][0] >= machine_records[index - 1][1]
            for index in range(1, len(machine_records))
        )


def main():
    """批量求解 50 个基准实例并输出 performance_mwkr_spt.csv。

    输入：项目 ``data`` 目录中的 vdata、Brandimarte 实例及 bks.json。
    输出：``Heuristics/performance_mwkr_spt.csv`` 性能结果表。
    """
    code_directory = Path(__file__).resolve().parent
    data_directory = code_directory.parent / "data"
    bks_path = data_directory / "bks.json"
    performance_path = code_directory / "performance_mwkr_spt.csv"

    with bks_path.open("r", encoding="utf-8-sig") as file:
        benchmark_records = DocuProcess.find_benchmark_records(json.load(file))

    performance = []
    for dataset, instance, instance_path in DocuProcess.get_instance_files(
        data_directory
    ):
        row = _empty_performance_row(dataset, instance)
        try:
            result = solve(instance_path.read_text(encoding="utf-8-sig"))
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
            print(f"{dataset}/{instance}: {row['makespan_gap']}")
        except Exception as error:
            row["status"] = "error"
            row["message"] = f"{type(error).__name__}: {error}"
            print(f"{dataset}/{instance}: ERROR - {error}")
        performance.append(row)

    DocuProcess.save_performance(performance, performance_path)
    successful = sum(row["status"] == "ok" for row in performance)
    print(f"\n完成：{successful}/{len(performance)}个实例求解成功")
    print(f"performance表格：{performance_path}")


def _empty_performance_row(dataset, instance):
    """返回一条具有固定 CSV 输入输出字段的空性能记录。"""
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


if __name__ == "__main__":
    main()
