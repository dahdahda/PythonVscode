import csv
import json
from pathlib import Path
import DocuProcess

"""
MOPNR-SPT 柔性作业车间调度算法。

本模块使用两条启发式规则构造调度：
1. MOPNR（Most Operations Remaining）：剩余工序数最多的工件优先。
2. SPT（Shortest Processing Time）：选择能让当前工序加工时间最短的候选机器。

输入：FJSP 文本实例，格式由 ``DocuProcess.parse`` 解析。
输出：单实例调度结果字典；批量运行时输出 performance_mopnr_spt.csv。
"""


def earliest_gap(intervals, ready, duration):
    """
    在给定的时间间隔中寻找最早的空隙
    intervals: 已占用的时间间隔列表，每个间隔为 (start, end, ...)
    ready: 工件准备好开始的时间
    duration: 工序所需的加工时间
    返回最早的可行开始时间
    """
    start = ready
    for old_start, old_end, *_ in sorted(intervals):
        if start + duration <= old_start:
            return start
        if start < old_end:
            start = old_end
    return start


def solve(data):
    """使用MOPNR-SPT求解一个实例，返回makespan及调度信息。"""
    n_jobs, n_machines, jobs = DocuProcess.parse(data)

    # MOPNR：剩余工序数最多者优先；同分时工件编号较小者优先。
    remaining = [len(x) for x in jobs]
    next_op = [0] * n_jobs
    priority = []
    while any(remaining):
        j = min(
            (j for j in range(n_jobs) if remaining[j]),
            key=lambda j: (-remaining[j], j),
        )
        priority.append((j, next_op[j]))
        next_op[j] += 1
        remaining[j] -= 1

    machine_intervals = [[] for _ in range(n_machines)]
    job_end = [0] * n_jobs
    records = []

    for j, o in priority:
        # SPT：加工时间最短者优先；同分时机器编号较小者优先。
        m, p = min(jobs[j][o], key=lambda mp: (mp[1], mp[0]))
        start = earliest_gap(machine_intervals[m], job_end[j], p)
        end = start + p
        rec = (
            start,
            end,
            j,
            o,
            m,
            p,
        )  # (start, end, job, operation, machine, processing_time)
        machine_intervals[m].append(rec)
        machine_intervals[m].sort()
        job_end[j] = end
        records.append(rec)

    # 可行性检查：所有工序均已调度。
    assert len(records) == sum(len(job) for job in jobs)

    # 可行性检查：同一工件的后续工序不能早于前序工序完成。
    for j in range(n_jobs):
        rs = sorted((r for r in records if r[2] == j), key=lambda r: r[3])
        assert all(rs[k][0] >= rs[k - 1][1] for k in range(1, len(rs)))

    # 可行性检查：同一机器上的工序不能重叠。
    for rs in machine_intervals:
        assert all(rs[k][0] >= rs[k - 1][1] for k in range(1, len(rs)))

    return {
        "n_jobs": n_jobs,
        "n_machines": n_machines,
        "makespan": max(job_end),
        "priority": priority,
        "machine_intervals": machine_intervals,
        "job_end": job_end,
    }


def main():
    """
    文件路径处理
    """
    code_directory = Path(__file__).resolve().parent
    project_directory = code_directory.parent
    data_directory = project_directory / "data"
    bsk_path = data_directory / "bks.json"
    performance_path = code_directory / "performance_mopnr_spt.csv"

    """
benchmark_records = [
    {
        "instance": "mk01",
        "size": "10 x 6",
        "type": "flexible jobshop",
        "lower_bound": 40,
        "upper_bound": 40,
        ...
    },
    {
        "instance": "mk02",
        "size": "10 x 6",
        "type": "flexible jobshop",
        "lower_bound": 26,
        "upper_bound": 26,
        ...
    },
    ...
]
    """
    with bsk_path.open("r", encoding="utf-8-sig") as file:
        benchmark_records = DocuProcess.find_benchmark_records(json.load(file))

    performance = []
    for dataset, instance, instance_path in DocuProcess.get_instance_files(
        data_directory
    ):
        row = {
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

        try:
            data = instance_path.read_text(encoding="utf-8-sig")
            result = solve(data)
            lower_bound, bsk_instance = DocuProcess.get_lower_bound(
                benchmark_records, dataset, instance
            )
            makespan = result["makespan"]
            gap = (makespan - lower_bound) / lower_bound * 100

            row.update(
                {
                    "bsk_instance": bsk_instance,
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


if __name__ == "__main__":
    main()
