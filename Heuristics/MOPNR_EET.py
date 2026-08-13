"""MOPNR-EET 柔性作业车间调度算法。

本模块使用两条启发式规则构造调度：
1. MOPNR（Most Operations Remaining）：剩余工序数最多的工件优先。
2. EET（Earliest End Time）：选择能让当前工序最早结束的候选机器。

输入：FJSP 文本实例，格式由 ``DocuProcess.parse`` 解析。
输出：单实例调度结果字典；批量运行时输出 performance_mopnr_eet.csv。

1.完成工件工序队列
2.完成机器候选队列

"""

import json
from pathlib import Path
import DocuProcess


def earliest_gap(intervals, ready, duration):
    """查找工序在指定机器上的最早可行开始时间。

    参数：
        intervals (list[tuple]): 机器已有调度区间。每条记录格式为
            ``(start, end, job, operation, machine, processing_time)``。
        ready (int | float): 当前工件的就绪时间，即前一道工序的结束时间。
        duration (int | float): 当前工序在该机器上的加工时间。

    返回：
        int | float: 不早于 ``ready`` 且不与已有区间重叠的最早开始时间。
    """
    start = ready
    for old_start, old_end, *_ in sorted(intervals):
        # 当前工序可以完整插入该已排工序之前的空隙。
        if start + duration <= old_start:
            return start
        # 存在重叠时，将候选开始时间移动到已排工序结束之后。
        if start < old_end:
            start = old_end
    return start


def select_machine_eet(choices, machine_intervals, ready):
    """按照 EET 规则为一道工序选择机器。

    参数：
        choices (list[tuple[int, int]]): 候选 ``(机器编号, 加工时间)`` 列表。
        machine_intervals (list[list[tuple]]): 所有机器当前的调度区间。
        ready (int | float): 当前工件的就绪时间。

    返回：
        tuple: ``(machine, processing_time, start, end)``，依次表示所选机器、
        加工时间、最早开始时间和最早结束时间。

    同分规则：
        优先结束时间更早者；若相同，依次选择开始时间更早、加工时间更短、
        机器编号更小的方案，使算法结果保持确定性。
    """
    candidates = []
    for machine, processing_time in choices:
        start = earliest_gap(machine_intervals[machine], ready, processing_time)
        end = start + processing_time
        candidates.append((end, start, processing_time, machine))

    if not candidates:
        raise ValueError("工序没有可用的候选机器")

    end, start, processing_time, machine = min(candidates)
    return machine, processing_time, start, end


def solve(data):
    """使用 MOPNR-EET 求解一个 FJSP 实例。

    参数：
        data (str): 一个完整的 FJSP 文本实例。第一行是工件数和机器数，
        后续每行描述一个工件的工序及候选机器加工时间。

    返回：
        dict: 调度结果，格式如下：

        ``n_jobs`` (int)
            工件数量。
        ``n_machines`` (int)
            机器数量。
        ``makespan`` (int | float)
            所有工件的最大完工时间。
        ``priority`` (list[tuple[int, int]])
            MOPNR 生成的 ``(工件编号, 工序编号)`` 优先序列。
        ``machine_intervals`` (list[list[tuple]])
            每台机器的调度记录，单条格式为
            ``(start, end, job, operation, machine, processing_time)``。
        ``job_end`` (list[int | float])
            每个工件的最终完工时间。
    """
    n_jobs, n_machines, jobs = DocuProcess.parse(data)

    # MOPNR：每次选择剩余工序数最多的工件；同分时工件编号较小者优先。
    remaining = [len(job) for job in jobs]
    next_op = [0] * n_jobs
    priority = []
    while any(remaining):
        job = min(
            (job for job in range(n_jobs) if remaining[job] > 0),
            key=lambda job: (-remaining[job], job),
        )
        priority.append((job, next_op[job]))
        next_op[job] += 1
        remaining[job] -= 1
        # 工件以及工序选择完成，接下来进行对应工序的机器选择

    # machine_intervals[machine] 保存该机器上已经排定的全部工序。
    machine_intervals = [[] for _ in range(n_machines)]
    # job_end[job] 是该工件上一道已排工序的结束时间。
    job_end = [0] * n_jobs
    records = []

    for job, operation in priority:
        # EET 同时考虑候选机器的空闲区间和当前工件的就绪时间。
        machine, processing_time, start, end = select_machine_eet(
            jobs[job][operation], machine_intervals, job_end[job]
        )
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

    # 可行性检查 1：输入中的每道工序都必须且只能被调度一次。
    assert len(records) == sum(len(job) for job in jobs)

    # 可行性检查 2：同一工件的后续工序不能早于前序工序完成。
    for job in range(n_jobs):
        job_records = sorted(
            (record for record in records if record[2] == job),
            key=lambda record: record[3],
        )
        assert all(
            job_records[index][0] >= job_records[index - 1][1]
            for index in range(1, len(job_records))
        )

    # 可行性检查 3：同一机器上的任意两道工序不能重叠。
    for machine_records in machine_intervals:
        assert all(
            machine_records[index][0] >= machine_records[index - 1][1]
            for index in range(1, len(machine_records))
        )

    return {
        "n_jobs": n_jobs,
        "n_machines": n_machines,
        "makespan": max(job_end, default=0),
        "priority": priority,
        "machine_intervals": machine_intervals,
        "job_end": job_end,
    }


def main():
    """批量求解项目中的基准实例并输出性能表。

    输入：
        ``data/hurink/vdata/la01-la40.txt``、
        ``data/brandimarte/mk01-mk10.txt`` 和 ``data/bks.json``。

    输出：
        ``Heuristics/performance_mopnr_eet.csv``。每行包含实例规模、下界、
        makespan、相对差距、运行状态和错误消息。
    """
    code_directory = Path(__file__).resolve().parent
    project_directory = code_directory.parent
    data_directory = project_directory / "data"
    bks_path = data_directory / "bks.json"
    performance_path = code_directory / "performance_mopnr_eet.csv"

    # benchmark_records 的格式为 list[dict]，每个字典至少包含
    # instance 和 lower_bound 两个字段。
    with bks_path.open("r", encoding="utf-8-sig") as file:
        benchmark_records = DocuProcess.find_benchmark_records(json.load(file))

    performance = []
    for dataset, instance, instance_path in DocuProcess.get_instance_files(
        data_directory
    ):
        # row 是 performance CSV 的一行；失败时保留实例信息并记录异常。
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


if __name__ == "__main__":
    main()
