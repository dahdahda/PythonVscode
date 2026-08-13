import csv
import json
from pathlib import Path


def parse(data):
    """数据处理。

        jobs = [
            [  # 工件J1
                [(机器编号, 加工时间), ...],  # O1
                [(机器编号, 加工时间), ...],  # O2
            ],
            [  # 工件J2
                [(机器编号, 加工时间), ...],
            ],
        ]
    n_jobs：工件数量。
    n_machines：机器数量。
    jobs[j][o]：工件 j 的第 o 道工序可选择的机器及加工时间
    """
    lines = data.strip().splitlines()
    n_jobs, n_machines = map(int, lines[0].split()[:2])
    jobs = []
    for line in lines[1:]:
        xs = list(map(int, line.split()))
        n_ops, pos, ops = xs[0], 1, []
        for _ in range(n_ops):
            count = xs[pos]
            pos += 1
            choices = []
            for _ in range(count):
                choices.append((xs[pos], xs[pos + 1]))
                pos += 2
            ops.append(choices)
        assert pos == len(xs)
        jobs.append(ops)
    assert len(jobs) == n_jobs
    return n_jobs, n_machines, jobs


def find_benchmark_records(value):
    """递归读取bsk.json中包含instance和lower_bound的记录。"""
    records = []
    if isinstance(value, dict):
        if "instance" in value and "lower_bound" in value:
            records.append(value)
        for child in value.values():
            records.extend(find_benchmark_records(child))
    elif isinstance(value, list):
        for child in value:
            records.extend(find_benchmark_records(child))
    return records


def normalize_instance_name(name):
    """忽略大小写、下划线和短横线比较实例名。"""
    return "".join(character for character in name.lower() if character.isalnum())


def get_lower_bound(benchmark_records, dataset, instance):
    """根据数据集和文件名查找唯一的lower_bound。"""
    if dataset == "vdata":
        # la01.txt 对应 bsk.json 中的 la01_vdata。
        expected_names = {
            normalize_instance_name(f"{instance}_vdata"),
            normalize_instance_name(f"vdata_{instance}"),
        }
    else:
        # Brandimarte通常直接使用Mk01至Mk10作为instance。
        expected_names = {
            normalize_instance_name(instance),
            normalize_instance_name(f"{instance}_brandimarte"),
            normalize_instance_name(f"brandimarte_{instance}"),
        }

    matches = [
        record
        for record in benchmark_records
        if normalize_instance_name(str(record["instance"])) in expected_names
    ]
    if len(matches) != 1:
        names = [record["instance"] for record in matches]
        raise ValueError(
            f"{dataset}/{instance}应匹配一个lower_bound，实际匹配{len(matches)}个：{names}"
        )

    lower_bound = matches[0]["lower_bound"]
    if not isinstance(lower_bound, (int, float)) or isinstance(lower_bound, bool):
        raise ValueError(f"{matches[0]['instance']}的lower_bound不是数字")
    if lower_bound <= 0:
        raise ValueError(f"{matches[0]['instance']}的lower_bound必须大于0")
    return lower_bound, matches[0]["instance"]


def get_instance_files(data_directory):
    """按la01-la40、mk01-mk10的固定顺序生成输入文件。
    (数据集名称, 实例名称, 文件路径)
    """
    files = []
    for number in range(1, 41):
        files.append(
            (
                "vdata",
                f"la{number:02d}",
                data_directory / "hurink" / "vdata" / f"la{number:02d}.txt",
            )
        )
    for number in range(1, 11):
        files.append(
            (
                "brandimarte",
                f"mk{number:02d}",
                data_directory / "brandimarte" / f"mk{number:02d}.txt",
            )
        )
    return files


def save_performance(rows, output_path):
    """把批量结果保存为performance.csv。"""
    columns = [
        "dataset",
        "instance",
        "bsk_instance",
        "size",
        "lower_bound",
        "makespan",
        "gap_percent",
        "makespan_gap",
        "status",
        "message",
    ]
    with output_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()  # 第一行
        writer.writerows(rows)
