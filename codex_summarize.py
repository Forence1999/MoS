"""
This script is used to summarize the finetuning results of llama2 in qlora github repo (https://github.com/artidoro/qlora.git).
Usage:s
    python summarize.py
"""

from pathlib import Path
import sys, os
import csv
import re
import json
import argparse
from pathlib import Path
from datetime import datetime, timedelta


def write_csv(names, data, file_path):
    with open(file_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(names)
        for row in data:
            writer.writerow(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Filter and sort files based on creation time"
    )
    parser.add_argument(
        "--ts",
        type=float,
        default=6,
        help="timespan (ts): Duration of the present time from its last modification time in hours (default: 6, -1 for all files)",
    )
    args = parser.parse_args()

    titles = [
        "dir",
        "train_loss",
        "eval_loss",
        "time_h",
        "pass1(0.1)",
        "pass5(0.1)",
        "pass10(0.1)",
        "pass20(0.1)",
        "pass10(0.8)",
    ]
    data = []
    timespan = args.ts if args.ts != -1 else 1000000
    time_threshold = datetime.now() - timedelta(hours=timespan)

    log_paths = [
        i
        for i in Path("/workspace/output/").rglob("all_results.json")
        if (
            ("checkpoint" not in i.parent.name)
            and (i.stat().st_mtime > time_threshold.timestamp())
        )
    ]
    log_paths = sorted(log_paths, reverse=False)
    for log_path in log_paths:
        try:
            print(str(log_path))
            items = []
            log_path = Path(log_path).absolute()
            subfolder_name = log_path.parent.name
            items.append(log_path.parent.name)

            log = json.load(open(log_path, "r"))
            pass1_log = json.load(
                open(
                    log_path.parent.joinpath("codex_eval_results/pass1/metrics.json"),
                    "r",
                )
            )
            pass10_log = json.load(
                open(
                    log_path.parent.joinpath("codex_eval_results/pass10/metrics.json"),
                    "r",
                )
            )

            (
                train_loss,
                eval_loss,
                time_h,
                pass1_01,
                pass5_01,
                pass10_01,
                pass20_01,
                pass10_08,
            ) = [None] * 8

            train_loss = log["train_loss"]
            eval_loss = log["eval_loss"]
            time_h = log["train_runtime"] / 3600
            pass1_01 = pass1_log["pass@1"]
            pass5_01 = pass1_log["pass@5"]
            pass10_01 = pass1_log["pass@10"]
            pass20_01 = pass1_log["pass@20"]
            pass10_08 = pass10_log["pass@10"]
            items.extend(
                [
                    train_loss,
                    eval_loss,
                    time_h,
                    pass1_01,
                    pass5_01,
                    pass10_01,
                    pass20_01,
                    pass10_08,
                ]
            )
            data.append(items)

        except Exception as e:
            print(e)

    write_csv(titles, data, "/workspace/summary.csv")
