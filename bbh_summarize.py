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
        "cot",
        "no_cot",
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
            cot_log = json.load(
                open(
                    log_path.parent.joinpath("bbh_results/cot/metrics.json"),
                    "r",
                )
            )
            no_cot_log = json.load(
                open(
                    log_path.parent.joinpath("bbh_results/no_cot/metrics.json"),
                    "r",
                )
            )

            (
                train_loss,
                eval_loss,
                time_h,
                cot,
                no_cot,
            ) = [None] * 5

            train_loss = log["train_loss"]
            eval_loss = log["eval_loss"]
            time_h = log["train_runtime"] / 3600
            cot_em = cot_log["average_exact_match"]
            no_cot_em = no_cot_log["average_exact_match"]
            items.extend(
                [
                    train_loss,
                    eval_loss,
                    time_h,
                    cot_em,
                    no_cot_em,
                ]
            )
            data.append(items)

        except Exception as e:
            print(e)

    write_csv(titles, data, "/workspace/summary.csv")
