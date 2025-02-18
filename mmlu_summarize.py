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

# from qlora_repo.qlora import train


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
        "MMLU_acc_STEM",
        "MMLU_acc_humanities",
        "MMLU_acc_socialsciences",
        "MMLU_acc_other",
        "MMLU_acc",
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
            mmlu_log = json.load(
                open(log_path.parent.joinpath("mmlu_results/metrics.json"), "r")
            )

            (
                train_loss,
                eval_loss,
                time_h,
                MMLU_acc_STEM,
                MMLU_acc_humanities,
                MMLU_acc_socialsciences,
                MMLU_acc_other,
                MMLU_acc,
            ) = [None] * 8
            try:
                train_loss = log["train_loss"]
                eval_loss = log["eval_loss"]
                time_h = log["train_runtime"] / 3600
            except Exception as e:
                train_loss = 0
                eval_loss = 0
                time_h = 0
                print(e)

            MMLU_acc_STEM = mmlu_log["cat_acc"]["STEM"]
            MMLU_acc_humanities = mmlu_log["cat_acc"]["humanities"]
            MMLU_acc_socialsciences = mmlu_log["cat_acc"]["social sciences"]
            MMLU_acc_other = mmlu_log["cat_acc"]["other (business, health, misc.)"]
            MMLU_acc = mmlu_log["average_acc"]

            items.extend(
                [
                    train_loss,
                    eval_loss,
                    time_h,
                    MMLU_acc_STEM,
                    MMLU_acc_humanities,
                    MMLU_acc_socialsciences,
                    MMLU_acc_other,
                    MMLU_acc,
                ]
            )
            data.append(items)

        except Exception as e:
            print(e)

    write_csv(titles, data, "/workspace/summary.csv")
