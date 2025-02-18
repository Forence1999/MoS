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
        "MMLU test acc",
        "MMLU test loss",
    ]
    data = []
    timespan = args.ts if args.ts != -1 else 1000000
    time_threshold = datetime.now() - timedelta(hours=timespan)

    log_paths = [
        i
        for i in Path("./output/").rglob("eval_test_results.json")
        if (
            ("_eval" in i.parent.name)
            and (i.stat().st_mtime > time_threshold.timestamp())
        )
    ]
    log_paths = sorted(log_paths, reverse=False)
    for log_path in log_paths:
        print(str(log_path))
        items = []
        log_path = Path(log_path).absolute()
        subfolder_name = log_path.parent.name
        items.append(log_path.parent.name)

        log = json.load(open(log_path, "r"))
        (
            MMLU_test_acc,
            MMLU_test_loss,
        ) = [None] * 2
        try:
            MMLU_test_acc = log["mmlu_test_accuracy"]
            MMLU_test_loss = log["mmlu_test_loss"]
        except:
            print("No MMLU test acc/loss found in {}".format(log_path))

        items.extend(
            [
                MMLU_test_acc,
                MMLU_test_loss,
            ]
        )
        data.append(items)

    write_csv(titles, data, "summary.csv")
