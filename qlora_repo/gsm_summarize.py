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
        "eval_loss (invalid)",
        "train_loss",
        "gsm_test_emsocre",
        "time/s",
    ]
    data = []
    timespan = args.ts if args.ts != -1 else 1000000
    time_threshold = datetime.now() - timedelta(hours=timespan)

    log_paths = [
        i
        for i in Path("./output/").rglob("trainer_state.json")
        if (
            ("checkpoint" not in i.parent.name)
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

        logs = json.load(open(log_path, "r"))["log_history"]

        (
            gsm_test_emsocre,
            eval_loss,
            train_loss,
            time_h,
        ) = [None] * 4
        try:
            eval_, train_, GSM_test = logs[-3:]
            gsm_test_emsocre = GSM_test["gsm_test_emsocre"]
            eval_loss = eval_["eval_loss"]
            train_loss = train_["train_loss"]
            time_h = train_["train_runtime"] / 3600
        except:
            pass

        items.extend(
            [
                gsm_test_emsocre,
                eval_loss,
                train_loss,
                time_h,
            ]
        )
        data.append(items)

    write_csv(titles, data, "summary.csv")
