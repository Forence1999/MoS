# -*- coding: utf-8 _*-
# @License: MIT Licence
# @Author: Forence
# @Contact: wang00sheng@gmail.com
# @GitHub: https://github.com/Forence1999
# @Time: 19/11/2023
# @Description:


import os
import sys
import time
import random
import warnings
import numpy as np
from pathlib import Path
import evaluate
from modules.open_instruct_utils import generate_completions, KeyWordsCriteria
import re
import torch
from tqdm import tqdm


def calculate_emscore(predictions, targets):
    exact_match = evaluate.load("exact_match")
    em_score = exact_match.compute(
        predictions=predictions,
        references=targets,
        ignore_case=True,
        ignore_punctuation=True,
    )["exact_match"]

    return em_score


def eval_gsm(model, tokenizer, dataset):
    # dataset = dataset.select(range(10))
    # data_loader = trainer.get_eval_dataloader(dataset)
    # source_max_len = trainer.data_collator.source_max_len
    # trainer.data_collator.source_max_len = args.val_source_max_len
    # predict_with_generate_data = trainer.data_collator.predict_with_generate
    # trainer.data_collator.predict_with_generate = True
    # predict_with_generate = trainer.args.predict_with_generate
    # trainer.args.predict_with_generate = True

    # dataset = dataset.select(range(5))
    new_line_token = tokenizer.encode("\n", add_special_tokens=False)[
        -1
    ]  # get the last token because the tokenizer may add space tokens at the start.
    stop_id_sequences = [[new_line_token]]
    stopping_criteria = [KeyWordsCriteria(stop_id_sequences)]

    model.eval()
    preds = []
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        outputs = generate_completions(
            model=model,
            tokenizer=tokenizer,
            prompts=dataset["input"],
            max_new_tokens=512,
            batch_size=1,
            stop_id_sequences=[[new_line_token]],
            do_sample=False,
        )

    # post-process outputs
    predictions = []
    for output in outputs:
        # replace numbers like `x,xxx` with `xxxx`
        output = re.sub(r"(\d),(\d)", r"\1\2", output)
        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", output)
        if numbers:
            predictions.append(numbers[-1])
        else:
            predictions.append(output)

    targets = dataset["output"]
    emsocre = calculate_emscore(predictions, targets)

    # trainer.data_collator.source_max_len = source_max_len
    # trainer.args.predict_with_generate = predict_with_generate
    # trainer.data_collator.predict_with_generate = predict_with_generate_data

    return {
        "gsm_test_emsocre": emsocre,
    }


def idle():
    # with torch.cuda.amp.autocast():
    #     prediction_output = trainer.predict(
    #         test_dataset=gsm_dataset,
    #         metric_key_prefix="predict",
    #         do_sample=False,
    #         max_new_tokens=512,
    #         num_return_sequences=1,
    #     )

    # predictions_beam = np.where(
    #     prediction_output.predictions != -100,
    #     prediction_output.predictions,
    #     trainer.tokenizer.pad_token_id,
    # )
    # preds_batch_beam = trainer.tokenizer.batch_decode(
    #     predictions_beam, skip_special_tokens=False
    # )

    # for batch in tqdm(data_loader, total=len(data_loader)):
    #     with torch.cuda.amp.autocast():
    #         (_, tokens, _) = trainer.prediction_step(
    #             trainer.model,
    #             batch,
    #             prediction_loss_only=False,
    #             do_sample=False,
    #             max_new_tokens=512,
    #             stopping_criteria=stopping_criteria,
    #         )
    #     preds.append(tokens)
    # preds = [trainer.tokenizer.decode(i[0], skip_special_tokens=True) for i in preds]

    pass


if __name__ == "__main__":
    print("Hello World!")
