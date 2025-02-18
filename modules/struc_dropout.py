# -*- coding: utf-8 _*-
# @License: MIT Licence
# @Author: Forence
# @Contact: wang00sheng@gmail.com
# @GitHub: https://github.com/Forence1999
# @Time: 18/10/2023
# @Description: structurally dropout LoRA modules


import os
import sys
import time
import random
import warnings
import numpy as np
from pathlib import Path
from peft.tuners.lora import LoraModel, Linear4bit
import types
import torch
import torch.functional as F


def LoraLinear4bit_forward(self, x: torch.Tensor):
    result = super(Linear4bit, self).forward(x)

    if self.disable_adapters or self.active_adapter not in self.lora_A.keys():
        return result
    elif self.r[self.active_adapter] > 0:
        result = result.clone()
        if not torch.is_autocast_enabled():
            assert False, "LoRA is not supported without autocast"
            # expected_dtype = result.dtype
            # x = x.to(self.lora_A[self.active_adapter].weight.dtype)
            # output = (
            #     self.lora_B[self.active_adapter](
            #         self.lora_A[self.active_adapter](
            #             self.lora_dropout[self.active_adapter](x)
            #         )
            #     ).to(expected_dtype)
            #     * self.scaling[self.active_adapter]
            # )
        else:
            # generate mask # it doesn't matter if mask is 0 for all elements.
            mask = torch.ones(
                self.r[self.active_adapter],
                requires_grad=False,
                device=self.lora_A[self.active_adapter].weight.device,
            )
            mask = self.lora_dropout[self.active_adapter](mask)
            # process lora_A and lora_B  # bool() is necessary for rescaling
            lora_A = self.lora_A[self.active_adapter].weight.t() * mask
            lora_B = (self.lora_B[self.active_adapter].weight * mask.bool()).t()

            output = (  # disable dropout
                torch.matmul(
                    torch.matmul(x, lora_A),
                    lora_B,
                )
                * self.scaling[self.active_adapter]
            )

            # # original implementation
            # output = (
            #     self.lora_B[self.active_adapter](
            #         self.lora_A[self.active_adapter](
            #             self.lora_dropout[self.active_adapter](x)
            #         )
            #     )
            #     * self.scaling[self.active_adapter]
            # )
        result += output
    return result


def LoraLinear4bit_reduce_rank_forward(self, x: torch.Tensor):
    result = super(Linear4bit, self).forward(x)

    if self.disable_adapters or self.active_adapter not in self.lora_A.keys():
        return result
    elif self.r[self.active_adapter] > 0:
        result = result.clone()
        if not torch.is_autocast_enabled():
            assert False, "LoRA is not supported without autocast"
            # expected_dtype = result.dtype
            # x = x.to(self.lora_A[self.active_adapter].weight.dtype)
            # output = (
            #     self.lora_B[self.active_adapter](
            #         self.lora_A[self.active_adapter](
            #             self.lora_dropout[self.active_adapter](x)
            #         )
            #     ).to(expected_dtype)
            #     * self.scaling[self.active_adapter]
            # )
        else:
            # generate mask # it doesn't matter if mask is 0 for all elements.
            rank = self.r[self.active_adapter]
            mask = torch.zeros(
                rank,
                requires_grad=False,
                device=self.lora_A[self.active_adapter].weight.device,
            )
            mask[: self.new_rank] = 1
            mask = mask / sum(mask) * rank
            # mask = self.lora_dropout[self.active_adapter](mask)
            # process lora_A and lora_B  # bool() is necessary for rescaling
            lora_A = self.lora_A[self.active_adapter].weight.t() * mask
            lora_B = (self.lora_B[self.active_adapter].weight * mask.bool()).t()

            output = (  # disable dropout
                torch.matmul(
                    torch.matmul(x, lora_A),
                    lora_B,
                )
                * self.scaling[self.active_adapter]
            )

            # # original implementation
            # output = (
            #     self.lora_B[self.active_adapter](
            #         self.lora_A[self.active_adapter](
            #             self.lora_dropout[self.active_adapter](x)
            #         )
            #     )
            #     * self.scaling[self.active_adapter]
            # )
        result += output
    return result


def dropout_struc(model):
    print("-" * 20, "Modify LoRA forward for structural Dropout", "-" * 20)
    for k, v in model.named_modules():
        if isinstance(v, Linear4bit):
            print(k)
            v.forward = types.MethodType(LoraLinear4bit_forward, v)
    print("-" * 20, "Finish modifying LoRA forward for structural Dropout", "-" * 20)


def reduce_LoRA_r(model, rank=-1):
    if rank == -1:
        warnings.warn(
            "rank is -1, which means no reduction. LoRA_rank_reduction will be skipped."
        )
        return
    print("-" * 20, "Modify LoRA rank for structural Dropout", "-" * 20)
    for k, v in model.named_modules():
        if isinstance(v, Linear4bit):
            print(k)
            v.new_rank = rank
            v.forward = types.MethodType(LoraLinear4bit_reduce_rank_forward, v)
    print("-" * 20, "Finish modifying LoRA rank for structural Dropout", "-" * 20)


if __name__ == "__main__":
    print("Hello World!")
