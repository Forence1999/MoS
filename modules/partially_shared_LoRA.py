# -*- coding: utf-8 _*-
# @License: MIT Licence
# @Author: Forence
# @Contact: wang00sheng@gmail.com
# @GitHub: https://github.com/Forence1999
# @Time: 03/11/2023
# @Description:


import os
import sys
import time
import random
import warnings
import numpy as np
from pathlib import Path
from peft.tuners.lora import LoraModel, Linear4bit
import types
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, PeftModel
from peft.tuners.lora import LoraLayer
from dataclasses import asdict, dataclass, field, replace
import torch
import torch.nn as nn
import torch.functional as F
import math
from typing import Any, Dict, List, Optional, Union
from peft import (
    get_peft_model_state_dict,
    PromptLearningConfig,
)
from peft.utils import (
    SAFETENSORS_WEIGHTS_NAME,
    WEIGHTS_NAME,
)
from safetensors.torch import save_file as safe_save_file


@dataclass
class LoraConfig_PartiallyShared(LoraConfig):
    r_shared: int = field(
        default=0, metadata={"help": "Shared LoRA attention dimension"}
    )


def LoraLinear4bitSharedRank_forward(self, x: torch.Tensor):
    """This func will replace the original forward function of LoraLinear4bit.
    It will perform Shared Rank for LoRA modules with the following features:
        1. Still save the full rank matrix
        2. Using mask to disable the former part of rank matrix
        3. Add shared matrix and mask outside the LoRA module
    """
    result = super(Linear4bit, self).forward(x)

    if self.disable_adapters or self.active_adapter not in self.lora_A.keys():
        assert False, "Disabling LoRA is not supported"
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
            assert hasattr(self, "mask"), "self.mask is not defined"
            assert hasattr(self, "lora_A_shared"), "self.lora_A_shared is not defined"
            assert hasattr(self, "lora_B_shared"), "self.lora_B_shared is not defined"

            mask = (self.mask).detach()
            lora_A = self.lora_A[self.active_adapter].weight.t() * mask
            lora_B = (self.lora_B[self.active_adapter].weight * mask).t()

            drop_x = self.lora_dropout[self.active_adapter](x)
            output = torch.matmul(torch.matmul(drop_x, lora_A), lora_B)
            output += self.lora_B_shared(self.lora_A_shared(drop_x))
            output *= self.scaling[self.active_adapter]
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


def PEFT_save_pretrained(
    self,
    save_directory: str,
    safe_serialization: bool = False,
    selected_adapters: Optional[List[str]] = None,
    **kwargs: Any,
):
    r"""
    This function saves the adapter model and the adapter configuration files to a directory, so that it can be
    reloaded using the [`LoraModel.from_pretrained`] class method, and also used by the [`LoraModel.push_to_hub`]
    method.

    Args:
        save_directory (`str`):
            Directory where the adapter model and configuration files will be saved (will be created if it does not
            exist).
        kwargs (additional keyword arguments, *optional*):
            Additional keyword arguments passed along to the `push_to_hub` method.
    """
    if os.path.isfile(save_directory):
        raise ValueError(
            f"Provided path ({save_directory}) should be a directory, not a file"
        )

    if selected_adapters is None:
        selected_adapters = list(self.peft_config.keys())
    else:
        if any(
            selected_adapter_name not in list(self.peft_config.keys())
            for selected_adapter_name in selected_adapters
        ):
            raise ValueError(
                f"You passed an invalid `selected_adapters` arguments, current supported adapter names are"
                f" {list(self.peft_config.keys())} - got {selected_adapters}."
            )

    os.makedirs(save_directory, exist_ok=True)
    self.create_or_update_model_card(save_directory)

    for adapter_name in selected_adapters:
        peft_config = self.peft_config[adapter_name]
        # save only the trainable weights
        output_state_dict = get_peft_model_state_dict(
            self, state_dict=kwargs.get("state_dict", None), adapter_name=adapter_name
        )
        # add shared rank matrix
        output_state_dict[
            f"base_model.model.lora_A_shared.weight"
        ] = self.base_model.model.lora_A_shared.weight.data
        output_state_dict[
            f"base_model.model.lora_B_shared.weight"
        ] = self.base_model.model.lora_B_shared.weight.data
        output_dir = (
            os.path.join(save_directory, adapter_name)
            if adapter_name != "default"
            else save_directory
        )
        os.makedirs(output_dir, exist_ok=True)

        if safe_serialization:
            safe_save_file(
                output_state_dict,
                os.path.join(output_dir, SAFETENSORS_WEIGHTS_NAME),
                metadata={"format": "pt"},
            )
        else:
            torch.save(output_state_dict, os.path.join(output_dir, WEIGHTS_NAME))

        # save the config and change the inference mode to `True`
        if peft_config.base_model_name_or_path is None:
            peft_config.base_model_name_or_path = (
                self.base_model.__dict__.get("name_or_path", None)
                if isinstance(peft_config, PromptLearningConfig)
                else self.base_model.model.__dict__.get("name_or_path", None)
            )
        inference_mode = peft_config.inference_mode
        peft_config.inference_mode = True

        if peft_config.task_type is None:
            # deal with auto mapping
            base_model_class = self._get_base_model_class(
                is_prompt_tuning=isinstance(peft_config, PromptLearningConfig)
            )
            parent_library = base_model_class.__module__

            auto_mapping_dict = {
                "base_model_class": base_model_class.__name__,
                "parent_library": parent_library,
            }
        else:
            auto_mapping_dict = None

        peft_config.save_pretrained(output_dir, auto_mapping_dict=auto_mapping_dict)
        peft_config.inference_mode = inference_mode


def share_lora_rank(model, rank_shared=-1):
    assert (
        isinstance(rank_shared, int) and rank_shared >= 0
    ), "rank_shared must be a non-negative integer."

    if rank_shared == 0:
        warnings.warn(
            "rank_shared is 0, which means no reduction. LoRA_rank_sharing will be skipped."
        )
        return model

    print("-" * 20, f"Sharing LoRA rank ({rank_shared})... ", "-" * 20)
    # init shared rank matrix
    in_features, out_features, dtype, active_adapter = None, None, None, None
    for k, v in model.named_modules():
        if isinstance(v, Linear4bit) and k.split(".")[-1] in [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ]:
            in_features = v.in_features
            out_features = v.out_features
            active_adapter = v.active_adapter
            dtype = v.lora_A[active_adapter].weight.dtype
            break
    # in_features, out_features = 4096, 4096
    lora_A_shared = nn.Linear(in_features, rank_shared, bias=False)
    lora_B_shared = nn.Linear(rank_shared, out_features, bias=False)
    nn.init.kaiming_uniform_(lora_A_shared.weight, a=math.sqrt(5))
    nn.init.zeros_(lora_B_shared.weight)
    lora_A_shared.to(model.device).to(dtype)
    lora_B_shared.to(model.device).to(dtype)
    # add to model state dict
    model.base_model.model.lora_A_shared = lora_A_shared
    model.base_model.model.lora_B_shared = lora_B_shared
    # model.state_dict()[f"base_model.model.lora_A_shared.weight"] = lora_A_shared.weight
    # model.state_dict()[f"base_model.model.lora_B_shared.weight"] = lora_B_shared.weight
    for k, v in model.named_modules():
        if isinstance(v, Linear4bit) and k.split(".")[-1] in [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ]:
            print(f"Adding shared rank matrix to {k}...")
            # add shared rank matrix
            v.lora_A_shared = lora_A_shared
            v.lora_B_shared = lora_B_shared
            # add mask
            v.mask = torch.ones(
                model.active_peft_config.r,
                requires_grad=False,
                device=model.device,
                dtype=torch.bool,
            )
            v.mask[:rank_shared] = False
            # correct scaler
            v.forward = types.MethodType(
                LoraLinear4bitSharedRank_forward, v
            )  # modify forward function
    model.save_pretrained = types.MethodType(
        PEFT_save_pretrained, model
    )  # modify save_pretrained function
    print("-" * 20, "Finish the modification of Shared LoRA Rank", "-" * 20)


if __name__ == "__main__":
    print("Hello World!")
