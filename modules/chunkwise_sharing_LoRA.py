import os
from peft.tuners.lora import Linear4bit
import types
from peft import LoraConfig
from peft.tuners.lora import LoraLayer
import torch
import torch.nn as nn
import torch.functional as F
from torch.nn.modules.dropout import Dropout
import math
from typing import Any, Dict, List, Optional
from peft import (
    get_peft_model_state_dict,
    PromptLearningConfig,
)
from peft.utils import (
    SAFETENSORS_WEIGHTS_NAME,
    WEIGHTS_NAME,
)
from safetensors.torch import save_file as safe_save_file
import bitsandbytes as bnb
from dataclasses import asdict, dataclass, field
import dataclasses
from copy import deepcopy
import transformers
import math


@dataclass
class LoraConfig_Chunkwise_Sharing(LoraConfig):
    enable_lora_vec: bool = field(
        default=False, metadata={"help": "Enable LoRA combination vector."}
    )
    enable_lora_bias: bool = field(
        default=False, metadata={"help": "Enable LoRA bias."}
    )
    enable_lora_rotation: bool = field(
        default=False, metadata={"help": "Enable LoRA rotation."}
    )
    reduce_lora_A_x: int = field(
        default=1, metadata={"help": "Multiples of LoRA_A sharing."}
    )
    reduce_lora_B_x: int = field(
        default=1, metadata={"help": "Multiples of LoRA_B sharing."}
    )
    init2zero_via_vec: bool = field(
        default=False, metadata={"help": "Initialize LoRA via vector."}
    )
    # lora_A_chunk_size: int = field(
    #     default=None, metadata={"help": "Chunk size along LoRA_B direction."}
    # )
    # lora_B_chunk_size: int = field(
    #     default=None, metadata={"help": "Chunk size along LoRA_B direction."}
    # )
    # lora_A_shift_size: int = field(
    #     default=None, metadata={"help": "Shift size along LoRA_A direction."}
    # )
    # lora_B_shift_size: int = field(
    #     default=None, metadata={"help": "Shift size along LoRA_B direction."}
    # )
    # lora_chunk_num: int = field(
    #     default=None, metadata={"help": "Total number of chunks."}
    # )


# generate chunkwise sharing LoRA config
def gen_chunkwise_sharing_lora_config(chunk_config, lora_config):
    config = LoraConfig_Chunkwise_Sharing(
        enable_lora_vec=chunk_config.enable_lora_vec,
        enable_lora_bias=chunk_config.enable_lora_bias,
        enable_lora_rotation=chunk_config.enable_lora_rotation,
        reduce_lora_A_x=chunk_config.reduce_lora_A_x,
        reduce_lora_B_x=chunk_config.reduce_lora_B_x,
        init2zero_via_vec=chunk_config.init2zero_via_vec,
        **asdict(lora_config),
    )

    assert (
        chunk_config.init2zero_via_vec == False
    ), "init2zero_via_vec is banned. The relevant code is not checked yet."
    # assert (
    #     chunk_config.enable_lora_vec == False
    # ), "enable_lora_vec is banned. The relevant code is not checked yet."
    assert (
        chunk_config.enable_lora_bias == False
    ), "enable_lora_bias is banned. The relevant code is not checked yet."

    return config


def LoraLayer_update_layer(
    self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, chunk_config
):
    assert chunk_config[
        "enable_lora_vec"
    ], "enable_lora_vec should be True in this version."

    if r <= 0:
        raise ValueError(
            f"`r` should be a positive integer value but the value passed is {r}"
        )

    self.r[adapter_name] = r
    self.lora_alpha[adapter_name] = lora_alpha
    if lora_dropout > 0.0:
        lora_dropout_layer = nn.Dropout(p=lora_dropout)
    else:
        lora_dropout_layer = nn.Identity()

    self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
    # Actual trainable parameters
    if r > 0:
        self.lora_A.update(
            nn.ModuleDict(
                {
                    adapter_name: nn.Linear(
                        chunk_config["lora_A_chunk_size"], r, bias=False
                    )
                }
            )
        )
        self.lora_B.update(
            nn.ModuleDict(
                {
                    adapter_name: nn.Linear(
                        r, chunk_config["lora_B_chunk_size"], bias=False
                    )
                }
            )
        )
        self.lora_A_vec.update(
            nn.ParameterDict(
                {
                    adapter_name: nn.Parameter(
                        torch.empty(r, chunk_config["reduce_lora_A_x"]),
                        requires_grad=chunk_config["enable_lora_vec"],
                    )
                }
            )
        )
        self.lora_B_vec.update(
            nn.ParameterDict(
                {
                    adapter_name: nn.Parameter(
                        torch.empty(r, chunk_config["reduce_lora_B_x"]),
                        requires_grad=chunk_config["enable_lora_vec"],
                    )
                }
            )
        )
        # self.lora_bias.update(
        #     nn.ParameterDict(
        #         {
        #             adapter_name: nn.Parameter(
        #                 torch.empty(1, chunk_config["lora_chunk_num"]),
        #                 requires_grad=chunk_config["enable_lora_bias"],
        #             )
        #         }
        #     )
        # )
        self.scaling[adapter_name] = lora_alpha / r
    if init_lora_weights:
        self.reset_lora_parameters(adapter_name, chunk_config)

    weight = getattr(self, "weight", None)
    if weight is not None:
        # the layer is already completely initialized, this is an update
        if weight.dtype.is_floating_point or weight.dtype.is_complex:
            self.to(weight.device, dtype=weight.dtype)
        else:
            self.to(weight.device)
    self.set_adapter(self.active_adapters)


def LoraLayer_reset_lora_parameters(self, adapter_name, chunk_config):
    assert (
        adapter_name in self.lora_A.keys()
    ), "adapter_name is not in self.lora_A.keys()"
    if chunk_config["init2zero_via_vec"]:
        assert (
            False
        ), "init2zero_via_vec is banned. The relevant code is not checked yet."
        nn.init.kaiming_uniform_(self.lora_A[adapter_name].weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_B[adapter_name].weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_vec[adapter_name])
        nn.init.zeros_(self.lora_bias[adapter_name])
    else:
        # initialize A the same way as the default for nn.Linear and B to zero
        nn.init.kaiming_uniform_(self.lora_A[adapter_name].weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B[adapter_name].weight)
        nn.init.ones_(self.lora_A_vec[adapter_name])
        nn.init.ones_(self.lora_B_vec[adapter_name])
        # nn.init.zeros_(self.lora_bias[adapter_name])


# re_init Linear4bit
def Linear4bit__Re_Init__(
    self,
    adapter_name,
    base_layer,
    r,
    lora_alpha,
    lora_dropout,
    chunk_config,
    **kwargs,
):
    # update chunk_config
    in_features = base_layer.in_features
    out_features = base_layer.out_features
    chunk_config = deepcopy(asdict(chunk_config))
    chunk_config["lora_A_size"] = in_features
    chunk_config["lora_B_size"] = out_features
    assert (
        isinstance(chunk_config["reduce_lora_A_x"], int)
        and isinstance(chunk_config["reduce_lora_B_x"], int)
        and chunk_config["reduce_lora_A_x"] > 0
        and chunk_config["reduce_lora_B_x"] > 0
        # and chunk_config["lora_A_size"] % chunk_config["reduce_lora_A_x"] == 0
        # and chunk_config["lora_B_size"] % chunk_config["reduce_lora_B_x"] == 0
    ), "The reduction multipliers should be positive integars."  # , \
    # and The chunk size should be divisible by the original size on both A and B low-rank matrices."
    assert not (
        chunk_config["init2zero_via_vec"] and (not chunk_config["enable_lora_vec"])
    ), "init2zero_via_vec is banned when enable_lora_vec is False."
    chunk_config["lora_A_chunk_size"] = math.ceil(
        chunk_config["lora_A_size"] / chunk_config["reduce_lora_A_x"]
    )
    chunk_config["lora_B_chunk_size"] = math.ceil(
        chunk_config["lora_B_size"] / chunk_config["reduce_lora_B_x"]
    )
    # chunk_config["lora_chunk_num"] = (
    #     chunk_config["reduce_lora_A_x"] * chunk_config["reduce_lora_B_x"]
    # )
    # chunk_config["lora_A_shift_size"] = max(
    #     chunk_config["lora_B_chunk_size"] // chunk_config["reduce_lora_A_x"],
    #     1,
    # )
    # chunk_config["lora_B_shift_size"] = max(
    #     chunk_config["lora_A_chunk_size"] // chunk_config["reduce_lora_B_x"],
    #     1,
    # )
    self.chunk_config = chunk_config

    # initialize the LoRA layer from the original code

    # bnb.nn.Linear4bit.__init__(
    #     self,
    #     in_features,
    #     out_features,
    #     bias=kwargs.get("bias", True),
    #     compute_dtype=kwargs.get("compute_dtype", torch.float32),
    #     compress_statistics=kwargs.get("compress_statistics", True),
    #     quant_type=kwargs.get("quant_type", "nf4"),
    # )
    torch.nn.Module.__init__(self)
    LoraLayer.__init__(
        self,
        in_features=chunk_config["lora_A_chunk_size"],
        out_features=chunk_config["lora_B_chunk_size"],
    )
    self.base_layer = base_layer

    init_lora_weights = kwargs.pop("init_lora_weights", True)
    # Freezing the pre-trained weight matrix
    self.lora_A_vec = nn.ParameterDict()
    self.lora_B_vec = nn.ParameterDict()
    self.lora_bias = nn.ParameterDict()
    self.update_layer = types.MethodType(LoraLayer_update_layer, self)
    self.reset_lora_parameters = types.MethodType(LoraLayer_reset_lora_parameters, self)
    self.update_layer(
        adapter_name,
        r,
        lora_alpha,
        lora_dropout,
        init_lora_weights,
        chunk_config=chunk_config,
    )
    self.set_adapter(adapter_name)


def de_reduce(chunk, reduce_x, r, enable_rotation, dim_len=None, vec_chunk=None):
    device = chunk.device
    chunks = chunk.unsqueeze(-1).repeat(1, 1, reduce_x)
    # neglect vec and bias for lora_A_chunks
    # lora_A_chunks = lora_A_chunks * vec.unsqueeze(1)
    # lora_A = torch.cat(torch.split(lora_A_chunks, 1, dim=-1), dim=-2).squeeze(
    #     -1
    # )
    if enable_rotation:
        with torch.no_grad():
            idx = (
                torch.arange(r, device=device, requires_grad=False)
                .unsqueeze(-1)
                .unsqueeze(-1)
                .expand_as(chunks)
            )
            idx_delta = (
                torch.arange(reduce_x, device=device, requires_grad=False)
                .unsqueeze(0)
                .unsqueeze(0)
            )
            idx = (idx + idx_delta) % r
        chunks = torch.gather(chunks, 0, idx)

    if vec_chunk is not None:
        chunks = chunks * vec_chunk.unsqueeze(1)
    res = torch.cat(torch.split(chunks, 1, dim=-1), dim=-2).squeeze(-1).t()

    if dim_len is not None:
        res = res[:dim_len]

    return res


# modify the forward function of LoraLinear4bit
def LoraLinear4bit_Chunkwise_Sharing_forward(
    self, x: torch.Tensor, *args, **kwargs
) -> torch.Tensor:
    """This func will replace the original forward function of LoraLinear4bit.
    It will perform Shared Rank for LoRA modules with the following features:
        1. Still save the full rank matrix
        2. Using mask to disable the former part of rank matrix
        3. Add shared matrix and mask outside the LoRA module
    """
    result = self.base_layer.forward(x, *args, **kwargs)

    assert len(self.active_adapters) == 1, "Only support single adapter now."
    active_adapter = self.active_adapters[0]

    if (
        self.disable_adapters
        or self.merged
        or (active_adapter not in self.lora_A.keys())
    ):
        assert (
            False
        ), "Disabling/Merged LoRA is not supported, or the active_adapter is not in the LoRA_A keys."
    elif self.r[active_adapter] > 0:
        result = result.clone()

        # prepare variables
        device = self.base_layer.weight.device
        reduce_lora_A_x = self.chunk_config["reduce_lora_A_x"]
        reduce_lora_B_x = self.chunk_config["reduce_lora_B_x"]
        r = self.r[active_adapter]
        # vec = self.lora_vec[active_adapter]
        # bias = self.lora_bias[active_adapter]
        lora_A_chunk = self.lora_A[active_adapter].weight
        lora_B_chunk = self.lora_B[active_adapter].weight
        lora_A_vec = self.lora_A_vec[active_adapter]
        lora_B_vec = self.lora_B_vec[active_adapter]
        enable_lora_rotation = self.chunk_config["enable_lora_rotation"]

        # # operate on the A and B seperately
        # gen lora_A/B
        lora_A = de_reduce(
            lora_A_chunk,
            reduce_lora_A_x,
            r,
            enable_lora_rotation,
            dim_len=self.chunk_config["lora_A_size"],
            vec_chunk=lora_A_vec,
        )
        lora_B = de_reduce(
            lora_B_chunk.t(),
            reduce_lora_B_x,
            r,
            enable_lora_rotation,
            dim_len=self.chunk_config["lora_B_size"],
            vec_chunk=lora_B_vec,
        ).t()

        # forward
        requires_conversion = not torch.is_autocast_enabled()
        if requires_conversion:
            expected_dtype = result.dtype
            x = x.to(lora_A.weight.dtype)

        output = (
            torch.matmul(
                torch.matmul(self.lora_dropout[active_adapter](x), lora_A),
                lora_B,
            )
            * self.scaling[active_adapter]
        )

        if requires_conversion:
            output = output.to(expected_dtype)

        # # original implementation
        # output = (
        #     self.lora_B[active_adapter](
        #         self.lora_A[active_adapter](
        #             self.lora_dropout[active_adapter](x)
        #         )
        #     )
        #     * self.scaling[active_adapter]
        # )
        result += output

    return result


def save_chunk_lora(trainer):
    print("Saving chunk-wise shared PEFT checkpoint manually...")
    model = trainer.model
    args = trainer.args

    def touch(fname, times=None):
        with open(fname, "a"):
            os.utime(fname, times)

    touch(os.path.join(args.output_dir, "completed"))

    chunk_config = getattr(model, "chunk_config", None)
    if chunk_config is None:
        print("Warning: chunk_config is None. Skip saving.")
        return

    ckpt_pth = os.path.join(args.output_dir, "adapter_model.bin")
    adapter_name = model.active_adapters[0]

    # save only the trainable weights
    output_state_dict = get_peft_model_state_dict(
        model,
        state_dict=None,
        adapter_name=adapter_name,
    )
    new_state_dict = {}
    for k, v in output_state_dict.items():  # dereduce the LoRA_A/B
        if "lora_A.weight" in k:
            lora_A_vec = output_state_dict[k.replace("lora_A.weight", "lora_A_vec")]
            dim_len = (
                4096
                if any(
                    [
                        i in k
                        for i in [
                            "q_proj",
                            "k_proj",
                            "v_proj",
                            "o_proj",
                            "up_proj",
                            "gate_proj",
                        ]
                    ]
                )
                else 11008
            )
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                v = de_reduce(
                    v,
                    chunk_config.reduce_lora_A_x,
                    chunk_config.r,
                    chunk_config.enable_lora_rotation,
                    dim_len=dim_len,
                    vec_chunk=lora_A_vec,
                ).t()
            new_state_dict[k] = v

        elif "lora_B.weight" in k:
            lora_B_vec = output_state_dict[k.replace("lora_B.weight", "lora_B_vec")]
            dim_len = (
                4096
                if any(
                    [
                        i in k
                        for i in [
                            "q_proj",
                            "k_proj",
                            "v_proj",
                            "o_proj",
                            "down_proj",
                        ]
                    ]
                )
                else 11008
            )
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                v = de_reduce(
                    v.t(),
                    chunk_config.reduce_lora_B_x,
                    chunk_config.r,
                    chunk_config.enable_lora_rotation,
                    dim_len=dim_len,
                    vec_chunk=lora_B_vec,
                )
            new_state_dict[k] = v

    torch.save(new_state_dict, ckpt_pth)


# share LoRA chunkwisely
def share_lora_chunkwisely(model, chunk_config):
    """
    This function is used to re_initialize the Linear4bit class with the LoRA being shared chunk-wisely.
    The config of the replaced Linear4bit module has the priority over the argumented parameters.
    """

    print("-" * 20, f"Sharing LoRA Rank Chunk-wisely... ", "-" * 20)
    for k, v in model.named_modules():
        if isinstance(v, Linear4bit):
            print(f"Adding shared rank matrix to {k}...")

            # generate chunkwise sharing LoRA config
            adapter_name = v.active_adapter[0]
            base_layer = v.base_layer
            kwargs = {
                "adapter_name": adapter_name,
                "base_layer": base_layer,
                "r": v.r[adapter_name],
                "lora_alpha": v.lora_alpha[adapter_name],
                "lora_dropout": v.lora_dropout[adapter_name].p
                if isinstance(v.lora_dropout[adapter_name], Dropout)
                else 0.0,
                "fan_in_fan_out": model.active_peft_config.fan_in_fan_out,
                "init_lora_weights": True,
                "compute_dtype": base_layer.compute_dtype,
                "compress_statistics": base_layer.weight.compress_statistics,
                "quant_type": base_layer.weight.quant_type,
            }
            dtype = v.lora_A[adapter_name].weight.dtype
            device = v.lora_A[adapter_name].weight.device
            # re_init Linear4bit module
            Linear4bit__Re_Init__(v, chunk_config=chunk_config, **kwargs)
            # change to the same device and dtype as the original Linear4bit module
            v.to(device=device, dtype=dtype)
            if hasattr(v, "lora_A_vec"):
                v.lora_A_vec.to(dtype=torch.float32)
            if hasattr(v, "lora_B_vec"):
                v.lora_B_vec.to(dtype=torch.float32)

            # modify the forward function of LoraLinear4bit
            v.forward = types.MethodType(LoraLinear4bit_Chunkwise_Sharing_forward, v)

    # modify save_pretrained function: do not need any change
    # model.save_pretrained = types.MethodType(PEFT_save_pretrained, model)
    print(
        "-" * 20,
        "Finish the modification of Sharing LoRA Rank Chunk-wisely. ",
        "-" * 20,
    )


if __name__ == "__main__":
    print("Hello World!")
