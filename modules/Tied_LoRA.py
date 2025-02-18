import os
import math
import warnings
from copy import deepcopy

import types
from typing import Any, Dict, List, Optional

from peft import LoraConfig
from peft.tuners.lora import LoraLayer
from peft.tuners.lora import Linear4bit

import torch
import torch.nn as nn
import torch.functional as F
from torch.nn.modules.dropout import Dropout

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



def kaiming_uniform_(
    tensor: torch.Tensor,
    a: float = 0,
    mode: str = "fan_in",
    nonlinearity: str = "leaky_relu",
    dim: int = None,
):
    r"""Fills the input `Tensor` with values according to the method
    described in `Delving deep into rectifiers: Surpassing human-level
    performance on ImageNet classification` - He, K. et al. (2015), using a
    uniform distribution. The resulting tensor will have values sampled from
    :math:`\mathcal{U}(-\text{bound}, \text{bound})` where

    .. math::
        \text{bound} = \text{gain} \times \sqrt{\frac{3}{\text{fan\_mode}}}

    Also known as He initialization.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        a: the negative slope of the rectifier used after this layer (only
            used with ``'leaky_relu'``)
        mode: either ``'fan_in'`` (default) or ``'fan_out'``. Choosing ``'fan_in'``
            preserves the magnitude of the variance of the weights in the
            forward pass. Choosing ``'fan_out'`` preserves the magnitudes in the
            backwards pass.
        nonlinearity: the non-linear function (`nn.functional` name),
            recommended to use only with ``'relu'`` or ``'leaky_relu'`` (default).

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.kaiming_uniform_(w, mode='fan_in', nonlinearity='relu')
    """
    from torch.nn.init import _calculate_correct_fan, calculate_gain

    if torch.overrides.has_torch_function_variadic(tensor):
        assert False, "kaiming_uniform_: not checked yet."
        return torch.overrides.handle_torch_function(
            kaiming_uniform_,
            (tensor,),
            tensor=tensor,
            a=a,
            mode=mode,
            nonlinearity=nonlinearity,
        )

    if 0 in tensor.shape:
        warnings.warn("Initializing zero-element tensors is a no-op")
        return tensor
    fan = _calculate_correct_fan(tensor, mode) if dim is None else dim
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    with torch.no_grad():
        return tensor.uniform_(-bound, bound)


@dataclass
class LoraConfig_Tied_Sharing(LoraConfig):
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
    init_lora_A_vec_value: float = field(
        default=1, metadata={"help": "The initial value of VeRA's LoRA_A vector."}
    )
    init_lora_A_vec_std: float = field(
        default=None,
        metadata={
            "help": "The std value of MoS's LoRA_A vector's normal distribution initialization.\
            If provided, lora_A_vec will be initialized with normal distribution.\
            Otherwise, lora_A_vec will be initialized with constant distribution."
        },
    )
    valid_param_lora_r: int = field(
        default=1, metadata={"help": "The number of active LoRA pairs."}
    )
    valid_param_private_r: int = field(
        default=1, metadata={"help": "The number of private LoRA pairs."}
    )
    ft_mode: str = field(
        default="mos",
        metadata={
            "help": "Finetuning mode. Choose between 'mos', 'lora'.\
                mos: create a lora-pair pool, and randomly select `lora_r` pairs for each lora module."
        },
    )
    num_chunk_per_vec: int = field(
        default=None,
        metadata={"help": "Number of chunks per vector."},
    )


# generate chunkwise sharing LoRA config
def gen_tied_lora_config(tied_config, lora_config):
    config = LoraConfig_Tied_Sharing(
        enable_lora_vec=tied_config.enable_lora_vec,
        enable_lora_bias=tied_config.enable_lora_bias,
        enable_lora_rotation=tied_config.enable_lora_rotation,
        reduce_lora_A_x=tied_config.reduce_lora_A_x,
        reduce_lora_B_x=tied_config.reduce_lora_B_x,
        init2zero_via_vec=tied_config.init2zero_via_vec,
        init_lora_A_vec_value=tied_config.init_lora_A_vec_value,
        init_lora_A_vec_std=tied_config.init_lora_A_vec_std,
        valid_param_lora_r=tied_config.valid_param_lora_r,
        valid_param_private_r=tied_config.valid_param_private_r,
        num_chunk_per_vec=tied_config.num_chunk_per_vec,
        ft_mode=tied_config.ft_mode,
        **asdict(lora_config),
    )
    config.r = tied_config.lora_r  # set the rank for VeRA
    assert (
        tied_config.init2zero_via_vec == False
    ), "init2zero_via_vec is banned. The relevant code is not checked yet."
    assert (
        tied_config.enable_lora_bias == False
    ), "enable_lora_bias is banned. The relevant code is not checked yet."

    assert (
        tied_config.ft_mode == "mos"
    ), "Only support MoS in this version. Other modes have not been checked yet!"

    if tied_config.valid_param_private_r == tied_config.valid_param_lora_r:
        assert (
            tied_config.valid_param_lora_r == tied_config.lora_r
        ), "valid_param_lora_r should be equal to lora_r when valid_param_private_r == valid_param_lora_r."

    assert (tied_config.init_lora_A_vec_std is None) or (
        tied_config.init_lora_A_vec_std == 0
    ), "init_lora_A_vec_std is banned."
    assert tied_config.init_lora_A_vec_value == 1, "init_lora_A_vec_value should be 1."

    return config


def transform_chunks(mat, mask, num_chunk_per_vec):
    mat = mat[mask]
    sliced_tensors = [mat[i::num_chunk_per_vec] for i in range(num_chunk_per_vec)]
    result = torch.cat(sliced_tensors, dim=1)

    return result


def LoraLayer_update_layer(
    self,
    adapter_name,
    r,
    lora_alpha,
    lora_dropout,
    init_lora_weights,
    tied_config,
    lora_module,
):
    assert tied_config[
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
        chunk_pool_r = lora_module["lora_A"].out_features
        num_chunk_per_vec = tied_config["num_chunk_per_vec"]
        self.lora_A.update(nn.ModuleDict({adapter_name: lora_module["lora_A"]}))
        self.lora_B.update(nn.ModuleDict({adapter_name: lora_module["lora_B"]}))
        self.lora_A_vec.update(
            nn.ParameterDict(
                {
                    adapter_name: nn.Parameter(
                        torch.empty(r),
                        requires_grad=False,
                    )
                }
            )
        )
        self.lora_A_anti_vec.update(
            nn.ParameterDict(
                {
                    adapter_name: nn.Parameter(
                        torch.empty(r),
                        requires_grad=False,
                    )
                }
            )
        )
        self.lora_pair_mask.update(
            nn.ParameterDict(
                {
                    adapter_name: nn.Parameter(
                        torch.empty(r * num_chunk_per_vec, dtype=torch.long),
                        requires_grad=False,
                    )
                }
            )
        )
        self.lora_pair_mask_B.update(
            nn.ParameterDict(
                {
                    adapter_name: nn.Parameter(
                        torch.empty(r * num_chunk_per_vec, dtype=torch.long),
                        requires_grad=False,
                    )
                }
            )
        )
        self.scaling[adapter_name] = lora_alpha / r
    if init_lora_weights:
        self.reset_lora_parameters(adapter_name, tied_config)

    weight = getattr(self, "weight", None)
    if weight is not None:
        # the layer is already completely initialized, this is an update
        if weight.dtype.is_floating_point or weight.dtype.is_complex:
            self.to(weight.device, dtype=weight.dtype)
        else:
            self.to(weight.device)
    self.set_adapter(self.active_adapters)


def LoraLayer_reset_lora_parameters(self, adapter_name, tied_config):
    assert (
        adapter_name in self.lora_A.keys()
    ), "adapter_name is not in self.lora_A.keys()"

    num_chunk_per_vec = tied_config["num_chunk_per_vec"]
    with torch.no_grad():
        # initialize LoRA_A and LoRA_B
        # nn.init.kaiming_uniform_(self.lora_A[adapter_name].weight, a=math.sqrt(5))
        kaiming_uniform_(
            self.lora_A[adapter_name].weight,
            a=math.sqrt(5),
            dim=self.lora_A[adapter_name].weight.shape[-1] * num_chunk_per_vec,
        )
        nn.init.zeros_(self.lora_B[adapter_name].weight)

        # initialization lora_A_vec
        vec_init = (
            "constant" if tied_config["init_lora_A_vec_std"] is None else "normal"
        )
        if vec_init == "constant":  # Constant initialization of activated vector
            nn.init.constant_(
                self.lora_A_vec[adapter_name],
                val=tied_config["init_lora_A_vec_value"],
            )
        else:  # Gaussian initialization of activated vector
            nn.init.normal_(
                self.lora_A_vec[adapter_name],
                mean=tied_config["init_lora_A_vec_value"],
                std=tied_config["init_lora_A_vec_std"],
            )
        nn.init.constant_(
            self.lora_A_anti_vec[adapter_name],
            val=1.0 / tied_config["init_lora_A_vec_value"],
        )

        # initialize lora_pair_mask: it has two purposes:
        # 1. Select the active lora pairs
        # 2. For each lora pair, mapping the lora_A_vec * lora_A_anti_vec to 1.0

        if self.ft_mode == "mos":
            print(f"Initialization mode for LoRA module: MoS.")
            # chunk_pool_r = self.lora_A[adapter_name].out_features
            layer_idx = self.layer_idx
            private_start_idx = (
                (
                    tied_config["valid_param_lora_r"]
                    - tied_config["valid_param_private_r"]
                )
                * self.layer_num
                * tied_config["num_chunk_per_vec"]
            )
            private_indices = (
                private_start_idx
                + torch.arange(
                    layer_idx
                    * tied_config["valid_param_private_r"]
                    * tied_config["num_chunk_per_vec"],
                    (layer_idx + 1)
                    * tied_config["valid_param_private_r"]
                    * tied_config["num_chunk_per_vec"],
                ).int()
            )
            if private_start_idx == 0:
                assert (
                    tied_config["valid_param_private_r"]
                    == tied_config["valid_param_lora_r"]
                ), "`private_start_idx==0`, but tied_config['valid_param_private_r']!=tied_config['valid_param_lora_r']"
                shared_indices = torch.tensor([]).int()
            else:
                shared_indices = torch.randperm(private_start_idx)[
                    : (tied_config["r"] - tied_config["valid_param_private_r"])
                    * tied_config["num_chunk_per_vec"]
                ].int()
            indices = torch.concatenate([shared_indices, private_indices])
        # elif self.ft_mode == "lora":  # Original LoRA
        #     print(f"Initialization mode for LoRA module: Vanilla LoRA.")
        #     assert (
        #         tied_config["r"] == tied_config["valid_param_lora_r"]
        #     ), f"rank ({tied_config['r']}) and valid_param_lora_r ({tied_config['valid_param_lora_r']}) should be equal for vanilla lora."
        #     layer_idx = self.layer_idx
        #     indices = torch.arange(
        #         layer_idx * tied_config["r"] * tied_config["num_chunk_per_vec"],
        #         (layer_idx + 1) * tied_config["r"] * tied_config["num_chunk_per_vec"],
        #     ).int()
        self.lora_pair_mask[adapter_name].copy_(indices)
        assert (
            len(indices) == tied_config["r"] * tied_config["num_chunk_per_vec"]
        ), f"Error: lora_r != len(indices) when initializing.\
                    lora_pair_mask: {len(indices)}\n vs {tied_config['r']}"

        # initialize lora_pair_mask_B
        if self.ft_mode == "mos":
            print(f"Initialization mode for LoRA module: MoS.")
            # chunk_pool_r = self.lora_A[adapter_name].out_features
            layer_idx = self.layer_idx
            private_start_idx = (
                (
                    tied_config["valid_param_lora_r"]
                    - tied_config["valid_param_private_r"]
                )
                * self.layer_num
                * tied_config["num_chunk_per_vec"]
            )
            private_indices = (
                private_start_idx
                + torch.arange(
                    layer_idx
                    * tied_config["valid_param_private_r"]
                    * tied_config["num_chunk_per_vec"],
                    (layer_idx + 1)
                    * tied_config["valid_param_private_r"]
                    * tied_config["num_chunk_per_vec"],
                ).int()
            )
            if private_start_idx == 0:
                assert (
                    tied_config["valid_param_private_r"]
                    == tied_config["valid_param_lora_r"]
                ), "`private_start_idx==0`, but tied_config['valid_param_private_r']!=tied_config['valid_param_lora_r']"
                shared_indices = torch.tensor([]).int()
            else:
                shared_indices = torch.randperm(private_start_idx)[
                    : (tied_config["r"] - tied_config["valid_param_private_r"])
                    * tied_config["num_chunk_per_vec"]
                ].int()
            indices = torch.concatenate([shared_indices, private_indices])
        # elif self.ft_mode == "lora":  # Original LoRA
        #     print(f"Initialization mode for LoRA module: Vanilla LoRA.")
        #     assert (
        #         tied_config["r"] == tied_config["valid_param_lora_r"]
        #     ), f"rank ({tied_config['r']}) and valid_param_lora_r ({tied_config['valid_param_lora_r']}) should be equal for vanilla lora."
        #     layer_idx = int(self.layer_name.split(".")[-3])
        #     indices = torch.arange(
        #         layer_idx * tied_config["r"] * tied_config["num_chunk_per_vec"],
        #         (layer_idx + 1) * tied_config["r"] * tied_config["num_chunk_per_vec"],
        #     ).int()
        self.lora_pair_mask_B[adapter_name].copy_(indices)
        assert (
            len(indices) == tied_config["r"] * tied_config["num_chunk_per_vec"]
        ), f"Error: lora_r != len(indices) when initializing.\
                    lora_pair_mask_B: {len(indices)}\n vs {tied_config['r']}"


# re_init Linear4bit
def Linear4bit__Re_Init__(  # FIXME
    self,
    adapter_name,
    base_layer,
    r,
    lora_alpha,
    lora_dropout,
    tied_config,
    **kwargs,
):
    # update tied_config
    tied_config = deepcopy(asdict(tied_config))
    tied_config["lora_A_size"] = base_layer.in_features
    tied_config["lora_B_size"] = base_layer.out_features
    assert not (
        tied_config["init2zero_via_vec"] and (not tied_config["enable_lora_vec"])
    ), "init2zero_via_vec is banned when enable_lora_vec is False."
    self.tied_config = tied_config
    self.ft_mode = tied_config["ft_mode"]
    self.layer_num = kwargs.pop("layer_num", None)
    self.layer_idx = kwargs.pop("layer_idx", None)

    torch.nn.Module.__init__(self)
    LoraLayer.__init__(
        self,
        in_features=tied_config["lora_A_size"],
        out_features=tied_config["lora_B_size"],
    )
    self.base_layer = base_layer

    init_lora_weights = kwargs.pop("init_lora_weights", True)
    self.layer_name = kwargs.pop("layer_name", None)
    lora_module = kwargs.pop("lora_module", None)
    # Freezing the pre-trained weight matrix
    self.lora_A_vec = nn.ParameterDict()
    self.lora_A_anti_vec = nn.ParameterDict()
    self.lora_pair_mask = nn.ParameterDict()
    self.lora_pair_mask_B = nn.ParameterDict()
    # self.lora_B_vec = nn.ParameterDict()
    self.lora_bias = nn.ParameterDict()
    self.update_layer = types.MethodType(LoraLayer_update_layer, self)
    self.reset_lora_parameters = types.MethodType(LoraLayer_reset_lora_parameters, self)
    self.update_layer(
        adapter_name,
        r,
        lora_alpha,
        lora_dropout,
        init_lora_weights,
        tied_config=tied_config,
        lora_module=lora_module,
    )
    self.set_adapter(adapter_name)

    # set the trainable attributes of parameters
    self.lora_A[adapter_name].weight.requires_grad = True
    self.lora_B[adapter_name].weight.requires_grad = True

    if self.ft_mode == "lora":
        self.lora_A_vec[adapter_name].requires_grad = False
    elif self.ft_mode == "mos":
        self.lora_A_vec[adapter_name].requires_grad = (
            False  # TODO: modify this attribute if needed.
        )
    else:
        raise ValueError(f"ft_mode {self.ft_mode} is not supported.")
    self.lora_A_anti_vec[adapter_name].requires_grad = False
    self.lora_pair_mask[adapter_name].requires_grad = False
    self.lora_pair_mask_B[adapter_name].requires_grad = False


# modify the forward function of LoraLinear4bit
def LoraLinear4bit_Tied_LoRA_forward(
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
        # bias = self.lora_bias[active_adapter]
        lora_A = self.lora_A[active_adapter].weight
        lora_B = self.lora_B[active_adapter].weight
        lora_A_vec = self.lora_A_vec[active_adapter]
        lora_A_anti_vec = self.lora_A_anti_vec[active_adapter]
        lora_pair_mask = self.lora_pair_mask[active_adapter]
        lora_pair_mask_B = self.lora_pair_mask_B[active_adapter]

        # forward
        requires_conversion = not torch.is_autocast_enabled()
        if requires_conversion:
            expected_dtype = result.dtype
            x = x.to(lora_A.weight.dtype)

        # apply LoRA
        lora_A = transform_chunks(
            lora_A, lora_pair_mask, self.tied_config["num_chunk_per_vec"]
        )
        lora_B = transform_chunks(
            lora_B.t(), lora_pair_mask_B, self.tied_config["num_chunk_per_vec"]
        ).t()
        output = (
            torch.matmul(
                torch.matmul(
                    self.lora_dropout[active_adapter](x),
                    lora_A.t() * (lora_A_vec * lora_A_anti_vec),
                ),
                lora_B.t(),
            )
            * self.scaling[active_adapter]
        )

        if requires_conversion:
            output = output.to(expected_dtype)

        result += output

    return result


def save_tied_lora(trainer):
    print("Saving Tied LoRA checkpoint manually...")
    model = trainer.model
    args = trainer.args

    def touch(fname, times=None):
        with open(fname, "a"):
            os.utime(fname, times)

    touch(os.path.join(args.output_dir, "completed"))

    tied_config = getattr(model, "chunk_config", None)
    if tied_config is None:
        print("Warning: tied_config is None. Skip saving.")
        return
    num_chunk_per_vec = tied_config.num_chunk_per_vec

    ckpt_pth = os.path.join(args.output_dir, "adapter_model.bin")
    adapter_name = model.active_adapters[0]

    # save only the trainable weights
    output_state_dict = get_peft_model_state_dict(
        model,
        state_dict=None,
        adapter_name=adapter_name,
    )
    new_state_dict = {}
    for k, v in output_state_dict.items():
        if "lora_A.weight" in k:
            lora_A_vec = output_state_dict[k.replace("lora_A.weight", "lora_A_vec")]
            lora_A_anti_vec = output_state_dict[
                k.replace("lora_A.weight", "lora_A_anti_vec")
            ]
            lora_pair_mask = output_state_dict[
                k.replace("lora_A.weight", "lora_pair_mask")
            ]
            lora_A = transform_chunks(v, lora_pair_mask, num_chunk_per_vec)
            lora_A = (lora_A.t() * (lora_A_vec * lora_A_anti_vec)).t().to(device="cpu")
            new_state_dict[k] = lora_A

        elif "lora_B.weight" in k:
            lora_pair_mask_B = output_state_dict[
                k.replace("lora_B.weight", "lora_pair_mask_B")
            ]
            v = (
                transform_chunks(v.t(), lora_pair_mask_B, num_chunk_per_vec)
                .t()
                .to(device="cpu")
            )
            new_state_dict[k] = v

    torch.save(new_state_dict, ckpt_pth)


def share_lora_tiedly(model, tied_config):
    """
    This function is used to re_initialize the Linear4bit class with the specific modification.
    The config of the replaced Linear4bit module has the priority over the argumented parameters.
    """

    # get the basic info of the model
    num_key_value_heads = getattr(
        model.config,
        "num_key_value_heads",
        model.config.num_attention_heads,
    )
    num_grouped_head = model.config.num_attention_heads // num_key_value_heads
    model_info = {
        "lora_r": tied_config.r,
        "q_dims": (model.config.hidden_size, model.config.hidden_size),
        "k_dims": (
            model.config.hidden_size,
            model.config.hidden_size // num_grouped_head,
        ),
        "v_dims": (
            model.config.hidden_size,
            model.config.hidden_size // num_grouped_head,
        ),
        "o_dims": (model.config.hidden_size, model.config.hidden_size),
        "up_dims": (model.config.hidden_size, model.config.intermediate_size),
        "gate_dims": (model.config.hidden_size, model.config.intermediate_size),
        "down_dims": (model.config.intermediate_size, model.config.hidden_size),
        "layer_num": model.config.num_hidden_layers,
    }

    # instantiate the MoS modules
    pool_r = tied_config.valid_param_lora_r * model_info["layer_num"]
    num_chunk_per_vec = tied_config.num_chunk_per_vec
    chunk_pool_r = pool_r * num_chunk_per_vec
    chunk_model_info = {}
    # check the divisibility of the model_info
    for k, v in model_info.items():
        if isinstance(v, tuple):
            assert (v[0] % num_chunk_per_vec == 0) and (
                v[1] % num_chunk_per_vec == 0
            ), f"Error: {k} should be divisible by num_chunk_per_vec."
            chunk_model_info[k] = (v[0] // num_chunk_per_vec, v[1] // num_chunk_per_vec)

    # define the tied parameters
    tied_param_dict = {
        "q_proj": {
            "lora_A": nn.Linear(
                chunk_model_info["q_dims"][0], chunk_pool_r, bias=False
            ),
            "lora_B": nn.Linear(
                chunk_pool_r, chunk_model_info["q_dims"][1], bias=False
            ),
        },
        "k_proj": {
            "lora_A": nn.Linear(
                chunk_model_info["k_dims"][0], chunk_pool_r, bias=False
            ),
            "lora_B": nn.Linear(
                chunk_pool_r, chunk_model_info["k_dims"][1], bias=False
            ),
        },
        "v_proj": {
            "lora_A": nn.Linear(
                chunk_model_info["v_dims"][0], chunk_pool_r, bias=False
            ),
            "lora_B": nn.Linear(
                chunk_pool_r, chunk_model_info["v_dims"][1], bias=False
            ),
        },
        "o_proj": {
            "lora_A": nn.Linear(
                chunk_model_info["o_dims"][0], chunk_pool_r, bias=False
            ),
            "lora_B": nn.Linear(
                chunk_pool_r, chunk_model_info["o_dims"][1], bias=False
            ),
        },
        "up_proj": {
            "lora_A": nn.Linear(
                chunk_model_info["up_dims"][0], chunk_pool_r, bias=False
            ),
            "lora_B": nn.Linear(
                chunk_pool_r, chunk_model_info["up_dims"][1], bias=False
            ),
        },
        "gate_proj": {
            "lora_A": nn.Linear(
                chunk_model_info["gate_dims"][0], chunk_pool_r, bias=False
            ),
            "lora_B": nn.Linear(
                chunk_pool_r, chunk_model_info["gate_dims"][1], bias=False
            ),
        },
        "down_proj": {
            "lora_A": nn.Linear(
                chunk_model_info["down_dims"][0], chunk_pool_r, bias=False
            ),
            "lora_B": nn.Linear(
                chunk_pool_r, chunk_model_info["down_dims"][1], bias=False
            ),
        },
    }
    print("-" * 20, f"Injecting MoS ... ", "-" * 20)
    for k, v in model.named_modules():
        if isinstance(v, Linear4bit):
            print(f"Adding MoS to {k}...")

            # generate config
            adapter_name = v.active_adapter[0]
            base_layer = v.base_layer
            layer_name, layer_idx, layer_type = (
                k,
                int(k.split(".")[-3]),
                k.split(".")[-1],
            )
            lora_module = tied_param_dict[layer_type]
            kwargs = {
                "adapter_name": adapter_name,
                "base_layer": base_layer,
                "r": tied_config.r,
                "lora_alpha": v.lora_alpha[adapter_name],
                "lora_dropout": (
                    v.lora_dropout[adapter_name].p
                    if isinstance(v.lora_dropout[adapter_name], Dropout)
                    else 0.0
                ),
                "fan_in_fan_out": model.active_peft_config.fan_in_fan_out,
                "init_lora_weights": True,
                "compute_dtype": base_layer.compute_dtype,
                "compress_statistics": base_layer.weight.compress_statistics,
                "quant_type": base_layer.weight.quant_type,
                "layer_name": layer_name,
                "layer_idx": layer_idx,
                "lora_module": lora_module,
                "layer_num": model_info["layer_num"],
            }
            dtype = v.lora_A[adapter_name].weight.dtype
            device = v.lora_A[adapter_name].weight.device
            # re_init Linear4bit module
            Linear4bit__Re_Init__(v, tied_config=tied_config, **kwargs)
            model.active_peft_config.r = tied_config.r
            # change to the same device and dtype as the original Linear4bit module
            v.to(device=device, dtype=dtype)
            v.lora_A.to(dtype=torch.float32)
            v.lora_B.to(dtype=torch.float32)
            v.lora_A_vec.to(dtype=torch.float32)
            v.lora_A_anti_vec.to(dtype=torch.float32)

            # modify the forward function of LoraLinear4bit
            v.forward = types.MethodType(LoraLinear4bit_Tied_LoRA_forward, v)


    print(
        "-" * 20,
        "Finish the modification for MoS. ",
        "-" * 20,
    )


if __name__ == "__main__":
    new_state_dict = torch.load(
        "/workspace/output/superni_20231217-165108_GPU_3_r_64_x_A1B1_lr_0.0002_sd_0/adapter_model.bin"
    )

    for k, v in new_state_dict.items():
        if "lora_A.weight" in k:
            lora_A = v
            lora_B = new_state_dict[k.replace("lora_A.weight", "lora_B.weight")]
            res = torch.matmul(lora_A.t(), lora_B.t())
            if torch.all(res == 0):
                pass
            else:
                print(k)

    print("Hello World!")
