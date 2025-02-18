#!/usr/bin/env python
# coding=utf-8
"""
This file is modified from the huggingface example for finetuning language models
[run_clm.py](https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py)
"""

import os, sys, json
import logging, warnings
import random
import torch

import datasets
from datasets import load_dataset
from dataclasses import dataclass, field
from typing import Optional
from functools import partial

import transformers
from transformers import (
    set_seed,
    Trainer,
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaTokenizer,
    LlamaTokenizerFast,
    HfArgumentParser,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig,
)

from transformers.trainer_utils import get_last_checkpoint
from open_instruct.finetune import (
    encode_with_prompt_completion_format,
    encode_with_messages_format,
)

from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, PeftModel
from peft.tuners.lora import LoraLayer
from modules.Tied_LoRA import gen_tied_lora_config as gen_chunkwise_sharing_lora_config
from modules.Tied_LoRA import share_lora_tiedly as share_lora_chunkwisely
from modules.Tied_LoRA import save_tied_lora as save_chunk_lora


logger = logging.getLogger(__name__)

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default="meta-llama/Llama-2-7b-hf",
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )
    use_fast_tokenizer: bool = field(
        default=False,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not. It may cause errors"
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    use_auth_token: bool = field(
        default=None,
        metadata={
            "help": "The `use_auth_token` argument is deprecated and will be removed in Transformers v4.34. Please use `token`."
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option"
                "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
                "execute code present on the Hub on your local machine."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default="bfloat16",
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    use_flash_attn: bool = field(
        default=False,
        metadata={"help": "Whether to use flash attention in the model training"},
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        },
    )
    train_file: Optional[str] = field(
        default=None,
        metadata={"help": "The input training data file (a json/jsonl file)."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            )
        },
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None:
            raise ValueError("Need either a dataset name or a training file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in [
                    "json",
                    "jsonl",
                ], "`train_file` should be a json or a jsonl file."


@dataclass
class TrainingArguments(transformers.Seq2SeqTrainingArguments):
    seed: int = field(default=42, metadata={"help": "Random seed for reproduciblity."})
    data_seed: int = field(
        default=None,
        metadata={
            "help": "Random seed to be used with data samplers. If not set, random generators for data sampling will use the same seed as `seed`. This can be used to ensure reproducibility of data sampling, independent of the model seed."
        },
    )
    report_to: str = field(
        default="tensorboard",
        metadata={
            "help": 'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        },
    )
    resume_from_checkpoint: Optional[str] = field(
        default=None,
        metadata={
            "help": "The path to a folder with checkpoints to resume training from."
        },
    )
    output_dir: str = field(
        default="./output",
        metadata={
            "help": "The output directory where the model predictions and checkpoints will be written."
        },
    )
    overwrite_output_dir: bool = field(
        default=False,
        metadata={
            "help": "Overwrite the content of the output directory. Use this to continue training if `output_dir` points to a checkpoint directory."
        },
    )
    do_train: bool = field(
        default=True, metadata={"help": "Whether to run training or not."}
    )
    do_eval: bool = field(
        default=True,
        metadata={"help": "Whether to run evaluation on the validation set or not."},
    )
    do_predict: bool = field(
        default=False,
        metadata={"help": "Whether to run predictions on the test set or not."},
    )
    evaluation_strategy: str = field(
        default="steps",
        metadata={
            "help": "The evaluation strategy to adopt during training. Possible values are:"
            " - `no`: No evaluation is done during training."
            " - `steps`: Evaluation is done (and logged) every `eval_steps`."
            " - `epoch`: Evaluation is done at the end of each epoch."
        },
    )
    prediction_loss_only: bool = field(
        default=False,
        metadata={
            "help": "When performing evaluation and generating predictions, only returns the loss."
        },
    )
    per_device_train_batch_size: int = field(
        default=16,
        metadata={
            "help": "The batch size per GPU/XPU/TPU/MPS/NPU core/CPU for training."
        },
    )
    per_device_eval_batch_size: int = field(
        default=16,
        metadata={
            "help": "The batch size per GPU/XPU/TPU/MPS/NPU core/CPU for evaluation."
        },
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={
            "help": "Number of updates steps to accumulate the gradients for, before performing a backward/update pass."
        },
    )
    learning_rate: float = field(
        default=2e-4,
        metadata={
            "help": "Initial learning rate (after the potential warmup period) to use."
        },
    )
    optim: str = field(
        default="paged_adamw_32bit", metadata={"help": "The optimizer to be used"}
    )
    max_steps: int = field(
        default=0,
        metadata={
            "help": "If > 0: set total number of optimizer steps to perform. Override num_train_epochs."
        },
    )
    weight_decay: float = field(
        default=0.0, metadata={"help": "The L2 weight decay rate of AdamW"}
    )  # use lora dropout instead for regularization if needed
    remove_unused_columns: bool = field(
        default=True,
        metadata={
            "help": "Remove columns not required by the model when using an nlp.Dataset."
        },
    )
    gradient_checkpointing: bool = field(
        default=True,
        metadata={
            "help": "If True, use gradient checkpointing to save memory at the expense of slower backward pass."
        },
    )
    lr_scheduler_type: str = field(
        default="linear",
        metadata={
            "help": "Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis"
            "could be one of 'linear', 'cosine', 'cosine_with_restarts', 'polynomial', 'constant', 'constant_with_warmup'"
        },
    )
    warmup_ratio: float = field(
        default=0.0, metadata={"help": "Ratio of total training steps used for warmup."}
    )
    logging_steps: int = field(
        default=10,
        metadata={
            "help": "Log the training loss and learning rate every logging_steps steps."
        },
    )
    group_by_length: bool = field(
        default=True,
        metadata={
            "help": "Whether or not to group samples of roughly the same length together when batching."
        },
    )
    length_column_name: Optional[str] = field(
        default="length",
        metadata={
            "help": "Column name with precomputed lengths to use when grouping by length."
        },
    )
    save_strategy: str = field(
        default="steps",
        metadata={"help": "The checkpoint save strategy to use."},
    )
    save_steps: float = field(
        default=1000,
        metadata={
            "help": (
                "Save checkpoint every X updates steps. Should be an integer or a float in range `[0,1)`. "
                "If smaller than 1, will be interpreted as ratio of total training steps."
            )
        },
    )
    save_total_limit: int = field(
        default=1,
        metadata={
            "help": "How many checkpoints to save before the oldest is overwritten"
        },
    )
    num_train_epochs: int = field(
        default=0,
        metadata={"help": "Total number of training epochs to perform."},
    )
    max_grad_norm: float = field(
        default=0.3,
        metadata={
            "help": "Gradient clipping max norm. This is tuned and works well for all models tested."
        },
    )
    eval_steps: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "Run an evaluation every X steps. Should be an integer or a float in range `[0,1)`. "
                "If smaller than 1, will be interpreted as ratio of total training steps."
            )
        },
    )
    logging_strategy: str = field(
        default="steps",
        metadata={"help": "The logging strategy to use."},
    )
    tf32: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to enable tf32 mode, available in Ampere and newer GPU architectures. This is an experimental"
                " API and it may change."
            )
        },
    )
    bf16: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to use bf16 (mixed) precision instead of 32-bit. Requires Ampere or higher NVIDIA"
                " architecture or using CPU (use_cpu) or Ascend NPU. This is an experimental API and it may change."
            )
        },
    )


@dataclass
class SelfDefinedArguments:
    use_lora: bool = field(default=True, metadata={"help": "Whether to use LoRA."})
    use_qlora: bool = field(
        default=True,
        metadata={
            "help": "Use qLoRA training - main thing is initialising model in quantised form. Not compatible with deepspeed."
        },
    )
    lora_r: int = field(default=64, metadata={"help": "Lora R dimension."})
    lora_alpha: float = field(default=16, metadata={"help": " Lora alpha."})
    lora_dropout: float = field(default=0.1, metadata={"help": "Lora dropout."})
    lora_modules: str = field(default="all", metadata={"help": "Lora modules to use."})
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
        default=1, metadata={"help": "The initial value of MoS's LoRA_A vector."}
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
        default=0, metadata={"help": "The number of private LoRA pairs."}
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


def prepare_model_tokenizer(training_args, model_args, selfdefined_args):
    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
        "padding_side": "right",
        "tokenizer_type": "llama"
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name, **tokenizer_kwargs
        )
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path, **tokenizer_kwargs
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this finetuning script."
        )

    if model_args.model_name_or_path:
        if selfdefined_args.use_qlora:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                load_in_8bit=False,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                token=model_args.token,
                quantization_config=bnb_config,
                torch_dtype=torch.bfloat16,
                use_flash_attention_2=model_args.use_flash_attn,
            )
            assert (
                model.config.torch_dtype == torch.bfloat16
            ), "Model dtype should be bf16"  # check dtype
        else:
            torch_dtype = (
                model_args.torch_dtype
                if model_args.torch_dtype in ["auto", None]
                else getattr(torch, model_args.torch_dtype)
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                torch_dtype=torch_dtype,
                use_flash_attention_2=model_args.use_flash_attn
            )
    else:
        assert (
            False
        ), "We don't support training from scratch yet. Please set `model_name_or_path`."

    assert isinstance(tokenizer, LlamaTokenizer) or isinstance(
        tokenizer, LlamaTokenizerFast
    ), "Only llama Model is supported yet."
    num_added_tokens = tokenizer.add_special_tokens(
        {
            "bos_token": "<s>",
            "eos_token": "</s>",
            "unk_token": "<unk>",
            "pad_token": "<pad>",
        }
    )
    assert num_added_tokens in [
        0,
        1,
    ], "LlamaTokenizer should only add one special token - the pad_token, or no tokens if pad token present."

    # resize embeddings if needed (e.g. for LlamaTokenizer)
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    if selfdefined_args.use_lora:
        if selfdefined_args.use_qlora:
            logger.info("Initializing model for QLORA kbit_training ...")
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=training_args.gradient_checkpointing
            )

        logger.info("Initializing LORA model...")
        peft_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=selfdefined_args.lora_r,
            lora_alpha=selfdefined_args.lora_alpha,
            lora_dropout=selfdefined_args.lora_dropout,
            bias="none",
            target_modules=[
                "q_proj",
                "o_proj",
                "v_proj",
                "k_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            inference_mode=False,
        )
        model = get_peft_model(model, peft_config)

        if selfdefined_args.use_qlora:
            for name, module in model.named_modules():
                if isinstance(module, LoraLayer):
                    module = module.to(torch.bfloat16)


        chunk_config = gen_chunkwise_sharing_lora_config(
            selfdefined_args,
            lora_config=model.active_peft_config,
        )
        share_lora_chunkwisely(model, chunk_config)
        model.chunk_config = chunk_config
        model.config.use_cache = False
        logger.info(f"Prepare model successfully!")

    else:
        assert False, "Only LoRA/QLoRA is supported yet. Must set `use_lora` to True."

    # Verifying the datatypes and parameter counts before training.
    print("Trainable parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.shape, param.dtype)

    model.print_trainable_parameters()

    dtypes = {}
    for _, p in model.named_parameters():
        dtype = p.dtype
        if dtype not in dtypes:
            dtypes[dtype] = 0
        dtypes[dtype] += p.numel()
    total = 0
    for k, v in dtypes.items():
        total += v
    for k, v in dtypes.items():
        print(k, v, v / total)

    return model, tokenizer


def prepare_data(training_args, data_args, model_args, tokenizer):
    # Preprocessing the datasets.
    if data_args.dataset_name is not None:
        assert (
            False
        ), "We don't support loading dataset from hub yet, only support tulu dataset."
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
            streaming=data_args.streaming,
        )
    else:
        raw_datasets = load_dataset(
            "json",
            data_files={"train": data_args.train_file},
            cache_dir=model_args.cache_dir,
        )
        # # For Debug only: Modify the train split to contain only the first 2048 samples
        # raw_datasets["train"] = raw_datasets["train"].select(range(2048))
        # warnings.warn(
        #     "`Modify the train split to contain only the first 2048 samples` should be run only in debug mode!"
        # )

    if (
        "prompt" in raw_datasets["train"].column_names
        and "completion" in raw_datasets["train"].column_names
    ):
        encode_function = partial(
            encode_with_prompt_completion_format,
            tokenizer=tokenizer,
            max_seq_length=data_args.max_seq_length,
        )
    elif "messages" in raw_datasets["train"].column_names:
        encode_function = partial(
            encode_with_messages_format,
            tokenizer=tokenizer,
            max_seq_length=data_args.max_seq_length,
        )
    else:
        raise ValueError(
            "You need to have either 'prompt'&'completion' or 'messages' in your column names."
        )

    # To speed up this part, we use multiprocessing.
    with training_args.main_process_first(desc="Processing instruction data"):
        if not data_args.streaming:
            lm_datasets = raw_datasets.map(
                encode_function,
                batched=False,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                remove_columns=[
                    name
                    for name in raw_datasets["train"].column_names
                    if name not in ["input_ids", "labels", "attention_mask"]
                ],
                desc="Tokenizing and reformatting instruction data",
            )
        else:
            lm_datasets = raw_datasets.map(
                encode_function,
                remove_columns=[
                    name
                    for name in raw_datasets["train"].column_names
                    if name not in ["input_ids", "labels", "attention_mask"]
                ],
                batched=False,
                desc="Tokenizing and reformatting instruction data",
            )
        lm_datasets.set_format(type="pt")
        lm_datasets = lm_datasets.filter(
            lambda example: (example["labels"] != -100).any(),
            desc="Filtering useless samples with invalid labels",
        )
        # Prepare group by length
        if training_args.group_by_length:
            lm_datasets = lm_datasets.map(
                lambda x: {"length": int(len(x["input_ids"]))},
                desc="Computing dataset lengths",
            )

    if training_args.do_eval:
        if "eval" in lm_datasets:
            eval_dataset = lm_datasets["eval"]
        else:
            print(
                "Splitting train dataset in train and validation according to `max_eval_samples`"
            )
            lm_datasets = lm_datasets["train"].train_test_split(
                test_size=data_args.max_eval_samples, shuffle=True, seed=42
            )
            eval_dataset = lm_datasets["test"]
        if (
            data_args.max_eval_samples is not None
            and len(eval_dataset) > data_args.max_eval_samples
        ):
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))

    if training_args.do_train:
        train_dataset = lm_datasets["train"]
        if (
            data_args.max_train_samples is not None
            and len(train_dataset) > data_args.max_train_samples
        ):
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
        # Log a few random samples from the training set:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    return {
        "train_dataset": train_dataset if training_args.do_train else None,
        "eval_dataset": eval_dataset if training_args.do_eval else None,
    }


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments, SelfDefinedArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args, selfdefined_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        (
            model_args,
            data_args,
            training_args,
            selfdefined_args,
        ) = parser.parse_args_into_dataclasses()

    # Prepare the logger file
    os.makedirs(training_args.output_dir, exist_ok=True)
    print(
        f"Output directory: {training_args.output_dir} is created {os.path.isdir(training_args.output_dir)}."
    )
    file_handler = logging.FileHandler(
        os.path.join(training_args.output_dir, "run.log")
    )
    # Set the formatter for the file handler (optional)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info(
        f"{'-'*25 + 'Configurations' + '-'*25}\n"
        f"Self-defined args: {selfdefined_args}\n"
        f"Model args: {model_args}\n"
        f"Data args: {data_args}\n"
        f"Training args: {training_args}\n"
        f"{'-'*25 + 'Configurations Complete' + '-'*25}\n"
    )

    if model_args.use_auth_token is not None:
        warnings.warn(
            "The `use_auth_token` argument is deprecated and will be removed in Transformers v4.34.",
            FutureWarning,
        )
        if model_args.token is not None:
            raise ValueError(
                "`token` and `use_auth_token` are both specified. Please set only the argument `token`."
            )
        model_args.token = model_args.use_auth_token

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Detecting last checkpoint.
    last_checkpoint = None
    assert (
        last_checkpoint is None and training_args.resume_from_checkpoint is None
    ), "Don't support resuming from checkpoint yet. It may cause ckpt loading errors of self-definded LoRA."

    # Prepare model and tokenizer
    model, tokenizer = prepare_model_tokenizer(
        training_args, model_args, selfdefined_args
    )

    # Prepare the datasets.
    lm_datasets = prepare_data(training_args, data_args, model_args, tokenizer)

    # initalize a trainer
    # here we use a custom trainer that moves the model to CPU when saving the checkpoint in FSDP mode
    # we can switch to the default trainer after moving to deepspeed (let's don't change too much for now)
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=lm_datasets["train_dataset"] if training_args.do_train else None,
        eval_dataset=lm_datasets["eval_dataset"] if training_args.do_eval else None,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model),
    )

    all_metrics = {"run_name": training_args.run_name}

    # Training
    if training_args.do_train:
        logger.info("*** Train ***")
        train_dataset = lm_datasets["train_dataset"]
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        assert checkpoint is None, "We don't support resuming from checkpoint yet."
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples
            if data_args.max_train_samples is not None
            else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))
        metrics["eval_samples"] = data_args.max_eval_samples
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        all_metrics.update(metrics)
        trainer.save_state()
        save_chunk_lora(trainer)

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(metric_key_prefix="eval")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        all_metrics.update(metrics)

    if training_args.do_train or training_args.do_eval or training_args.do_predict:
        with open(os.path.join(training_args.output_dir, "metrics.json"), "w") as fout:
            fout.write(json.dumps(all_metrics))


if __name__ == "__main__":
    main()
