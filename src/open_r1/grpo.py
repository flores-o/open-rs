# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import sys
from dataclasses import dataclass, field

import datasets
import torch
import transformers
from datasets import load_dataset
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint

from open_r1.configs import GRPOConfig
from open_r1.rewards import (
    accuracy_reward,
    code_reward,
    format_reward,
    get_code_format_reward,
    get_cosine_scaled_reward,
    get_repetition_penalty_reward,
    len_reward,
    reasoning_steps_reward,
    tag_count_reward,
)
from open_r1.utils import get_tokenizer
from open_r1.utils.callbacks import get_callbacks
from open_r1.utils.wandb_logging import init_wandb_training
from open_r1.callbacks.push_each_checkpoint import PushEachCheckpointCallback
from trl import GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config
# ────────── Dynamic-filter helper ──────────
from collections import defaultdict
from transformers import TrainerCallback, TrainerControl, TrainerState
from transformers.models.qwen3.modeling_qwen3 import Qwen3Model
from huggingface_hub import HfFolder, HfApi
from pathlib import Path
import re
import types





import numpy as np, torch; torch.serialization.add_safe_globals([np.core.multiarray._reconstruct])
# --- replace the existing derive_base_repo ----------------------------------
def derive_base_repo(model_args):
    """
    Return a suitable repo prefix for the current token.

    • If `model_name_or_path` already looks like   user/repo   → keep as-is
    • Otherwise prepend <token_owner>/.
    """
    raw = str(model_args.model_name_or_path).rstrip("/")

    if "/" in raw and not raw.startswith("."):
        return raw                                  # already a Hub path

    # discover the username tied to the token
    try:
        owner = HfApi().whoami()["name"]            # e.g. "oanaflores"
    except Exception:
        owner = os.getenv("HF_USERNAME", "your-name")

    from pathlib import Path, PurePosixPath
    import re
    clean = re.sub(r"[^A-Za-z0-9._-]", "-", Path(raw).name)  # slug-ify
    return f"{owner}/{clean}"
# ---------------------------------------------------------------------------


class DynamicFilterCallback(TrainerCallback):
    """
    Counts how often each training example receives a zero reward
    (after Clip-Higher masking) and lets the trainer know so we can
    drop 'hopeless' ones after N tries.
    """

    def __init__(self, max_zero=3):
        super().__init__()
        self.max_zero = max_zero
        self._zero_count = defaultdict(int)

    # This hook name is defined in TRL's GRPOTrainer
    def on_post_reward(
        self,
        args,
        state: TrainerState,
        control: TrainerControl,
        rewards=None,
        batch_indices=None,
        **kwargs,
    ):
        for idx, rw in zip(batch_indices, rewards):
            if rw == 0.0:
                self._zero_count[idx] += 1

    # small helper we will reference when building the filtered dataset
    def keep_example(self, example, idx):
        return self._zero_count[idx] < self.max_zero


logger = logging.getLogger(__name__)


@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format', 'reasoning_steps', 'cosine', 'repetition_penalty', 'length', 'tag_count', 'code', 'code_format'.
        cosine_min_value_wrong (`float`):
            Minimum reward for cosine scaling for wrong answers.
        cosine_max_value_wrong (`float`):
            Maximum reward for cosine scaling for wrong answers.
        cosine_min_value_correct (`float`):
            Minimum reward for cosine scaling for correct answers.
        cosine_max_value_correct (`float`):
            Maximum reward for cosine scaling for correct answers.
        cosine_max_len (`int`):
            Maximum length for cosine scaling.
        code_language (`str`):
            Language for code format reward.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format", "tag_count"],
        metadata={
            "help": "List of reward functions. Possible values: 'accuracy', 'format', 'reasoning_steps', 'cosine', 'repetition_penalty', 'length', tag_count', 'code', 'code_format'"
        },
    )
    cosine_min_value_wrong: float = field(
        default=0.0,
        metadata={"help": "Minimum reward for wrong answers"},
    )
    cosine_max_value_wrong: float = field(
        default=-0.5,
        metadata={"help": "Maximum reward for wrong answers"},
    )
    cosine_min_value_correct: float = field(
        default=0.5,
        metadata={"help": "Minimum reward for correct answers"},
    )
    cosine_max_value_correct: float = field(
        default=1.0,
        metadata={"help": "Maximum reward for correct answers"},
    )
    cosine_max_len: int = field(
        default=1000,
        metadata={"help": "Maximum length for scaling"},
    )
    repetition_n_grams: int = field(
        default=3,
        metadata={"help": "Number of n-grams for repetition penalty reward"},
    )
    repetition_max_penalty: float = field(
        default=-1.0,
        metadata={"help": "Maximum (negative) penalty for for repetition penalty reward"},
    )
    code_language: str = field(
        default="python",
        metadata={
            "help": "Language for code format reward. Based on E2B supported languages https://e2b.dev/docs/code-interpreting/supported-languages",
            "choices": ["python", "javascript", "r", "java", "bash"],
        },
    )
    # ---------- Dapo flags ----------
    clip_higher: bool = field(default=False)
    dynamic_filter: bool = field(default=False)



def main(script_args, training_args, model_args):
    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Training parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    if "wandb" in training_args.report_to:
        init_wandb_training(training_args)

    # Load the dataset
    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

    ################
    # Load tokenizer
    ################
    tokenizer = get_tokenizer(model_args, training_args)

    # 🩹 Qwen-3 + FlashAttention needs LEFT padding ────────────────
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:          # keep pad-id valid
        tokenizer.pad_token = tokenizer.eos_token
    # ──────────────────────────────────────────────────────────────

    # -------- Clip-Higher wrapper ----------
    def _mask_if_truncated(base_fn):
        if not script_args.clip_higher:
            return base_fn

        def wrapped(completions, **kw):
            r = base_fn(completions, **kw)
            ceiling = training_args.max_completion_length
            for i, c in enumerate(completions):
                if len(c[0]["content"]) >= ceiling:
                    r[i] = 0.0
            return r
        return wrapped


    # Get reward functions
    REWARD_FUNCS_REGISTRY = {
        "accuracy": _mask_if_truncated(accuracy_reward),
        "format": format_reward,
        "reasoning_steps": reasoning_steps_reward,
        "cosine": get_cosine_scaled_reward(
            min_value_wrong=script_args.cosine_min_value_wrong,
            max_value_wrong=script_args.cosine_max_value_wrong,
            min_value_correct=script_args.cosine_min_value_correct,
            max_value_correct=script_args.cosine_max_value_correct,
            max_len=script_args.cosine_max_len,
        ),
        "repetition_penalty": get_repetition_penalty_reward(
            ngram_size=script_args.repetition_n_grams,
            max_penalty=script_args.repetition_max_penalty,
        ),
        "length": len_reward,
        "code": code_reward,
        "code_format": get_code_format_reward(language=script_args.code_language),
        "tag_count": tag_count_reward,
    }
    reward_funcs = [REWARD_FUNCS_REGISTRY[func] for func in script_args.reward_funcs]

    # Format into conversation
    def make_conversation(example):
        prompt = []

        if training_args.system_prompt is not None:
            prompt.append({"role": "system", "content": training_args.system_prompt})

        prompt.append({"role": "user", "content": example["problem"]})
        return {"prompt": prompt}

    dataset = dataset.map(make_conversation)

    for split in dataset:
        if "messages" in dataset[split].column_names:
            dataset[split] = dataset[split].remove_columns("messages")

    logger.info("*** Initializing model kwargs ***")
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
    )
    training_args.model_init_kwargs = model_kwargs


    # Get a reference to the original method for learning its implementation
    original_update_causal_mask = Qwen3Model._update_causal_mask

    def patched_update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        past_key_values_length: int,
        seq_length: int,
        dtype: torch.dtype = None,
        *args,  # Accept any additional positional arguments
        **kwargs  # Accept any additional keyword arguments
    ):
        # Force the model to use left padding regardless of its setting
        self.padding_side = "left"
        if hasattr(self, "config"):
            self.config.padding_side = "left"
        
        # Handle the case where attention_mask is None (happens during initialization)
        if attention_mask is None:
            # Create a default batch size of 1 for initialization
            batch_size = 1
            device = next(self.parameters()).device if hasattr(self, "parameters") else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            batch_size = attention_mask.shape[0]
            device = attention_mask.device
        
        # Get the correct dtype
        mask_dtype = dtype or (attention_mask.dtype if attention_mask is not None else torch.float32)
        
        # Pure tensor operations for CUDA graph capture compatibility
        if isinstance(seq_length, torch.Tensor):
            # Use tensor operations instead of .item()
            seq_length_value = seq_length.shape[-1] if seq_length.dim() > 0 else seq_length
        else:
            seq_length_value = seq_length
        
        # Handle past key values length without .item()
        past_len = 0
        if isinstance(past_key_values_length, torch.Tensor):
            past_len = past_key_values_length.shape[0] if past_key_values_length.dim() > 0 else past_key_values_length
        else:
            past_len = past_key_values_length
        
        # Create causal mask with proper tensor construction
        ones_tensor = torch.ones(seq_length_value, seq_length_value, device=device, dtype=mask_dtype)
        causal_mask = torch.tril(ones_tensor).unsqueeze(0).unsqueeze(1)
        
        # Always create the full causal mask with past values (if any)
        if past_len > 0:
            past_mask = torch.ones(1, 1, seq_length_value, past_len, device=device, dtype=mask_dtype)
            causal_mask = torch.cat([past_mask, causal_mask], dim=-1)
        
        # Expand for batch dimension
        if batch_size > 1:
            final_seq_len = seq_length_value + past_len
            causal_mask = causal_mask.expand(batch_size, 1, seq_length_value, final_seq_len)
        
        return causal_mask

    # Apply the complete replacement - FIX: This was incorrectly indented inside the function
    Qwen3Model._update_causal_mask = patched_update_causal_mask

    logger.warning("Completely replaced Qwen3Model._update_causal_mask to force left padding")

    logger.warning("Patched Qwen3Model._update_causal_mask to force left padding")

    #############################
    # Initialize the GRPO trainer
    #############################
    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        callbacks=get_callbacks(training_args, model_args),
        processing_class=tokenizer,
    )


    trainer.loss_reduction = "token_average"  

    # === Apply comprehensive Qwen3 padding fixes for distributed training ===
    trainer.loss_reduction = "token_average"

    # Force padding_side="left" in tokenizer before any processing
    tokenizer.padding_side = "left"
    if hasattr(tokenizer, "model_input_names"):
        for key in tokenizer.model_input_names:
            if hasattr(tokenizer, f"{key}_side"):
                setattr(tokenizer, f"{key}_side", "left")

    # 1. Fix all model components
    if hasattr(trainer.model, "config"):
        trainer.model.config.padding_side = "left"
    trainer.model.padding_side = "left"

    # Also fix any nested model objects
    if hasattr(trainer.model, "model"):
        trainer.model.model.padding_side = "left"
        if hasattr(trainer.model.model, "config"):
            trainer.model.model.config.padding_side = "left"

    # Fix trainer's tokenizer
    trainer.tokenizer.padding_side = "left"

    # Fix VLLM pipeline
    if hasattr(trainer, "vllm_pipeline") and hasattr(trainer.vllm_pipeline, "tokenizer"):
        trainer.vllm_pipeline.tokenizer.padding_side = "left"

    # Fix generation config
    if hasattr(trainer.model, "generation_config"):
        trainer.model.generation_config.padding_side = "left"

    # 2. Patch the generation method
    original_generate = trainer._generate_and_score_completions
    def wrapped_generate(self, inputs):
        # Force all tokenizers to left padding before generating
        for obj in [self, self.model, getattr(self, "vllm_pipeline", None)]:
            if hasattr(obj, "tokenizer") and obj.tokenizer is not None:
                obj.tokenizer.padding_side = "left"
        
        # Ensure model's internal padding is left too
        self.model.padding_side = "left"
        if hasattr(self.model, "config"):
            self.model.config.padding_side = "left"
        
        # Call the original method
        return original_generate(inputs)
    trainer._generate_and_score_completions = types.MethodType(wrapped_generate, trainer)

    # 3. Patch the prepare inputs method
    original_prepare_inputs = trainer._prepare_inputs
    def _prepare_inputs_with_padding_check(self, inputs):
        # Ensure tokenizer is using left padding
        if hasattr(self, "tokenizer"):
            self.tokenizer.padding_side = "left"
        
        # Ensure model's padding is left too (critical for Flash Attention)
        self.model.padding_side = "left"
        if hasattr(self.model, "config"):
            self.model.config.padding_side = "left"
        
        return original_prepare_inputs(inputs)
    trainer._prepare_inputs = types.MethodType(_prepare_inputs_with_padding_check, trainer)

    logger.warning("Applied comprehensive Qwen3 padding fixes for distributed training")

    base_repo = derive_base_repo(model_args)


    # 2️⃣  Push-every-checkpoint callback
    push_cb = PushEachCheckpointCallback(
        base_repo_name=None,                 # leave None → will respect CKPT_REPO
        hf_token=HfFolder.get_token(),
        private=False,
    )
    trainer.add_callback(push_cb)



    # -------- Dynamic filter ----------
    if script_args.dynamic_filter:
        dyn_cb = DynamicFilterCallback(max_zero=3)   # or any threshold you like
        trainer.add_callback(dyn_cb)

        # rebuild the training split so that the DataLoader skips
        # examples that have already hit the zero-reward cap
        dataset[script_args.dataset_train_split] = dataset[
            script_args.dataset_train_split
        ].filter(dyn_cb.keep_example, with_indices=True)



    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset[script_args.dataset_train_split])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "dataset_name": script_args.dataset_name,
        "tags": ["open-r1"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    ##########
    # Evaluate
    ##########
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(dataset[script_args.dataset_test_split])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    #############
    # push to hub
    #############
    if training_args.push_to_hub:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
