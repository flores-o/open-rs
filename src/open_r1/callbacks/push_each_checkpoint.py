# src/open_r1/callbacks/push_each_checkpoint.py
from __future__ import annotations

import os
import shutil
from pathlib import Path

from huggingface_hub import HfApi, upload_folder
from transformers import TrainerCallback


class PushEachCheckpointCallback(TrainerCallback):
    """
    • Loads *from* whatever model you want, but pushes every saved checkpoint
      to a repo you control.

    Priority for the target repo name (first non-empty wins):
        1. The ``base_repo_name`` argument
        2. Environment variable ``CKPT_REPO``
        3. Derivation from the loaded model name (old behaviour)

    It supports every recent Transformers version:

        * ≤ 4.41  →  ``kwargs["checkpoint_folder"]``
        * 4.42–4.48 → ``kwargs["checkpoint"]``
        * ≥ 4.49  →  ``kwargs["checkpoint_dir"]``
    """

    def __init__(
        self,
        base_repo_name: str | None = None,
        hf_token: str | None = None,
        private: bool = False,
        keep_local: int = 1,
    ):
        super().__init__()
        self.base_repo_name = base_repo_name or os.getenv("CKPT_REPO")
        self.hf_token = hf_token
        self.private = private
        self.keep_local = keep_local
        self.api = HfApi(token=hf_token)

    # --------------------------------------------------------------------- #
    # Hooks                                                                 #
    # --------------------------------------------------------------------- #
    def on_save(self, args, state, control, **kwargs):
        ckpt_dir = Path(
            kwargs.get("checkpoint_dir")        # transformers ≥ 4.49
            or kwargs.get("checkpoint")         # transformers 4.42-4.48
            or kwargs.get("checkpoint_folder")  # transformers ≤ 4.41
            or Path(args.output_dir) / f"checkpoint-{state.global_step}"
        )

        if not ckpt_dir.exists():
            print(f"[push-cb] expected checkpoint at {ckpt_dir} but it wasn’t found")
            return control

        # ------------------------------------------------------------------ #
        # 1.  Push this checkpoint to the Hub                                #
        # ------------------------------------------------------------------ #
        repo_name = f"{self.base_repo_name}-step{state.global_step}"
        print(f"[push-cb] uploading {ckpt_dir}  →  {repo_name}")

        self.api.create_repo(repo_name, exist_ok=True, private=self.private)
        upload_folder(
            repo_id=repo_name,
            folder_path=str(ckpt_dir),
            commit_message=f"checkpoint {state.global_step}",
            token=self.hf_token,
        )

        # ------------------------------------------------------------------ #
        # 2.  Optionally prune older local checkpoints                       #
        # ------------------------------------------------------------------ #
        if self.keep_local >= 0:
            all_ckpts = sorted(
                Path(args.output_dir).glob("checkpoint-*"),
                key=lambda p: int(p.name.split("-")[-1]),
            )
            for old in all_ckpts[:-self.keep_local]:
                print(f"[push-cb] deleting local {old}")
                shutil.rmtree(old, ignore_errors=True)

        return control
