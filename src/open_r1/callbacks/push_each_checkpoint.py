from __future__ import annotations

import os
import re
import shutil
from pathlib import Path
from typing import Optional

from huggingface_hub import HfApi, upload_folder
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments


class PushEachCheckpointCallback(TrainerCallback):
    """
    Keep disk usage bounded while still pushing every checkpoint to the Hub.

    Sequence per save cycle:
      1. on_step_end      → delete previous checkpoint (frees space)
      2. Trainer / DS     → write new checkpoint
      3. on_save          → push new checkpoint, prune extras, remember as prev
    """

    _ckpt_re = re.compile(r"^checkpoint-(\d+)$")

    # ------------------------------------------------------------------ #
    # init                                                               #
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        base_repo_name: Optional[str] = None,
        hf_token: Optional[str] = None,
        private: bool = False,
        keep_local: int = 0,
    ):
        super().__init__()

        self.base_repo_name = base_repo_name or os.getenv("CKPT_REPO")
        if not self.base_repo_name:
            raise ValueError("Provide `base_repo_name` or set the $CKPT_REPO env var.")

        self.hf_token = hf_token
        self.private = private
        self.keep_local = max(keep_local, 0)
        self.api = HfApi(token=hf_token)

        self._prev_ckpt: Optional[Path] = None  # path of “last” checkpoint folder

    # ------------------------------------------------------------------ #
    # helpers                                                            #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _ckpt_dir_from_kwargs(
        args: TrainingArguments,
        state: TrainerState,
        kwargs: dict,
    ) -> Path:
        """
        Resolve the checkpoint directory path irrespective of transformers
        version (checkpoint_dir / checkpoint / checkpoint_folder changed over
        time).
        """
        return Path(
            kwargs.get("checkpoint_dir")        # transformers ≥ 4.49
            or kwargs.get("checkpoint")         # transformers 4.42-4.48
            or kwargs.get("checkpoint_folder")  # transformers ≤ 4.41
            or Path(args.output_dir) / f"checkpoint-{state.global_step}"
        )

    @staticmethod
    def _find_latest_ckpt(output_dir: Path) -> Optional[Path]:
        ckpts = sorted(
            (p for p in output_dir.iterdir() if p.is_dir() and re.match(r"^checkpoint-\d+$", p.name)),
            key=lambda p: int(p.name.split("-")[1]),
        )
        return ckpts[-1] if ckpts else None

    @staticmethod
    def _delete(path: Path, tag: str = "deleting"):
        if path.exists():
            print(f"[push-cb] {tag}: {path}")
            shutil.rmtree(path, ignore_errors=True)

    # ------------------------------------------------------------------ #
    # callback hooks                                                     #
    # ------------------------------------------------------------------ #
    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        Runs *after* transformers/DeepSpeed have (optionally) loaded a
        `resume_from_checkpoint`.  We scan the output directory and treat the
        latest folder as “previous” so that it can be deleted before the first
        new save.
        """
        if self._prev_ckpt is None:  # only the first time
            latest = self._find_latest_ckpt(Path(args.output_dir))
            if latest is not None:
                self._prev_ckpt = latest
                print(f"[push-cb] will treat {latest} as previous checkpoint")
        return control

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        Called after the backward/update step, **before** the trainer proceeds
        to save/eval/log.  If a save is scheduled and we still have a previous
        folder on disk, delete it now to free space for the upcoming write.
        """
        if control.should_save and self._prev_ckpt is not None:
            self._delete(self._prev_ckpt, tag="deleting previous checkpoint")
            self._prev_ckpt = None            # avoid double-deletes
        return control

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        Runs after the trainer finished writing the new checkpoint.
        1. Push the folder to the Hub
        2. Optionally prune older local checkpoints
        3. Remember this folder for the next cycle
        """
        ckpt_dir = self._ckpt_dir_from_kwargs(args, state, kwargs)
        if not ckpt_dir.exists():
            print(f"[push-cb] expected checkpoint at {ckpt_dir} but it does not exist")
            return control

        # 1️⃣  Push to Hub ---------------------------------------------------
        repo_name = f"{self.base_repo_name}-step{state.global_step}"
        print(f"[push-cb] uploading {ckpt_dir} → {repo_name}")
        self.api.create_repo(repo_name, exist_ok=True, private=self.private)
        upload_folder(
            repo_id=repo_name,
            folder_path=str(ckpt_dir),
            commit_message=f"checkpoint {state.global_step}",
            token=self.hf_token,
        )

        # 2️⃣  prune extra local checkpoints --------------------------------
        if self.keep_local >= 0:
            all_ckpts = sorted(
                (p for p in Path(args.output_dir).iterdir() if self._ckpt_re.match(p.name)),
                key=lambda p: int(p.name.split("-")[1]),
            )
            for old in all_ckpts[:-self.keep_local]:
                if old != ckpt_dir:
                    self._delete(old, tag="pruning checkpoint")

        # 3️⃣  remember for next cycle --------------------------------------
        self._prev_ckpt = ckpt_dir
        return control