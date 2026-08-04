#!/usr/bin/env python3
"""Export a LightX2V FastWAM checkpoint for FastWAM LIBERO evaluation.

The FastWAM LIBERO evaluator expects a checkpoint loadable by
`FastWAM.load_checkpoint()` and a nearby `dataset_stats.json`. LightX2V
training checkpoints already save the model payload in `fastwam.pt`; this
script validates that payload and builds a small evaluation bundle:

    <output-dir>/
      dataset_stats.json
      export_manifest.json
      checkpoints/weights/step_XXXXXX.pt

The generated checkpoint path can be passed directly to
`experiments/libero/run_libero_manager.py`.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch


FASTWAM_ROOT = Path(__file__).resolve().parents[1]
FASTWAM_VENV_PYTHON = FASTWAM_ROOT / ".venv" / "bin" / "python"
DEFAULT_LIGHTX2V_CKPT_DIR = Path(
    "/mnt/afs_1/charles/codes/LightX2V_fastwam/"
    "lightx2v_train/runs/fastwam_libero_full/checkpoint-000020000"
)
DEFAULT_OUTPUT_DIR = FASTWAM_ROOT / "runs" / "lightx2v_fastwam_libero_full_export"


def expand_path(path: str | Path) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(str(path)))).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a FastWAM LIBERO evaluation bundle from a LightX2V "
            "FastWAM checkpoint directory or checkpoint file."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_LIGHTX2V_CKPT_DIR,
        help=(
            "LightX2V checkpoint directory containing fastwam.pt, or a direct "
            ".pt checkpoint path."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the FastWAM evaluation bundle will be written.",
    )
    parser.add_argument(
        "--dataset-stats",
        type=Path,
        default=None,
        help=(
            "Path to dataset_stats.json. If omitted, parent directories of the "
            "checkpoint are searched."
        ),
    )
    parser.add_argument(
        "--task",
        default="libero_uncond_2cam224_1e-4",
        help="FastWAM Hydra task config to print or run.",
    )
    parser.add_argument(
        "--python-executable",
        type=Path,
        default=FASTWAM_VENV_PYTHON if FASTWAM_VENV_PYTHON.exists() else Path(sys.executable),
        help=(
            "Python executable used for run_libero_manager.py. Defaults to "
            "FastWAM/.venv/bin/python when it exists."
        ),
    )
    parser.add_argument(
        "--step",
        type=int,
        default=None,
        help="Override the exported step number used in the output filename.",
    )
    parser.add_argument(
        "--mode",
        choices=("hardlink", "copy", "symlink", "resave"),
        default="hardlink",
        help=(
            "How to write the checkpoint. hardlink saves disk when possible; "
            "resave rewrites a clean payload containing only model weights."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing exported checkpoint file.",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=8,
        help="MULTIRUN.num_gpus value for the printed or launched command.",
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=None,
        help="Optional EVALUATION.num_trials override.",
    )
    parser.add_argument(
        "--max-tasks-per-gpu",
        type=int,
        default=None,
        help="Optional MULTIRUN.max_tasks_per_gpu override.",
    )
    parser.add_argument(
        "--output-eval-dir",
        type=Path,
        default=None,
        help="Optional EVALUATION.output_dir override for LIBERO results.",
    )
    parser.add_argument(
        "--extra-override",
        action="append",
        default=[],
        help="Additional Hydra override forwarded to run_libero_manager.py.",
    )
    parser.add_argument(
        "--run-manager",
        action="store_true",
        help="Launch experiments/libero/run_libero_manager.py after export.",
    )
    return parser.parse_args()


def resolve_checkpoint_path(checkpoint: Path) -> Path:
    checkpoint = expand_path(checkpoint)
    if checkpoint.is_file():
        return checkpoint
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Checkpoint path does not exist: {checkpoint}")

    preferred = checkpoint / "fastwam.pt"
    if preferred.exists():
        return preferred

    candidates = sorted(checkpoint.glob("*.pt"))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(f"No .pt checkpoint found under: {checkpoint}")
    raise ValueError(
        "Multiple .pt files found under checkpoint directory; pass one with "
        f"--checkpoint. Candidates: {[str(path) for path in candidates]}"
    )


def load_payload_for_validation(checkpoint_path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(
            checkpoint_path,
            map_location="meta",
            mmap=True,
            weights_only=False,
        )
    except TypeError:
        payload = torch.load(checkpoint_path, map_location="meta")
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint payload must be a dict, got: {type(payload)}")
    return payload


def tensor_summary(state: Any) -> tuple[int, int]:
    if not isinstance(state, dict):
        return 0, 0
    count = 0
    numel = 0
    for value in state.values():
        if torch.is_tensor(value):
            count += 1
            numel += value.numel()
    return count, numel


def validate_payload(payload: dict[str, Any], checkpoint_path: Path) -> dict[str, Any]:
    if "mot" not in payload:
        raise ValueError(f"Checkpoint missing required `mot` key: {checkpoint_path}")
    if not isinstance(payload["mot"], dict):
        raise ValueError("Checkpoint `mot` entry must be a state_dict-like mapping.")

    mot_tensors, mot_numel = tensor_summary(payload["mot"])
    prop_tensors, prop_numel = tensor_summary(payload.get("proprio_encoder"))
    if mot_tensors == 0:
        raise ValueError("Checkpoint `mot` state_dict contains no tensors.")

    return {
        "step": payload.get("step"),
        "torch_dtype": payload.get("torch_dtype"),
        "mot_tensors": mot_tensors,
        "mot_numel": mot_numel,
        "proprio_encoder_tensors": prop_tensors,
        "proprio_encoder_numel": prop_numel,
    }


def infer_step(
    explicit_step: int | None,
    payload_summary: dict[str, Any],
    checkpoint_path: Path,
) -> int | None:
    if explicit_step is not None:
        return explicit_step

    payload_step = payload_summary.get("step")
    if isinstance(payload_step, int):
        return payload_step
    if isinstance(payload_step, str) and payload_step.isdigit():
        return int(payload_step)

    for path in [checkpoint_path, *checkpoint_path.parents]:
        match = re.search(r"(?:step_|checkpoint-)(\d+)", path.name)
        if match:
            return int(match.group(1))
    return None


def resolve_dataset_stats(
    explicit_stats: Path | None,
    checkpoint_path: Path,
) -> Path:
    candidates: list[Path] = []
    if explicit_stats is not None:
        candidates.append(expand_path(explicit_stats))

    for parent in [checkpoint_path.parent, *checkpoint_path.parents]:
        candidates.append(parent / "dataset_stats.json")

    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Could not find dataset_stats.json. Pass it explicitly with "
        "--dataset-stats /path/to/dataset_stats.json."
    )


def prepare_output_path(output_dir: Path, step: int | None) -> tuple[Path, Path]:
    output_dir = expand_path(output_dir)
    weights_dir = output_dir / "checkpoints" / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    filename = f"step_{step:06d}.pt" if step is not None else "fastwam_eval.pt"
    return output_dir, weights_dir / filename


def remove_existing(path: Path) -> None:
    if path.is_symlink() or path.exists():
        path.unlink()


def export_checkpoint(
    source: Path,
    destination: Path,
    mode: str,
    overwrite: bool,
) -> str:
    if destination.exists() or destination.is_symlink():
        if not overwrite:
            raise FileExistsError(
                f"Output checkpoint already exists: {destination}. "
                "Pass --overwrite to replace it."
            )
        remove_existing(destination)

    if mode == "symlink":
        destination.symlink_to(source)
        return "symlink"

    if mode == "hardlink":
        try:
            os.link(source, destination)
            return "hardlink"
        except OSError as exc:
            print(f"Hardlink failed ({exc}); falling back to copy.", file=sys.stderr)
            shutil.copy2(source, destination)
            return "copy"

    if mode == "copy":
        shutil.copy2(source, destination)
        return "copy"

    if mode == "resave":
        payload = torch.load(source, map_location="cpu", weights_only=False)
        clean_payload = {
            "mot": payload["mot"],
            "step": payload.get("step"),
            "torch_dtype": payload.get("torch_dtype"),
        }
        if "proprio_encoder" in payload:
            clean_payload["proprio_encoder"] = payload["proprio_encoder"]
        torch.save(clean_payload, destination)
        return "resave"

    raise ValueError(f"Unsupported export mode: {mode}")


def copy_dataset_stats(source: Path, output_dir: Path, overwrite: bool) -> Path:
    destination = output_dir / "dataset_stats.json"
    if destination.resolve() == source.resolve():
        return destination
    if destination.exists():
        if not overwrite:
            raise FileExistsError(
                f"dataset_stats.json already exists: {destination}. "
                "Pass --overwrite to replace it."
            )
        destination.unlink()
    shutil.copy2(source, destination)
    return destination


def write_manifest(
    output_dir: Path,
    source_checkpoint: Path,
    exported_checkpoint: Path,
    source_stats: Path,
    exported_stats: Path,
    payload_summary: dict[str, Any],
    export_method: str,
) -> Path:
    manifest = {
        "source_checkpoint": str(source_checkpoint),
        "exported_checkpoint": str(exported_checkpoint),
        "source_dataset_stats": str(source_stats),
        "exported_dataset_stats": str(exported_stats),
        "export_method": export_method,
        "payload_summary": payload_summary,
    }
    destination = output_dir / "export_manifest.json"
    destination.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return destination


def build_manager_command(args: argparse.Namespace, ckpt: Path, stats: Path) -> list[str]:
    command = [
        str(expand_path(args.python_executable)),
        "experiments/libero/run_libero_manager.py",
        f"task={args.task}",
        f"ckpt={ckpt}",
        f"EVALUATION.dataset_stats_path={stats}",
        f"MULTIRUN.num_gpus={int(args.num_gpus)}",
    ]
    if args.num_trials is not None:
        command.append(f"EVALUATION.num_trials={int(args.num_trials)}")
    if args.max_tasks_per_gpu is not None:
        command.append(f"MULTIRUN.max_tasks_per_gpu={int(args.max_tasks_per_gpu)}")
    if args.output_eval_dir is not None:
        command.append(f"EVALUATION.output_dir={expand_path(args.output_eval_dir)}")
    command.extend(args.extra_override)
    return command


def print_command(command: list[str]) -> None:
    print("\nRun LIBERO evaluation with:")
    print(f"cd {FASTWAM_ROOT}")
    print(shlex.join(command))


def main() -> None:
    args = parse_args()
    source_checkpoint = resolve_checkpoint_path(args.checkpoint)
    payload = load_payload_for_validation(source_checkpoint)
    payload_summary = validate_payload(payload, source_checkpoint)
    step = infer_step(args.step, payload_summary, source_checkpoint)
    output_dir, exported_checkpoint = prepare_output_path(args.output_dir, step)
    source_stats = resolve_dataset_stats(args.dataset_stats, source_checkpoint)

    export_method = export_checkpoint(
        source=source_checkpoint,
        destination=exported_checkpoint,
        mode=args.mode,
        overwrite=bool(args.overwrite),
    )
    exported_stats = copy_dataset_stats(
        source=source_stats,
        output_dir=output_dir,
        overwrite=bool(args.overwrite),
    )
    manifest = write_manifest(
        output_dir=output_dir,
        source_checkpoint=source_checkpoint,
        exported_checkpoint=exported_checkpoint,
        source_stats=source_stats,
        exported_stats=exported_stats,
        payload_summary=payload_summary,
        export_method=export_method,
    )

    print("Export complete.")
    print(f"Source checkpoint: {source_checkpoint}")
    print(f"Exported checkpoint: {exported_checkpoint}")
    print(f"Dataset stats: {exported_stats}")
    print(f"Manifest: {manifest}")
    print(
        "Payload: "
        f"step={payload_summary.get('step')}, "
        f"dtype={payload_summary.get('torch_dtype')}, "
        f"mot_tensors={payload_summary['mot_tensors']}, "
        f"proprio_encoder_tensors={payload_summary['proprio_encoder_tensors']}"
    )

    command = build_manager_command(args, exported_checkpoint, exported_stats)
    print_command(command)

    if args.run_manager:
        subprocess.run(command, cwd=FASTWAM_ROOT, check=True)


if __name__ == "__main__":
    main()
