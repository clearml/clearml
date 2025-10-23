"""
Fine-tune a base LLM with LoRA adapters on HyperDataset Q&A entries using Hugging Face's Trainer.

This variant keeps the ClearML multi-node bootstrap from the original script but delegates the
training loop to `transformers.Trainer` for a more compact implementation.

Example local run (single node, all GPUs):
    python finetune_qa_lora.py --devices-per-node -1

Example ClearML (enqueue):
    python finetune_qa_lora.py --queue <your_queue> --num-nodes 2 --devices-per-node -1 --dist-port 29500
"""

from __future__ import annotations

import argparse
import os
import uuid
from typing import Any, Dict, Iterable, Optional

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import IterableDataset

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
)
from transformers.trainer_utils import set_seed

from clearml import Task, OutputModel
from clearml.hyperdatasets import (
    DataEntry,
    DataSubEntry,
    DataView,
    HyperDatasetManagement,
)

try:
    from peft import LoraConfig, get_peft_model
except ImportError as exc:
    raise RuntimeError("peft is required. Install with `pip install peft`.") from exc


class QADataSubEntry(DataSubEntry):
    """Simple text sub-entry carrying a role label for each snippet."""

    def __init__(self, name: str, text: str, role: str):
        super().__init__(
            name=name,
            source=f"text://{uuid.uuid4().hex}",
            metadata={"text": text, "role": role},
        )


class QADataEntry(DataEntry):
    """Structured representation of a Q&A pair stored on a HyperDataset."""

    def __init__(
        self,
        question: str,
        answer: str,
        *,
        reference: Optional[str] = None,
        tags: Optional[Iterable[str]] = None,
    ) -> None:
        metadata: Dict[str, Any] = {"question": question, "answer": answer}
        if reference:
            metadata["reference"] = reference
        if tags:
            metadata["tags"] = list(tags)
        super().__init__(metadata=metadata)

    @property
    def question(self) -> str:
        return self._metadata.get("question", "")

    @property
    def answer(self) -> str:
        return self._metadata.get("answer", "")


class QALoraIterableDataset(IterableDataset):
    """Stream Q&A pairs from a DataView and tokenize for causal LM fine-tuning."""

    def __init__(
        self,
        query_kwargs: Dict[str, Any],
        tokenizer: AutoTokenizer,
        *,
        max_length: int,
        prompt_template: str,
        label_pad_token_id: int = -100,
        dist_world_size: int = 1,
        dist_rank: int = 0,
    ) -> None:
        super().__init__()
        self._query_kwargs = dict(query_kwargs)
        self._tokenizer = tokenizer
        self._max_length = max_length
        self._prompt_template = prompt_template
        self._label_pad_token_id = label_pad_token_id
        self._dist_world_size = max(1, int(dist_world_size))
        self._dist_rank = int(dist_rank)

        dv = DataView(auto_connect_with_task=True)
        dv.add_query(**self._query_kwargs)
        worker_info = torch.utils.data.get_worker_info()
        if worker_info:
            per_process_workers = max(worker_info.num_workers, 1)
            total_workers = self._dist_world_size * per_process_workers
            worker_index = self._dist_rank * per_process_workers + worker_info.id
        elif self._dist_world_size > 1:
            total_workers = self._dist_world_size
            worker_index = self._dist_rank
        else:
            total_workers = None
            worker_index = None
        self._dataview_iterator = dv.get_iterator(
            num_workers=total_workers,
            worker_index=worker_index,
            cache_in_memory=True
        )

    def __iter__(self) -> Iterable[Dict[str, torch.Tensor]]:
        for entry in self._dataview_iterator:
            if not isinstance(entry, QADataEntry):
                continue
            qa_entry = entry
            prompt = self._prompt_template.format(
                question=qa_entry.question,
                answer=qa_entry.answer,
            )
            encoding = self._tokenizer(
                prompt,
                truncation=True,
                max_length=self._max_length,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids = encoding["input_ids"].squeeze(0)
            attention_mask = encoding["attention_mask"].squeeze(0)
            labels = input_ids.clone()
            labels[attention_mask == 0] = self._label_pad_token_id
            yield {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }


def resolve_dataset(args: argparse.Namespace, *, prefix: str = ""):
    dataset_name = getattr(args, f"{prefix}dataset_name")
    dataset_id = getattr(args, f"{prefix}dataset_id")
    version_name = getattr(args, f"{prefix}version_name")
    version_id = getattr(args, f"{prefix}version_id")
    project_name = getattr(args, f"{prefix}project", None) or args.project

    if dataset_name and dataset_id:
        raise ValueError("Provide either --dataset-name or --dataset-id, not both")
    if version_name and version_id:
        raise ValueError("Provide either --version-name or --version-id, not both")

    if dataset_id:
        return HyperDatasetManagement.get(dataset_id=dataset_id, version_id=version_id)
    if not dataset_name:
        raise ValueError("Either --dataset-id or --dataset-name must be supplied")
    return HyperDatasetManagement.get(
        dataset_name=dataset_name,
        version_name=version_name,
        project_name=project_name,
    )


def prepare_model(args: argparse.Namespace):
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    elif tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

    model = AutoModelForCausalLM.from_pretrained(args.model)
    if getattr(model.config, "use_cache", None):
        model.config.use_cache = False

    lora_config = LoraConfig(
        r=args.lora_r if hasattr(args, "lora_r") else 8,
        lora_alpha=args.lora_alpha if hasattr(args, "lora_alpha") else 32.0,
        lora_dropout=args.lora_dropout if hasattr(args, "lora_dropout") else 0.05,
        target_modules=list(getattr(args, "lora_target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    if int(os.environ.get("RANK", "0")) == 0:
        model.print_trainable_parameters()
    return model, tokenizer


class _LimitedIterableDataset(IterableDataset):
    """Wrap an iterable dataset and stop after yielding ``max_items`` samples."""

    def __init__(self, dataset: IterableDataset, max_items: Optional[int] = None) -> None:
        self._dataset = dataset
        self._max_items = max_items

    def __iter__(self) -> Iterable[Dict[str, torch.Tensor]]:
        if self._max_items is None:
            yield from self._dataset
            return
        for idx, sample in enumerate(self._dataset):
            if idx >= self._max_items:
                break
            yield sample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune a causal LM using HyperDataset Q&A entries with LoRA (Trainer variant)"
    )
    name_group = parser.add_argument_group("Name-based selection")
    name_group.add_argument("--project", default=None, help="ClearML project containing the HyperDataset")
    name_group.add_argument("--dataset-name", default=None, help="HyperDataset collection name")
    name_group.add_argument("--version-name", default=None, help="HyperDataset version name")

    id_group = parser.add_argument_group("ID-based selection")
    id_group.add_argument("--dataset-id", default=None, help="HyperDataset collection id")
    id_group.add_argument("--version-id", default=None, help="HyperDataset version id")

    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct", help="Base causal LM to fine-tune")
    parser.add_argument("--batch-size", type=int, default=2, help="Per-device training batch size")
    parser.add_argument("--max-length", type=int, default=1024, help="Maximum sequence length")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Optimizer learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay value")
    parser.add_argument("--max-steps", type=int, default=100, help="Total optimization steps (set -1 to disable)")
    parser.add_argument("--warmup-steps", type=int, default=10, help="Scheduler warmup steps")
    parser.add_argument("--gradient-accumulation", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker processes")
    parser.add_argument("--max-epochs", type=int, default=3, help="Maximum passes over the dataset")
    parser.add_argument(
        "--prompt-template", default="### Question:\n{question}\n\n### Answer:\n{answer}\n", help="Prompt template"
    )
    parser.add_argument("--output-dir", default=None, help="Where to store the LoRA adapters after training")
    parser.add_argument("--device", default=None, help="Torch device override (must be 'cuda' or omitted)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--log-interval", type=int, default=10, help="Trainer logging interval (in optimizer steps)")
    parser.add_argument("--lora-r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=float, default=32.0, help="LoRA alpha scaling")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout probability")
    parser.add_argument(
        "--lora-target-modules",
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj"],
        help="LoRA target modules within the attention blocks",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        help="Merge LoRA adapters into the base model and upload it as a ClearML OutputModel",
    )

    # ClearML / distributed
    parser.add_argument("--queue", type=str, default=None, help="ClearML queue to enqueue to (runs remotely)")
    parser.add_argument("--num-nodes", type=int, default=1, help="Total number of nodes")
    parser.add_argument("--devices-per-node", type=int, default=-1, help="-1: all visible GPUs; else number")
    parser.add_argument("--dist-port", type=int, default=29500, help="Rendezvous port (local multi-GPU)")

    # Eval dataset (optional)
    parser.add_argument("--skip-eval", action="store_true", help="Skip evaluation after training")
    parser.add_argument("--eval-batch-limit", type=int, default=None, help="Evaluate on at most this many batches")
    parser.add_argument(
        "--eval-project", default=None, help="Project for the evaluation dataset (defaults to training project)"
    )
    parser.add_argument("--eval-dataset-name", default=None, help="Name of the evaluation dataset")
    parser.add_argument("--eval-version-name", default=None, help="Version name of the evaluation dataset")
    parser.add_argument("--eval-dataset-id", default=None, help="ID of the evaluation dataset")
    parser.add_argument("--eval-version-id", default=None, help="Version ID of the evaluation dataset")

    return parser.parse_args()


def _build_datasets(
    args: argparse.Namespace,
    tokenizer,
    query_kwargs: Dict[str, str],
    eval_query_kwargs: Optional[Dict[str, str]],
) -> tuple[IterableDataset, Optional[IterableDataset]]:
    train_dataset = QALoraIterableDataset(
        query_kwargs=query_kwargs,
        tokenizer=tokenizer,
        max_length=args.max_length,
        prompt_template=args.prompt_template,
        dist_world_size=1,
        dist_rank=0,
    )

    if args.skip_eval or eval_query_kwargs is None:
        return train_dataset, None

    eval_dataset: IterableDataset = QALoraIterableDataset(
        query_kwargs=eval_query_kwargs,
        tokenizer=tokenizer,
        max_length=args.max_length,
        prompt_template=args.prompt_template,
        dist_world_size=1,
        dist_rank=0,
    )
    if args.eval_batch_limit:
        max_items = args.eval_batch_limit * args.batch_size
        eval_dataset = _LimitedIterableDataset(eval_dataset, max_items=max_items)

    return train_dataset, eval_dataset


def _trainer_process_entry(
    local_rank: int,
    args: argparse.Namespace,
    task_id: str,
    node_rank: int,
    gpus_per_node: int,
    world_size: int,
) -> None:
    global_rank = int(os.environ.get("RANK", "0"))
    if world_size > 1:
        os.environ["LOCAL_RANK"] = str(local_rank)

    if args.device and args.device.lower() not in {None, "cuda"}:
        raise ValueError("Only CUDA devices are supported in this script. Omit --device or set it to 'cuda'.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA must be available to run this script; NCCL backend requires GPUs.")

    torch.cuda.set_device(local_rank)

    if world_size > 1 and dist.is_available() and not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=global_rank,
        )

    task = Task.current_task() or Task.init(project_name="Finetune", task_name="Finetune", task_id=task_id)
    task.get_logger()

    set_seed(args.seed + global_rank)

    dataset = resolve_dataset(args)
    eval_dataset = dataset
    has_eval_override = any(
        getattr(args, name) is not None
        for name in ("eval_dataset_name", "eval_dataset_id", "eval_version_name", "eval_version_id", "eval_project")
    )
    if has_eval_override:
        eval_dataset = resolve_dataset(args, prefix="eval_")

    model, tokenizer = prepare_model(args)

    query_kwargs = {
        "project_id": dataset.project_id or "*",
        "dataset_id": dataset.dataset_id,
        "version_id": dataset.version_id,
    }

    eval_query_kwargs = None
    if not args.skip_eval:
        eval_query_kwargs = {
            "project_id": eval_dataset.project_id or query_kwargs["project_id"],
            "dataset_id": eval_dataset.dataset_id,
            "version_id": eval_dataset.version_id,
        }

    train_dataset, eval_dataset_obj = _build_datasets(args, tokenizer, query_kwargs, eval_query_kwargs)

    default_output_dir = args.output_dir or os.path.join("outputs", task_id)
    if global_rank == 0:
        os.makedirs(default_output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=default_output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        num_train_epochs=args.max_epochs,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        logging_steps=max(1, args.log_interval),
        logging_first_step=True,
        save_strategy="no",
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=True,
        report_to=["clearml"],
        lr_scheduler_type="linear",
        fp16=torch.cuda.is_available(),
        remove_unused_columns=False,
        seed=args.seed,
        gradient_checkpointing=False,
        max_grad_norm=1.0,
        disable_tqdm=not (global_rank == 0),
        ddp_find_unused_parameters=False,
        ddp_backend="nccl" if world_size > 1 else None,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset_obj,
        data_collator=default_data_collator,
    )

    train_result = trainer.train()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_state()

    if not args.skip_eval and eval_dataset_obj is not None:
        metrics = trainer.evaluate(eval_dataset=eval_dataset_obj)
        trainer.log_metrics("eval", metrics)

    if trainer.is_world_process_zero():
        trainer.save_model(default_output_dir)
        tokenizer.save_pretrained(default_output_dir)
        if getattr(args, "save_model", False):
            merged_dir = os.path.join(default_output_dir, "merged_model")
            os.makedirs(merged_dir, exist_ok=True)
            try:
                merged_model = trainer.model.merge_and_unload()
                merged_model.save_pretrained(merged_dir)
                tokenizer.save_pretrained(merged_dir)
                model_name = f"{os.path.basename(args.model)}-finetuned"
                output_model = OutputModel(task=task, name=model_name, framework="PyTorch")
                output_model.update_weights_package(weights_path=merged_dir, async_enable=False)
                task.get_logger().report_text(
                    f"Merged model uploaded to ClearML model repository as '{output_model.name or model_name}'."
                )
            except Exception as exc:  # pragma: no cover
                task.get_logger().report_text(f"Failed to merge and upload merged model: {exc}")
                raise

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def main() -> None:
    args = parse_args()
    if args.num_nodes < 1:
        raise ValueError("--num-nodes must be at least 1")
    if args.devices_per_node == 0 or args.devices_per_node < -1:
        raise ValueError("--devices-per-node must be -1 (all GPUs) or a positive integer")

    Task.set_resource_monitor_iteration_timeout(
        seconds_from_start=1,
        wait_for_first_iteration_to_start_sec=1,
        max_wait_for_first_iteration_to_start_sec=1,
    )
    task = Task.init(project_name="Finetune", task_name="Finetune", output_uri=True)

    if args.queue:
        task.set_packages("./requirements.txt")
        task.set_base_docker(
            docker_image="pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime",
        )
        task.execute_remotely(queue_name=args.queue)

    node_rank = 0
    if args.num_nodes > 1:
        task.launch_multi_node(args.num_nodes, port=args.dist_port, devices=args.devices_per_node, wait=True)
        node_rank = int(os.environ.get("NODE_RANK", "0"))

    visible_gpus = torch.cuda.device_count()
    if visible_gpus < 1:
        raise RuntimeError("No CUDA devices visible inside the process/container.")
    gpus_per_node = visible_gpus if args.devices_per_node in (-1, None) else min(args.devices_per_node, visible_gpus)
    if gpus_per_node < 1:
        raise RuntimeError("devices_per_node resolved to <1 GPU. Set --devices-per-node or expose GPUs.")

    world_size = gpus_per_node * max(1, args.num_nodes)

    if world_size > 1:
        mp.spawn(
            _trainer_process_entry,
            nprocs=gpus_per_node,
            args=(args, task.id, node_rank, gpus_per_node, world_size),
            join=True,
        )
    else:
        _trainer_process_entry(0, args, task.id, node_rank, gpus_per_node, world_size)


if __name__ == "__main__":
    main()
