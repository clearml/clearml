import argparse
import logging
from transformers import AutoConfig
from clearml import Task, Dataset, InputModel, OutputModel
from pathlib import Path

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    default_data_collator,
)
from peft import LoraConfig, get_peft_model, TaskType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="LoRA fine-tuning script for Qwen2.5-Coder-0.5B-Instruct"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="ID of the ClearML dataset used for finetuning",
    )
    parser.add_argument("--model", type=str, help="ID of the ClearML model to finetune")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="lora-qwen-output",
        help="Where to save LoRA adapters and final/merged model",
    )
    parser.add_argument(
        "--num_epochs", type=int, default=3, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Training batch size per device"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=3e-4, help="Learning rate for optimizer"
    )
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument(
        "--lora_dropout", type=float, default=0.05, help="LoRA dropout rate"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum sequence length (prompt+completion+eos)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    task = Task.init(project_name="Example", task_name="Finetune", output_uri=True)
    task.set_packages("requirements.txt")
    task.execute_remotely(queue_name="some_queue")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load ClearML dataset
    logger.info(f"Loading dataset from {args.dataset}")
    dataset = Dataset.get(args.dataset)
    dataset_files = dataset.get_local_copy()
    raw_datasets = load_dataset(
        "json", data_files={"train": dataset_files + "/dataset.jsonl"}
    )

    logger.info(f"Loading model and tokenizer: {args.model}")
    clearml_model = InputModel(args.model)
    clearml_model_files = clearml_model.get_local_copy()
    tokenizer = AutoTokenizer.from_pretrained(
        clearml_model_files, trust_remote_code=True
    )
    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({"eos_token": "</s>"})

    model = AutoModelForCausalLM.from_pretrained(
        clearml_model_files,
        quantization_config=None,
        device_map="auto",
        trust_remote_code=True,
    )

    # Configure LoRA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Tokenization
    max_length = args.max_length

    def tokenize_fn(example):
        text = example["prompt"] + example["completion"] + tokenizer.eos_token
        tokenized = tokenizer(
            text, truncation=True, max_length=max_length, padding="max_length"
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    tokenized_datasets = raw_datasets["train"].map(
        tokenize_fn, batched=False, remove_columns=raw_datasets["train"].column_names
    )

    # Data collator for padded inputs
    data_collator = default_data_collator

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(out_dir / "adapter"),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        fp16=True,
        logging_steps=50,
        save_steps=500,
        save_total_limit=3,
        remove_unused_columns=False,
        report_to="none",
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
        data_collator=data_collator,
    )

    # Train
    trainer.train()

    # Save LoRA adapter
    adapter_dir = out_dir / "adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    # Merge LoRA weights into base model and save merged
    logger.info("Merging LoRA weights into the base model")
    merged_model = model.merge_and_unload()
    merged_dir = out_dir / "merged"
    merged_dir.mkdir(exist_ok=True)
    merged_model.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)

    config = AutoConfig.from_pretrained(clearml_model_files, trust_remote_code=True)
    config.save_pretrained(merged_dir)

    logger.info(
        f"LoRA fine-tuning complete. Adapter saved to {adapter_dir}, merged model saved to {merged_dir}. Uploading to ClearML..."
    )
    merged_model = OutputModel(name="Example Merged Weights")
    merged_model.update_weights_package(weights_path=merged_dir)


if __name__ == "__main__":
    main()
