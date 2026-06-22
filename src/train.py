import json
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
import os
import argparse

parser = argparse.ArgumentParser(description="Fine-tune a language model using QLoRA")

parser.add_argument("--train_data", type=str, default="training_data.jsonl", help="Path to training data in JSONL format")
parser.add_argument("--output_dir", type=str, default="./qwen3-4b-finetuned", help="Directory to save the fine-tuned model")
parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-4B-Instruct-2507", help="Pre-trained model ID or path")
parser.add_argument("--val_split", type=float, default=0.1, help="Fraction of data to use for validation")

# Config
MODEL_ID   = parser.parse_args().model_id
JSONL_PATH = parser.parse_args().train_data
OUTPUT_DIR = parser.parse_args().output_dir
VAL_SPLIT  = parser.parse_args().val_split

# Verify GPU
assert torch.cuda.is_available(), "No GPU detected! Check your CUDA installation."
print(f"Using GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Load and split
with open(JSONL_PATH, "r") as f:
    records = [json.loads(line) for line in f]

dataset = Dataset.from_list(records)
split = dataset.train_test_split(test_size=VAL_SPLIT, seed=42)
train_dataset = split["train"]
val_dataset   = split["test"]

print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")

if not os.path.exists('val_data.jsonl'):
    with open("val_data.jsonl", "w") as f:
        for i in range(len(val_dataset)):
            f.write(json.dumps(val_dataset[i]) + "\n")
    print("Val set saved to val_data.jsonl")

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Model with QLoRA
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="cuda",
    trust_remote_code=True,
)
model.config.use_cache = False

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# SFTConfig (TRL 1.5.x)
sft_config = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    bf16=True,
    fp16=False,
    tf32=True,
    optim="paged_adamw_8bit",
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
    gradient_checkpointing=True,                          # now a native SFTConfig param
    gradient_checkpointing_kwargs={"use_reentrant": False},
    logging_steps=10,
    save_strategy="epoch",
    eval_strategy="epoch",
    eval_on_start=True,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    max_length=2048,                                      # replaces max_seq_length in 1.5.x
    packing=False,
    dataset_text_field=None,
    assistant_only_loss=True,                             # only train on assistant tokens
)

# Train
trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    peft_config=lora_config,
    processing_class=tokenizer,                           # replaces tokenizer=
)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"Trainable params: {trainable:,} ({100 * trainable / total:.2f}% of {total:,})")

trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Done! Best checkpoint saved to {OUTPUT_DIR}")