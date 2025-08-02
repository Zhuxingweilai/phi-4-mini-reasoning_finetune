from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model
import torch
import json

# Paths (assume model and data are in current working directory)
model_path = "./phi-4-mini-reasoning"
train_file = "./finetune_data.json"
output_dir = "./finetuned_phi4"

# Load dataset
with open(train_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

dataset = [{"instruction": item["instruction"], "input": item["input"], "output": item["output"]} for item in data]
ds = Dataset.from_list(dataset)

# Format prompt as plain text
def format_example(example):
    return {
        "text": f"{example['instruction']}\n{example['input']}\nAnswer: {example['output']}"
    }
ds = ds.map(format_example)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
tokenizer.padding_side = "right"

# Tokenization and label masking
def tokenize(example):
    full_text = example["text"]
    tokens = tokenizer(full_text, truncation=True, padding="max_length", max_length=512)
    input_ids = tokens["input_ids"]
    labels = [-100] * len(input_ids)

    answer_start = full_text.find("Answer:")
    if answer_start != -1:
        answer_tokens = tokenizer(full_text[answer_start:], truncation=True, padding="max_length", max_length=512)["input_ids"]
        labels[-len(answer_tokens):] = answer_tokens

    tokens["labels"] = labels
    return tokens

tokenized_ds = ds.map(tokenize)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    trust_remote_code=True
).cuda()

# LoRA configuration
peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_config)

# Training arguments
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="epoch",
    learning_rate=1e-5,
    fp16=True,
    max_grad_norm=1.0,
    report_to="none"
)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds,
    tokenizer=tokenizer
)

# Start training
trainer.train()

# Save results
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
