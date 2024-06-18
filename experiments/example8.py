from transformers import Trainer, TrainingArguments
from datasets import load_dataset
from transformers import AutoTokenizer, OPTForCausalLM
import numpy as np
from src import SoftDEEPR, convert_to_deep_rewireable, convert_from_deep_rewireable
from src.utils import measure_sparsity

# Load dataset and tokenizer
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Load model
model = OPTForCausalLM.from_pretrained("facebook/opt-125m")
convert_to_deep_rewireable(model, handle_biases='second_bias')

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",  # Set save strategy to "epoch"
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=10,
    weight_decay=0.01,
    save_total_limit=2,
    save_steps=10_000,  # This is redundant now, but can be kept for additional checkpoints
    fp16=True,
    logging_dir='./logs',
    logging_steps=500,
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    greater_is_better=False
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer
)

# Train model with early stopping callback
trainer.train()
