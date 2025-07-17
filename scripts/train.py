

import os

from datasets import load_dataset
from stepwise_reward_model import evaluate_stepwise
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

from trainer.stepwise_dpo_trainer import StepwiseDPOTrainer

MODEL_NAME = "gpt2" # or mistralai/Mistral-7B if enough VRAM
MAX_SAMPLES = 100  # for quick testing — increase later

def preprocess(example):
    prompt = example["problem"]
    steps = example["solution"].split(". ")  # naive split into steps
    steps = [s.strip() for s in steps if s.strip()]
    example["text"] = f"{prompt}\nSteps:\n" + "\n".join([f"{i+1}. {step}" for i, step in enumerate(steps)])
    return example

def main():
    dataset = load_dataset("Mai0313/prm800k", split="train[:{}]".format(MAX_SAMPLES))
    print(dataset[0]) 
    dataset = dataset.map(preprocess)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize(example):
        encoding = tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)
        encoding["labels"] = encoding["input_ids"].copy()
        return encoding

    tokenized_dataset = dataset.map(tokenize)

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=2,
        num_train_epochs=1,
        logging_steps=5,
        save_strategy="no",
        report_to="none",
    )

    trainer = StepwiseDPOTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        reward_model_fn=evaluate_stepwise,
    )

    trainer.train()

if __name__ == "__main__":
    main()
