from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
import torch

# Load model and tokenizer
model_name = "meta-llama/Llama-2-7b-hf"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# GPT2 has no pad token → set it manually
tokenizer.pad_token = tokenizer.eos_token

# Load dataset
dataset = load_dataset("json", data_files="finetune_data.json")["train"]

def tokenize(batch):
    input_texts = []
    label_texts = []

    for instr, inp, out in zip(batch["instruction"], batch["input"], batch["output"]):
        if inp.strip():
            prompt = f"Instruction: {instr}\nInput: {inp}\nAnswer:"
        else:
            prompt = f"Instruction: {instr}\nAnswer:"
        
        input_texts.append(prompt)
        label_texts.append(out)

    # Tokenize inputs
    model_inputs = tokenizer(
        input_texts,
        truncation=True,
        padding="max_length",
        max_length=128,
    )

    # Tokenize labels separately
    labels = tokenizer(
        label_texts,
        truncation=True,
        padding="max_length",
        max_length=128,
    )["input_ids"]

    model_inputs["labels"] = labels
    return model_inputs

dataset = dataset.map(tokenize, batched=True)

# LoRA config
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# Training args
training_args = TrainingArguments(
    output_dir="./lora-gpt2",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    learning_rate=2e-4,
    logging_steps=10,
    save_steps=200,
    save_total_limit=2,
    fp16=True,
)

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
)
trainer.train()

# Save
model.save_pretrained("./lora-gpt2")
tokenizer.save_pretrained("./lora-gpt2")
