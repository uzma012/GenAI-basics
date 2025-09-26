from transformers import AutoTokenizer, pipeline
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM

# Load fine-tuned model
base_model = AutoModelForCausalLM.from_pretrained("gpt2")
model = PeftModel.from_pretrained(base_model, "./lora-gpt2")
tokenizer = AutoTokenizer.from_pretrained("./lora-gpt2")
tokenizer.pad_token = tokenizer.eos_token

generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Example query
prompt = "Instruction: Translate this English sentence to French.\nInput: I enjoy studying AI.\nOutput:"
print(generator(prompt, max_new_tokens=50)[0]["generated_text"])
