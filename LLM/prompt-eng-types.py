from llm_client import get_groq_client

client = get_groq_client()


model = "llama-3.3-70b-versatile"


# simple zero-shot example
resp = client.chat.completions.create(
    model=model,  # a lightweight Groq model
    messages=[{"role": "user", "content": "Translate to French: I enjoy studying AI."}]
)

print("Zero-shot:", resp.choices[0].message.content)

print("\n" + "="*50 + "\n")

# few-shot example
prompt = """Translate English to French:
English: I love programming.
French: J'adore programmer.

English: How are you?
French: Comment ça va?

English: I enjoy studying AI.
French:"""

resp = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": prompt}]
)

print("Few-shot:", resp.choices[0].message.content)
print("\n" + "="*50 + "\n")

# chain-of-thought example
prompt = """Let's reason step by step.
Task: Translate 'I enjoy studying AI' into French.
Answer:"""

resp = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": prompt}]
)

print("Chain-of-Thought:", resp.choices[0].message.content)
print("\n" + "="*50 + "\n")

#Role-play example

prompt = """You are an expert French translator.
Translate the following into French: I enjoy studying AI."""

resp = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": prompt}]
)

print("Role-based:", resp.choices[0].message.content)

print("\n" + "="*50 + "\n")

#output formate control example
prompt = """Translate this into French and return in JSON:
Text: I enjoy studying AI."""

resp = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": prompt}]
)

print("JSON Output:", resp.choices[0].message.content)



# temprature variation example

prompt = "Write a short sentence about AI."

for temp in [0.0, 0.7, 1.2]:
    resp = client.chat.completions.create(
        model= model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temp,
        max_tokens=30
    )
    print(f"Temp={temp}:", resp.choices[0].message.content)


#Output
# Temp=0.0 (deterministic): Artificial intelligence is rapidly changing the way we live and work by automating tasks and providing innovative solutions.
# Temp=0.7 (default): Artificial intelligence (AI) is rapidly transforming numerous industries with its advanced capabilities and machine learning algorithms.
# Temp=1.2 (highly creative): Artificial intelligence (AI) has revolutionized various industries with its ability to learn, analyze, and make decisions.