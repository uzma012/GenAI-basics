

from llm_client import get_groq_client

client = get_groq_client()

# Now you can use `client` normally

def call_groq(prompt: str, model="llama-3.3-70b-versatile", max_tokens=256, temperature=0.0):
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers questions using provided context."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return resp.choices[0].message.content

# usage
from rag_query import assemble_prompt, retrieve, rerank
q = "provide Key aspects of AI"
retrieved = retrieve(q, k=8)
reranked = rerank(q, retrieved)
prompt = assemble_prompt(q, reranked[:4])
answer = call_groq(prompt)
print(answer)
