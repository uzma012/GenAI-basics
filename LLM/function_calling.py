from llm_client import get_groq_client
import json

client = get_groq_client()

# Define available functions
functions = [
    {
        "name": "get_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City and country, e.g., 'Paris, France'"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature unit"
                }
            },
            "required": ["location", "unit"]
        }
    }
]
model = "llama-3.3-70b-versatile"  # update to your available Groq model
# Fake weather function (you can connect to real API later)
def get_weather(location, unit):
    return {"location": location, "temperature": "22", "unit": unit}


def chat_with_groq(user_prompt: str):
    """Handles conversation with Groq and function calling"""
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": user_prompt}],
        functions=functions,
        function_call="auto"  # let model decide
    )

    msg = response.choices[0].message

    # If model triggered a function call
    if hasattr(msg, "function_call") and msg.function_call:
        function_name = msg.function_call.name
        args = json.loads(msg.function_call.arguments)

        print(f"\n🔧 Model decided to call function: {function_name} with args {args}")

        if function_name == "get_weather":
            result = get_weather(**args)

            # Call model again with function result
            second_response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": user_prompt},
                    msg,  
                    {"role": "function", "name": function_name, "content": json.dumps(result)}
                ]
            )
            return second_response.choices[0].message.content

    # If no function call, just return model’s answer
    return msg.content


# ---- Run Chat ----
if __name__ == "__main__":
    user_prompt = input("Enter your prompt: ")
    answer = chat_with_groq(user_prompt)
    print("\n🤖 LLM Response:", answer)
