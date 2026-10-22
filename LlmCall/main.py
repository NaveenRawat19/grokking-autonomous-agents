import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
from json_repair import loads

load_dotenv("../.env")

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("API Key cannot be NULL")

client = Groq(api_key=api_key)

messages = [
    {
        "role": "system",
        "content": "You are a greatest ever Psychologist counseling the patient. You are understanding the mental situation of the person and trying to understand the root cause of the suffering by asking the relevant questions."
    }
]

while True:
    user_input = input("Your Input: ")
    
    if user_input.lower() in {"exit", "quit", "stop", "end"}:
        break
    
    messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    response = client.chat.completions.create(
        messages=messages,
        model="openai/gpt-oss-120b",
    )

    assistant_reply = response.choices[0].message.content

    print("Assistant: ", assistant_reply)

    messages.append({
        "role": "assistant",
        "content": assistant_reply
    })
