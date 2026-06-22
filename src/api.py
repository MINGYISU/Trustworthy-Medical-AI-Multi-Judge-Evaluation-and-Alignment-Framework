from openai import OpenAI
import os
import json
from dotenv import load_dotenv
import time

from utils import *

load_dotenv()

def read_jsonl(file_path):
    try:
        with open(file_path, "r") as f:
            return [json.loads(line) for line in f]
    except Exception as e:
        prRed(f"Error reading file {file_path}: {e}")
        return []

def save_jsonl(data, file_path):
    data = dict(sorted(data.items(), key=lambda item: int(item[0])))
    with open(file_path, "w") as f:
        for item in data:
            json.dump(item, f)
            f.write("\n")

def generator_prompt(
        system_msg, 
        user_msg, 
        template="{}\nThe following is the user's message:\n{}"):
    return template.format(system_msg, user_msg)

def evaluation_prompt(
        evaluator, 
        input, 
        generated_resp):
    return evaluator.format(model_input=input, model_output=generated_resp)

def init_client(url="https://openrouter.ai/api/v1"):
    # Using services from openrouter.ai by default, change to your provider's base URL if needed
    return OpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url=url
    )

def api_call(client, model, prompt):
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
            "role": "user",
            "content": prompt
            }
        ]
    )
    return completion.choices[0].message.content.strip()
    
def safe_call(client, model, prompt, retries=3, wait_time=1):
    for i in range(retries):
        try:
            return api_call(client, model, prompt)
        except Exception as e:
            prRed(f"API call failed: {e}")
            if i < retries - 1:
                prYellow(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                wait_time *= 2  # Exponential backoff
    prRed("All retries failed. Returning None.")
    return None

def test_call(*args, **kwargs):
    prGreen("This is a test call to verify the API client is working.")
    return "Test successful!"