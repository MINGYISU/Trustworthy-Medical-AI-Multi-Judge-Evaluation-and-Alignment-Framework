import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Config
FINETUNED_MODEL_DIR = "./qwen3-4b-finetuned"
VAL_DATA_PATH       = "val_data.jsonl"
EVAL_OUTPUT_PATH    = "eval_results.json"
MAX_EVAL_SAMPLES    = 50

# Verify GPU
assert torch.cuda.is_available(), "No GPU detected!"
print(f"Using GPU: {torch.cuda.get_device_name(0)}")

# Load fine-tuned model on GPU
tokenizer = AutoTokenizer.from_pretrained(FINETUNED_MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(
    FINETUNED_MODEL_DIR,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
)
model.eval()

# Inference helper
def generate_response(messages: list[dict]) -> str:
    input_messages = [m for m in messages if m["role"] != "assistant"]
    
    # apply_chat_template returns a BatchEncoding when return_tensors="pt"
    # so we need to extract input_ids explicitly
    inputs = tokenizer.apply_chat_template(
        input_messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,           # returns a BatchEncoding with input_ids + attention_mask
    ).to("cuda")

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,               # unpacks input_ids and attention_mask
            max_new_tokens=512,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

with open(VAL_DATA_PATH, "r") as f:
    records = [json.loads(line) for line in f]

eval_results = []
for i, record in enumerate(records[:MAX_EVAL_SAMPLES]):
    print(f"Evaluating sample {i+1}/{min(MAX_EVAL_SAMPLES, len(records))}...")
    messages = record["messages"]
    reference = next(m["content"] for m in messages if m["role"] == "assistant")
    user_msg  = next(m["content"] for m in messages if m["role"] == "user")

    prediction = generate_response(messages)
    eval_results.append({
    "index":      i,
    "user":       user_msg,
    "reference":  reference,
    "prediction": prediction,
    })

with open(EVAL_OUTPUT_PATH, "w") as f:
    json.dump(eval_results, f, indent=2)
print(f"Evaluation results saved to {EVAL_OUTPUT_PATH}")
