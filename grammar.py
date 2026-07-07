
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "microsoft/Phi-3-mini-4k-instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)            

def improve_sentence(text):
    prompt = f"""
You are an English assistant.

Convert the following sign language text into a natural English sentence.

Input:
{text}

Output:
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=60,
        temperature=0.3
    )

    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )

    return response


if __name__ == "__main__":
    print("Program started")

    text = "I HAPPY TODAY"

    answer = improve_sentence(text)

    print("Answer")
    print(answer)


