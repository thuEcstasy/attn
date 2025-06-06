import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-1.5B",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="sdpa"
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-1.5B")

prompt = "Give me a short introduction to large language models."
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to("cuda")
print(model)

model.model.layers[0].self.attn = ManualSdpaAttention(

generated_ids = model.generate(
    model_inputs.input_ids,
    cache_implementation="static",
    max_new_tokens=512,
    do_sample=True,
    temperature=0.6,
    top_p=0.95
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)