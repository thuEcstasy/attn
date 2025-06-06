import os
from datasets import load_dataset
import torch
import json
from transformers import (
    AutoTokenizer,
    AutoConfig,
    LlamaTokenizer,
    LlamaForCausalLM,
    AutoModelForCausalLM,
)
from tqdm import tqdm
import numpy as np
import random
import argparse
from evaluation.quest_attention import enable_quest_attention_eval
from evaluation.llama import enable_tuple_kv_cache_for_llama 
from evaluation.mistral import enable_tuple_kv_cache_for_mistral
from evaluation.qwen2 import enable_tuple_kv_cache_for_qwen2
import re
import torch.multiprocessing as mp

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="/home/haizhonz/Zhaofeng/models/DeepSeek-R1-Distill-Qwen-1.5B",
        choices=[
            "llama2-7b-chat-4k",
            "longchat-v1.5-7b-32k",
            "xgen-7b-8k",
            "internlm-7b-8k",
            "chatglm2-6b",
            "chatglm2-6b-32k",
            "chatglm3-6b-32k",
            "vicuna-v1.5-7b-16k",
            "Mistral-7B-Instruct-v0.3",
            "Meta-Llama-3.1-8B-Instruct",
            "/home/haizhonz/Zhaofeng/models/DeepSeek-R1-Distill-Qwen-1.5B",
            "EpistemeAI/Reasoning-Llama-3.1-CoT-RE1-NMT"
        ],
    )
    parser.add_argument("--e", action="store_true", help="Evaluate on LongBench-E")
    parser.add_argument("--token_budget", type=int, default=1024)
    parser.add_argument("--chunk_size", type=int, default=16)
    parser.add_argument("--quest", action="store_true", help="Enable Quest Attention")
    parser.add_argument("--reasoning", action="store_true", help="Enable Reasoning Mode")
    return parser.parse_args(args)

def load_model_and_tokenizer(path, model_name, device):
    if 'llama' in model_name.lower() or 'longchat' in model_name.lower():
        print("enable!")
        enable_tuple_kv_cache_for_llama()
    if 'mistral' in model_name.lower():
        enable_tuple_kv_cache_for_mistral()
    if 'qwen' in model_name.lower():
        enable_tuple_kv_cache_for_qwen2()
        
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="cuda:0"
    )
    model = model.eval()

    orig_model = AutoModelForCausalLM.from_pretrained(
        path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="cuda:0"
    )
    orig_model = orig_model.eval()
    print(orig_model.model.layers[0].self_attn.sliding_window, flush=True)
    if args.quest:
        enable_quest_attention_eval(model, args)
        print("Quest Attention enabled.", args, flush=True)

    return model, orig_model, tokenizer

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    seed_everything(42)
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = args.model
    # define your model
    model, orig_model, tokenizer = load_model_and_tokenizer(
       "/home/haizhonz/Zhaofeng/models/DeepSeek-R1-Distill-Qwen-1.5B", model_name, device
    )
    print(model)
    # test the output consistency
    # fake_input has 1000 tokens
    fake_input = "Recent research in artificial intelligence has demonstrated significant breakthroughs in natural language processing. The study conducted by researchers at leading universities revealed important insights about machine learning algorithms. Scientists have discovered new methods for improving neural network performance and efficiency. Advanced computational techniques are revolutionizing our understanding of complex data patterns. Experimental results indicate that deep learning models can achieve remarkable accuracy in various tasks. The development of transformer architectures has fundamentally changed the landscape of AI research. Interdisciplinary collaboration between computer scientists and domain experts has led to innovative solutions. Quantum computing represents a paradigm shift in computational power and problem-solving capabilities. Machine learning applications in healthcare are showing promising results for disease diagnosis and treatment. The integration of artificial intelligence into everyday technology continues to accelerate rapidly."
    input = tokenizer(fake_input, return_tensors="pt").to(device)
    print(input.input_ids.shape, flush=True)
    outputs = model(
        input_ids=input.input_ids,
        past_key_values=None,
        use_cache=True,
    )
    past_key_values = outputs.past_key_values
    pred_token_idx = (
        outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
    )
    print("1st", pred_token_idx, flush=True)
    outputs = model(
        input_ids=pred_token_idx,
        past_key_values=past_key_values,
        use_cache=True,
    )
    past_key_values = outputs.past_key_values
    pred_token_idx = (
        outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
    )
    print("1st", pred_token_idx, flush=True)
    outputs = model(
        input_ids=pred_token_idx,
        past_key_values=past_key_values,
        use_cache=True,
    )
    pred_token_idx = (
        outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
    )
    print("Output", outputs.logits, flush=True)


    orig_outputs = orig_model(
        input_ids=input.input_ids,
        past_key_values=None,
        use_cache=True,
    )
    orig_past_key_values = orig_outputs.past_key_values


    pred_token_idx = (
        orig_outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
    )
    print("2nd", pred_token_idx, flush=True)
    orig_outputs = orig_model(
        input_ids=pred_token_idx,
        past_key_values=orig_past_key_values,
        use_cache=True,
    )
    orig_past_key_values = orig_outputs.past_key_values

    pred_token_idx = (
        orig_outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
    )
    print("2nd", pred_token_idx, flush=True)
    orig_outputs = orig_model(
        input_ids=pred_token_idx,
        past_key_values=orig_past_key_values,
        use_cache=True,
    )
    print("Original Output", orig_outputs.logits, flush=True)
    assert torch.allclose(
        outputs.logits, orig_outputs.logits, atol=1e-3
    ), "The output logits are not consistent with the original model."

