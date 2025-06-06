import torch
import torch.nn as nn
import torch.nn.functional as F
import json

import sys
import torch._dynamo
torch._dynamo.config.suppress_errors = True
LOCAL_TRANSFORMERS_PATH = "/home/haizhonz/Zhaofeng/quest/evaluation/LongBench/transformers/src"
sys.path.insert(0, LOCAL_TRANSFORMERS_PATH)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,5"
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
print(f"Using transformers from: {transformers.__file__}")
from tqdm import tqdm
import os
import math
from typing import Optional, Tuple
import copy
import numpy as np
import random
import torch.multiprocessing as mp
import re
import argparse
from datasets import load_dataset

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--token_budget", type=int, default=256)
    return parser.parse_args()

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def load_model_and_tokenizer(path, model_name, device, args):
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        path, trust_remote_code=True, torch_dtype=torch.float32, device_map=device
    )
    model = model.eval()
    return model, tokenizer


def run_inference(rank, args, model_path, model_name, data_chunk, out_path, chunk):
    torch.cuda.set_device(rank)
    seed_everything(42 + rank)
    device = torch.device(f"cuda:{rank}")
    model, tokenizer = load_model_and_tokenizer(model_path, model_name, device, args)
    print(f"[GPU {rank}] Start inference for {data_chunk} budget...")
    preds = []
    for json_obj in tqdm(chunk):
        text = f"<｜begin▁of▁sentence｜><｜User｜>Solve the following math problem efficiently and clearly.  The last line of your response should be of the following format: 'Therefore, the final answer is: $\\boxed {{ANSWER}}$. I hope it is correct' (without quotes) where ANSWER is just the final number or expression that solves the problem. Think step by step before answering.\n\n{json_obj['Problem']} <｜Assistant｜><think>"
        model_inputs = tokenizer(text, return_tensors="pt").to(device)
        pred = model.generate(
            **model_inputs,
            max_new_tokens=8192,
            cache_implementation="static",
            num_beams=1,
            do_sample=True,
            temperature=0.6,
            top_p=0.95
        )[0]
        decoded_pred = tokenizer.decode(pred, skip_special_tokens=True)
        print(f"[GPU {rank}] Generated text: {decoded_pred}")
        preds.append({
            "response": decoded_pred,
            "response_length": len(pred),
            "gold": json_obj["Solution"],
            "question": json_obj["Problem"]
        })
    partial_out_path = out_path.replace(".jsonl", f".part{rank}.jsonl")
    with open(partial_out_path, "w", encoding="utf-8") as f:
        for pred in preds:
            json.dump(pred, f, ensure_ascii=False)
            f.write("\n")
    print(f"[GPU {rank}] Finished. Output written to {partial_out_path}")

if __name__ == "__main__":
    seed_everything(42)
    args = parse_args()
    model = AutoModelForCausalLM.from_pretrained(
        "/home/haizhonz/Zhaofeng/models/DeepSeek-R1-Distill-Qwen-1.5B",
        torch_dtype=torch.float32,
        device_map="auto",
        attn_implementation="sdpa"
    )
    tokenizer = AutoTokenizer.from_pretrained("/home/haizhonz/Zhaofeng/models/DeepSeek-R1-Distill-Qwen-1.5B")
    data = load_dataset("Maxwell-Jia/AIME_2024", "default", split="train").select(range(10))
    out_path = f"pred/{args.token_budget}.jsonl"
    ctx = mp.get_context("spawn")
    processes = []
    for rank in range(4):
        chunk = data.shard(num_shards=4, index=rank)
        p = ctx.Process(target=run_inference, args=(rank, args, "/home/haizhonz/Zhaofeng/models/DeepSeek-R1-Distill-Qwen-1.5B", "DeepSeek-R1-Distill-Qwen-1.5B", 256, out_path, chunk))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    with open(out_path, "w", encoding="utf-8") as outfile:
        for rank in range(4):
            part_path = out_path.replace(".jsonl", f".part{rank}.jsonl")
            with open(part_path, "r", encoding="utf-8") as infile:
                for line in infile:
                    outfile.write(line)
            os.remove(part_path)
    print(f"[Main] Final output saved to {out_path}")