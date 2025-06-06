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
        default=None,
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
        ],
    )
    parser.add_argument("--e", action="store_true", help="Evaluate on LongBench-E")

    parser.add_argument("--task", type=str, help="task name", required=True)

    parser.add_argument("--token_budget", type=int, default=None)
    parser.add_argument("--chunk_size", type=int, default=None)
    parser.add_argument("--quest", action="store_true", help="Enable Quest Attention")
    parser.add_argument("--reasoning", action="store_true", help="Enable Reasoning Mode")
    return parser.parse_args(args)


# This is the customized building prompt for chat models
def build_chat(tokenizer, prompt, model_name):
    if "chatglm3" in model_name:
        prompt = tokenizer.build_chat_input(prompt)
    elif "chatglm" in model_name:
        prompt = tokenizer.build_prompt(prompt)
    elif "longchat" in model_name or "vicuna" in model_name:
        from fastchat.model import get_conversation_template

        conv = get_conversation_template("vicuna")
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    elif "llama2" in model_name:
        prompt = f"[INST]{prompt}[/INST]"
    elif "xgen" in model_name:
        header = (
            "A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n"
        )
        prompt = header + f" ### Human: {prompt}\n###"
    elif "internlm" in model_name:
        prompt = f"<|User|>:{prompt}<eoh>\n<|Bot|>:"
    
    elif "Qwen" in model_name:
        enable_system_blog = False
        if enable_system_blog:
            system_prompt = (
                "You are a helpful, respectful and honest assistant. "
                "Always answer as helpfully as possible, while being safe. "
                "Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. "
                "Please ensure that your responses are socially unbiased and positive in nature.\n"
            )
            prompt = f"<|im_start|>system\n{system_prompt} <|im_start|>user\n{prompt}.<|im_end|> \n<|im_start|>assistant\n<think>\n"
        else:
            prompt = prompt

    return prompt


def post_process(response, model_name):
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    return response

from torch.nn import functional as F

def stable_top_p_sampling(logits, temperature=0.6, top_p=0.95, eps=1e-8):
    """
    稳定的top-p采样实现
    
    Args:
        logits: 模型输出的logits [batch_size, vocab_size]
        temperature: 温度参数，控制随机性
        top_p: nucleus sampling的概率阈值
        eps: 数值稳定性的小常数
    
    Returns:
        sampled_tokens: 采样的token索引 [batch_size, 1]
    """
    # 1. 应用温度缩放
    logits = logits / temperature
    
    # 2. 转换为概率分布
    probs = F.softmax(logits, dim=-1)
    
    # 3. 按概率降序排序
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    
    # 4. 计算累积概率
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # 5. 创建mask，保留累积概率在top_p以内的tokens
    # 关键：使用 <= 而不是 < 来确保包含刚好达到阈值的token
    mask = cumulative_probs <= top_p
    
    # 6. 确保至少保留一个token（概率最高的）
    mask[:, 0] = True
    
    # 7. 应用mask，将不需要的概率设为0
    filtered_probs = sorted_probs * mask.float()
    
    # 8. 检查是否有有效概率
    prob_sum = filtered_probs.sum(dim=-1, keepdim=True)
    
    # 9. 处理边界情况：如果所有概率都被过滤掉了
    valid_mask = (prob_sum > eps).squeeze(-1)
    if not valid_mask.all():
        # 对于无效的行，只保留最高概率的token
        invalid_rows = ~valid_mask
        filtered_probs[invalid_rows] = 0.0
        filtered_probs[invalid_rows, 0] = 1.0
        prob_sum[invalid_rows] = 1.0
    
    # 10. 重新归一化
    filtered_probs = filtered_probs / prob_sum
    
    # 11. 数值稳定性检查
    filtered_probs = torch.clamp(filtered_probs, min=eps)
    filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)
    
    # 12. 从过滤后的分布中采样
    try:
        sampled_indices = torch.multinomial(filtered_probs, num_samples=1)
    except RuntimeError as e:
        # 如果采样失败，回退到贪心解码
        print(f"采样失败，回退到贪心解码: {e}")
        sampled_indices = torch.zeros((filtered_probs.shape[0], 1), 
                                    dtype=torch.long, device=filtered_probs.device)
    
    # 13. 映射回原始token索引
    sampled_tokens = torch.gather(sorted_indices, dim=-1, index=sampled_indices)
    
    return sampled_tokens

def get_pred(
    model,
    tokenizer,
    data,
    max_length,
    max_gen,
    prompt_format,
    dataset,
    device,
    model_name,
):
    preds = []
    for json_obj in tqdm(data):
        prompt = prompt_format.format(**json_obj)
        # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
        tokenized_prompt = tokenizer(
            prompt, truncation=False, return_tensors="pt"
        ).input_ids[0]
        if "chatglm3" in model_name:
            tokenized_prompt = tokenizer(
                prompt, truncation=False, return_tensors="pt", add_special_tokens=False
            ).input_ids[0]
        if len(tokenized_prompt) > max_length:
            half = int(max_length / 2)
            prompt = tokenizer.decode(
                tokenized_prompt[:half], skip_special_tokens=True
            ) + tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        if dataset not in [
            "trec",
            "triviaqa",
            "samsum",
            "lsht",
            "lcc",
            "repobench-p",
        ]:  # chat models are better off without build prompts on these tasks
            prompt = build_chat(tokenizer, prompt, model_name)

        q_pos = None

        if q_pos != None:
            question = prompt[q_pos:]
            prompt = prompt[:q_pos]
        else:
            # input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
            input = tokenizer(prompt, truncation=False, return_tensors="pt").to("cuda")

        context_length = input.input_ids.shape[-1]

        if (
            dataset == "samsum"
        ):  # prevent illegal output on samsum (model endlessly repeat "\nDialogue"), might be a prompting issue
            pred = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=0.6,
                top_p=0.95,
                min_length=context_length + 1,
                eos_token_id=[
                    tokenizer.eos_token_id,
                ],
            )[0]
        else:
            with torch.no_grad():
                output = model(
                    input_ids=input.input_ids,
                    past_key_values=None,
                    use_cache=True,
                )
                past_key_values = output.past_key_values
                pred_token_idx = output.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
                logits = output.logits[:, -1, :]
                
                generated_content = [pred_token_idx.item()]
                for _ in range(max_gen - 1):
                    outputs = model(
                        input_ids=pred_token_idx,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )

                    past_key_values = outputs.past_key_values
                    # pred_token_idx = (
                    #     outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
                    # )
                    pred_token_idx = stable_top_p_sampling(
                        outputs.logits[:, -1, :],
                        temperature=0.6, 
                        top_p=0.95
                    )
                    generated_content += [pred_token_idx.item()]
                    if pred_token_idx.item() == tokenizer.eos_token_id:
                        break

            # output = model.generate(
            #     **input,
            #     max_new_tokens=max_gen,
            #     num_beams=1,
            #     do_sample=False,
            #     temperature=1.0,
            # )[0]

            pred = tokenizer.decode(generated_content, skip_special_tokens=True)
            pred_len = len(generated_content)
            # pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
            pred = post_process(pred, model_name)
        out = {
                "pred": pred,
                "pred_len": pred_len,
                "answers": json_obj["answers"] if "answers" in json_obj else json_obj["answer"],
                "all_classes": json_obj["all_classes"] if "all_classes" in json_obj else None,
                "prompt_length": len(tokenized_prompt)
            }
        preds.append(out)
        print(out, flush=True)
    return preds


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def load_model_and_tokenizer(path, model_name, device, args):
    if 'llama' in model_name.lower() or 'longchat' in model_name.lower():
        enable_tuple_kv_cache_for_llama()
    if 'mistral' in model_name.lower():
        enable_tuple_kv_cache_for_mistral()
    if 'qwen' in model_name.lower():
        enable_tuple_kv_cache_for_qwen2()
        
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map=device
    )
    model = model.eval()

    if args.quest:
        enable_quest_attention_eval(model, args)

    return model, tokenizer

def run_inference(rank, args, model_path, model_name, dataset_name, data_chunk, out_path):
    torch.cuda.set_device(rank)
    seed_everything(42 + rank)
    device = torch.device(f"cuda:{rank}")

    model, tokenizer = load_model_and_tokenizer(model_path, model_name, device, args)
    
    prompt_format = json.load(open("config/dataset2prompt_reasoning.json", "r"))[dataset_name]
    max_length = json.load(open("config/model2maxlen.json", "r"))[model_name]
    max_gen = json.load(open("config/dataset2maxlen_reasoning.json", "r"))[dataset_name]

    print(f"[GPU {rank}] Start inference for {len(data_chunk)} samples...")

    preds = get_pred(
        model,
        tokenizer,
        data_chunk,
        max_length,
        max_gen,
        prompt_format,
        dataset_name,
        device,
        model_name,
    )

    partial_out_path = out_path.replace(".jsonl", f".part{rank}.jsonl")
    with open(partial_out_path, "w", encoding="utf-8") as f:
        for pred in preds:
            json.dump(pred, f, ensure_ascii=False)
            f.write("\n")
    print(f"[GPU {rank}] Finished. Output written to {partial_out_path}")


if __name__ == "__main__":
    seed_everything(42)
    args = parse_args()
    model2path = json.load(open("config/model2path.json", "r"))
    model2maxlen = json.load(open("config/model2maxlen.json", "r"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = args.model
    max_length = model2maxlen[model_name]
    if args.e:
        datasets = [
            "qasper",
            "multifieldqa_en",
            "hotpotqa",
            "2wikimqa",
            "gov_report",
            "multi_news",
            "trec",
            "triviaqa",
            "samsum",
            "passage_count",
            "passage_retrieval_en",
            "lcc",
            "repobench-p",
        ]
    else:
        datasets = [args.task]
    # we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
    dataset2prompt = json.load(open("config/dataset2prompt_reasoning.json", "r"))
    dataset2maxlen = json.load(open("config/dataset2maxlen_reasoning.json", "r"))
    if args.reasoning:
        dataset2maxlen = json.load(open("config/dataset2maxlen_reasoning.json", "r"))
    # predict on each dataset
    if not os.path.exists("pred"):
        os.makedirs("pred")
    if not os.path.exists("pred_e"):
        os.makedirs("pred_e")
    for dataset in datasets:
        if dataset == "MATH-500":
            data = load_dataset("HuggingFaceH4/MATH-500", split="test")
            if not os.path.exists(f"pred/{model_name}"):
                os.makedirs(f"pred/{model_name}")
            if args.quest:
                out_path = f"pred/{model_name}/{dataset}-{args.token_budget}.jsonl"
            else:
                out_path = f"pred/{model_name}/{dataset}-full.jsonl"
        elif dataset == "gsm8k":
            data = load_dataset("openai/gsm8k", "main", split="test")
            # choose first 50 samples for testing
            data = data.select(range(50))
            if not os.path.exists(f"pred/{model_name}"):
                os.makedirs(f"pred/{model_name}")
            if args.quest:
                out_path = f"pred/{model_name}/{dataset}-{args.token_budget}.jsonl"
            else:
                out_path = f"pred/{model_name}/{dataset}-full.jsonl"
        elif args.e:
            data = load_dataset("THUDM/LongBench", f"{dataset}_e", split="test")
            if not os.path.exists(f"pred_e/{model_name}"):
                os.makedirs(f"pred_e/{model_name}")
            out_path = f"pred_e/{model_name}/{dataset}.jsonl"
            if args.quest:
                out_path = f"pred_e/{model_name}/{dataset}-{args.token_budget}.jsonl"
            else:
                out_path = f"pred_e/{model_name}/{dataset}-full.jsonl"
        else:
            data = load_dataset("THUDM/LongBench", dataset, split="test")
            if not os.path.exists(f"pred/{model_name}"):
                os.makedirs(f"pred/{model_name}")
            if args.quest:
                out_path = f"pred/{model_name}/{dataset}-{args.token_budget}.jsonl"
            else:
                out_path = f"pred/{model_name}/{dataset}-full.jsonl"
        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        ctx = mp.get_context("spawn")
        processes = []
        for rank in range(4):
            chunk = data.shard(num_shards=4, index=rank)
            p = ctx.Process(target=run_inference, args=(rank, args, model2path[model_name], model_name, dataset, chunk, out_path))
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
