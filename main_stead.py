import os

# 设置HuggingFace缓存路径
# os.environ["HF_HOME"] = "~/cache"
# os.environ["TRANSFORMERS_CACHE"] = "~/cache"

# 配置HF镜像源
# os.environ["HF_ENDPOINT"] = "https://huggingface.co"

# 禁用HF本地目录自动同步
# os.environ["HF_HUB_LOCAL_DIR_AUTO_SYNC"] = "0"

import sys
import json
import time
import torch
import random
import numpy as np
from tqdm import tqdm
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, AutoTokenizer

from stego_utils import *
from config import Settings, text_default_settings_stead, text_default_settings_sample
from utils import set_seed


def text_encode(settings: Settings, model, tokenizer, message, prompt, key):
    if settings.algo == "sample":
        from random_sample_cy import encode_text
    else:
        from stega import encode_text
    settings.seed = key
    single_example_output: SingleExampleOutput = encode_text(
        model, tokenizer, message, prompt, settings
    )
    return single_example_output.stego_object, single_example_output.ave_kld


def text_decode(settings: Settings, model, tokenizer, stego, prompt, key, flag=False):
    from stega import decode_text

    settings.seed = key

    stego = stego.replace("<s>", " <s>")
    stego = stego.replace("</s>", " </s>")
    if flag:
        stego = "<s>" + stego

    stego_ids = tokenizer(stego, return_tensors="pt")["input_ids"][0].tolist()

    if flag:
        stego_ids = stego_ids[1:]
    if stego_ids[0] == 1:
        stego_ids = stego_ids[1:]
    if settings.model_name == "LLaMA-8B" and stego_ids[0] == 128000:
        stego_ids = stego_ids[1:]
    elif settings.model_name == "Qwen-7B" and stego_ids[0] == 151646:
        stego_ids = stego_ids[1:]
    elif settings.model_name == "LLaMA-7B" and stego_ids[0] == 50256:
        stego_ids = stego_ids[1:]

    message_decoded = decode_text(model, tokenizer, stego_ids, prompt, settings)
    return message_decoded


def load_diff_model(device):
    # model_path = "dream-dllm/Dream-v0-Instruct-7B"
    model_path = "Dream-org/Dream-v0-Instruct-7B"
    model = AutoModel.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = model.to(device).eval()
    return model, tokenizer


def test_dataset():
    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    import argparse
    import jsonlines

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="diffusion",
        choices=[
            "diffusion",
            "Qwen/Qwen2.5-7B",
            "deepseek-ai/deepseek-llm-7b-base",
            "huggyllama/llama-7b",
        ],
    )
    parser.add_argument("--prompt_file", type=str, default="data/imdb.jsonl")
    parser.add_argument("--message_file", type=str, default="data/message.txt")
    parser.add_argument("--test_size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--temp", type=float, default=1.0)
    parser.add_argument("--topp", type=float, default=1.0)
    parser.add_argument("--topk", type=int, default=None)
    parser.add_argument("--length", type=int, default=512)

    args = parser.parse_args()
    print(args)

    text_default_settings = Settings(
        "text",
        model_name="LLaMA-7B",
        algo="Stead",
        top_p=args.topp,
        top_k=args.topk,
        temp=args.temp,
        length=args.length,
    )

    setup_seed(args.seed)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    text_default_settings.device = device

    if "diffusion" in args.model_name:
        model, tokenizer = load_diff_model(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            device_map=device,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    print(f"device: {model.device}")

    with jsonlines.open(args.prompt_file, "r") as f:
        prompt_dataset = [item["content"] for item in list(f)]

    sample_size = min(len(prompt_dataset), args.test_size)
    print(f"Sample size: {sample_size}")
    prompt_data = random.sample(prompt_dataset, sample_size)

    with open(args.message_file, "r") as file:
        binary_string = file.read().strip()

    binary_message = []
    for _ in range(sample_size):
        start_index = random.randint(0, len(binary_string) - 3000)
        segment = binary_string[start_index : start_index + 3000]
        binary_message.append(segment)

    stego_prompt = []
    for text in prompt_data:
        sentences = text.split(". ")
        length = len(sentences)
        prompt = ". ".join(sentences[:2]) if length > 2 else text
        stego_prompt.append(prompt.replace("\n", " "))

    result = []
    for idx in tqdm(range(sample_size)):
        try:
            set_seed(args.seed + idx)

            time1 = time.time()
            from stead import encode_text

            text_default_settings.seed = args.seed
            single_example_output, embed_time = encode_text(
                model,
                tokenizer,
                binary_message[idx],
                stego_prompt[idx],
                text_default_settings,
            )
            time2 = time.time()

            stego_token = single_example_output.generated_ids
            stego_text = single_example_output.stego_object

            stego_ids = tokenizer(stego_text, return_tensors="pt")["input_ids"][
                0
            ].tolist()

            from stead import decode_text

            time3 = time.time()
            message_decoded = decode_text(
                model, tokenizer, stego_ids, stego_prompt[idx], text_default_settings
            )
            time4 = time.time()

            if len(message_decoded) == single_example_output.total_capacity:
                if single_example_output.total_capacity == 0:
                    correctness = -1.0
                else:
                    message_numpy = np.array(
                        [
                            int(b)
                            for b in binary_message[idx][
                                : single_example_output.total_capacity
                            ]
                        ]
                    )
                    decoded_numpy = np.array([int(b) for b in message_decoded])
                    correctness = (message_numpy == decoded_numpy).mean()
            else:
                correctness = 0.0

            result_data = {
                "stego_prompt": stego_prompt[idx],
                "stego_text": single_example_output.stego_object,
                "stego_token": single_example_output.generated_ids,
                "embed_message": binary_message[idx][
                    : single_example_output.total_capacity
                ],
                "extracted_message": message_decoded,
                "correctness": correctness,
                "total_capacity": single_example_output.total_capacity,
                "total_entropy": single_example_output.total_entropy,
                "perplexity": single_example_output.perplexity,
                "embedding_rate": single_example_output.embedding_rate,
                "total_minimum_entropy": single_example_output.total_minimum_entropy,
                "utilization_rate": single_example_output.utilization_rate,
                "time_encode": time2 - time1,
                "time_decode": time4 - time3,
                "time_embed": embed_time,
            }
            result.append(result_data)
        except Exception as e:
            print(f"Error: {e}")
            continue

        output_file = f"rebuttal_result/{args.length}/stead_{args.model_name.split('/')[0]}_t{args.temp}_p{args.topp}_k{args.topk}.json"

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    # test_single_example()
    with torch.no_grad():
        test_dataset()
