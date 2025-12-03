import time
import torch
from torch.nn import functional as F
import random
from scipy.stats import entropy
import numpy as np

from utils import (
    top_k_logits,
    top_p_logits,
)
import pdb

from config import Settings, text_default_settings_stead
from math import log2
from typing import List, Dict, Optional, Union
from utils import get_probs_indices_past, SingleExampleOutput, set_seed

from tqdm import tqdm

@torch.no_grad()
def compute_capacity_with_sort(probs):
    probs, indices = probs.sort(descending=True, dim=-1)
    capacity = [int(np.log2(1 / x[0].cpu())) for x in probs]
    capacity = min(capacity) if len(capacity) > 2 else 0
    return probs, indices, capacity


@torch.no_grad()
def encode_one_single_token(probs, indices, capacity, message_bits, ptr):

    probs_cumsum = probs.cumsum(dim=0)
    probs = probs.tolist()
    indices = indices.int().tolist()

    max_entropy = entropy(probs, base=2)

    capacity_upper_bound = capacity + 1

    tbl = {}

    if capacity == 0:
        n_bits = capacity

    rotate_step_size = 2.0**-capacity
    is_available = True
    tbl_new = {}

    for k in range(2**capacity):
        ptr_i = ptr + k * rotate_step_size
        if ptr_i >= 1.0:
            ptr_i -= 1
        index_idx = (probs_cumsum > ptr_i).nonzero()[0].item()

        index = indices[index_idx]
        tbl_new[k] = index

    tbl = tbl_new
    n_bits = capacity

    if n_bits < 1:
        sampled_index = indices[(probs_cumsum > ptr).nonzero()[0].item()]

    else:
        cur_message_bits_decimal = 0
        base = 1
        for d in range(n_bits - 1, -1, -1):
            if message_bits[d] == "1":
                cur_message_bits_decimal += base
            base *= 2
        sampled_index = tbl[cur_message_bits_decimal]

    return sampled_index, n_bits, max_entropy


@torch.no_grad()
def encode_diff(
    model,
    input_ids,
    tokenizer,
    attention_mask,
    message_bits,
    max_new_tokens,
    steps,
    device="cuda",
    temperature=0.0,
    top_p=1.0,
    top_k=None,
):

    stat_time = 0
    embed_time = 0

    max_length = max_new_tokens + input_ids.shape[1]
    mask_token_id = tokenizer.mask_token_id
    eps = 0.001

    x = F.pad(input_ids, (0, max_length - input_ids.shape[1]), value=mask_token_id)
    tok_idx = None
    attention_mask = "full"

    timesteps = torch.linspace(1, eps, steps + 1, device=x.device)

    total_capacity = 0
    total_entropy = 0

    total_minimum_entropy = 0
    total_log_probs = 0

    for i in tqdm(range(steps)):
        mask_index = x == mask_token_id
        logits = model(x, attention_mask, tok_idx).logits
        logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

        mask_logits = logits[mask_index]
        t = timesteps[i]
        s = timesteps[i + 1]

        p_transfer = 1 - s / t if i < steps - 1 else 1
        x0 = (
            torch.zeros_like(x[mask_index], device=device, dtype=torch.long)
            + mask_token_id
        )
        transfer_index_t_s = torch.rand(*x0.shape, device=device) < p_transfer

        generated_ids = []
        one_step_capacity = 0
        max_step_capacity = 0

        if transfer_index_t_s.sum() == 0:
            continue

        if transfer_index_t_s.sum() > 1:
            ECC_FLAG = True
        else:
            ECC_FLAG = False

        x0_logits = mask_logits[transfer_index_t_s]
        x0_logits = x0_logits.to(torch.double)
        x0_logits[:, tokenizer.eos_token_id] = -float("inf")

        if temperature > 0:
            x0_logits = x0_logits / temperature
        if top_p is not None and top_p < 1:
            x0_logits = top_p_logits(x0_logits, top_p)
        if top_k is not None:
            x0_logits = top_k_logits(x0_logits, top_k)
        probs = torch.softmax(x0_logits, dim=-1)

        ecc_message_bits = []
        one_step_capacity = 0

        probs, indices, capacity = compute_capacity_with_sort(probs)

        t0 = time.time()
        for j in range(probs.shape[0]):

            message_bits_one_token = message_bits[: capacity + 1]
            ptr_j = random.random()

            total_minimum_entropy += -log2(probs[j].max())

            sampled_index, one_token_capacity, max_token_capacity = (
                encode_one_single_token(
                    probs=probs[j],
                    indices=indices[j],
                    capacity=capacity,
                    message_bits=message_bits_one_token,
                    ptr=ptr_j,
                )
            )
            token_index = torch.where(indices[j] == sampled_index)[0]
            total_log_probs += log2(probs[j][token_index].item())

            ecc_message_bits.append(one_token_capacity)

            generated_ids.append(sampled_index)
            max_step_capacity += max_token_capacity

        if ECC_FLAG:
            nonzero = [x for x in ecc_message_bits if x != 0]
            if len(nonzero) > 2:
                one_step_capacity = min(nonzero)
            else:
                one_step_capacity = 0
        else:
            one_step_capacity = 0

        message_bits = message_bits[one_step_capacity:]
        t1 = time.time()

        total_capacity += one_step_capacity
        total_entropy += max_step_capacity

        generated_ids = torch.tensor(generated_ids, device=device, dtype=torch.long)

        x0[transfer_index_t_s] = generated_ids

        x[mask_index] = x0.clone()
        embed_time += t1 - t0

    perplexity = 2 ** (-1 / max_new_tokens * total_log_probs)
    return x[0][-max_new_tokens:].tolist(), total_capacity, total_entropy, perplexity, total_minimum_entropy, embed_time

def check_length(stego, max_new_tokens):
    return stego.shape[1] - max_new_tokens

@torch.no_grad()
def decode_diff(
    model,
    tokenizer,
    input_ids,
    attention_mask,
    stego,
    max_new_tokens,
    steps,
    device="cuda",
    temperature=1.0,
    top_p=1.0,
    top_k=None,
    mu=2,
):

    max_length = max_new_tokens + input_ids.shape[1]
    mask_token_id = tokenizer.mask_token_id
    eps = 0.001

    x = F.pad(input_ids, (0, max_length - input_ids.shape[1]), value=mask_token_id)

    tok_idx = None
    attention_mask = "full"

    timesteps = torch.linspace(1, eps, steps + 1, device=x.device)

    total_capacity = 0
    total_entropy = 0

    message_decoded = ""
    stego_retoken = stego
    mu = max(mu, np.abs(check_length(stego_retoken, max_length)))

    if stego_retoken.shape[1] < max_length:
        stego_recovered = F.pad(
            stego_retoken, (0, max_length - stego_retoken.shape[1]), value=mask_token_id
        )
    else:
        stego_recovered = stego_retoken

    offset = torch.zeros_like(stego_recovered[0], device=device, dtype=torch.int)

    for i in tqdm(range(steps)):
        mask_index = x == mask_token_id
        logits = model(x, attention_mask, tok_idx).logits
        logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

        mask_logits = logits[mask_index]
        t = timesteps[i]
        s = timesteps[i + 1]

        p_transfer = 1 - s / t if i < steps - 1 else 1
        x0 = (
            torch.zeros_like(x[mask_index], device=device, dtype=torch.long)
            + mask_token_id
        )
        transfer_index_t_s = torch.rand(*x0.shape, device=device) < p_transfer

        if transfer_index_t_s.sum() == 0:
            continue

        x0_logits = mask_logits[transfer_index_t_s]
        x0_logits = x0_logits.to(torch.double)
        x0_logits[:, tokenizer.eos_token_id] = -float("inf")

        if temperature > 0:
            x0_logits = x0_logits / temperature
        if top_p is not None and top_p < 1:
            x0_logits = top_p_logits(x0_logits, top_p)
        if top_k is not None:
            x0_logits = top_k_logits(x0_logits, top_k)
        probs = torch.softmax(x0_logits, dim=-1)

        probs, indices, capacity = compute_capacity_with_sort(probs)
        ptrs = [random.random() for _ in range(probs.shape[0])]

        index = torch.nonzero(mask_index)[transfer_index_t_s].tolist()

        stegos = []
        for j in index:
            stegos.append(stego_recovered[0][j[1]].item())

        neighborhoods = []

        for _, i_ in index:
            # 计算 neighborhood 的范围
            start_index = max(0, i_ - mu)
            end_index = min(max_length, i_ + mu + 1)
            neighborhood = stego_recovered[0][start_index:end_index].tolist()
            neighborhoods.append(neighborhood)

        corrected_stegos = []
        ecc_message = []
        ECC_FLAG = False

        count_for_error_list = dict()
        for j in range(probs.shape[0]):

            try:
                stegos[j] = stego_recovered[0][index[j][1] + offset[index[j][1]]].item()
            except:
                stegos[j] = stego_recovered[0][index[j][1]].item()

            ecc_message_t, _, _, correct_random_sampled_index = decode_one_single_token(
                probs=probs[j],
                indices=indices[j],
                capacity=capacity,
                stego_t=stegos[j],
                ptr=ptrs[j],
            )
            ecc_message.append(ecc_message_t)
            count_for_error_list[str(index[j][1])] = 0

        voted_message, error_list = find_most_common(ecc_message, capacity)

        generated_ids = stegos.copy()

        if len(error_list) == 0:

            message_decoded_t = voted_message
            correct_random_sampled_index = None

        error_count = 0
        
        while error_list != []:
            error_j = error_list.pop(0) # 0,144

            while count_for_error_list[str(index[error_j][1])] >= 20: # 2,4
                try:
                    error_j = error_list.pop(0)
                except:
                    return message_decoded

            if offset[index[error_j][1]] != 0:

                stego_offset = stego_recovered[0][
                    index[error_j][1] + offset[index[error_j][1]]
                ].item()

                ecc_message_t, _, _, _ = decode_one_single_token(
                    probs=probs[error_j],
                    indices=indices[error_j],
                    capacity=capacity,
                    stego_t=stego_offset,
                    ptr=ptrs[error_j],
                )
                ecc_message[error_j] = ecc_message_t

                voted_message, error_list = find_most_common(ecc_message, capacity)

                if ecc_message_t != "x" and ecc_message_t == voted_message:
                    generated_ids[error_j] = stego_offset
                    continue

            if voted_message != "x":
                corrected_stego, _, _ = encode_one_single_token(
                    probs=probs[error_j],
                    indices=indices[error_j],
                    capacity=capacity,
                    message_bits=voted_message,
                    ptr=ptrs[error_j],
                )

                ecc_message[error_j] = voted_message
                voted_message, error_list = find_most_common(ecc_message, capacity)

            else:
                if capacity == 0:
                    corrected_stego, _, _ = encode_one_single_token(
                        probs=probs[error_j],
                        indices=indices[error_j],
                        capacity=capacity,
                        message_bits="",
                        ptr=ptrs[error_j],
                    )
                    ecc_message[error_j] = ""
                    voted_message, error_list = find_most_common(ecc_message, capacity)

                else:
                    if error_count >= 10:
                        return message_decoded

                    temp_stegos_j = []

                    for temp_stego in neighborhoods[error_j]:
                        if temp_stego != stegos[error_j]:

                            ecc_message_t, _, _, _ = decode_one_single_token(
                                probs=probs[error_j],
                                indices=indices[error_j],
                                capacity=capacity,
                                stego_t=temp_stego,
                                ptr=ptrs[error_j],
                            )

                            if ecc_message_t != "x":
                                temp_stegos_j.append(temp_stego)
                                corrected_stegos.append(temp_stego)
                                voted_message, error_list = find_most_common(
                                    ecc_message, capacity
                                )

                    if len(temp_stegos_j) == 0:
                        error_count += 1
                        continue

            if corrected_stego in neighborhoods[error_j]:

                begin_index = index[error_j][1]
                unmask_index = (mask_index == 0).nonzero(as_tuple=True)[1][
                    input_ids.shape[1] :
                ]

                end_index = (
                    unmask_index[unmask_index > begin_index][0]
                    if len(unmask_index[unmask_index > begin_index]) > 0
                    else None
                )

                offset[begin_index:end_index] = (
                    -mu
                    + neighborhoods[error_j].index(corrected_stego)
                )

            generated_ids[error_j] = corrected_stego
            count_for_error_list[str(index[error_j][1])] += 1

        message_decoded += voted_message

        generated_ids = torch.tensor(generated_ids, device=device, dtype=torch.long)
        x0[transfer_index_t_s] = generated_ids

        x[mask_index] = x0.clone()

    return message_decoded


def decode_one_single_token(probs, indices, capacity, stego_t, ptr):

    probs_cumsum = probs.cumsum(dim=0)
    probs = probs.tolist()
    indices = indices.int().tolist()
    max_entropy = entropy(probs, base=2)

    tbl = {}

    if capacity == 0:
        n_bits = capacity

    rotate_step_size = 2.0**-capacity
    is_available = True
    tbl_new = {}

    for k in range(2**capacity):
        ptr_i = ptr + k * rotate_step_size
        if ptr_i >= 1.0:
            ptr_i -= 1
        index_idx = (probs_cumsum > ptr_i).nonzero()[0].item()

        index = indices[index_idx]
        tbl_new[k] = index

    tbl = tbl_new
    n_bits = capacity

    if n_bits < 1:
        correct_random_sampled_index = indices[(probs_cumsum > ptr).nonzero()[0].item()]
        if stego_t == correct_random_sampled_index:
            ecc_message = ""
        else:
            ecc_message = "x"
        return ecc_message, n_bits, max_entropy, correct_random_sampled_index
    else:
        if stego_t not in tbl.values():
            ecc_message = "x"
        else:
            tbl_swapped = dict(zip(tbl.values(), tbl.keys()))
            ecc_message = bin(tbl_swapped[stego_t])[2:].zfill(n_bits)
          
    return ecc_message, n_bits, max_entropy, None

def find_most_common(lst, length):
    valid = []
    error = []
    for t in range(len(lst)):
        if lst[t] == "x":
            error.append(t)
        else:
            if len(lst[t]) != length:
                error.append(t)
            if all(c in {"0", "1"} for c in lst[t]):
                valid.append(t)

    if len(valid) == 0:
        return "x", error

    counter = Counter([lst[t] for t in valid])
    max_count = max(counter.values())
    max_elements = {k for k, v in counter.items() if v == max_count}

    for t in valid:
        if lst[t] in max_elements:
            elem = lst[t]
        else:
            error.append(t)

    return elem, error

@torch.no_grad()
def encode_text(model,
                tokenizer,
                message_bits: Optional[str] = None,
                prompt: str = None,
                settings: Settings = Settings(),
                segment: Optional[int] = None) -> SingleExampleOutput:
    # General architecture of Steganography Encoding (message_bits -> English text)
    algo, temp, top_p, top_k, length, seed = settings()
    chat_template = [
        {
            "role": "user",
            "content": prompt,
        }
    ]
    inputs = tokenizer.apply_chat_template(
        chat_template, return_tensors="pt", return_dict=True, add_generation_prompt=True
    )
    input_ids = inputs.input_ids.to(device=model.device)
    attention_mask = inputs.attention_mask.to(device=model.device)

    set_seed(seed)
    start = time.time()
    
    stego_tokens, total_capacity, total_entropy, perplexity, total_minimum_entropy, embed_time = encode_diff(
        model=model,
        tokenizer=tokenizer,
        input_ids=input_ids,
        attention_mask=attention_mask,
        message_bits=message_bits,
        steps=512,
        max_new_tokens=length,  # 512
        temperature=temp, # 1.2
        top_p=top_p, # 1.0
        top_k=top_k,
    )
    end = time.time()
    stego_object = tokenizer.decode(stego_tokens)
    ave_kld, max_kld = 0, 0

    return SingleExampleOutput(stego_tokens, stego_object, 
                               total_capacity, total_entropy, 
                               ave_kld, max_kld, 
                               perplexity, end - start, 
                               settings, total_minimum_entropy), embed_time


@torch.no_grad()
def decode_text(model,
                tokenizer,
                stego: Union[str, List[int]],
                prompt: str,
                settings: Settings = Settings()) -> str:
    # General architecture of Steganography Decoding (English text -> message_bits)
    # Returns `message_decoded`
    algo, temp, top_p, top_k, length, seed = settings()

    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]
    inputs = tokenizer.apply_chat_template(
        messages, return_tensors="pt", return_dict=True, add_generation_prompt=True
    )
    input_ids = inputs.input_ids.to(device=model.device)
    attention_mask = inputs.attention_mask.to(device=model.device)

    stego_tensor = torch.tensor([stego], device=model.device)
    stego = torch.cat([input_ids, stego_tensor], dim=1)

    start = time.time()
    set_seed(seed)
    decoded_message = decode_diff(
        model=model,
        tokenizer=tokenizer,
        input_ids=input_ids,
        attention_mask=attention_mask,
        stego=stego,
        steps=512,
        max_new_tokens=length,  # 512
        temperature=temp, # 1.2
        top_p=top_p, # 1.0
        top_k=top_k,
    )

    # print(decoded_message)
    return decoded_message
