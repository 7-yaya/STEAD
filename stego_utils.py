import os
import torch
from collections import Counter
from scipy.stats import entropy

class StegaConfig(torch.nn.Module):
    def __init__(self, algo, message_bits, decode, stego):
        super(StegaConfig, self).__init__()

        self.algo = algo
        self.message_bits = message_bits
        self.decode = decode
        self.stego = stego

    def forward(self):
        return {"algo": self.algo, "message_bits": self.message_bits}

# Sampling (Encoding) results and statistics for single example
class SingleExampleOutput:
    def __init__(self, generated_ids, stego_object, n_bits, total_entropy, ave_kld, max_kld, perplexity, time_cost, settings,
                 total_minimum_entropy):
        self.generated_ids = generated_ids
        self.stego_object = stego_object
        self.algo = settings.algo
        self.temp = settings.temp
        self.top_p = settings.top_p
        self.n_bits = n_bits
        if generated_ids is not None:
            self.n_tokens = len(generated_ids)
        else:
            self.n_tokens = len(stego_object)
        self.total_entropy = total_entropy
        self.ave_kld = ave_kld
        self.max_kld = max_kld
        self.embedding_rate = n_bits / self.n_tokens
        self.utilization_rate = n_bits / total_entropy if total_entropy != 0 else 0
        self.perplexity = perplexity
        self.time_cost = time_cost
        self.total_minimum_entropy = total_minimum_entropy

    def __str__(self) -> str:
        d = self.__dict__
        excluded_attr = ['generated_ids']
        selected_attr = list(d.keys())
        for x in excluded_attr:
            selected_attr.remove(x)
        return '\n'.join('{} = {}'.format(key, d[key]) for key in selected_attr)
    
def encoder_str2bin(string_data):
    encoding_map = {
        '0': '11', '1': '0001', '2': '0010', '3': '0011',
        '4': '0100', '5': '0101', '6': '0110', '7': '0111',
        '8': '1000', '9': '1001', ' ': '101'
    }
    binary_string = ''.join(encoding_map[char] for char in string_data)
    binary_string += '0000'

    return binary_string

def decoder_bin2str(binary_string):
    decoding_map = {
        '0001': '1', '0010': '2', '0011': '3', '0100': '4',
        '0101': '5', '0110': '6', '0111': '7', '1000': '8',
        '1001': '9', '1010': ' '
    }
    
    temp = ''
    binary_data = ''
    for i in range(len(binary_string)):
        if binary_string[i] == 'x':
            break
        temp += binary_string[i]
        if temp == '11':
            binary_data += '0'
            temp = ''
        elif temp == '101':
            binary_data += ' '
            temp = ''
        elif len(temp) == 4:
            if temp == '0000':
                break
            binary_data += decoding_map[temp]
            temp = ''

    return binary_data
    
def verify_text(message_text, decoded_message_text):
    for i in range(len(decoded_message_text)):
        if i > len(message_text):
            print(f"核验越界，提取信息长度大于秘密信息长度!")
            return False
        
        if message_text[i] != decoded_message_text[i]:
            print(f"秘密信息核验错误!")
            print(f"秘密信息: {message_text[:len(decoded_message_text)]}")
            print(f"提取信息: {decoded_message_text}")
            return False

    print(f"秘密信息提取成功，核验正确!")
    return True

def calculate_loading_rate(stego_text, message_text):
    if len(stego_text) == 0:
        print("错误：载密文本为空！")
    
    loading_rate = len(message_text) / len(stego_text)
    print(f"载荷率: {loading_rate}")
    
    return loading_rate

def calculate_kl_divergence(cover_text, stego_text):
    cover_words = [word for text in cover_text for word in text.split()]
    stego_words = [word for text in stego_text for word in text.split()]

    cover_word_counts = Counter(cover_words)
    stego_word_counts = Counter(stego_words)

    all_words = set(cover_words + stego_words)

    cover_probs = [cover_word_counts.get(word, 0) / len(cover_words) for word in all_words]
    stego_probs = [stego_word_counts.get(word, 0) / len(stego_words) for word in all_words]

    kl_divergence = entropy(cover_probs, stego_probs)
    print(f"KL散度: {kl_divergence}")

    return kl_divergence

import heapq

class Node:
    def __init__(self, char=None, freq=0, left=None, right=None):
        self.char = char
        self.freq = freq
        self.left = left
        self.right = right

    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_tree_from_freq(freq):
    heap = [Node(char=ch, freq=f) for ch, f in freq.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        node1 = heapq.heappop(heap)
        node2 = heapq.heappop(heap)
        merged = Node(char=None, freq=node1.freq + node2.freq, left=node1, right=node2)
        heapq.heappush(heap, merged)

    return heap[0] if heap else None

def build_huffman_tree(text):
    freq = {}
    for ch in text:
        freq[ch] = freq.get(ch, 0) + 1
    return build_huffman_tree_from_freq(freq)

def build_huffman_code_table(root, code='', table=None):
    if table is None:
        table = {}
    if root is not None:
        if root.char is not None:
            table[root.char] = code if code != '' else '0'
        else:
            build_huffman_code_table(root.left, code + '0', table)
            build_huffman_code_table(root.right, code + '1', table)
    return table

def build_global_huffman(texts):
    global_freq = {}
    for text in texts:
        for ch in text:
            global_freq[ch] = global_freq.get(ch, 0) + 1

    tree = build_huffman_tree_from_freq(global_freq)
    code_table = build_huffman_code_table(tree)
    return tree, code_table

def huffman_encode(text, code_table):
    return ''.join(code_table[ch] for ch in text)

def huffman_decode(encoded_text, tree):
    decoded_text = ''
    node = tree
    for bit in encoded_text:
        node = node.left if bit == '0' else node.right
        if node and node.char is not None:
            decoded_text += node.char
            node = tree
    return decoded_text
