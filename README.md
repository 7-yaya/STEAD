# STEAD: Robust Provably Secure Linguistic Steganography with Diffusion Language Model


<div align="center">

[![NeurIPS](https://img.shields.io/badge/NeurIPS%202025-Poster-blue.svg)](https://neurips.cc/virtual/2025/loc/san-diego/poster/117948)&nbsp;
[![OpenReview](https://img.shields.io/badge/OpenReview-b31b1b.svg)](https://openreview.net/forum?id=SF2POTDz2o)&nbsp;

</div>


This repo contains steganographic embedding/extracting algorithms proposed in
> [**STEAD: Robust Provably Secure Linguistic Steganography with Diffusion Language Model**](https://neurips.cc/virtual/2025/loc/san-diego/poster/117948)<br>
> Yuang Qi · Na Zhao · Qiyi Yao · Benlong Wu · Weiming Zhang · Nenghai Yu · Kejiang Chen
> <br>Anhui Province Key Laboratory of Digital Security, University of Science and Technology of China<br>

The code is implemented based on [**Dream**](https://github.com/DreamLM/Dream). If you want to reproduce the experiments in our paper, you need to deploy Dream-7B first. THANK THEM! The implementation of Dream is based on the [Huggingface `transformers`](https://github.com/huggingface/transformers) library. You should first install transformers by `pip install transformers==4.46.2` and `torch==2.5.1` as Dream uses the [SdpaAttention](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) built in torch. Other versions of transformers and torch are not been fully tested. Run the model requires a GPU with at least 20GB memory. 





