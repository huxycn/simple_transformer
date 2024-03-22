# Simple Transformer

A repo for transformer from theory (paper) to practice (code), make it easy to implement an Attention-Mechanism after reading the paper, and get familiar with the real Transformer implementation in PyTorch.

- **simple attention** implementation referring to https://github.com/hyunwoongko/transformer
- **similar transformer** implementation referring to https://github.com/pytorch/pytorch/blob/v1.12.1/torch/nn/modules/transformer.py
- **newer dataset** interface of torchtext and **seq2seq network** implementation referring to https://pytorch.org/tutorials/beginner/translation_transformer.html



## Installation

```
conda create -n transformer python=3.8
conda activate transformer

pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1+cu113 torchtext==0.13.1 -f https://download.pytorch.org/whl/torch_stable.html
```

## Data Preparation

```
pip install -U torchdata
pip install -U spacy
python -m spacy download en_core_web_sm
python -m spacy download de_core_news_sm
```

## Usage

```
python train_simple_transformer.py
```

## Code Reading Order

1. simple_transformer/attention.py
2. simple_transformer/transformer.py
3. simple_transformer/model.py
4. data.py
5. train_simple_transformer.py
