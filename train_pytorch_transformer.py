"""
This code is copied from https://pytorch.org/tutorials/beginner/translation_transformer.html
    - subsequent word mask & source and target padding masks
    - hyperparameters, model, loss function, optimizer
    - training and evaluation loop
    - run training and prediction
"""

import torch
import torch.nn as nn

from timeit import default_timer as timer
from bleu import get_bleu


from data import (
    PAD_IDX, BOS_IDX, EOS_IDX,
    SRC_LANGUAGE, TGT_LANGUAGE,
    SRC_VOCAB_SIZE, TGT_VOCAB_SIZE,
    vocab_transform, text_transform,
    build_dataloader
)

from pytorch_transformer.model import Seq2SeqTransformer


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(0)

NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3
EMB_SIZE = 512
NHEAD = 8
FFN_HID_DIM = 512

BATCH_SIZE = 128

NUM_EPOCHS = 3


def _gen_pad_mask_base(x):
    """ Example (F position is masked):
    >>> PAD_IDX = 1
    >>> _gen_pad_mask_base(torch.tensor([[2, 4, 5, 3], [2, 4, 3, 1]]))
    tensor([[F, F, F, F],
            [F, F, F, T]])
    """
    return (x == PAD_IDX)


def _gen_sub_mask_base(sz):
    """ Example (F position is masked):
    >>> _gen_sub_mask_base(3)
    tensor([[F, T, T],
            [F, F, T],
            [F, F, F]])
    """
    return torch.triu(torch.ones(sz, sz), diagonal=1).type(torch.bool)


def generate_square_subsequent_mask(sz):
    mask = _gen_sub_mask_base(sz)
    return mask
    # mask = mask.float().masked_fill(~mask, float('-inf')).masked_fill(mask, float(0.0))
    # return mask


def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len)).type(torch.bool)

    mem_mask = torch.zeros((tgt_seq_len, src_seq_len)).type(torch.bool)

    src_padding_mask = _gen_pad_mask_base(src).transpose(0, 1)
    tgt_padding_mask = _gen_pad_mask_base(tgt).transpose(0, 1)
    return src_mask, tgt_mask, mem_mask, src_padding_mask, tgt_padding_mask, src_padding_mask


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                 NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)

print(f'The model has {count_parameters(transformer):,} trainable parameters')

for name, p in transformer.named_parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)
    elif 'proj' in name:
        nn.init.constant_(p, 0.)

transformer = transformer.to(DEVICE)


criterion = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)



def train_epoch(model, optimizer):
    model.train()
    losses = 0
    train_dataloader = build_dataloader('train', batch_size=BATCH_SIZE)

    for src, tgt in train_dataloader:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, mem_mask, src_padding_mask, tgt_padding_mask, mem_padding_mask = create_mask(src, tgt_input)
        src_mask = src_mask.to(DEVICE)
        tgt_mask = tgt_mask.to(DEVICE)
        mem_mask = mem_mask.to(DEVICE)
        src_padding_mask = src_padding_mask.to(DEVICE)
        tgt_padding_mask = tgt_padding_mask.to(DEVICE)
        mem_padding_mask = mem_padding_mask.to(DEVICE)

        logits = model(src, tgt_input, src_mask, tgt_mask, mem_mask, src_padding_mask, tgt_padding_mask, mem_padding_mask)

        optimizer.zero_grad()

        tgt_out = tgt[1:, :]
        loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()

    return losses / len(list(train_dataloader))


@torch.no_grad()
def evaluate(model):
    model.eval()
    losses = 0

    val_dataloader = build_dataloader('valid', batch_size=BATCH_SIZE)

    for src, tgt in val_dataloader:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, mem_mask, src_padding_mask, tgt_padding_mask, mem_padding_mask = create_mask(src, tgt_input)
        src_mask = src_mask.to(DEVICE)
        tgt_mask = tgt_mask.to(DEVICE)
        mem_mask = mem_mask.to(DEVICE)
        src_padding_mask = src_padding_mask.to(DEVICE)
        tgt_padding_mask = tgt_padding_mask.to(DEVICE)
        mem_padding_mask = mem_padding_mask.to(DEVICE)

        logits = model(src, tgt_input, src_mask, tgt_mask, mem_mask, src_padding_mask, tgt_padding_mask, mem_padding_mask)

        tgt_out = tgt[1:, :]
        loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()

    return losses / len(list(val_dataloader))


def run():
    for epoch in range(1, NUM_EPOCHS+1):
        start_time = timer()
        train_loss = train_epoch(transformer, optimizer)
        end_time = timer()
        val_loss = evaluate(transformer)
        print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))


# function to generate output sequence using greedy algorithm
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)
    src_padding_mask = (src == PAD_IDX).transpose(0, 1).to(DEVICE)

    memory = model.encode(src, src_mask, src_padding_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
    for i in range(max_len-1):
        memory = memory.to(DEVICE)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0))
                    .type(torch.bool)).to(DEVICE)
        mem_mask = (torch.zeros(ys.size(0), memory.size(0))
                    .type(torch.bool)).to(DEVICE)
        tgt_key_padding_mask = (ys == PAD_IDX).transpose(0, 1).to(DEVICE)
        mem_key_padding_mask = src_padding_mask.to(DEVICE)
        out = model.decode(ys, memory, tgt_mask, mem_mask, tgt_key_padding_mask, mem_key_padding_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == EOS_IDX:
            break
    return ys


# actual function to translate input sentence into target language
def translate(model: torch.nn.Module, src_sentence: str):
    model.eval()
    src = text_transform[SRC_LANGUAGE](src_sentence).view(-1, 1)
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = greedy_decode(
        model,  src, src_mask, max_len=num_tokens + 5, start_symbol=BOS_IDX).flatten()
    return " ".join(vocab_transform[TGT_LANGUAGE].lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<bos>", "").replace("<eos>", "")


if __name__ == '__main__':
    run()
    src_words = "A group of people stand in front of an igloo ."
    tgt_words = "Eine Gruppe von Menschen steht vor einem Iglu ."
    output_words = translate(transformer, src_words)
    bleu = get_bleu(hypotheses=output_words.split(), reference=tgt_words.split())

    print(f'src_words: {src_words}')
    print(f'tgt_words: {tgt_words}')
    print(f'output_words: {output_words}')
    print(f'BLEU: {bleu}')
