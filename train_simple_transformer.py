"""
@author : Hyunwoong
@when : 2019-10-22
@homepage : https://github.com/gusdnd852
"""
import math
import time

import torch
from torch import nn, optim
from torch.optim import Adam


from simple_transformer.model import Seq2SeqTransformer
from bleu import idx_to_word, get_bleu

from tqdm import tqdm
from loguru import logger

from data import PAD_IDX, BOS_IDX, EOS_IDX, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, build_dataloader, SRC_LANGUAGE, TGT_LANGUAGE, vocab_transform


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# GPU device setting
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# model parameter setting
batch_size = 128
max_len = 256
d_model = 512
n_layers = 3
n_heads = 8
ffn_hidden = 512
drop_prob = 0.1

# optimizer parameter setting
init_lr = 1e-5
factor = 0.9
adam_eps = 5e-9
patience = 10
warmup = 100
epoch = 3
clip = 1.0
weight_decay = 5e-4
inf = float('inf')

EVAL_INTERVAL = 100


def make_src_mask(src):
    src_mask = (src != PAD_IDX).unsqueeze(1).unsqueeze(2)
    return src_mask


def make_tgt_mask(tgt):
    tgt_pad_mask = (tgt != PAD_IDX).unsqueeze(1).unsqueeze(3)
    tgt_len = tgt.shape[1]
    # torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)
    tgt_sub_mask = torch.tril(torch.ones(tgt_len, tgt_len)).type(torch.ByteTensor).to(tgt_pad_mask.device)
    tgt_mask = tgt_pad_mask & tgt_sub_mask.type(torch.bool)
    return tgt_mask


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.kaiming_uniform(m.weight.data)


model = Seq2SeqTransformer(num_encoder_layers=n_layers,
                 num_decoder_layers=n_layers,
                 emb_size=d_model,
                 nhead=n_heads,
                 src_vocab_size=SRC_VOCAB_SIZE,
                 tgt_vocab_size=TGT_VOCAB_SIZE,
                 dim_feedforward=ffn_hidden,
                 dropout=drop_prob).to(device)

print(f'The model has {count_parameters(model):,} trainable parameters')
model.apply(initialize_weights)
optimizer = Adam(params=model.parameters(),
                 lr=init_lr,
                 weight_decay=weight_decay,
                 eps=adam_eps)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                 verbose=True,
                                                 factor=factor,
                                                 patience=patience)

criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)


def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    pbar = tqdm(enumerate(iterator))
    iters = 0
    for i, (src, tgt) in pbar:
        src = src.transpose(0, 1).to(DEVICE)
        tgt = tgt.transpose(0, 1).to(DEVICE)

        optimizer.zero_grad()
        src_mask = make_src_mask(src)
        tgt_mask = make_tgt_mask(tgt[:, :-1])
        output = model(src, tgt[:, :-1], src_mask, tgt_mask)
        output_reshape = output.contiguous().view(-1, output.shape[-1])
        tgt = tgt[:, 1:].contiguous().view(-1)

        loss = criterion(output_reshape, tgt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()
        pbar.set_description(f'Train Loss : {epoch_loss / (i + 1):.3f}')
        iters += 1
        # print('step :', round((i / len(iterator)) * 100, 2), '% , loss :', loss.item())

    return epoch_loss / iters

@torch.no_grad()
def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    batch_bleu = []
    iters = 0
    # with torch.no_grad():
    for src, tgt in tqdm(iterator):
        src = src.transpose(0, 1).to(device)
        tgt = tgt.transpose(0, 1).to(device)
        tgt_ori = tgt.clone()

        src_mask = make_src_mask(src)
        tgt_mask = make_tgt_mask(tgt[:, :-1])
        output = model(src, tgt[:, :-1], src_mask, tgt_mask)
        output_reshape = output.contiguous().view(-1, output.shape[-1])
        tgt = tgt[:, 1:].contiguous().view(-1)

        loss = criterion(output_reshape, tgt)
        epoch_loss += loss.item()

        total_bleu = []
        for j in range(batch_size):
            try:
                tgt_words = idx_to_word(tgt_ori[j], vocab_transform[TGT_LANGUAGE])
                output_words = output[j].max(dim=1)[1]
                output_words = idx_to_word(output_words, vocab_transform[TGT_LANGUAGE])

                # print('-----------------------------------')
                # print(f'Ground Truth: {tgt_words}')
                # print(f'Prediction: {output_words}')
                # print()

                bleu = get_bleu(hypotheses=output_words.split(), reference=tgt_words.split())
                total_bleu.append(bleu)
            except:
                pass

        total_bleu = sum(total_bleu) / len(total_bleu)
        batch_bleu.append(total_bleu)
        iters += 1

    batch_bleu = sum(batch_bleu) / len(batch_bleu)
    return epoch_loss / iters, batch_bleu


def run(total_epoch, best_loss):
    train_losses, test_losses, bleus = [], [], []
    for step in range(total_epoch):
        start_time = time.time()
        train_iter = build_dataloader('train', batch_size=batch_size)
        
        train_loss = train(model, train_iter, optimizer, criterion, clip)
        logger.info(f'Epoch: {step+1} - Train Loss: {train_loss}')

        if (step+1) % EVAL_INTERVAL == 0 or step == total_epoch - 1:
            valid_iter = build_dataloader('valid', batch_size=batch_size)
            valid_loss, bleu = evaluate(model, valid_iter, criterion)
            
            logger.info(f'Epoch: {step+1} - Val Loss: {valid_loss} - BLEU: {bleu}')
        # end_time = time.time()

            if step > warmup:
                scheduler.step(valid_loss)
            
            if valid_loss < best_loss:
                best_loss = valid_loss
                torch.save(model.state_dict(), 'saved/model-{0}.pt'.format(valid_loss))

        # train_losses.append(train_loss)
        # test_losses.append(valid_loss)
        # bleus.append(bleu)
        # epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        
        # f = open('result/train_loss.txt', 'w')
        # f.write(str(train_losses))
        # f.close()

        # f = open('result/bleu.txt', 'w')
        # f.write(str(bleus))
        # f.close()

        # f = open('result/test_loss.txt', 'w')
        # f.write(str(test_losses))
        # f.close()

        # print(f'Epoch: {step + 1} | Time: {epoch_mins}m {epoch_secs}s')
        # print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        # print(f'\tVal Loss: {valid_loss:.3f} |  Val PPL: {math.exp(valid_loss):7.3f}')
        # print(f'\tBLEU Score: {bleu:.3f}')


if __name__ == '__main__':
    run(total_epoch=epoch, best_loss=inf)
