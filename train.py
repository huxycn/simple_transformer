"""
@author : Hyunwoong
@when : 2019-10-22
@homepage : https://github.com/gusdnd852
"""
import math
import time

from torch import nn, optim
from torch.optim import Adam

# from data import *
from conf import *
from models.model.transformer import Transformer
from util.bleu import idx_to_word, get_bleu
from util.epoch_timer import epoch_time

from tqdm import tqdm

from data2 import PAD_IDX, BOS_IDX, EOS_IDX, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, build_dataloader, SRC_LANGUAGE, TGT_LANGUAGE, vocab_transform


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.kaiming_uniform(m.weight.data)


model = Transformer(src_pad_idx=PAD_IDX,
                    trg_pad_idx=PAD_IDX,
                    trg_sos_idx=BOS_IDX,
                    d_model=d_model,
                    enc_voc_size=SRC_VOCAB_SIZE,
                    dec_voc_size=TGT_VOCAB_SIZE,
                    max_len=max_len,
                    ffn_hidden=ffn_hidden,
                    n_head=n_heads,
                    n_layers=n_layers,
                    drop_prob=drop_prob,
                    device=device).to(device)

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
    for i, (src, trg) in pbar:
        src = src.transpose(0, 1).to(DEVICE)
        trg = trg.transpose(0, 1).to(DEVICE)

        optimizer.zero_grad()
        output = model(src, trg[:, :-1])
        output_reshape = output.contiguous().view(-1, output.shape[-1])
        trg = trg[:, 1:].contiguous().view(-1)

        loss = criterion(output_reshape, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()
        pbar.set_description(f'Epoch Loss : {epoch_loss / (i + 1):.3f}')
        iters += 1
        # print('step :', round((i / len(iterator)) * 100, 2), '% , loss :', loss.item())

    return epoch_loss / iters


def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    batch_bleu = []
    iters = 0
    with torch.no_grad():
        for src, trg in iterator:
            src = src.transpose(0, 1).to(device)
            trg = trg.transpose(0, 1).to(device)
            trg_ori = trg.clone()

            output = model(src, trg[:, :-1])
            output_reshape = output.contiguous().view(-1, output.shape[-1])
            trg = trg[:, 1:].contiguous().view(-1)

            loss = criterion(output_reshape, trg)
            epoch_loss += loss.item()

            total_bleu = []
            for j in range(batch_size):
                try:
                    trg_words = idx_to_word(trg_ori[j], vocab_transform[TGT_LANGUAGE])
                    output_words = output[j].max(dim=1)[1]
                    output_words = idx_to_word(output_words, vocab_transform[TGT_LANGUAGE])

                    # print('-----------------------------------')
                    # print(f'Ground Truth: {trg_words}')
                    # print(f'Prediction: {output_words}')
                    # print()

                    bleu = get_bleu(hypotheses=output_words.split(), reference=trg_words.split())
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
        valid_iter = build_dataloader('valid', batch_size=batch_size)
        train_loss = train(model, train_iter, optimizer, criterion, clip)
        valid_loss, bleu = evaluate(model, valid_iter, criterion)
        end_time = time.time()

        if step > warmup:
            scheduler.step(valid_loss)

        train_losses.append(train_loss)
        test_losses.append(valid_loss)
        bleus.append(bleu)
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), 'saved/model-{0}.pt'.format(valid_loss))

        f = open('result/train_loss.txt', 'w')
        f.write(str(train_losses))
        f.close()

        f = open('result/bleu.txt', 'w')
        f.write(str(bleus))
        f.close()

        f = open('result/test_loss.txt', 'w')
        f.write(str(test_losses))
        f.close()

        print(f'Epoch: {step + 1} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\tVal Loss: {valid_loss:.3f} |  Val PPL: {math.exp(valid_loss):7.3f}')
        print(f'\tBLEU Score: {bleu:.3f}')


if __name__ == '__main__':
    run(total_epoch=epoch, best_loss=inf)
