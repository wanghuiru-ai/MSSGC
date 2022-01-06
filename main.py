import argparse
from tqdm import tqdm

from numpy import mean, argmax
import numpy as np
import random
import copy

import torch
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, AutoModel

from data.loader import load_corpus as load_mm_corpus
from text.loader import load_corpus as load_text_corpus
from image.loader import load_corpus as load_image_corpus
from model.resnet import resnet152
from model.mtm import MTM
from model.maml import optimize_w_sgd, optimize_w_adam, copy_model_params, update_vs_ss

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--cuda', type=int, default=6)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--bs', type=int, default=16)
parser.add_argument('--hidden_size', type=int, default=2816)
parser.add_argument('--hidden_layer', type=int, default=2)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--update_linear_lr', type=float, default=3e-4)
parser.add_argument('--update_embedding_lr', type=float, default=1e-5)
parser.add_argument('--senti_weight', type=float, default=0.3)
parser.add_argument('--fusion', type=str, default='concat', choices=('text', 'image', 'concat'))
parser.add_argument('--lm', type=str, default='bert', choices=('bert', 'roberta'))
parser.add_argument('--use_optimize', action='store_true', default=True)
parser.add_argument('--freeze_bert', action='store_true', default=False)
parser.add_argument('--freeze_resnet', action='store_true', default=False)
args = parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


seed_everything(args.seed)

mm_corpus = load_mm_corpus('/home/data1/liuyi/MSD/dataset')
mm_data_loader = DataLoader(mm_corpus.train, batch_size=args.bs, collate_fn=list, shuffle=True)

text_corpus = load_text_corpus('/home/data1/liuyi/MSD/dataset')
text_data_loader = DataLoader(text_corpus.train, batch_size=args.bs, collate_fn=list, shuffle=True)

image_corpus = load_image_corpus('/home/data1/liuyi/MSD/dataset')
image_data_loader = DataLoader(image_corpus.train, batch_size=args.bs, collate_fn=list, shuffle=True)

lm_path = {
    'bert': '/home/data1/liuyi/models/transformers/bert-base-uncased',
    'roberta': '/home/data1/liuyi/models/transformers/roberta-base',
}
device = torch.device(f'cuda:{args.cuda}')
tokenizer = AutoTokenizer.from_pretrained(lm_path[args.lm])
sentence_embedding = AutoModel.from_pretrained(lm_path[args.lm])
image_embedding = resnet152(pretrained=True)
model = MTM(device, tokenizer, sentence_embedding, image_embedding, hidden_size=args.hidden_size, hidden_layer=args.hidden_layer)
if args.freeze_bert:
    for parameter in sentence_embedding.parameters():
        parameter.requires_grad = False
if args.freeze_resnet:
    for parameter in image_embedding.parameters():
        parameter.requires_grad = False

lm_ids = list(map(id, sentence_embedding.parameters()))
resnet_ids = list(map(id, image_embedding.parameters()))
other_parameters = filter(lambda p: id(p) not in lm_ids + resnet_ids, model.parameters())
optimizer = torch.optim.Adam([
    {'params': other_parameters, 'lr': args.lr},
    {'params': sentence_embedding.parameters(), 'lr': args.lr / 100},
    {'params': image_embedding.parameters(), 'lr': args.lr / 100},
], weight_decay=args.weight_decay)

recalls_TSA, metrics_TSA = [], []
f1_scores_ISA, metrics_ISA = [], []
f1_scores_MSD, metrics_MSD = [], []
best_f1_msd = 0
best_f1_isa = 0
best_recall_tsa = 0
iteration = 1
text_data_loader_iterator = iter(text_data_loader)
image_data_loader_iterator = iter(image_data_loader)
for epoch in range(1, args.epochs + 1):
    loss_list_TSA = []
    loss_list_ISA = []
    loss_list_MSD = []
    model.train()
    for mm in tqdm(mm_data_loader, unit='batch'):
        optimizer.zero_grad()
        # get text batch
        try:
            text = next(text_data_loader_iterator)
        except StopIteration:
            text_data_loader_iterator = iter(text_data_loader)
            text = next(text_data_loader_iterator)
        # get image batch
        try:
            image = next(image_data_loader_iterator)
        except StopIteration:
            image_data_loader_iterator = iter(image_data_loader)
            image = next(image_data_loader_iterator)

        text_loss = model.forward_loss_text(text)
        loss_list_TSA.append(text_loss.item())

        image_loss = model.forward_loss_image(image, second_forward=True)
        loss_list_ISA.append(image_loss.item())

        mm_loss_first = model.forward_loss_mm(mm)
        loss_list_MSD.append(mm_loss_first.item())

        model_before = copy.deepcopy(model)
        model = optimize_w_sgd(
            model, optimizer=optimizer, epsilon=1, loss=text_loss+image_loss, update_linear_lr=args.update_linear_lr,
            update_embedding_lr=args.update_embedding_lr
        )

        mm_loss_second = model.forward_loss_mm(mm, second_forward=True)
        L = (1 - args.senti_weight) * mm_loss_first + args.senti_weight * mm_loss_second
        model = copy_model_params(model, model_before)
        L.backward()
        optimizer.step()
        del model_before, text, image, mm, L, text_loss, image_loss, mm_loss_first, mm_loss_second
        torch.cuda.empty_cache()

    loss1 = mean(loss_list_TSA)
    recall_TSA, f1_pn_TSA, accuracy_TSA = model.evaluate_text(text_corpus.test)
    recalls_TSA.append(recall_TSA)
    metrics_TSA.append((recall_TSA, f1_pn_TSA, accuracy_TSA))
    print(f'epoch #{epoch}, loss: {loss1:.2f}, recall: {recall_TSA:2.2%}')

    if recall_TSA > best_recall_tsa:
        print('best text epoch: ', epoch)
        best_recall_tsa = recall_TSA
    del loss1, recall_TSA, f1_pn_TSA, accuracy_TSA
    torch.cuda.empty_cache()

    loss2 = mean(loss_list_ISA)
    f1_score_ISA, precision_ISA, recall_ISA, accuracy_ISA = model.evaluate_image(image_corpus.test)
    f1_scores_ISA.append(f1_score_ISA)
    metrics_ISA.append((f1_score_ISA, precision_ISA, recall_ISA, accuracy_ISA))
    print(f'epoch #{epoch}, loss: {loss2:.2f}, f1_score: {f1_score_ISA:2.2%}')

    if f1_score_ISA > best_f1_isa:
        print('best image epoch: ', epoch)
        best_f1_isa = f1_score_ISA
    del loss2, f1_score_ISA, precision_ISA, recall_ISA, accuracy_ISA
    torch.cuda.empty_cache()

    loss3 = mean(loss_list_MSD)
    f1_score_MSD, precision_MSD, recall_MSD, accuracy_MSD = model.evaluate_mm(mm_corpus.test)
    f1_scores_MSD.append(f1_score_MSD)
    metrics_MSD.append((f1_score_MSD, precision_MSD, recall_MSD, accuracy_MSD))

    print(f'epoch #{epoch}, loss: {loss3:.2f}, f1_score: {f1_score_MSD:2.2%}')

    if f1_score_MSD > best_f1_msd:
        print('best msd epoch: ', epoch)
        best_f1_msd = f1_score_MSD

    del loss3, f1_score_MSD, precision_MSD, recall_MSD, accuracy_MSD
    torch.cuda.empty_cache()

print('-----------------------------------------------------')
print('epoch\tf1_score\tprecision\trecall\t\taccuracy')
print('-----------------------------------------------------')
for epoch, metric in zip(range(1, args.epochs + 1), metrics_MSD):
    f1_score, precision, recall, accuracy = metric
    print(f'{epoch}\t\t{f1_score:2.2%}\t\t{precision:2.2%}\t\t{recall:2.2%}\t\t{accuracy:2.2%}')
print('-----------------------------------------------------')

max_index = argmax(f1_scores_MSD).item()
print(f'msd: best f1 score({f1_scores_MSD[max_index]:2.2%}) at epoch#{max_index + 1}')

print('-----------------------------------------------------')
print('epoch\tf1_score\tprecision\trecall\t\taccuracy')
print('-----------------------------------------------------')
for epoch, metric in zip(range(1, args.epochs + 1), metrics_ISA):
    f1_score, precision, recall, accuracy = metric
    print(f'{epoch}\t\t{f1_score:2.2%}\t\t{precision:2.2%}\t\t{recall:2.2%}\t\t{accuracy:2.2%}')
print('-----------------------------------------------------')

max_index = argmax(f1_scores_ISA).item()
print(f'image: best f1 score({f1_scores_ISA[max_index]:2.2%}) at epoch#{max_index + 1}')

print('-----------------------------------------------------')
print('epoch\trecall\tf1_pn\t\taccuracy')
print('-----------------------------------------------------')
for epoch, metric in zip(range(1, args.epochs + 1), metrics_TSA):
    recall, f1_pn, accuracy = metric
    print(f'{epoch}\t\t{recall:2.2%}\t\t{f1_pn:2.2%}\t\t{accuracy:2.2%}')
print('-----------------------------------------------------')

max_index = argmax(recalls_TSA).item()
print(f'text: best f1 score({recalls_TSA[max_index]:2.2%}) at epoch#{max_index + 1}')
