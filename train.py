from config import Config
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
from model import Casrel
import torch.optim as optim
import time

import torch.nn.functional as F
import numpy as np

import json

def train(train_dataset, dev_dataset, id2rel):
    con = Config()
    model = Casrel()
    model.cuda()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=con.lr) # 优化器
    model.train()

    loss_sum = 0
    global_step = 0
    period = con.period
    start_time = time.time()

    train_batch_size = train_dataset.batch_size
    train_batch_nums = len(train_dataset) // train_batch_size
    if len(train_dataset) % train_batch_size > 0:
        train_batch_nums += 1

    dev_batch_size = dev_dataset.batch_size
    dev_batch_nums = len(dev_dataset) // dev_batch_size
    if len(dev_dataset) % dev_batch_size > 0:
        dev_batch_nums += 1

    def loss(gold, pred, mask):  # 损失函数：binary交叉熵，mask用来遮住padding
        pred = pred.squeeze(-1)
        los = F.binary_cross_entropy(pred, gold, reduction='none')
        if los.shape != mask.shape:
            mask = mask.unsqueeze(-1)
        los = torch.sum(los * mask) / torch.sum(mask)
        return los

    def to_tup(triple_list):
        ret = []
        for triple in triple_list:
            ret.append(tuple(triple))
        return ret


    for epoch_i in range(con.epochs):
        results = []
        for batch_i in range(train_batch_nums):
            dataset_i = train_dataset[batch_i]
            sub_heads_prediction, sub_tails_prediction, obj_heads_prediction, obj_tails_prediction = model(dataset_i)

            sub_heads_loss = loss(dataset_i['sub_heads'].to('cuda'), sub_heads_prediction, dataset_i['masks'].to('cuda'))
            sub_tails_loss = loss(dataset_i['sub_tails'].to('cuda'), sub_tails_prediction, dataset_i['masks'].to('cuda'))
            obj_heads_loss = loss(dataset_i['obj_head'].to('cuda'), obj_heads_prediction, dataset_i['masks'].to('cuda'))
            obj_tails_loss = loss(dataset_i['obj_tail'].to('cuda'), obj_tails_prediction, dataset_i['masks'].to('cuda'))
            total_loss = sub_heads_loss + sub_tails_loss + obj_heads_loss + obj_tails_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()  # 正常优化迭代

            loss_sum += total_loss.item()
            global_step += 1

            if global_step % period == 0:
                cur_loss = loss_sum / period
                elapsed = time.time() - start_time
                print("epoch: {:3d}, step: {:4d}, speed: {:5.2f}ms/b, train loss: {:5.3f}".
                      format(epoch_i, global_step, elapsed * 1000 / period, cur_loss))
                print(max(sub_heads_prediction[0]).item(), max(sub_tails_prediction[0]).item())
                print(obj_heads_prediction.max().item(), obj_tails_prediction[0].max().item())
                print(sub_heads_loss.item(), sub_tails_loss.item(), obj_heads_loss.item(), obj_tails_loss.item())

                loss_sum = 0
                start_time = time.time()

        for batch_i in range(dev_batch_nums):
            with torch.no_grad():
                correct_num, predict_num, gold_num = 0, 0, 0
                dataset_i = dev_dataset[batch_i]

                # 首先取得token_id
                token_ids = dataset_i['token_ids'].cuda()
                tokens = dataset_i['tokens'][0]
                masks = dataset_i['masks'].cuda()

                hidden_emb = model.bert_encoder(token_ids, attention_mask=masks)[0]

                # 还是先计算主语词向量
                sub_heads_prediction = torch.sigmoid(model.sub_head_linear(hidden_emb))
                sub_tails_prediction = torch.sigmoid(model.sub_tail_linear(hidden_emb))

                sub_heads, sub_tails = np.where(sub_heads_prediction.cpu()[0] > con.h_bar)[0], np.where(sub_tails_prediction.cpu()[0] > con.h_bar)[0]  # 找出sub可标注点

                subjects = []
                for sub_head in sub_heads:
                    sub_tail = sub_tails[sub_tails >= sub_head]
                    if len(sub_tail) > 0:
                        sub_tail = sub_tail[0]
                        subject = tokens[sub_head: sub_tail]
                        subjects.append((subject, sub_head, sub_tail))
                if subjects:
                    triple_list = []

                    # repeat：沿着指定的维度重复对应次数
                    # 此处的含义为每有一个候补sub重复一次，注意test的batchsize=1，所以0维度自动变为len(sub)，即len(sub) * seq_len * bert_len
                    repeated_hidden_emb = hidden_emb.repeat(len(subjects), 1, 1)

                    sub_head_mapping = torch.Tensor(len(subjects), 1,
                                                    hidden_emb.size(1)).zero_()  # hidden_emb.size(1)为seq_len
                    sub_tail_mapping = torch.Tensor(len(subjects), 1, hidden_emb.size(1)).zero_()

                    for subject_idx, subject in enumerate(subjects):  # 每个subject都是整个tokens, sub_head, sub_tail三部分组成的
                        sub_head_mapping[subject_idx][0][subject[1]] = 1
                        sub_tail_mapping[subject_idx][0][subject[2]] = 1

                    sub_tail_mapping = sub_tail_mapping.to(repeated_hidden_emb)
                    sub_head_mapping = sub_head_mapping.to(repeated_hidden_emb)

                    sub_heads_mm = torch.matmul(sub_tail_mapping, hidden_emb)
                    sub_tails_mm = torch.matmul(sub_head_mapping, hidden_emb)
                    sub_both = (sub_heads_mm + sub_tails_mm) / 2
                    hidden_emb_sub = sub_both + hidden_emb
                    obj_heads_prediction = torch.sigmoid(model.obj_head_linear(hidden_emb_sub))
                    obj_tails_prediction = torch.sigmoid(model.obj_tail_linear(hidden_emb_sub))

                    for subject_idx, subject in enumerate(subjects):
                        sub = subject[0]
                        sub = ''.join([i.lstrip("##") for i in sub])
                        sub = ' '.join(sub.split('[unused1]'))
                        obj_heads, obj_tails = np.where(obj_heads_prediction.cpu()[subject_idx] > con.t_bar), np.where(obj_tails_prediction.cpu()[subject_idx] > con.t_bar)
                        for obj_head, rel_head in zip(*obj_heads):
                            for obj_tail, rel_tail in zip(*obj_tails):
                                if obj_head <= obj_tail and rel_head == rel_tail:
                                    rel = id2rel[str(int(rel_head))]
                                    obj = tokens[obj_head: obj_tail]
                                    obj = ''.join([i.lstrip("##") for i in obj])
                                    obj = ' '.join(obj.split('[unused1]'))
                                    triple_list.append((sub, rel, obj))
                                    break

                        triple_set = set()
                        for s, r, o in triple_list:
                            triple_set.add((s, r, o))
                        pred_list = list(triple_set)
                else:
                    pred_list = []

                pred_triples = set(pred_list)
                gold_triples = set(to_tup(dataset_i['triple'][0]))
                correct_num += len(pred_triples & gold_triples)
                predict_num += len(pred_triples)
                gold_num += len(gold_triples)
                orders = ['subject', 'relation', 'object']
                result = {
                    # 'text': ' '.join(tokens),
                    'triple_list_gold': [
                        dict(zip(orders, triple)) for triple in gold_triples
                    ],
                    'triple_list_pred': [
                        dict(zip(orders, triple)) for triple in pred_triples
                    ],
                    'new': [
                        dict(zip(orders, triple)) for triple in pred_triples - gold_triples
                    ],
                    'lack': [
                        dict(zip(orders, triple)) for triple in gold_triples - pred_triples
                    ]
                }
                results.append(result)
        with open('dev_result/' + con.dataset + '_' + str(epoch_i) + '.txt', 'a', encoding='utf8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

    torch.save(model, 'model_result/model.pth')