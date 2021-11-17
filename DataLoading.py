import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

from collections import defaultdict
from random import choice

from config import Config


class CasRel_Dataset(Dataset):
    def __init__(self, data, id2rel, rel2id, batch_size=10):
        self.data = data
        self.id2rel = id2rel
        self.rel2id = rel2id
        self.batch_size = batch_size
        self.con = Config()

        self.tokenizer = BertTokenizer.from_pretrained(self.con.bert_model_path)

        self.triple_lists = [data[i]['triple_list'] for i in range(len(self.data))]
        self.text_list = [data[i]['text'][:self.con.max_len] for i in range(len(self.data))]

        self.tokens = [self.tokenizer.tokenize(self.text_list[i]) for i in range(len(self.data))]
        self.token_ids = [self.tokenizer.encode(self.text_list[i]) for i in range(len(self.data))]

    #         self.token_ids = [x[0] for x in self.tokenizer_ids]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        def find_head_idx(source, target):
            target_len = len(target)
            for i in range(len(source)):
                if source[i: i + target_len] == target:
                    return i
            return -1

        if idx * self.batch_size >= len(self.data):
            return None
        start, end = idx * self.batch_size, (idx + 1) * self.batch_size

        if end > len(self.data):
            end = len(self.data)

        triple_lists_idx = self.triple_lists[start:end]
        text_list_idx = self.text_list[start:end]
        tokens_idx = self.tokens[start:end]
        token_ids_idx = self.token_ids[start:end]
        max_length = max([len(x) for x in token_ids_idx])
        token_ids_idx = [x + [0] * (max_length - len(x)) for x in token_ids_idx]
        masks = [[1] * (len(x) + 2) + [0] * (max_length - len(x) - 2) for x in tokens_idx]

        sub_head_idx = torch.zeros(len(triple_lists_idx), max_length)
        sub_tail_idx = torch.zeros(len(triple_lists_idx), max_length)
        sub_heads_idx = torch.zeros(len(triple_lists_idx), max_length)
        sub_tails_idx = torch.zeros(len(triple_lists_idx), max_length)
        obj_heads_idx = torch.zeros(len(triple_lists_idx), max_length, len(self.rel2id))
        obj_tails_idx = torch.zeros(len(triple_lists_idx), max_length, len(self.rel2id))

        for i in range(len(triple_lists_idx)):
            so_dic = defaultdict(list)
            triple_list = triple_lists_idx[i]
            token = token_ids_idx[i]
            for sub_j, rel_j, obj_j in triple_list:
                sub_j_token_id = self.tokenizer.encode(sub_j)[1:-1]
                obj_j_token_id = self.tokenizer.encode(obj_j)[1:-1]
                sub_pos = find_head_idx(token, sub_j_token_id) - 1
                obj_pos = find_head_idx(token, obj_j_token_id) - 1
                if sub_pos > -1:
                    sub_heads_idx[i, sub_pos] = 1
                    sub_tails_idx[i, sub_pos + len(sub_j_token_id)] = 1
                if obj_pos > -1:
                    so_dic[(sub_pos, sub_pos + len(sub_j_token_id))].append(
                        (obj_pos, obj_pos + len(obj_j_token_id), self.rel2id[rel_j]))

            if so_dic:
                sub_head_idx_temp, sub_tail_idx_temp = choice(list(so_dic.keys()))
                sub_head_idx[i, sub_head_idx_temp] = 1
                sub_tail_idx[i, sub_tail_idx_temp] = 1
                for ro in so_dic.get((sub_head_idx_temp, sub_tail_idx_temp), []):
                    obj_heads_idx[i, ro[0], ro[2]] = 1
                    obj_tails_idx[i, ro[1], ro[2]] = 1

        return {
            'tokens': tokens_idx,
            'text': text_list_idx,
            'triple': triple_lists_idx,
            'token_ids': torch.tensor(token_ids_idx),
            'masks': torch.tensor(masks),
            'sub_heads': sub_head_idx,
            'sub_tails': sub_tail_idx,
            'sub_head': sub_head_idx,
            'sub_tail': sub_tail_idx,
            'obj_head': obj_heads_idx,
            'obj_tail': obj_tails_idx
        }