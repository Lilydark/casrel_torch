import torch
from torch import nn
from transformers import BertModel
from config import Config

class Casrel(nn.Module):
    def __init__(self):
        super(Casrel, self).__init__()  # 继承，之后查一下继承的作用

        self.con = Config()

        # 语言模型，用albert
        self.bert_encoder = BertModel.from_pretrained(self.con.bert_model_path)  # 隐藏层维度：batch_size * seq_len * bert_size(312)
        self.hidden_size = self.con.bert_dim

        # 主语线性层
        self.sub_head_linear = nn.Linear(self.hidden_size, 1)
        self.sub_tail_linear = nn.Linear(self.hidden_size, 1)
        self.obj_head_linear = nn.Linear(self.hidden_size, self.con.num_rel)
        self.obj_tail_linear = nn.Linear(self.hidden_size, self.con.num_rel)

    def forward(self, data):
        token_ids = data['token_ids'].to('cuda')
        masks = data['masks'].to('cuda')

        hidden_emb = self.bert_encoder(token_ids, attention_mask=masks)[0]  # batch_size * seq_len * hidden_size

        # 预测主语位置
        sub_heads = data['sub_head'].to('cuda')  # batch_size * seq_length
        sub_tails = data['sub_tail'].to('cuda')
        sub_heads_prediction = torch.sigmoid(self.sub_head_linear(hidden_emb))
        sub_tails_prediction = torch.sigmoid(self.sub_tail_linear(hidden_emb))

        # 预测宾语位置与关系
        sub_heads_mm = torch.matmul(sub_heads.unsqueeze(1), hidden_emb)
        sub_tails_mm = torch.matmul(sub_tails.unsqueeze(1), hidden_emb)
        sub_both = (sub_heads_mm + sub_tails_mm) / 2
        hidden_emb_sub = sub_both + hidden_emb
        obj_heads_prediction = torch.sigmoid(self.obj_head_linear(hidden_emb_sub))
        obj_tails_prediction = torch.sigmoid(self.obj_tail_linear(hidden_emb_sub))

        return sub_heads_prediction, sub_tails_prediction, obj_heads_prediction, obj_tails_prediction

