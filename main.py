from config import Config
import json
from DataLoading import CasRel_Dataset
from train import train

con = Config()

def data_get(data_path):
    file_name = open(data_path, encoding='utf8')
    data = json.load(file_name)
    return data

train_fn = "dataset/"+con.dataset+"/train_data.json"
dev_fn = "dataset/"+con.dataset+"/dev_data.json"
test_fn = "dataset/"+con.dataset+"/test_data.json"

train_data, dev_data, test_data = [ data_get(x) for x in [train_fn, dev_fn, test_fn] ]
rel2id, id2rel = json.load(open(con.rel, encoding='utf8'))

train_dataset = CasRel_Dataset(train_data, id2rel=id2rel, rel2id=rel2id, batch_size=con.batch_size)
dev_dataset = CasRel_Dataset(dev_data, id2rel=id2rel, rel2id=rel2id, batch_size=1)

train(train_dataset, dev_dataset, id2rel)

