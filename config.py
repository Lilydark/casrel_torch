class Config():
    def __init__(self):
        self.dataset = "baidu"
        self.bert_dim = 312
        self.bert_model_path = "voidful/albert_chinese_tiny"
        self.num_rel = 18
        self.schemas = "dataset/"+self.dataset+"/schemas.json"
        self.max_len = 512
        self.batch_size = 32
        self.lr = 1e-4
        self.h_bar = 0.5
        self.t_bar = 0.5
        self.save_model_name = "checkpoint/" + self.dataset + "_casrel_model.pth"
        self.rel = "dataset/"+self.dataset+"/rel.json"
        self.period = 100
        self.epochs = 1