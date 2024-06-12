import torch
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TORCH_SEED = 129
DATA_DIR = '../data'
TRAIN_FILE = 'fold%s_train.json'
VALID_FILE = 'fold%s_valid.json'
TEST_FILE  = 'fold%s_test.json'
SENTIMENTAL_CLAUSE_DICT = 'sentimental_clauses.pkl'


class Config(object):
    def __init__(self):
        self.split = 'split10'
        self.bert_cache_path = 'bert-base-chinese'
        self.feat_dim = 768
        self.rgcn_layer = 1
        self.rgcn_hid_dims = 256
        self.rgcn_out_feats = 256

        self.gnn_dims = '192'
        self.tagset_size = 7
        self.att_heads = '4'
        self.K = 3
        self.pos_emb_dim = 50
        self.emo_emb_dim = 50
        self.pairwise_loss = False

        self.epochs = 15
        self.lr = 1e-5
        self.batch_size = 1
        self.gradient_accumulation_steps = 2
        self.dp = 0.1
        self.l2 = 1e-5
        self.l2_bert = 0.01
        self.warmup_proportion = 0.1
        self.adam_epsilon = 1e-8

