import torch.nn as nn


class Predictions_E(nn.Module):
    def __init__(self, configs):
        super(Predictions_E, self).__init__()
        self.feat_dim = int(configs.gnn_dims.strip().split(',')[-1]) * int(configs.att_heads)
        self.out_e = nn.Linear(self.feat_dim, 1)


    def forward(self, doc_sents_h):
        pred_e = self.out_e(doc_sents_h)
        return pred_e.squeeze(2)

class Predictions_C(nn.Module):
    def __init__(self, configs):
        super(Predictions_C, self).__init__()
        self.feat_dim = int(configs.gnn_dims.strip().split(',')[-1]) * int(configs.att_heads)
        self.out_c = nn.Linear(self.feat_dim, 1)

    def forward(self, doc_sents_h):
        pred_c = self.out_c(doc_sents_h)
        return pred_c.squeeze(2)

class Pre_Emotion(nn.Module):
    def __init__(self, configs):
        super(Pre_Emotion, self).__init__()
        self.feat_dim = int(configs.gnn_dims.strip().split(',')[-1]) * int(configs.att_heads)
        self.tagset_size = configs.tagset_size
        self.h_to_tag = nn.Linear(self.feat_dim, self.tagset_size)

    def forward(self, doc_sents_h):
        pred_emo = self.h_to_tag(doc_sents_h)
        return pred_emo