import numpy as np
import torch
import sys
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from src.networks.supervisor_layer import *
from src.networks.gat_layer import *
from src.networks.RGCN import *

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Network(nn.Module):
    def __init__(self, configs):
        super(Network, self).__init__()
        self.bert = BertModel.from_pretrained(configs.bert_cache_path)
        self.gnn = GraphNN(configs)
        self.pred_E = Predictions_E(configs)
        self.pred_C = Predictions_C(configs)
        self.pred_e_layer = Pre_Emotion(configs)
        self.PG = Pair_Generate(configs)
        self.iergcn = IERGCN(configs)
        self.pairwise_loss = configs.pairwise_loss
        self.tagset_size = configs.tagset_size

    def forward(self, bert_token_b, bert_segment_b, bert_masks_b, bert_clause_b, doc_len, adj):
        bert_output = self.bert(input_ids=bert_token_b.to(DEVICE), attention_mask=bert_masks_b.to(DEVICE), token_type_ids = bert_segment_b.to(DEVICE))
        all_cls = bert_output[1]

        doc_sents_h = self.batched_index_select(bert_output, bert_clause_b.to(DEVICE))

        doc_sents_h = self.gnn(doc_sents_h, doc_len, adj)

        doc_sents_he = doc_sents_h
        doc_sents_hc = doc_sents_h
        pred_e = self.pred_E(doc_sents_he)
        pred_c = self.pred_C(doc_sents_hc)
        pred_emo = self.pred_e_layer(doc_sents_he)

        couples_pos_emo, emo_cau_pos = self.PG(doc_sents_he, doc_sents_hc, pred_emo)
        couples_pred = self.iergcn(couples_pos_emo, doc_sents_he, doc_sents_hc, all_cls)


        return couples_pred, emo_cau_pos, pred_e, pred_emo, pred_c

    def batched_index_select(self, bert_output, bert_clause_b):
        hidden_state = bert_output[0]
        dummy = bert_clause_b.unsqueeze(2).expand(bert_clause_b.size(0), bert_clause_b.size(1), hidden_state.size(2))
        doc_sents_h = hidden_state.gather(1, dummy)
        return doc_sents_h

    def loss_rank(self, couples_pred, emo_cau_pos, doc_couples, y_mask, test=False):
        couples_true, couples_mask, doc_couples_pred = self.output_util(couples_pred, emo_cau_pos, doc_couples, y_mask, test)

        if not self.pairwise_loss:
            couples_mask = torch.ByteTensor(couples_mask).to(DEVICE)
            couples_true = torch.FloatTensor(couples_true).to(DEVICE)
            criterion = nn.BCEWithLogitsLoss(reduction='mean')
            couples_true = couples_true.masked_select(couples_mask.bool())
            couples_pred = couples_pred.masked_select(couples_mask.bool())
            loss_couple = criterion(couples_pred, couples_true)
        else:
            x1, x2, y = self.pairwise_util(couples_pred, couples_true, couples_mask)
            criterion = nn.MarginRankingLoss(margin=1.0, reduction='mean')
            loss_couple = criterion(F.tanh(x1), F.tanh(x2), y)

        return loss_couple, doc_couples_pred

    def output_util(self, couples_pred, emo_cau_pos, doc_couples, y_mask, test=False):
        batch, n_couple = couples_pred.size()
        couples_true, couples_mask = [], []

        doc_couples_pred = []
        for i in range(batch):
            y_mask_i = y_mask[i]
            max_doc_idx = sum(y_mask_i)
            doc_couples_i = doc_couples[i]
            couples_true_i = []
            couples_mask_i = []
            for couple_idx, emo_cau in enumerate(emo_cau_pos):

                if emo_cau[0] > max_doc_idx or emo_cau[1] > max_doc_idx:
                    couples_mask_i.append(0)
                    couples_true_i.append(0)
                else:
                    couples_mask_i.append(1)
                    couples_true_i.append(1 if emo_cau in doc_couples_i else 0)

            couples_pred_i = couples_pred[i]

            doc_couples_pred_i = []
            if test:
                if torch.sum(torch.isnan(couples_pred_i)) > 0:
                    k_idx = [0] * 3
                else:
                    _, k_idx = torch.topk(couples_pred_i, k=3, dim=0)
                doc_couples_pred_i = [(emo_cau_pos[idx], couples_pred_i[idx].tolist()) for idx in k_idx]

            couples_true.append(couples_true_i)
            couples_mask.append(couples_mask_i)
            doc_couples_pred.append(doc_couples_pred_i)

        return couples_true, couples_mask, doc_couples_pred

    def loss_pre(self, pred_e, pred_c, y_emotions, y_causes, y_mask):
        y_mask = torch.BoolTensor(y_mask).to(DEVICE)
        y_emotions = torch.FloatTensor(y_emotions).to(DEVICE)
        y_causes = torch.FloatTensor(y_causes).to(DEVICE)
        criterion = nn.BCEWithLogitsLoss(reduction='mean')
        pred_e = pred_e.masked_select(y_mask)
        true_e = y_emotions.masked_select(y_mask)
        loss_e = criterion(pred_e, true_e)
        pred_c = pred_c.masked_select(y_mask)
        true_c = y_causes.masked_select(y_mask)
        loss_c = criterion(pred_c, true_c)
        return loss_e, loss_c

    def loss_pre_emo(self, pred_emo, y_emotions_label_b, y_mask):
        y_mask = np.expand_dims(y_mask, axis=-1)
        y_mask = torch.ByteTensor(y_mask).to(DEVICE)
        y_mask = torch.cat([y_mask] * self.tagset_size, dim=-1).bool()
        y_emotions_label_b = torch.FloatTensor(y_emotions_label_b).to(DEVICE)
        loss_function = nn.BCEWithLogitsLoss(reduction='mean')
        pred_emo = pred_emo.masked_select(y_mask)
        true_emo = y_emotions_label_b.masked_select(y_mask)
        loss_emo = loss_function(pred_emo, true_emo)
        return loss_emo

    def pairwise_util(self, couples_pred, couples_true, couples_mask):
        batch, n_couple = couples_pred.size()
        x1, x2 = [], []
        for i in range(batch):
            x1_i_tmp = []
            x2_i_tmp = []
            couples_mask_i = couples_mask[i]
            couples_pred_i = couples_pred[i]
            couples_true_i = couples_true[i]
            for pred_ij, true_ij, mask_ij in zip(couples_pred_i, couples_true_i, couples_mask_i):
                if mask_ij == 1:
                    if true_ij == 1:
                        x1_i_tmp.append(pred_ij.reshape(-1, 1))
                    else:
                        x2_i_tmp.append(pred_ij.reshape(-1))
            m = len(x2_i_tmp)
            n = len(x1_i_tmp)
            x1_i = torch.cat([torch.cat(x1_i_tmp, dim=0)] * m, dim=1).reshape(-1)
            x1.append(x1_i)
            x2_i = []
            for _ in range(n):
                x2_i.extend(x2_i_tmp)
            x2_i = torch.cat(x2_i, dim=0)
            x2.append(x2_i)

        x1 = torch.cat(x1, dim=0)
        x2 = torch.cat(x2, dim=0)
        y = torch.FloatTensor([1] * x1.size(0)).to(DEVICE)
        return x1, x2, y

class Pre_Predictions(nn.Module):
    def __init__(self, configs):
        super(Pre_Predictions, self).__init__()
        self.feat_dim = int(configs.gnn_dims.strip().split(',')[-1]) * int(configs.att_heads.strip().split(',')[-1])
        self.out_e = nn.Linear(self.feat_dim, 1)
        self.out_c = nn.Linear(self.feat_dim, 1)

    def forward(self, doc_sents_h):
        pred_e = self.out_e(doc_sents_h)
        pred_c = self.out_c(doc_sents_h)
        return pred_e.squeeze(2), pred_c.squeeze(2)

class Pre_Emotion(nn.Module):
    def __init__(self, configs):
        super(Pre_Emotion, self).__init__()
        self.feat_dim = int(configs.gnn_dims.strip().split(',')[-1]) * int(configs.att_heads.strip().split(',')[-1])
        self.tagset_size = configs.tagset_size
        self.h_to_tag = nn.Linear(self.feat_dim, self.tagset_size)
    def forward(self, doc_sents_h):
        pred_emo = self.h_to_tag(doc_sents_h)
        return pred_emo

class GraphNN(nn.Module):
    def __init__(self, configs):
        super(GraphNN, self).__init__()
        in_dim = configs.feat_dim
        self.gnn_dims = [in_dim] + [int(dim) for dim in configs.gnn_dims.strip().split(',')]

        self.gnn_layers = len(self.gnn_dims) - 1
        self.att_heads = [int(att_head) for att_head in configs.att_heads.strip().split(',')]
        self.gnn_layer_stack = nn.ModuleList()
        for i in range(self.gnn_layers):
            in_dim = self.gnn_dims[i] * self.att_heads[i - 1] if i != 0 else self.gnn_dims[i]
            self.gnn_layer_stack.append(GraphAttentionLayer(self.att_heads[i], in_dim, self.gnn_dims[i + 1], configs.dp))

    def forward(self, doc_sents_h, doc_len, adj):
        batch, max_doc_len, _ = doc_sents_h.size()
        assert max(doc_len) == max_doc_len
        for i, gnn_layer in enumerate(self.gnn_layer_stack):
            doc_sents_h = gnn_layer(doc_sents_h, adj)
        return doc_sents_h

class Pair_Generate(nn.Module):
    def __init__(self, configs):
        super(Pair_Generate, self).__init__()
        self.batch_size =configs.batch_size
        self.K = configs.K
        self.pos_emb_dim = configs.pos_emb_dim
        self.pos_layer = nn.Embedding(2*self.K + 1, self.pos_emb_dim)
        nn.init.xavier_uniform_(self.pos_layer.weight)

        self.emo_emb_dim = configs.emo_emb_dim
        self.tagset_size = configs.tagset_size
        self.emo_layer = nn.Embedding(self.tagset_size, self.emo_emb_dim)
        nn.init.xavier_uniform_(self.emo_layer.weight)
        self.feat_dim = int(configs.gnn_dims.strip().split(',')[-1]) * int(configs.att_heads.strip().split(',')[-1])
        self.rank_feat_dim = 2*self.feat_dim + self.emo_emb_dim + self.pos_emb_dim


    def forward(self, doc_sents_he, doc_sents_hc, pred_emo):
        pred_emo = pred_emo.reshape(-1, 7)
        _, pred_emo = torch.max(pred_emo, 1)
        pred_emo = pred_emo.reshape(self.batch_size, -1)
        emo_lab_embedding = self.emo_layer(pred_emo)
        couples_h, rel_pos, emo_cau_pos = self.couple_generator(doc_sents_he, doc_sents_hc, self.K, emo_lab_embedding)

        rel_pos = rel_pos + self.K
        rel_pos_emb = self.pos_layer(rel_pos)
        kernel = self.kernel_generator(rel_pos)
        kernel = kernel.unsqueeze(0).expand(self.batch_size, -1, -1)
        rel_pos_emb = torch.matmul(kernel, rel_pos_emb)
        couples_pos_emo = torch.cat([couples_h, rel_pos_emb], dim=2)

        return couples_pos_emo, emo_cau_pos

    def couple_generator(self, He, Hc, k, emo_lab_embedding):
        batch, seq_len, feat_dim = He.size()
        e_batch, e_seq_len, e_feat_dim = emo_lab_embedding.size()
        emo_lab_embedding = torch.cat([emo_lab_embedding] * seq_len, dim=2)
        emo_lab_embedding = emo_lab_embedding.reshape(-1, seq_len * seq_len, e_feat_dim)

        P_left = torch.cat([He] * seq_len, dim=2)
        P_left = P_left.reshape(-1, seq_len * seq_len, feat_dim)
        P_right = torch.cat([Hc] * seq_len, dim=1)
        P = torch.cat([P_left, P_right], dim=2)
        P = torch.cat([P, emo_lab_embedding], dim=2)

        base_idx = np.arange(1, seq_len + 1)
        emo_pos = np.concatenate([base_idx.reshape(-1, 1)] * seq_len, axis=1).reshape(1, -1)[0]
        cau_pos = np.concatenate([base_idx] * seq_len, axis=0)
        rel_pos = cau_pos - emo_pos
        rel_pos = torch.LongTensor(rel_pos).to(DEVICE)
        emo_pos = torch.LongTensor(emo_pos).to(DEVICE)
        cau_pos = torch.LongTensor(cau_pos).to(DEVICE)

        if seq_len > k + 1:
            rel_mask = np.array(list(map(lambda x: -k <= x <= k, rel_pos.tolist())), dtype=int)
            rel_mask = torch.BoolTensor(rel_mask).to(DEVICE)
            rel_pos = rel_pos.masked_select(rel_mask)
            emo_pos = emo_pos.masked_select(rel_mask)
            cau_pos = cau_pos.masked_select(rel_mask)
            rel_mask = rel_mask.unsqueeze(1).expand(-1, 2 * feat_dim + e_feat_dim)
            rel_mask = rel_mask.unsqueeze(0).expand(batch, -1, -1)

            P = P.masked_select(rel_mask)
            P = P.reshape(batch, -1, 2 * feat_dim + e_feat_dim)
        assert rel_pos.size(0) == P.size(1)
        rel_pos = rel_pos.unsqueeze(0).expand(batch, -1)

        emo_cau_pos = []
        for emo, cau in zip(emo_pos.tolist(), cau_pos.tolist()):
            emo_cau_pos.append([emo, cau])
        return P, rel_pos, emo_cau_pos

    def kernel_generator(self, rel_pos):
        n_couple = rel_pos.size(1)
        rel_pos_ = rel_pos[0].type(torch.FloatTensor).to(DEVICE)
        kernel_left = torch.cat([rel_pos_.reshape(-1, 1)] * n_couple, dim=1)
        kernel = kernel_left - kernel_left.transpose(0, 1)
        return torch.exp(-(torch.pow(kernel, 2)))

