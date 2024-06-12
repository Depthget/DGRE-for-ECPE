import torch
import torch.nn as nn
import dgl
import numpy as np
import dgl.nn as dglnn
import torch.nn.functional as F
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class RGCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super().__init__()
        rel_names = ['alle', 'allc', 'allp', 'cc', 'ec', 'ee', 'pe', 'pc', 'pp']
        self.conv1 = dglnn.HeteroGraphConv({rel: dglnn.GraphConv(in_feats, hid_feats) for rel in rel_names},
                                           aggregate='mean')
        self.conv2 = dglnn.HeteroGraphConv({rel: dglnn.GraphConv(hid_feats, out_feats) for rel in rel_names},
                                           aggregate='mean')

    def forward(self, graph, inputs):
        h = self.conv1(graph, inputs)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv2(graph, h)
        return h

class IERGCN(nn.Module):
    def __init__(self, configs):
        super(IERGCN, self).__init__()
        self.K = configs.K
        self.rgcn_layer = configs.rgcn_layer
        self.pos_emb_dim = configs.pos_emb_dim
        self.emo_emb_dim = configs.emo_emb_dim

        self.in_feats = configs.feat_dim
        self.hid_feats = configs.rgcn_hid_dims
        self.out_feats = configs.rgcn_out_feats

        self.pairs_input_embedding = nn.Linear(self.in_feats * 2 + self.pos_emb_dim + self.emo_emb_dim, self.in_feats)
        self.pairs_output1_embedding = nn.Linear(self.out_feats, self.out_feats)
        self.pairs_output2_embedding = nn.Linear(self.out_feats, 1)
        self.rgcn_layer_stack = nn.ModuleList()
        for i_layer_ in range(self.rgcn_layer):
            self.rgcn_layer_stack.append(RGCN(self.in_feats, self.hid_feats, self.out_feats))

    def forward(self, couples_pos_emo, doc_sents_he, doc_sents_hc, all_cls):
        batch_size, doc_sen_len, _ = doc_sents_he.size()
        for i in range(batch_size):
            len_sen_ = doc_sen_len
            p, e, c, c_u, c_v, all_node, all_e_node, all_c_node, all_p_node = self.dgl_adj_matrix_construction(len_sen_, self.K)
            emotion = doc_sents_he[i]
            cause = doc_sents_hc[i]
            pair = couples_pos_emo[i]
            all_f = all_cls[i].unsqueeze(0)


            G = self.pos_build_graph(p, e, c, c_u, c_v, emotion, cause, pair, all_f, all_node, all_e_node, all_c_node, all_p_node)
            all_feats = G.nodes['all'].data['feat']
            emotion_feats = G.nodes['emotion'].data['feat']
            cause_feats = G.nodes['cause'].data['feat']
            pair_feats = self.pairs_input_embedding(G.nodes['pair'].data['feat'])

            for i_r_gcn_layer, rgcn_layer in enumerate(self.rgcn_layer_stack):
                pred_dict = rgcn_layer(G, {'emotion': emotion_feats, 'cause': cause_feats, 'pair': pair_feats, 'all': all_feats})

            get_hidden_p = F.relu(self.pairs_output1_embedding(pred_dict['pair']))
            get_out_p = self.pairs_output2_embedding(get_hidden_p)
            out_p = get_out_p.squeeze(1)

            if i == 0:
                # emotions_pred = pred_dict['emotion'].unsqueeze(0)
                # causes_pred = pred_dict['cause'].unsqueeze(0)
                couples_pred = out_p.unsqueeze(0)
            else:
                # emotions_pred = torch.cat([emotions_pred, pred_dict['emotion'].unsqueeze(0)], dim=0)
                # causes_pred = torch.cat([causes_pred, pred_dict['cause'].unsqueeze(0)], dim=0)
                couples_pred = torch.cat([couples_pred, out_p.unsqueeze(0)], dim=0)

        return couples_pred

    def pos_build_graph(self, p, e, c, c_u, c_v, emotion, cause, pair, all, all_node, all_e_node, all_c_node, all_p_node):
        G = dgl.heterograph({
            ('all', 'alle', 'emotion'): (all_node, all_e_node),
            ('all', 'allc', 'cause'): (all_node, all_c_node),
            ('all', 'allp', 'pair'): (all_p_node, p),
            ('cause', 'cc', 'cause'): (c_u, c_v),
            ('emotion', 'ec', 'cause'): (e, c),
            ('emotion', 'ee', 'emotion'): (e, e),

            ('pair', 'pe', 'emotion'): (p, e),
            ('pair', 'pc', 'cause'): (p, c),
            ('pair', 'pp', 'pair'): (p, p)
        })

        G.nodes['emotion'].data['feat'] = emotion
        G.nodes['cause'].data['feat'] = cause
        G.nodes['pair'].data['feat'] = pair
        G.nodes['all'].data['feat'] = all

        return G

    def dgl_adj_matrix_construction(self, seq_len, k):
        you_want_gcn_len_cau = 1
        pos = torch.LongTensor(np.concatenate([np.arange(1, seq_len + 1)] * seq_len, axis=0) - np.concatenate([np.arange(1, seq_len + 1).reshape(-1, 1)] * seq_len, axis=1).reshape(1, -1)[0]).to(DEVICE)
        pair_idx = torch.tensor(np.array(list(map(lambda x: -k <= x <= k, pos.tolist())), dtype=int).reshape(seq_len, seq_len).tolist())
        cau_gcn_len_idx = torch.tensor(np.array(list(map(lambda y: -you_want_gcn_len_cau <= y <= you_want_gcn_len_cau, pos.tolist())), dtype=int).reshape(seq_len, seq_len).tolist())
        p, e, c, c_u, c_v = [], [], [], [], []
        r, prs_tmp = 0, 0
        for i in range(seq_len):
            for j in range(seq_len):
                if pair_idx[i][j] == 1:
                    pairs = prs_tmp
                    p.append(pairs)
                    prs_tmp += 1
                    e.append(i)
                    c.append(j)
        for rows in range(seq_len):
            for column in range(seq_len):
                if cau_gcn_len_idx[rows][column] == 1:
                    c_u.append(rows)
                    c_v.append(column)
        all_node = [0]*seq_len
        all_e_node = [i for i in range(seq_len)]
        all_c_node = all_e_node
        all_p_node = [0]*len(p)

        return p, e, c, c_u, c_v, all_node, all_e_node, all_c_node, all_p_node


