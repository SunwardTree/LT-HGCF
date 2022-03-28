#!/usr/bin/env python
# coding: utf-8
import random
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from utils import csr2tensor


class RelationalGraphConvLayer(Module):
    def __init__(self, input_dim, hidden_dim, rel_dict, device='cpu', dropout=[0, 0],
                 active_fun='leaky_relu', use_weight=False, use_rgcn=True):
        super(RelationalGraphConvLayer, self).__init__()
        self.rel_dict = rel_dict
        num_rel = len(self.rel_dict.keys())
        self.device = device
        self.use_weight = use_weight
        self.use_rgcn = use_rgcn

        if self.use_weight:
            # R-GCN weights
            self.w = Parameter(torch.FloatTensor(num_rel, input_dim, hidden_dim))
            if active_fun != 'none':
                nn.init.xavier_uniform_(self.w.data, gain=nn.init.calculate_gain(active_fun))
            else:
                nn.init.xavier_uniform_(self.w.data)

        net_drop = dropout[0]
        self.node_drop = dropout[1]
        self.drop_layer = nn.Dropout(net_drop)
        self.active_fun_str = active_fun

        if active_fun == 'relu':
            self.active_fun = nn.ReLU()
        else:
            self.active_fun = nn.LeakyReLU()

    def node_dropout(self, embs, dev):
        t_emb = embs.clone()
        index_long = torch.LongTensor(random.sample(range(embs.shape[0]), int(self.node_drop * embs.shape[0]))).to(dev)
        t_emb[index_long] = torch.zeros(embs.shape[1]).to(dev)
        return t_emb

    def forward(self, norm_A, norm_adjs, embs):
        embs = embs.to(self.device)
        all_embs = None
        if self.use_rgcn:
            for rel_str in self.rel_dict.keys():
                t_embs = torch.sparse.mm(csr2tensor(norm_A[self.rel_dict[rel_str]], self.device),
                                         self.node_dropout(embs, self.device))
                if self.active_fun_str != 'none':
                    if self.use_weight:
                        t_embs = self.drop_layer(self.active_fun(torch.mm(t_embs, self.w[self.rel_dict[rel_str]])))
                    else:
                        t_embs = self.active_fun(t_embs)
                else:
                    if self.use_weight:
                        t_embs = self.drop_layer(torch.mm(t_embs, self.w[self.rel_dict[rel_str]]))
                if all_embs is None:
                    all_embs = t_embs
                else:
                    all_embs += t_embs
        else:
            for t_adj in norm_adjs:
                t_embs = torch.sparse.mm(csr2tensor(t_adj, self.device),
                                         self.node_dropout(embs, self.device))
                if all_embs is None:
                    all_embs = t_embs
                else:
                    all_embs += t_embs
        return all_embs


class RelationalGraphConvModel(nn.Module):
    def __init__(self, input_size, out_size, hidden_size, rel_dict, num_layer,
                 dropout, device='cpu', active_fun='leaky_relu', use_weight=False,
                 use_rgcn=True):
        super(RelationalGraphConvModel, self).__init__()

        self.layers = nn.ModuleList()
        self.all_embs = None

        for i in range(num_layer):
            if i == 0:
                self.layers.append(RelationalGraphConvLayer(input_size, hidden_size, rel_dict,
                                                            device, dropout, active_fun, use_weight, use_rgcn))
            else:
                self.layers.append(RelationalGraphConvLayer(hidden_size, out_size, rel_dict,
                                                            device, dropout, active_fun, use_weight, use_rgcn))

    def forward(self, norm_A, norm_adjs, embs, use_residual=False, use_layer_weight=False):
        one_embs = 0
        embs_list = []
        for i, layer in enumerate(self.layers):
            embs = layer(norm_A, norm_adjs, embs)  # GCN propagation
            # 将第一层的embs加上
            if i == 0 and use_residual:
                one_embs = embs.clone()
            if i > 0 and use_residual:
                embs = embs + one_embs
            if use_layer_weight:
                embs_list.append(embs)
        if use_layer_weight:
            embs_tensor = torch.stack(embs_list, dim=1)
            self.all_embs = torch.mean(embs_tensor, dim=1)
        else:
            self.all_embs = embs
        return self.all_embs

    def lookup_emb(self, emb_index):
        if self.all_embs is not None:
            return self.all_embs[emb_index]
        else:
            return False


class PredictNet(nn.Module):
    def __init__(self, g_info, g_in_dim, g_out_dim, g_hidden_dim, p_hidden_dim, num_layer, dropout,
                 use_dr_pre, use_des, use_rev, pre_v_dict, pred_method, device='cpu',
                 active_fun='leaky_relu', use_residual=False, use_layer_weight=False,
                 use_weight=False, use_rgcn=True):
        super(PredictNet, self).__init__()

        num_nodes = g_info['n_node']
        rel_dict = g_info['rel_dict']
        self.use_residual = use_residual
        self.use_layer_weight = use_layer_weight
        self.active_fun_str = active_fun
        # self.active_fun_str = 'leaky_relu'
        if self.active_fun_str == 'relu':
            self.active_fun = nn.ReLU()
        else:
            self.active_fun = nn.LeakyReLU()

        if use_dr_pre:
            self.init_embs = nn.Parameter(torch.zeros(num_nodes, pre_v_dict['review'].shape[1]))
            num_user = g_info['n_user']
            num_item = g_info['n_item']
            num_des = g_info['n_des']
            if use_des:
                self.init_embs.data[num_user + num_item: num_user + num_item + num_des] = \
                    torch.FloatTensor(pre_v_dict['description'])
            if use_rev:
                self.init_embs.data[num_user + num_item + num_des:] = torch.FloatTensor(pre_v_dict['review'])
        else:
            self.init_embs = nn.Parameter(torch.zeros(num_nodes, g_in_dim))
            nn.init.normal_(self.init_embs.data)

        g_in_dim = self.init_embs.data.shape[1]
        print('Graph input dim is reset as:', g_in_dim)
        self.heteroGCN = RelationalGraphConvModel(input_size=g_in_dim, out_size=g_out_dim, hidden_size=g_hidden_dim,
                                                  rel_dict=rel_dict, num_layer=num_layer,
                                                  dropout=dropout, device=device,
                                                  active_fun=active_fun, use_weight=use_weight,
                                                  use_rgcn=use_rgcn)
        self.pred_method = pred_method
        if not use_weight:
            g_hidden_dim = g_in_dim
        if self.pred_method == 'mlp':
            self.predict_l_1 = nn.Linear(in_features=g_hidden_dim * 2, out_features=p_hidden_dim)
            self.predict_l_2 = nn.Linear(in_features=p_hidden_dim, out_features=p_hidden_dim)
            self.predict_l_3 = nn.Linear(in_features=p_hidden_dim, out_features=1)
        elif self.pred_method == 'joint':
            self.predict_l_1 = nn.Linear(in_features=g_hidden_dim * 2, out_features=p_hidden_dim)
            self.predict_l_2 = nn.Linear(in_features=p_hidden_dim, out_features=p_hidden_dim)
            self.predict_ul = nn.Linear(in_features=g_hidden_dim, out_features=p_hidden_dim)
            self.predict_il = nn.Linear(in_features=g_hidden_dim, out_features=p_hidden_dim)
            self.fusion_l = nn.Linear(in_features=p_hidden_dim * 2, out_features=1)

    def forward(self, norm_A, norm_adjs):
        # update emb
        self.heteroGCN(norm_A, norm_adjs, self.init_embs, self.use_residual, self.use_layer_weight)

    def predict(self, u_emb, i_emb, dropout=0.0):
        # predict part
        if self.pred_method == 'mlp':
            drop_layer = nn.Dropout(dropout)
            if self.active_fun_str != 'none':
                layer_1 = self.active_fun(drop_layer(
                    self.predict_l_1(torch.cat((u_emb, i_emb), dim=1))))
                layer_2 = self.active_fun(drop_layer(self.predict_l_2(layer_1)))
                predict_value = self.active_fun(self.predict_l_3(layer_2))
            else:
                layer_1 = drop_layer(
                    self.predict_l_1(torch.cat((u_emb, i_emb), dim=1)))
                layer_2 = drop_layer(self.predict_l_2(layer_1))
                predict_value = self.predict_l_3(layer_2)
        elif self.pred_method == 'joint':
            drop_layer = nn.Dropout(dropout)
            if self.active_fun_str != 'none':
                layer_1 = self.active_fun(drop_layer(
                    self.predict_l_1(torch.cat((u_emb, i_emb), dim=1))))
                layer_2 = self.active_fun(drop_layer(self.predict_l_2(layer_1)))
                mat_vec = layer_2
                u_layer = self.active_fun(drop_layer(self.predict_ul(u_emb)))
                i_layer = self.active_fun(drop_layer(self.predict_il(i_emb)))
                rep_vec = torch.mul(u_layer, i_layer)
                predict_value = self.active_fun(self.fusion_l(
                    torch.cat((mat_vec, rep_vec), dim=1)))
            else:
                layer_1 = drop_layer(
                    self.predict_l_1(torch.cat((u_emb, i_emb), dim=1)))
                layer_2 = drop_layer(self.predict_l_2(layer_1))
                mat_vec = layer_2
                u_layer = drop_layer(self.predict_ul(u_emb))
                i_layer = drop_layer(self.predict_il(i_emb))
                rep_vec = torch.mul(u_layer, i_layer)
                predict_value = self.fusion_l(
                    torch.cat((mat_vec, rep_vec), dim=1))
        else:
            # default dot pruduct
            embd = torch.mm(u_emb, i_emb.t()).sum(0)
            predict_value = embd.flatten()
        return predict_value
