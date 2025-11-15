# -*- coding: utf-8 -*-

import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss
from common.init import xavier_uniform_initialization
import torch.nn.functional as F
from typing import Tuple, List, Dict
from collections import defaultdict
import random


class MMKG(GeneralRecommender):
    def __init__(self, config, dataset):
        super(MMKG, self).__init__(config, dataset)
        self.dataset = dataset
        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(
            form='coo').astype(np.float32)

        # load parameters info
        self.embed_dim = config['embedding_size']
        self.origin_embed_dim = self.embed_dim
        self.n_layers = config['n_layers']
        self.reg_weight = config['reg_weight']
        self.config = config
        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        # lightgcn
        self.embedding_dict = self._init_model()
        self.norm_adj_matrix = self.get_norm_adj_mat().to(self.device)
        
        self.modal_num = self.count_modal_num()

        self.item_n_layers = config['item_n_layers']
        self.init_item_graph()

        #### MMKG ####
        self.init_mmkg()
        N = int(self.dataset.n_entity)
        self.kg_adj_norm = self._build_norm_adj_from_edge_index(
            self.kg_edge_index, N, add_self_loops=True
        ).to(self.device)
        
    def count_modal_num(self):
        modal_num = 0
        for modal in ['image', 'text', 'review', 'llm']:
            if self.config[f'use_{modal}']:
                modal_num += 1
        return modal_num

    def init_mmkg(self):
        config = self.config
        self.embed_dim *= 2

        self.item_emb = nn.Embedding(self.n_items, self.embed_dim)

        self.kg_edge_index = torch.stack([self.dataset.triplet[0], self.dataset.triplet[2]], dim=0).to(self.device)
        self.kg_edge_type  = self.dataset.triplet[1]
        self.n_rel = self.dataset.n_rel
        self.kg_layer_n = self.config['kg_layer_n']

        d = self.embed_dim // 2
        self.rel_emb = nn.Embedding(self.n_rel, d)
        self.kg_gamma = 12.0
        self.kg_epsilon = 2.0
        self.kg_embedding_range = (self.kg_gamma + self.kg_epsilon) / d
        self.kg_norm_range = (self.origin_embed_dim ** 0.5) * (self.kg_embedding_range / (3.0 ** 0.5))
        # item
        nn.init.uniform_(self.item_emb.weight,
                        a=-self.kg_embedding_range, b=+self.kg_embedding_range)
        # rel
        nn.init.uniform_(self.rel_emb.weight,
                            a=-self.kg_embedding_range, b=+self.kg_embedding_range)
        self._init_pretrained_entities_default()

    def _init_pretrained_entities_default(self):
        if self.config['use_image']:
            self.image_emb = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.v_proj_layer = nn.Linear(self.v_feat.shape[1], self.embed_dim)
        if self.config['use_text']:
            self.text_emb = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.t_proj_layer = nn.Linear(self.t_feat.shape[1], self.embed_dim)
        if self.config['use_review']:
            self.review_emb = nn.Embedding.from_pretrained(self.r_feat, freeze=False)
            self.r_proj_layer = nn.Linear(self.r_feat.shape[1], self.embed_dim)
        if self.config['use_llm']:
            self.llm_emb = nn.Embedding.from_pretrained(self.l_feat, freeze=False)
            self.l_proj_layer = nn.Linear(self.l_feat.shape[1], self.embed_dim)
        
    #### MMKG ###
    def entity_projection(self):
        modal_feat = []
        if self.config['use_image']:
            proj_feat = self.v_proj_layer(self.image_emb.weight)
            norm = proj_feat.norm(dim=1, keepdim=True).clamp_min(1e-12)
            z_unit = proj_feat / norm
            z_scaled = z_unit * self.kg_norm_range
            modal_feat.append(z_scaled)
        if self.config['use_text']:
            proj_feat = self.t_proj_layer(self.text_emb.weight)
            norm = proj_feat.norm(dim=1, keepdim=True).clamp_min(1e-12)
            z_unit = proj_feat / norm
            z_scaled = z_unit * self.kg_norm_range
            modal_feat.append(z_scaled)
        if self.config['use_review']:
            proj_feat = self.r_proj_layer(self.review_emb.weight)
            norm = proj_feat.norm(dim=1, keepdim=True).clamp_min(1e-12)
            z_unit = proj_feat / norm
            z_scaled = z_unit * self.kg_norm_range
            modal_feat.append(z_scaled)
        if self.config['use_llm']:
            proj_feat = self.l_proj_layer(self.llm_emb.weight)
            norm = proj_feat.norm(dim=1, keepdim=True).clamp_min(1e-12)
            z_unit = proj_feat / norm
            z_scaled = z_unit * self.kg_norm_range
            modal_feat.append(z_scaled)
        modal_pre_emb = torch.cat(modal_feat, dim=0)
        kg_embed = torch.cat([self.item_emb.weight, modal_pre_emb], dim=0)
        return kg_embed

    def get_gcn_embedding(self):
        return self.kg_lightgcn_forward()
        
    def kg_lightgcn_forward(self):
        x = self.entity_projection()  # [N_entity_total, d]
        embs = [x]
        for _ in range(self.kg_layer_n):
            if self.config['use_sparse']:
                x = torch.sparse.mm(self.kg_adj_norm, x)
            else:
                x = torch.mm(self.kg_adj_norm.to_dense(), x)
            embs.append(x)
        x_stack = torch.stack(embs, dim=1)   # [N, L+1, d]
        x_mean  = torch.mean(x_stack, dim=1) # [N, d]
        return x_mean
    #### MMKG ####

    @torch.no_grad()
    def _build_norm_adj_from_edge_index(self, edge_index: torch.Tensor, num_nodes: int, add_self_loops: bool = True):
        device = edge_index.device
        src, dst = edge_index[0], edge_index[1]

        if add_self_loops:
            loop = torch.arange(num_nodes, device=device, dtype=src.dtype)
            src = torch.cat([src, loop], dim=0)
            dst = torch.cat([dst, loop], dim=0)

        idx = torch.stack([dst, src], 0)
        val = torch.ones(idx.size(1), device=device, dtype=torch.float32)  
        A = torch.sparse_coo_tensor(idx, val, (num_nodes, num_nodes)).coalesce()

        idx = A.indices()
        val = A.values()

        deg = torch.sparse.sum(A, dim=1).to_dense().to(torch.float32)     
        inv_sqrt = torch.pow(deg.clamp_min(1e-12), -0.5)
        scale = inv_sqrt[idx[0]] * inv_sqrt[idx[1]]                     

        A_hat = torch.sparse_coo_tensor(idx, val * scale, A.size()).coalesce()
        A_hat = self._sorted_coalesce(A_hat)
        return A_hat

    def _sorted_coalesce(self, sp):
        sp = sp.coalesce()
        idx = sp.indices()        
        val = sp.values()
        order = torch.argsort(idx[0] * sp.size(1) + idx[1])
        return torch.sparse_coo_tensor(idx[:, order], val[order], sp.size()).coalesce()

    # LightGCN
    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.n_users, self.embed_dim))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.n_items, self.embed_dim)))
        })
        return embedding_dict
    
    def get_norm_adj_mat(self):
        r"""Get the normalized interaction matrix of users and items.

        Construct the square matrix from the training data and normalize it
        using the laplace-like normalization (symmetric).

        .. math::
            \hat{A} = D^{-0.5} A D^{-0.5}

        Returns:
            torch.sparse.FloatTensor: normalized adjacency (sparse COO)
        """
        n_users, n_items = self.n_users, self.n_items
        n = n_users + n_items
        inter = self.interaction_matrix.tocoo().astype(np.float32)

        rows = np.concatenate([inter.row, inter.col + n_users]).astype(np.int64)
        cols = np.concatenate([inter.col + n_users, inter.row]).astype(np.int64)
        data = np.ones(rows.shape[0], dtype=np.float32)

        A = sp.coo_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float32).tocsr()

        deg = np.array(A.sum(1)).ravel()
        d_inv_sqrt = np.power(deg + 1e-7, -0.5, where=(deg > 0))
        d_inv_sqrt[deg == 0] = 0.0
        D_inv_sqrt = sp.diags(d_inv_sqrt, dtype=np.float32)
        L = D_inv_sqrt @ A @ D_inv_sqrt
        L = L.tocoo()
        idx = np.vstack([L.row, L.col]).astype(np.int64)
        indices = torch.from_numpy(idx)
        values = torch.from_numpy(L.data.astype(np.float32))
        SparseL = torch.sparse_coo_tensor(indices, values, torch.Size(L.shape)) #.coalesce()
        SparseL = self._sorted_coalesce(SparseL)
        return SparseL
    
    def get_ego_embeddings(self):
        r"""Get the embedding of users and items and combine to an embedding matrix.

        Returns:
            Tensor of the embedding matrix. Shape of [n_items+n_users, embedding_dim]
        """
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        return ego_embeddings

    def lightgcn_forward(self):
        all_embeddings = self.get_ego_embeddings()
        
        embeddings_list = [all_embeddings]
        for layer_idx in range(self.n_layers):
            if self.config['use_sparse']:
                all_embeddings = torch.sparse.mm(self.norm_adj_matrix, all_embeddings)
            else:
                all_embeddings = torch.mm(self.norm_adj_matrix.to_dense(), all_embeddings)
            embeddings_list.append(all_embeddings)
        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)

        user_all_embeddings = lightgcn_all_embeddings[:self.n_users, :]
        item_all_embeddings = lightgcn_all_embeddings[self.n_users:, :]

        return user_all_embeddings, item_all_embeddings

    def init_item_graph(self):
        # config = self.config
        self.knn_k = self.config['knn_k']
        self.mm_image_weight = self.config['mm_image_weight']
        dataset_path = os.path.abspath(self.config['data_path'] + self.config['dataset'])
        v_feat_file = self.config['vision_feature_file'].split('.')[0] 
        t_feat_file = self.config['text_feature_file'].split('.')[0]
        mm_adj_file = os.path.join(dataset_path, 'unified_mm_adj_{}_{}_{}_{}.pt'.format(v_feat_file, t_feat_file, self.knn_k, int(10*self.mm_image_weight)))
        if os.path.exists(mm_adj_file):
            self.mm_adj = torch.load(mm_adj_file)
            self.mm_adj = self.mm_adj.to(self.device)
        else:
            print('new mm_adj matrix is building')
            if self.v_feat is not None:
                indices, image_adj = self.get_knn_adj_mat(self.v_feat)
                self.mm_adj = image_adj.coalesce().to(torch.float32)
            if self.t_feat is not None:
                indices, text_adj = self.get_knn_adj_mat(self.t_feat)
                self.mm_adj = text_adj.coalesce().to(torch.float32)
            if self.v_feat is not None and self.t_feat is not None:
                self.mm_adj = self.mm_image_weight * image_adj + (1.0 - self.mm_image_weight) * text_adj
                self.mm_adj = self.mm_adj.coalesce().to(torch.float32)
                del text_adj
                del image_adj
                if not self.config['adj_new_make']:
                    torch.save(self.mm_adj, mm_adj_file)
        self.mm_adj = self.mm_adj.to(self.device)
        
    def get_knn_adj_mat(self, mm_embeddings):
        context_norm = mm_embeddings.div(torch.norm(mm_embeddings, p=2, dim=-1, keepdim=True))
        sim = torch.mm(context_norm, context_norm.transpose(1, 0))
        _, knn_ind = torch.topk(sim, self.knn_k, dim=-1)
        adj_size = sim.size()
        del sim
        indices0 = torch.arange(knn_ind.shape[0]).to(self.device)
        indices0 = torch.unsqueeze(indices0, 1)
        indices0 = indices0.expand(-1, self.knn_k)
        indices = torch.stack((torch.flatten(indices0), torch.flatten(knn_ind)), 0)
        return indices, self.compute_normalized_laplacian(indices, adj_size)

    def compute_normalized_laplacian(self, indices, adj_size):
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        cols_inv_sqrt = r_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return torch.sparse.FloatTensor(indices, values, adj_size)
    
    def item_forward(self, item_emb):
        h = item_emb
        for i in range(self.item_n_layers):
            if self.config['use_sparse']:
                h = torch.sparse.mm(self.mm_adj, h)    
            else:
                h = torch.mm(self.mm_adj.to_dense(), h)
        return h

    ####main method#### 
    def score(self, h, r, t):
        pi = 3.141592653589793
        h_re, h_im = torch.chunk(h, 2, dim=1)
        phase = r / (self.kg_embedding_range / pi)
        re_r, im_r = torch.cos(phase), torch.sin(phase)
        t_re, t_im = torch.chunk(t, 2, dim=1)
        re = h_re*re_r - h_im*im_r - t_re
        im = h_re*im_r + h_im*re_r - t_im
        d  = torch.sqrt(re.pow(2) + im.pow(2) + 1e-9).sum(dim=1)
        return self.kg_gamma - d                                                                           
    
    def calc_kg_loss(self, interaction):
        h_ids = interaction[0]                
        r_ids = interaction[1]                
        t_ids = interaction[2]                
        neg_t_ids = interaction[3:]           
        node_emb = self.get_gcn_embedding()   
        h = node_emb[h_ids]                   
        r = self.rel_emb.weight[r_ids]        
        t = node_emb[t_ids]                   

        pos = self.score(h, r, t)
        K, B = neg_t_ids.shape
        neg_t = node_emb[neg_t_ids.view(-1)]             
        h_rep = h.unsqueeze(0).expand(K, B, -1).reshape(K*B, -1)
        r_rep = r.unsqueeze(0).expand(K, B, -1).reshape(K*B, -1)
        neg = self.score(h_rep, r_rep, neg_t).reshape(K, B).T
        margin = 1.0
        loss = F.relu(margin + neg - pos.unsqueeze(1)).mean()
        reg = self.reg_loss(h, r, t)
        return loss + self.config['kg_reg_weight'] * reg.squeeze()

    def final_forward(self):
        mmkg_item_embeddings = self.item_emb.weight[:, :self.embed_dim//2]
        id_user_embeddings, id_item_embeddings = self.lightgcn_forward()
        final_user_embeddings = id_user_embeddings
        item_emb = mmkg_item_embeddings + id_item_embeddings
        mm_item_embeddings = self.item_forward(item_emb)
        final_item_embeddings = id_item_embeddings + mm_item_embeddings     
        return final_user_embeddings, final_item_embeddings
    
    def calc_cf_loss(self, interaction):
        user = interaction[0]
        pos_item = interaction[1]
        neg_item = interaction[2]
        user_embeddings, item_embeddings = self.final_forward()
        u_embeddings = user_embeddings[user, :]
        posi_embeddings = item_embeddings[pos_item, :]
        negi_embeddings = item_embeddings[neg_item, :]
                
        # calculate BPR Loss
        pos_scores = torch.mul(u_embeddings, posi_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, negi_embeddings).sum(dim=1)
        mf_loss = self.mf_loss(pos_scores, neg_scores)

        u_ego_embeddings = self.embedding_dict['user_emb'][user, :]
        posi_ego_embeddings = self.embedding_dict['item_emb'][pos_item, :]
        negi_ego_embeddings = self.embedding_dict['item_emb'][neg_item, :]
        mmkg_item_pos, mmkg_item_neg = self.item_emb.weight[pos_item, :self.origin_embed_dim], self.item_emb.weight[neg_item, :self.origin_embed_dim]
        reg_emb_lst = [u_ego_embeddings, posi_ego_embeddings, negi_ego_embeddings, mmkg_item_pos, mmkg_item_neg]
        reg_loss = self.reg_loss(*reg_emb_lst)
        loss = mf_loss + self.reg_weight * reg_loss
        return loss
    
    def full_sort_predict(self, interaction):
        user = interaction[0]
        restore_user_e, restore_item_e = self.final_forward()
        u_embeddings = restore_user_e[user, :]
        scores = torch.matmul(u_embeddings, restore_item_e.transpose(0, 1))
        return scores 
        
    def calculate_loss(self, interaction):
        kg_loss = self.calc_kg_loss(interaction)
        cf_loss = self.calc_cf_loss(interaction)
        return kg_loss + cf_loss