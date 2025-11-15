# coding: utf-8
# @email: enoche.chow@gmail.com
"""
Wrap dataset into dataloader
################################################
"""
import math
import torch
import random
import numpy as np
from logging import getLogger
from scipy.sparse import coo_matrix
import collections 
import os
import pandas as pd
import pickle

class AbstractDataLoader(object):
    """:class:`AbstractDataLoader` is an abstract object which would return a batch of data which is loaded by
    :class:`~recbole.data.interaction.Interaction` when it is iterated.
    And it is also the ancestor of all other dataloader.

    Args:
        config (Config): The config of dataloader.
        dataset (Dataset): The dataset of dataloader.
        batch_size (int, optional): The batch_size of dataloader. Defaults to ``1``.
        dl_format (InputType, optional): The input type of dataloader. Defaults to
            :obj:`~recbole.utils.enum_type.InputType.POINTWISE`.
        shuffle (bool, optional): Whether the dataloader will be shuffle after a round. Defaults to ``False``.

    Attributes:
        dataset (Dataset): The dataset of this dataloader.
        shuffle (bool): If ``True``, dataloader will shuffle before every epoch.
        real_time (bool): If ``True``, dataloader will do data pre-processing,
            such as neg-sampling and data-augmentation.
        pr (int): Pointer of dataloader.
        step (int): The increment of :attr:`pr` for each batch.
        batch_size (int): The max interaction number for all batch.
    """
    def __init__(self, config, dataset, additional_dataset=None,
                 batch_size=1, neg_sampling=False, shuffle=False):
        self.config = config
        self.logger = getLogger()
        self.dataset = dataset
        self.dataset_bk = self.dataset.copy(self.dataset.df)
        # if config['model_type'] == ModelType.GENERAL:
        #     self.dataset.df.drop(self.dataset.ts_id, inplace=True, axis=1)
        # elif config['model_type'] == ModelType.SEQUENTIAL:
        #     # sort instances
        #     pass
        self.additional_dataset = additional_dataset
        self.batch_size = batch_size
        self.step = batch_size
        self.shuffle = shuffle
        self.neg_sampling = neg_sampling
        self.device = config['device']   

        self.sparsity = 1 - self.dataset.inter_num / self.dataset.user_num / self.dataset.item_num
        self.pr = 0
        self.inter_pr = 0
        
    def pretrain_setup(self):
        """This function can be used to deal with some problems after essential args are initialized,
        such as the batch-size-adaptation when neg-sampling is needed, and so on. By default, it will do nothing.
        """
        pass

    def data_preprocess(self):
        """This function is used to do some data preprocess, such as pre-neg-sampling and pre-data-augmentation.
        By default, it will do nothing.
        """
        pass

    def __len__(self):
        return math.ceil(self.pr_end / self.step)

    def __iter__(self):
        if self.shuffle:
            self._shuffle()
        return self

    def __next__(self):
        if self.pr >= self.pr_end:
            self.pr = 0
            self.inter_pr = 0
            raise StopIteration()
        return self._next_batch_data()

    @property
    def pr_end(self):
        """This property marks the end of dataloader.pr which is used in :meth:`__next__()`."""
        raise NotImplementedError('Method [pr_end] should be implemented')

    def _shuffle(self):
        """Shuffle the order of data, and it will be called by :meth:`__iter__()` if self.shuffle is True.
        """
        raise NotImplementedError('Method [shuffle] should be implemented.')

    def _next_batch_data(self):
        """Assemble next batch of data in form of Interaction, and return these data.

        Returns:
            Interaction: The next batch of data.
        """
        raise NotImplementedError('Method [next_batch_data] should be implemented.')


class TrainDataLoader(AbstractDataLoader):
    """
    General dataloader with negative sampling.
    """
    def __init__(self, config, dataset, batch_size=1, shuffle=False, batch_size_kg=1024):
        super().__init__(config, dataset, additional_dataset=None,
                         batch_size=batch_size, neg_sampling=True, shuffle=shuffle)

        # special for training dataloader
        self.history_items_per_u = dict()
        # full items in training.
        self.all_items = self.dataset.df[self.dataset.iid_field].unique().tolist()
        self.all_uids = self.dataset.df[self.dataset.uid_field].unique()
        self.all_items_set = set(self.all_items)
        self.all_users_set = set(self.all_uids)
        self.all_item_len = len(self.all_items)
        # if full sampling
        self.use_full_sampling = config['use_full_sampling']
        if config['use_kg']:
            self.triplet, self.n_entity, self.ht_dict, self.hrt_dict, self.rel_dict = self.construct_kgdata(mode=self.config['kg_structure'])
            self.logger.info('=================================')
            self.logger.info(f'n_entity: {self.n_entity}, n_rels: {self.triplet[1].max().item()+1}, n_edges:{len(self.triplet[0])}')
            self.n_rel = len(self.rel_dict)

        if config['use_neg_sampling']:
            if self.use_full_sampling:
                self.sample_func = self._get_full_uids_sample
            else:
                self.sample_func = self._get_neg_sample

        else:
            self.sample_func = self._get_non_neg_sample

        self._get_history_items_u()
        self.neighborhood_loss_required = config['use_neighborhood_loss']
        if self.neighborhood_loss_required:
            self.history_users_per_i = {}
            self._get_history_users_i()
            self.user_user_dict = self._get_my_neighbors(self.config['USER_ID_FIELD'])
            self.item_item_dict = self._get_my_neighbors(self.config['ITEM_ID_FIELD'])
        
        self._phase = 'cf'  
        self.batch_size_kg = batch_size_kg

    def pretrain_setup(self):
        """
        Reset dataloader. Outputing the same positive & negative samples with each training.
        :return:
        """
        # sort & random
        if self.shuffle:
            self.dataset = self.dataset_bk.copy(self.dataset_bk.df)
        self.all_items.sort()
        if self.use_full_sampling:
            self.all_uids.sort()
        random.shuffle(self.all_items)
        # reorder dataset as default (chronological order)
        #self.dataset.sort_by_chronological()

    def inter_matrix(self, form='coo', value_field=None):
        """Get sparse matrix that describe interactions between user_id and item_id.

        Sparse matrix has shape (user_num, item_num).

        For a row of <src, tgt>, ``matrix[src, tgt] = 1`` if ``value_field`` is ``None``,
        else ``matrix[src, tgt] = self.inter_feat[src, tgt]``.

        Args:
            form (str, optional): Sparse matrix format. Defaults to ``coo``.
            value_field (str, optional): Data of sparse matrix, which should exist in ``df_feat``.
                Defaults to ``None``.

        Returns:
            scipy.sparse: Sparse matrix in form ``coo`` or ``csr``.
        """
        if not self.dataset.uid_field or not self.dataset.iid_field:
            raise ValueError('dataset doesn\'t exist uid/iid, thus can not converted to sparse matrix')
        return self._create_sparse_matrix(self.dataset.df, self.dataset.uid_field,
                                          self.dataset.iid_field, form, value_field)

    def _create_sparse_matrix(self, df_feat, source_field, target_field, form='coo', value_field=None):
        """Get sparse matrix that describe relations between two fields.

        Source and target should be token-like fields.

        Sparse matrix has shape (``self.num(source_field)``, ``self.num(target_field)``).

        For a row of <src, tgt>, ``matrix[src, tgt] = 1`` if ``value_field`` is ``None``,
        else ``matrix[src, tgt] = df_feat[value_field][src, tgt]``.

        Args:
            df_feat (pandas.DataFrame): Feature where src and tgt exist.
            form (str, optional): Sparse matrix format. Defaults to ``coo``.
            value_field (str, optional): Data of sparse matrix, which should exist in ``df_feat``.
                Defaults to ``None``.
 

        Returns:
            scipy.sparse: Sparse matrix in form ``coo`` or ``csr``.
        """
        src = df_feat[source_field].values
        tgt = df_feat[target_field].values
        if value_field is None:
            data = np.ones(len(df_feat))
        else:
            if value_field not in df_feat.columns:
                raise ValueError('value_field [{}] should be one of `df_feat`\'s features.'.format(value_field))
            data = df_feat[value_field].values
        mat = coo_matrix((data, (src, tgt)), shape=(self.dataset.user_num, self.dataset.item_num))

        if form == 'coo':
            return mat
        elif form == 'csr':
            return mat.tocsr()
        else:
            raise NotImplementedError('sparse matrix format [{}] has not been implemented.'.format(form))

    def set_phase(self, phase):
        assert phase in ('cf', 'kg')
        self._phase = phase
        if self._phase == 'cf':
            self.step = int(self.batch_size)
        elif self._phase == 'kg':
            self.step = int(self.batch_size_kg)

    @property
    def pr_end(self):
        if self._phase == 'cf':
            return len(self.all_uids) if self.use_full_sampling else len(self.dataset)
        elif self._phase == 'kg':
            return self.triplet.size(1)

    def _shuffle(self):
        self.dataset.shuffle()
        if self.use_full_sampling:
            np.random.shuffle(self.all_uids)
        
        if self.config['use_kg']:
            perm = torch.randperm(self.triplet.size(1))
            self.triplet = self.triplet[:, perm]

    def _next_batch_data(self):
        if self._phase == 'cf':
            batch = self._get_neg_sample()
            return batch
        elif self._phase == 'kg':
            batch = self._get_neg_sample_kg()
            return batch
        else:
            raise NotImplementedError
        
    def _get_neg_sample(self): 
        cur_data = self.dataset[self.pr: self.pr + self.step]
        self.pr += self.step
        # to tensor
        user_tensor = torch.tensor(cur_data[self.config['USER_ID_FIELD']].values).type(torch.LongTensor).to(self.device)
        item_tensor = torch.tensor(cur_data[self.config['ITEM_ID_FIELD']].values).type(torch.LongTensor).to(self.device)
        batch_tensor = torch.cat((torch.unsqueeze(user_tensor, 0),
                                  torch.unsqueeze(item_tensor, 0)))
        u_ids = cur_data[self.config['USER_ID_FIELD']]
        # sampling negative items only in the dataset (train)
        neg_ids = self._sample_neg_ids(u_ids).to(self.device)
        # for neighborhood loss
        if self.neighborhood_loss_required:
            i_ids = cur_data[self.config['ITEM_ID_FIELD']]
            pos_neighbors, neg_neighbors = self._get_neighborhood_samples(i_ids, self.config['ITEM_ID_FIELD'])
            pos_neighbors, neg_neighbors = pos_neighbors.to(self.device), neg_neighbors.to(self.device)

            batch_tensor = torch.cat((batch_tensor, neg_ids.unsqueeze(0),
                                      pos_neighbors.unsqueeze(0), neg_neighbors.unsqueeze(0)))
        # merge negative samples
        else:
            batch_tensor = torch.cat((batch_tensor, neg_ids.unsqueeze(0)))
        return batch_tensor
    
    def _get_neg_sample_kg(self):
        cur_data = self.triplet[:, self.pr: self.pr + self.step].to(self.device)
        self.pr += self.step
        h_ids = cur_data[0].tolist() 
        # sampling negative items only in the dataset (train)
        neg_ids = []
        for h in h_ids:
            # random 1 item
            pos_tail = self.ht_dict[h]
            iid = random.randint(0, self.n_entity-1)
            while iid in pos_tail:
                iid = random.randint(0, self.n_entity-1)
            neg_ids.append(iid)
        neg_ids = torch.LongTensor(neg_ids).to(self.device)
        batch_tensor = torch.cat((cur_data, neg_ids.unsqueeze(0)))
        return batch_tensor
    
    def _get_non_neg_sample(self):
        cur_data = self.dataset[self.pr: self.pr + self.step]
        self.pr += self.step
        # to tensor
        user_tensor = torch.tensor(cur_data[self.config['USER_ID_FIELD']].values).type(torch.LongTensor).to(self.device)
        item_tensor = torch.tensor(cur_data[self.config['ITEM_ID_FIELD']].values).type(torch.LongTensor).to(self.device)
        batch_tensor = torch.cat((torch.unsqueeze(user_tensor, 0),
                                  torch.unsqueeze(item_tensor, 0)))
        return batch_tensor

    def _get_full_uids_sample(self):
        user_tensor = torch.tensor(self.all_uids[self.pr: self.pr + self.step]).type(torch.LongTensor).to(self.device)
        self.pr += self.step
        return user_tensor

    def _sample_neg_ids(self, u_ids):
        neg_ids = []
        for u in u_ids:
            # random 1 item
            iid = self._random()
            while iid in self.history_items_per_u[u]:
                iid = self._random()
            neg_ids.append(iid)
        return torch.tensor(neg_ids).type(torch.LongTensor)

    def _get_my_neighbors(self, id_str):
        ret_dict = {}
        a2b_dict = self.history_items_per_u if id_str == self.config['USER_ID_FIELD'] else self.history_users_per_i
        b2a_dict = self.history_users_per_i if id_str == self.config['USER_ID_FIELD'] else self.history_items_per_u
        for i, j in a2b_dict.items():
            k = set()
            for m in j:
                k |= b2a_dict.get(m, set()).copy()
            k.discard(i)                        # remove myself
            ret_dict[i] = k
        return ret_dict

    def _get_neighborhood_samples(self, ids, id_str):
        a2a_dict = self.user_user_dict if id_str == self.config['USER_ID_FIELD'] else self.item_item_dict
        all_set = self.all_users_set if id_str == self.config['USER_ID_FIELD'] else self.all_items_set
        pos_ids, neg_ids = [], []
        for i in ids:
            pos_ids_my = a2a_dict[i]
            if len(pos_ids_my) <= 0 or len(pos_ids_my)/len(all_set) > 0.8:
                pos_ids.append(0)
                neg_ids.append(0)
                continue
            pos_id = random.sample(pos_ids_my, 1)[0]
            pos_ids.append(pos_id)
            neg_id = random.sample(all_set, 1)[0]
            while neg_id in pos_ids_my:
                neg_id = random.sample(all_set, 1)[0]
            neg_ids.append(neg_id)
        return torch.tensor(pos_ids).type(torch.LongTensor), torch.tensor(neg_ids).type(torch.LongTensor)

    def _random(self):
        rd_id = random.sample(self.all_items, 1)[0]
        return rd_id

    def _get_history_items_u(self):
        uid_field = self.dataset.uid_field
        iid_field = self.dataset.iid_field
        # load avail items for all uid
        uid_freq = self.dataset.df.groupby(uid_field)[iid_field]
        for u, u_ls in uid_freq:
            self.history_items_per_u[u] = set(u_ls.values)
        return self.history_items_per_u

    def _get_history_users_i(self):
        uid_field = self.dataset.uid_field
        iid_field = self.dataset.iid_field
        # load avail items for all uid
        iid_freq = self.dataset.df.groupby(iid_field)[uid_field]
        for i, u_ls in iid_freq:
            self.history_users_per_i[i] = set(u_ls.values)
        return self.history_users_per_i
    
    # for kg data
    def construct_kgdata(self, mode):
        if self.config['dataset'] not in ['sports', 'clothing', 'toys', 'beauty']:
            self.config['use_llm'] = False
            print('automatically changed use_llm : ', self.config['use_llm'])

        n_item = self.dataset.item_num
        unq_item = list(range(0, n_item))

        modals = ['image', 'text', 'review', 'llm']
        selected_modal = []
        for m in modals:
            if self.config[f'use_{m}']:
                selected_modal.append(m)
        n_entity = 0
        n_entity += n_item
        pad_num = {'item':0}
        num_val = n_item 
        for m in selected_modal:
            pad_num[m] = num_val
            num_val += n_item
            n_entity += n_item
        
        head = []
        rel = []
        tail = []
        n_rel = 0
        rel_dict = {}

        # i-m
        for modal in selected_modal:
            h = torch.LongTensor(unq_item)
            t = torch.LongTensor(unq_item) + pad_num[modal]
            head.extend(h.tolist())
            tail.extend(t.tolist())
            rel.extend([n_rel]*len(h))
            rel_dict[f'item_{modal}'] = n_rel
            n_rel+=1
            head.extend(t.tolist())
            tail.extend(h.tolist())
            rel.extend([n_rel]*len(h))
            rel_dict[f'{modal}_item'] = n_rel
            n_rel+=1

        dataset_path = os.path.abspath(self.config['data_path'] + self.config['dataset'])
        v_feat_file = self.config['vision_feature_file'].split('.')[0]
        t_feat_file = self.config['text_feature_file'].split('.')[0]
        r_feat_file = self.config['review_feature_file'].split('.')[0]
        feat_files = {'image': v_feat_file, 'text':t_feat_file, 'review':r_feat_file}
        if self.config['llm_feature_file']:
            l_feat_file = self.config['llm_feature_file'].split('.')[0]
            feat_files['llm'] = l_feat_file
        
        knn_k = self.config['indices_k']
        for modal in selected_modal:
            feat_file_name = feat_files[modal]
            indicies_path = os.path.join(dataset_path, f'indices_{feat_file_name}_{knn_k}.pt')
            mm_edge_index = torch.load(indicies_path)
            mm_head = mm_edge_index[0] + pad_num[modal]
            mm_tail = mm_edge_index[1] + pad_num[modal]
            head.extend(mm_head.tolist())
            tail.extend(mm_tail.tolist())
            rel.extend([n_rel]*(len(mm_edge_index[0])))
            rel_dict[f'{modal}_{modal}'] = n_rel
            n_rel+=1

        triplet = torch.LongTensor([head, rel, tail])

        ht_dict = collections.defaultdict(list)
        for h, r, t in zip(head, rel, tail):
            ht_dict[h].append(t)
        
        hrt_dict = collections.defaultdict(set)
        for h, r, t in zip(head, rel, tail):
            hrt_dict[(h,r)].add(t)
        return triplet, n_entity, ht_dict, hrt_dict, rel_dict

class EvalDataLoader(AbstractDataLoader):
    """
        additional_dataset: training dataset in evaluation
    """
    def __init__(self, config, dataset, additional_dataset=None,
                 batch_size=1, shuffle=False):
        super().__init__(config, dataset, additional_dataset=additional_dataset,
                         batch_size=batch_size, neg_sampling=False, shuffle=shuffle)

        if additional_dataset is None:
            raise ValueError('Training datasets is nan')
        self.eval_items_per_u = []
        self.eval_len_list = []
        self.train_pos_len_list = []

        self.eval_u = self.dataset.df[self.dataset.uid_field].unique()
        # special for eval dataloader
        self.pos_items_per_u = self._get_pos_items_per_u(self.eval_u).to(self.device)
        self._get_eval_items_per_u(self.eval_u)
        # to device
        self.eval_u = torch.tensor(self.eval_u).type(torch.LongTensor).to(self.device)

    @property
    def pr_end(self):
        return self.eval_u.shape[0]

    def _shuffle(self):
        self.dataset.shuffle()

    def _next_batch_data(self):
        inter_cnt = sum(self.train_pos_len_list[self.pr: self.pr+self.step])
        batch_users = self.eval_u[self.pr: self.pr + self.step]
        batch_mask_matrix = self.pos_items_per_u[:, self.inter_pr: self.inter_pr+inter_cnt].clone()
        # user_ids to index
        batch_mask_matrix[0] -= self.pr
        self.inter_pr += inter_cnt
        self.pr += self.step

        return [batch_users, batch_mask_matrix]

    def _get_pos_items_per_u(self, eval_users):
        """
        history items in training dataset.
        masking out positive items in evaluation
        :return:
        user_id - item_ids matrix
        [[0, 0, ... , 1, ...],
         [0, 1, ... , 0, ...]]
        """
        uid_field = self.additional_dataset.uid_field
        iid_field = self.additional_dataset.iid_field
        # load avail items for all uid
        uid_freq = self.additional_dataset.df.groupby(uid_field)[iid_field]
        u_ids = []
        i_ids = []
        for i, u in enumerate(eval_users):
            u_ls = uid_freq.get_group(u).values
            i_len = len(u_ls)
            self.train_pos_len_list.append(i_len)
            u_ids.extend([i]*i_len)
            i_ids.extend(u_ls)
        return torch.tensor([u_ids, i_ids]).type(torch.LongTensor)

    def _get_eval_items_per_u(self, eval_users):
        """
        get evaluated items for each u
        :return:
        """
        uid_field = self.dataset.uid_field
        iid_field = self.dataset.iid_field
        # load avail items for all uid
        uid_freq = self.dataset.df.groupby(uid_field)[iid_field]
        for u in eval_users:
            u_ls = uid_freq.get_group(u).values
            self.eval_len_list.append(len(u_ls))
            self.eval_items_per_u.append(u_ls)
        self.eval_len_list = np.asarray(self.eval_len_list)

    # return pos_items for each u
    def get_eval_items(self):
        return self.eval_items_per_u

    def get_eval_len_list(self):
        return self.eval_len_list

    def get_eval_users(self):
        return self.eval_u.cpu()


