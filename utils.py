import scipy.sparse as sp
import numpy as np
import torch
import random
from data_prepare import load_data


def input_data(dirname, dev_str, emb_method, self_loop):
    A, graph_info, pre_vec_dict, train_data, test_data, p_ui_dic, all_adjs\
        = load_data(dirname, dev_str, emb_method, self_loop)
    return A, graph_info, pre_vec_dict, train_data, test_data, p_ui_dic, all_adjs


def getEmb(b_list, the_model, item_shift, dev):
    u_array, i_array, g_array = np.array(b_list).transpose()
    i_array += item_shift
    u_long, i_long, g_truth = torch.LongTensor(u_array).to(dev), torch.LongTensor(i_array).to(dev), \
                              torch.LongTensor(g_array).to(dev)

    u_emb = the_model.heteroGCN.lookup_emb(u_long)
    i_emb = the_model.heteroGCN.lookup_emb(i_long)

    return u_emb, i_emb, g_truth


# Random matching user negative sampling
def sample_neg(batch_pos_list, p_ui_dic, item_num, sample_num=1):
    batch_neg_list = []
    for (u_id, _, _) in batch_pos_list:
        neg_count = 0
        while True:
            if neg_count == sample_num:
                break
            n_iid = random.randint(0, item_num - 1)
            if n_iid not in p_ui_dic[u_id] and (u_id, n_iid, 0) not in batch_neg_list:
                batch_neg_list.append((u_id, n_iid, 0))
                neg_count += 1
    return batch_neg_list


def csr2tensor(X, device):
    coo = X.tocoo().astype(np.float32)
    row = torch.Tensor(coo.row).long()
    col = torch.Tensor(coo.col).long()
    index = torch.stack([row, col])
    data = torch.FloatTensor(coo.data)
    return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape)).to(device)


def row_normalize(A_mat):
    A_list = list()
    # print(len(A_mat))
    for i in range(len(A_mat)):
        d = np.array(A_mat[i].sum(1)).flatten()
        d_inv = 1. / (d + 1e-10)
        d_inv[np.isinf(d_inv)] = 0.
        D_inv = sp.diags(d_inv)
        norm_A = D_inv.dot(A_mat[i]).tocsr()
        A_list.append(norm_A)
    return A_list


# hit_rate nDCG precision recall -- pre-user
def result_evaluate(user_id: int, top_k_list: list, the_model, test_dict, item_shift, device):
    one_hit, one_ndcg = [], []
    h_test_items = test_dict[str(user_id) + '_p'].copy()
    test_candidate_items = h_test_items + test_dict[str(user_id) + '_n'].copy()
    # print('True Percent;', len(h_test_items) / len(test_candidate_items))
    random.shuffle(test_candidate_items)
    # Calculate first and then select Top-k
    c_score_list = list()

    u_array, i_array, _ = np.array(test_candidate_items).transpose()
    i_array += item_shift
    u_long, i_long = torch.LongTensor(u_array).to(device), torch.LongTensor(i_array).to(device)
    te_user_emb = the_model.heteroGCN.lookup_emb(u_long)
    te_item_emb = the_model.heteroGCN.lookup_emb(i_long)

    test_scores = the_model.predict(te_user_emb, te_item_emb, dropout=0).squeeze().cpu().detach().numpy().tolist()
    t_i = 0
    for (_, t_iid, _) in test_candidate_items:
        c_score_list.append([t_iid, test_scores[t_i]])
        t_i += 1
    recommend_list = []
    for ii in range(top_k_list[len(top_k_list) - 1]):
        r_item = -1
        max_score = -np.inf
        for c_score in c_score_list:
            c_item = c_score[0]
            score = c_score[1]
            if score > max_score and c_item not in recommend_list:
                max_score = score
                r_item = c_item
        recommend_list.append(r_item)
    # print(test_item)
    # print(test_candidate_items)
    for top_k in top_k_list:
        hit_count = 0
        hit_list = []
        dcg = 0
        idcg = 0
        for k in range(len(recommend_list[:top_k])):
            t_item = recommend_list[k]
            if (user_id, t_item, 1) in h_test_items:
                hit_count += 1
                dcg += 1 / np.log(k + 2)
                hit_list.append(1)
            else:
                hit_list.append(0)
        hit_list.sort(reverse=True)
        # print(hit_list)
        kk = 0
        for imp_rating in hit_list:
            idcg += imp_rating / np.log(kk + 2)
            kk += 1
        if hit_count > 0:
            one_hit.append(1)
            one_ndcg.append(dcg / idcg)
        else:
            one_hit.append(0)
            one_ndcg.append(0)
    return one_hit, one_ndcg
