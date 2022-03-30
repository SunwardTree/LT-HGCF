# To do with amazon data
# http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/
import os
import numpy as np
import random
import math
import scipy.sparse as sp
from tqdm import tqdm
from sentence_transformers import models, SentenceTransformer


def clean_data(texts: str):
    # lowdown cased
    texts = texts.lower()
    texts = texts.replace('(', ' ').replace(')', ' ').replace('[', ' ').replace(']', ' ').replace('*', ' ') \
        .replace('-', '').replace('......', ' ').replace('...', ' ').replace('?', ' ').replace(':', ' ') \
        .replace(';', ' ').replace(',', ' ').replace('.', ' ').replace('!', ' ').replace("/", ' ') \
        .replace('" ', ' ').replace("' ", ' ').replace(' "', ' ').replace(" '", ' ').replace("=", ' ') \
        .replace('  ', ' ').replace('  ', ' ')
    return texts


def restore_data(d_name: str):
    old_user2new, old_item2new = {}, {}
    pos_uir_list = []
    p_ui_dic = {}
    ui_pair = []
    ui_r_dict = {}
    reviews_dict = {}

    current_u_index = 0
    current_i_index = 0
    current_r_index = 0
    with open('./Data/' + d_name + '/' + d_name + '_5.json', 'r', encoding="utf-8") as f:  # reviews
        while True:
            temp_str = f.readline()
            if temp_str:
                temp_dict = eval(temp_str)
                o_user = temp_dict['reviewerID']
                o_item = temp_dict['asin']

                try:
                    n_user = old_user2new[o_user]
                    # print(n_user)
                except KeyError:
                    n_user = current_u_index
                    old_user2new[o_user] = current_u_index
                    p_ui_dic[n_user] = []
                    current_u_index += 1
                try:
                    n_item = old_item2new[o_item]
                    # print(n_item)
                except KeyError:
                    n_item = current_i_index
                    old_item2new[o_item] = current_i_index
                    current_i_index += 1

                # times = int(temp_dict['unixReviewTime'])
                pos_uir_list.append((n_user, n_item, 1))
                p_ui_dic[n_user].append(n_item)
                ui_pair.append((n_user, n_item))

                try:
                    reviews = clean_data(str(temp_dict['reviewText']))
                    reviews += clean_data(str(temp_dict['summary']))
                except KeyError:
                    try:
                        reviews = clean_data(str(temp_dict['summary']))
                    except KeyError:
                        continue
                reviews_dict[current_r_index] = reviews
                ui_r_dict[(n_user, n_item)] = current_r_index
                current_r_index += 1
            else:
                break
    print('valid_user:', current_u_index)
    print('valid_item:', current_i_index)
    print('valid_review:', current_r_index)
    return old_user2new, old_item2new, current_u_index, current_i_index, current_r_index, \
           pos_uir_list, ui_pair, ui_r_dict, reviews_dict, p_ui_dic


def get_descriptions(d_name: str, old_item2new):
    descriptions_dict = {}
    des_index = 0
    id_pair = []
    des_path = './Data/' + d_name + '/meta_' + d_name + '.json'  # descriptions
    with open(des_path, 'r', encoding="utf-8") as f:
        while True:
            temp_str = f.readline()
            if temp_str:
                temp_dict = eval(temp_str)
                t_asin = temp_dict['asin']
                try:
                    n_item = old_item2new[t_asin]
                    t_descriptions = clean_data(str(temp_dict['description']))
                    t_descriptions += clean_data(str(temp_dict['categories']))
                except KeyError:
                    try:
                        n_item = old_item2new[t_asin]
                        t_descriptions = clean_data(str(temp_dict['categories']))
                    except KeyError:
                        continue
                descriptions_dict[n_item] = t_descriptions
                id_pair.append((n_item, des_index))
                des_index += 1
            else:
                break
    print('Get ', des_index, ' ' + d_name + ' descriptions.')
    return des_index, id_pair, descriptions_dict


def get_train_test(pos_uir_list, p_ui_dic, item_num):
    train_pos = pos_uir_list.copy()
    test_uir_dict = {}
    for t_uid in p_ui_dic.keys():
        t_test_pos = []
        t_pos_list = p_ui_dic[t_uid]
        pos_num = len(t_pos_list)
        if pos_num > 1:
            test_p_num = 1
        else:
            test_p_num = 0
        train_n_num = pos_num - test_p_num
        # print(pos_num - test_p_num, train_n_num)
        if test_p_num > 0:
            t_test_pos.append((t_uid, t_pos_list[-1], 1))
            train_pos.remove((t_uid, t_pos_list[-1], 1))
            test_n_num = 99
            # print(test_p_num, test_n_num)
            t_test_neg = sample_negative(t_pos_list, t_uid, item_num, test_n_num)
            test_uir_dict[str(t_uid)+'_p'] = t_test_pos
            test_uir_dict[str(t_uid)+'_n'] = t_test_neg
        else:
            continue
    train_p_uir_dict = train_pos
    print('Get train and test.')
    return train_p_uir_dict, test_uir_dict


def sample_negative(u_pos_list, c_uid, item_num, neg_num):
    neg_triples = []
    neg_item_list = list(range(item_num))
    for pos_item in u_pos_list:
        neg_item_list.remove(pos_item)
    c_countor = 0
    while True:
        t_item_index = random.randint(0, len(neg_item_list)-1)
        neg_item = neg_item_list[t_item_index]
        # print(neg_item_list)
        # print(neg_item)
        neg_item_list.pop(t_item_index)
        neg_triples.append((c_uid, neg_item, 0))
        c_countor += 1
        if c_countor >= neg_num:
            break
    return neg_triples


def get_stop_word(stop_word_path: str):
    with open(stop_word_path) as stop_word_file:
        i_stop_words_list = (stop_word_file.read()).split()
    return i_stop_words_list


def get_glove_dict(glove_dict_path: str):
    with open(glove_dict_path, 'r', encoding="utf-8") as glove_file:
        in_glove_dict = {}
        for line in glove_file.readlines():
            t_list = line.split()
            if len(t_list) > 1:
                tt_list = []
                for number in t_list[1:]:
                    tt_list.append(float(number))
                in_glove_dict[t_list[0]] = np.array(tt_list)
    return in_glove_dict


def get_vector(record_texts_dict: dict, in_glove_dict, embedding_size, stop_word_list: list,
               record_num: int, save_path):
    record_embeddings = np.zeros((record_num, embedding_size))
    t_count = 0
    # print(item_num)
    for i in tqdm(range(record_num)):
        item_emb = np.zeros(embedding_size)
        try:
            word_str = str(record_texts_dict[i])
            word_list = word_str.split(" ")
            # print(word_list)
            t_div = 1
            for word in word_list:
                if word not in stop_word_list:
                    try:
                        word_glove_vector = in_glove_dict[word]
                        item_emb = item_emb + word_glove_vector
                    except KeyError:
                        continue
                    t_div += 1
                else:
                    continue
            # print(t_div, item_emb, item_emb / t_div)
            record_embeddings[i] = item_emb / t_div  # normalise
            t_count += 1
        except KeyError:
            continue
    if save_path != '':
        np.save(save_path, record_embeddings)
    return record_embeddings


def load_sparse_csr(filename):
    loader = np.load(filename)
    return sp.csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                         shape=loader['shape'], dtype=np.float32)


def save_sparse_csr(filename, array):
    np.savez(filename, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)


def load_data(data_name, device, emb_method='bert', self_loop=False):

    data_root = './Data/' + data_name + '/'
    if not os.path.exists(data_root): os.makedirs(data_root)
    glove_root = data_root + 'glove/'
    if not os.path.exists(glove_root): os.makedirs(glove_root)
    bert_root = data_root + 'bert/'
    if not os.path.exists(bert_root): os.makedirs(bert_root)

    # Directly read the generated
    try:
        with open(data_root + 'train_data', 'r') as train_f:
            train_data = eval(train_f.read())
        with open(data_root + 'test_data', 'r') as test_f:
            test_data = eval(test_f.read())
        with open(data_root + 'p_ui_dic', 'r') as p_ui_dic_f:
            p_ui_dic = eval(p_ui_dic_f.read())
        with open(data_root + 'old_item2new', 'r') as old_item2new_f:
            old_item2new = eval(old_item2new_f.read())
        num_item = len(old_item2new.keys())
        with open(data_root + 'old_user2new', 'r') as old_user2new_f:
            old_user2new = eval(old_user2new_f.read())
        num_user = len(old_user2new.keys())
    except Exception:
        old_user2new, old_item2new, num_user, num_item, num_com, pos_uir_list, \
        ui_pair, ui_r_dict, old_reviews_dict, p_ui_dic = restore_data(data_name)
        with open(data_root + 'p_ui_dic', 'w') as p_ui_dic_f:
            p_ui_dic_f.write(str(p_ui_dic))
        with open(data_root + 'old_item2new', 'w') as old_item2new_f:
            old_item2new_f.write(str(old_item2new))
        with open(data_root + 'old_user2new', 'w') as old_user2new_f:
            old_user2new_f.write(str(old_user2new))

        print('pos_uir_list', len(pos_uir_list))
        train_data, test_data = get_train_test(pos_uir_list, p_ui_dic, num_item)
        with open(data_root + 'train_data', 'w') as train_f:
            train_f.write(str(train_data))
        with open(data_root + 'test_data', 'w') as test_f:
            test_f.write(str(test_data))
    try:
        with open(data_root + 'graph.info', 'r') as info_f:
            g_info = eval(info_f.read())

        rel_dict = g_info['rel_dict']
        n_node = g_info['n_node']
        adj_list = list(range(len(rel_dict.keys())))
        for rel_str in rel_dict.keys():
            adj_list[rel_dict[rel_str]] = load_sparse_csr(data_root + 'adj_' + rel_str + '.npz')
        a_hat_mat = load_sparse_csr(data_root + 'a_hat_mat' + '.npz')
        all_adjs = [a_hat_mat]
        # add identity matrix (self-connections)
        if self_loop:
            adj_list.append(sp.identity(n_node).tocsr())
            rel_dict['self'] = len(rel_dict.keys())
            g_info['rel_dict'] = rel_dict
        # print(len(adj_list))
        print(g_info)

        r_vectors = np.load(data_root + emb_method + '/' + 'review_vectors.npy')
        d_vectors = np.load(data_root + emb_method + '/' + 'description_vectors.npy')
        pre_vec_dict = {'description': d_vectors, 'review': r_vectors}

        return adj_list, g_info, pre_vec_dict, train_data, test_data, p_ui_dic, all_adjs
    # Regenerate
    except Exception:
        ui_pair = []
        ui_r_dict = {}
        old_reviews_dict = {}

        current_r_index = 0
        with open('./Data/' + data_name + '/' + data_name + '_5.json', 'r', encoding="utf-8") as f:  # reviews
            while True:
                temp_str = f.readline()
                if temp_str:
                    temp_dict = eval(temp_str)
                    o_user = temp_dict['reviewerID']
                    o_item = temp_dict['asin']
                    n_user = old_user2new[o_user]
                    n_item = old_item2new[o_item]

                    # times = int(temp_dict['unixReviewTime'])
                    p_ui_dic[n_user].append(n_item)
                    ui_pair.append((n_user, n_item))

                    try:
                        reviews = clean_data(str(temp_dict['reviewText']))
                        reviews += clean_data(str(temp_dict['summary']))
                    except KeyError:
                        try:
                            reviews = clean_data(str(temp_dict['summary']))
                        except KeyError:
                            continue
                    old_reviews_dict[current_r_index] = reviews
                    ui_r_dict[(n_user, n_item)] = current_r_index
                    current_r_index += 1
                else:
                    break

        num_com = current_r_index
        for t_uid in range(num_user):
            try:
                test_tuples = test_data[str(t_uid) + '_p']
                for (t_uid, t_iid, t_r) in test_tuples:
                    # remove test
                    if t_r == 1:
                        ui_pair.remove((t_uid, t_iid))
                        old_reviews_dict.pop(ui_r_dict[(t_uid, t_iid)])
                        num_com -= 1
            except KeyError:
                continue
        # review resort
        new_rev_id = 0
        old_rev_id2new = {}
        reviews_dict = {}
        for old_rev_id in old_reviews_dict.keys():
            old_rev_id2new[old_rev_id] = new_rev_id
            reviews_dict[new_rev_id] = old_reviews_dict[old_rev_id]
            new_rev_id += 1
        del old_reviews_dict
        # print(old_rev_id2new)

        ur_pair = []
        ir_pair = []
        train_tuples = train_data
        for (t_uid, t_iid, t_r) in train_tuples:
            if t_r == 1:
                t_tuple = (t_uid, t_iid)
                # print(t_tuple, ui_r_dict[t_tuple])
                ur_pair.append((t_uid, old_rev_id2new[ui_r_dict[t_tuple]]))
                ir_pair.append((t_iid, old_rev_id2new[ui_r_dict[t_tuple]]))

        num_des, id_pair, descriptions_dict = get_descriptions(data_name, old_item2new)

        print('ui_pair:', len(ui_pair), ', ur_pair:', len(ur_pair),
              ', ir_pair:', len(ir_pair), ', id_pair:', len(id_pair))

        n_node = num_user + num_item + num_des + num_com
        adj_shape = (n_node, n_node)

        ui_edges = np.array(ui_pair)
        ur_edges = np.array(ur_pair)
        ir_edges = np.array(ir_pair)
        id_edges = np.array(id_pair)
        all_edges = {'ui': ui_edges, 'ur': ur_edges, 'ir': ir_edges, 'id': id_edges}

        rel_dict = {'id_t': 0, 'ir_t': 1, 'ur_t': 2, 'ui_t': 3, 'ui': 4, 'id': 5, 'ir': 6, 'ur': 7}
        adj_list = list(range(len(rel_dict.keys())))
        all_adj = None
        for e_key in all_edges.keys():
            edges = all_edges[e_key]
            if e_key == 'ui':
                row, col = np.transpose(edges)
                col += num_user
            elif e_key == 'ur':
                row, col = np.transpose(edges)
                col += num_user + num_item + num_des
            elif e_key == 'ir':
                row, col = np.transpose(edges)
                row += num_user
                col += num_user + num_item + num_des
            else:
                row, col = np.transpose(edges)
                row += num_user
                col += num_user + num_item

            data = np.ones(len(row))
            adj = sp.csr_matrix((data, (row, col)), shape=adj_shape)
            if all_adj is None:
                all_adj = adj
            else:
                all_adj += adj
            adj_trans = adj.transpose()
            # save
            save_sparse_csr(data_root + 'adj_' + e_key + '.npz', adj)
            save_sparse_csr(data_root + 'adj_' + e_key + '_t.npz', adj_trans)
            adj_list[rel_dict[e_key]] = adj
            adj_list[rel_dict[e_key + '_t']] = adj_trans
        all_adjs = [all_adj]
        save_sparse_csr(data_root + 'a_hat_mat' + '.npz', all_adj)
        # add identity matrix (self-connections)
        if self_loop:
            adj_list.append(sp.identity(n_node).tocsr())
            rel_dict['self'] = len(rel_dict.keys())
        # save
        graph_info = {'n_node': n_node, 'rel_dict': rel_dict, 'n_user': num_user, 'n_item': num_item,
                      'n_des': num_des, 'n_com': num_com}
        # print(graph_info)
        with open(data_root + 'graph.info', 'w') as info_f:
            info_f.write(str(graph_info))

        # glove
        print('----glove vector----')
        o_stop_word_list = get_stop_word(stop_word_path='./resource/stop_words.txt')
        emb_size = 300
        glove_dict = get_glove_dict(glove_dict_path='./resource/glove/glove.6B.' + str(emb_size) + 'd.txt')
        des_vectors = get_vector(descriptions_dict, glove_dict, emb_size, o_stop_word_list, num_des, "")
        rev_vectors = get_vector(reviews_dict, glove_dict, emb_size, o_stop_word_list, num_com, "")
        np.save(glove_root + 'description_vectors.npy', des_vectors)
        np.save(glove_root + 'review_vectors.npy', rev_vectors)
        g_pre_vec_dict = {'description': des_vectors, 'review': rev_vectors}

        # bert
        rev_vectors, des_vectors = None, None
        print('----sentence-bert vector----')
        # Sentence-BERT:
        # Sentence Embeddings using Siamese BERT-Networks https://arxiv.org/abs/1908.10084
        # https://github.com/UKPLab/sentence-transformers
        # google/bert_uncased_L-2_H-128_A-2(BERT-Tiny)
        # google/bert_uncased_L-12_H-256_A-4(BERT-Mini)
        # google/bert_uncased_L-4_H-512_A-8(BERT-Small)
        # google/bert_uncased_L-8_H-512_A-8(BERT-Medium)
        # google/bert_uncased_L-12_H-768_A-12(BERT-Base)
        word_embedding_model = models.Transformer('google/bert_uncased_L-12_H-256_A-4', max_seq_length=510)
        # Apply mean pooling to get one fixed sized sentence vector
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
        bert_model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device=device)
        one_req_num = 50
        des_list = list(descriptions_dict.values())
        req_times = int(math.ceil(len(des_list) / one_req_num))
        for ii in tqdm(range(req_times)):
            if ii == 0:
                des_vectors = bert_model.encode(des_list[ii * one_req_num: (ii + 1) * one_req_num])
            elif ii < req_times - 1:
                des_vectors = np.vstack(
                    (des_vectors, bert_model.encode(des_list[ii * one_req_num: (ii + 1) * one_req_num])))
            else:
                des_vectors = np.vstack((des_vectors, bert_model.encode(des_list[ii * one_req_num:])))
        rev_list = list(reviews_dict.values())
        req_times = int(math.ceil(len(rev_list) / one_req_num))
        for ii in tqdm(range(req_times)):
            if ii == 0:
                rev_vectors = bert_model.encode(rev_list[ii * one_req_num: (ii + 1) * one_req_num])
            elif ii < req_times - 1:
                rev_vectors = np.vstack(
                    (rev_vectors, bert_model.encode(rev_list[ii * one_req_num: (ii + 1) * one_req_num])))
            else:
                rev_vectors = np.vstack((rev_vectors, bert_model.encode(rev_list[ii * one_req_num:])))
        np.save(bert_root + 'description_vectors.npy', des_vectors)
        np.save(bert_root + 'review_vectors.npy', rev_vectors)
        b_pre_vec_dict = {'description': des_vectors, 'review': rev_vectors}

        if emb_method == 'bert':
            pre_vec_dict = b_pre_vec_dict
            del g_pre_vec_dict
        else:
            pre_vec_dict = g_pre_vec_dict
            del b_pre_vec_dict

        return adj_list, graph_info, pre_vec_dict, train_data, test_data, p_ui_dic, all_adjs
