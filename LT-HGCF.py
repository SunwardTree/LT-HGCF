import os
import random
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
from model import PredictNet
from utils import row_normalize, input_data, getEmb, sample_neg, result_evaluate
from params import parse_args

args = parse_args()
str_dev = "cuda:" + str(args.gpu_id)
device = torch.device(str_dev if (torch.cuda.is_available() and args.gpu_id >= 0) else "cpu")
print('Device:', device)
torch.manual_seed(args.seed)
np.random.seed(args.seed)


class RunProcess:
    def __init__(self, args):
        self.args = args
        self.model_state = None
        # Load data
        self.A, self.graph_info, self.pre_vec_dict, self.train_data, self.test_data, self.p_ui_dic, self.all_adjs =\
            input_data(self.args.data, device, self.args.emb_method, self.args.self_loop)
        self.num_nodes = self.graph_info['n_node']
        self.n_item = self.graph_info['n_item']
        self.n_user = self.graph_info['n_user']
        self.item_shift = self.n_user
        self.num_rel = len(self.A)
        self.topKs = eval(self.args.topKs)

        self.train_p_list = self.train_data
        print('Positive train data:', len(self.train_p_list))

        # Adjacency matrix normalization
        self.norm_A = row_normalize(self.A)
        self.norm_adjs = row_normalize(self.all_adjs)
        # Create Model
        self.model = PredictNet(g_info=self.graph_info, g_hidden_dim=self.args.hidden,
                                p_hidden_dim=self.args.p_hidden_dim, num_layer=self.args.num_layer,
                                use_dr_pre=self.args.use_dr_pre, use_rev=self.args.use_rev, use_des=self.args.use_des,
                                pre_v_dict=self.pre_vec_dict, dropout=eval(self.args.drop),
                                pred_method=self.args.pred_method, device=device, active_fun=self.args.active_fun,
                                use_residual=self.args.use_residual, use_layer_weight=self.args.use_layer_weight,
                                use_weight=self.args.use_weight, use_rgcn=self.args.use_rgcn)
        self.model.to(device)

        # optimizer weight_decay
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.l2)

    def train(self):
        # Start training
        self.model.train()
        self.model(norm_A=self.norm_A, norm_adjs=self.norm_adjs)
        net_dropout = eval(self.args.drop)[0]

        batch_pos_list = random.sample(self.train_p_list, self.args.batch_size // 2)
        batch_neg_list = sample_neg(batch_pos_list, self.p_ui_dic, self.n_item)
        p_user_emb, p_item_emb, _ = getEmb(batch_pos_list, self.model, self.item_shift, device)
        pos_scores = self.model.predict(p_user_emb, p_item_emb, dropout=net_dropout)
        n_user_emb, n_item_emb, _ = getEmb(batch_neg_list, self.model, self.item_shift, device)
        neg_scores = self.model.predict(n_user_emb, n_item_emb, dropout=net_dropout)
        loss = (-nn.LogSigmoid()(pos_scores - neg_scores)).sum()

        # Backward and optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def test(self, best_hit, best_ndcg):
        list_hits, list_ndcgs = [], []
        for _ in self.topKs:
            list_hits.append([])
            list_ndcgs.append([])
        with torch.no_grad():
            self.model.eval()
            for t_uid in range(self.n_user):
                try:
                    one_hit, one_ndcg = result_evaluate(
                        t_uid, self.topKs, self.model, self.test_data, self.item_shift, device)
                    # print(t_uid, one_hit, one_ndcg, one_precision, one_recall)
                    kk = 0
                    for _ in self.topKs:
                        list_hits[kk].append(one_hit[kk])
                        list_ndcgs[kk].append(one_ndcg[kk])
                        kk += 1
                except KeyError:
                    continue

            kk = 0
            str_log = ''
            for top_k in self.topKs:
                if len(list_hits) > 0:
                    t_hit = np.array(list_hits[kk]).mean()
                    t_ndcg = np.array(list_ndcgs[kk]).mean()
                else:
                    t_hit = 0
                    t_ndcg = 0
                best_hit[kk] = best_hit[kk] if best_hit[kk] > t_hit else t_hit
                best_ndcg[kk] = best_ndcg[kk] if best_ndcg[kk] > t_ndcg else t_ndcg
                str_log += 'Top %3d: ' \
                           'HR = %.6f, NDCG = %.6f \n' \
                           'Best HR = %.6f, NDCG = %.6f\n' % \
                           (top_k, t_hit, t_ndcg, best_hit[kk], best_ndcg[kk])
                kk += 1
        return str_log

    def save_checkpoint(self, filename='./checkpoints/' + args.data):
        if not os.path.exists('checkpoints'):
            os.makedirs('checkpoints')
        self.model_state = {
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        torch.save(self.model_state, filename)
        print('Successfully saved model_old\n...')

    def load_checkpoint(self, filename='./checkpoints/' + args.data):
        load_state = torch.load(filename)
        self.model.load_state_dict(load_state['state_dict'])
        self.optimizer.load_state_dict(load_state['optimizer'])
        print('Successfully Loaded model_old\n...')


if __name__ == '__main__':
    if not os.path.exists('./result_log/'):
        os.makedirs('./result_log/')
    result_log = open('./result_log/' + args.data + '_HGCF'
                      + '--emb_method-' + args.emb_method + '--pred_method-' + args.pred_method
                      + '--hidden-' + str(args.hidden) + '--self_loop-' + str(args.self_loop)
                      + '--use_residual-' + str(args.use_residual) + '--n_layer-' + str(args.num_layer)
                      + '--use_des-' + str(args.use_des) + '--use_rev-' + str(args.use_rev)
                      + '--dr_pre_train-' + str(args.use_dr_pre) + '--use_weight-' + str(args.use_weight) + '--'
                      + datetime.now().strftime("%Y%m%d_%H%M%S") + '.txt', 'w')
    result_log.write(str(args) + '\n')
    result_log.flush()

    print(args)
    process = RunProcess(args)
    if args.load_model:
        process.load_checkpoint()

    best_hit, best_ndcg = [], []
    for _ in eval(args.topKs):
        best_hit.append(0)
        best_ndcg.append(0)
    for epoch in range(args.epochs + 1):
        t_loss = process.train()
        if epoch % 10 == 0:
            train_log = "Epoch: {epoch}, Training Loss: {loss}".format(epoch=epoch, loss=str(t_loss))
            test_log = process.test(best_hit, best_ndcg)
            all_log = train_log + '\n' + test_log + '\n'
            print(all_log)
            result_log.write(all_log)
            result_log.flush()
    result_log.close()
    if args.save_model:
        process.save_checkpoint()
