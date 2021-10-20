import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    # Digital_Music, Beauty, Clothing_Shoes_and_Jewelry
    parser.add_argument('--data', type=str, default="Digital_Music",
                        help='dataset: Digital_Music, Beauty, Clothing_Shoes_and_Jewelry')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='Enables CUDA training. If use cpu, set -1.')
    parser.add_argument('--seed', type=int, default=2021, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=2000,
                        help='Number of epochs to train.')
    parser.add_argument('--batch_size', type=int, default=1024)  # 1024
    parser.add_argument('--lr', type=float, default=0.001,  # 0.001
                        help='Initial learning rate.')
    parser.add_argument('--l2', type=float, default=1e-4,  # 1e-4
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--self_loop', type=int, default=0,  # 0
                        help='If self loop.')
    parser.add_argument('--active_fun', type=str, default='none',  # none
                        help='leaky_relu, relu; none means no use.')
    parser.add_argument('--use_weight', type=int, default=0,  # 0
                        help='If use weight for RGCN.')

    parser.add_argument('--use_dr_pre', type=int, default=1,
                        help='If use pre-trained embeddings.')
    # The control variable introduces description and comment respectively, and the direct vector not introduced is set to 0
    parser.add_argument('--use_des', type=int, default=1, help='1 is True.')
    parser.add_argument('--use_rev', type=int, default=1, help='1 is True.')

    # If use initial residual
    parser.add_argument('--use_residual', type=int, default=1, help='1 is True.')  # 1
    # If use layer_weight
    parser.add_argument('--use_layer_weight', type=int, default=1, help='1 is True.')  # 1

    # inner_product mlp joint
    parser.add_argument('--pred_method', type=str, default='joint')
    # glove bert
    parser.add_argument('--emb_method', type=str, default='bert')
    parser.add_argument('--topKs', type=str, default='[10, 20]')
    # dropout rate of [net, node]
    parser.add_argument('--drop', type=str, default='[0, 0]')  # [0, 0]

    parser.add_argument('--hidden', type=int, default=128,  # 128
                        help='Number of hidden units.')
    parser.add_argument('--g_in_dim', type=int, default=128,
                        help='Number of graph input dim.')
    parser.add_argument('--p_hidden_dim', type=int, default=128,  # 128
                        help='Number of hidden units of PredictNet.')
    parser.add_argument('--num_layer', type=int, default=4,  # 4
                        help='Number of R-GCN layers.')
    parser.add_argument('--load_model', type=int, default=0)
    parser.add_argument('--save_model', type=int, default=0)
    parser.add_argument('--use_rgcn', type=int, default=1)  # 1
    return parser.parse_args()
