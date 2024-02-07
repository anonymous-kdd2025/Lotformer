import pandas as pd
import argparse
from model.lt_transformer import RTransformer, CTransformer
from model.utils import *
from model.splitter import Splitter, UniformSplitter
from experiment.train import train_r_transformer, train_c_transformer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # datasets
    parser.add_argument('-d', '--dataset', default=3, type=int, help='Test on the simulation dataset')
    parser.add_argument('-t', '--t0', default=14, type=int, help='Duration of short-term surrogate')
    parser.add_argument('-T', '--t_tar', default=100, type=int, help='Targeted long-term time point')
    parser.add_argument('-s', '--surrogate', type=str, default=11, help='number of surrogates incorporated in the model')
    # model
    parser.add_argument('-m', '--model', default='r_transformer', help='Test model')
    parser.add_argument('-de', '--dim_embedding', type=int, default=64, help='dimension of embedding layer')
    parser.add_argument('-dk', '--dim_k', type=int, default=32, help='dimension of key and query')
    parser.add_argument('-dv', '--dim_v', type=int, default=32, help='dimension of value')
    parser.add_argument('-dh', '--dim_hidden', type=int, default=64, help='dimension of hidden layer')
    parser.add_argument('-dc', '--dim_class', type=int, default=128, help='dimension splitter class, for c transformer only(default 128)')
    parser.add_argument('-nl', '--n_layers', type=int, default=4, help='number of encoder and decoder layers')
    parser.add_argument('-nh', '--n_heads', type=int, default=8, help='number of attention heads')
    parser.add_argument('-drop', '--drop', type=float, default=0.1, help='drop rate')
    # train
    parser.add_argument('-b', '--batch_size', type=int, default=64, help='batch size(default 64)')
    parser.add_argument('-e', '--epoch', type=int, default=20, help='number of epochs(default 20)')
    parser.add_argument('-r', '--repeat', type=int, default=5, help='replications(default 5)')
    parser.add_argument('-l', '--lr', type=float, default=0.0002, help='learning rate (default 1e-3)')
    parser.add_argument('-tl', '--tune_lr_every', type=int, default=20, help='tuning lr step-wisely(default step 20)')
    parser.add_argument('-g', '--gamma', type=float, default=0.2, help='decaying rate of lr(default 0.1)')
    parser.add_argument('-wd', '--weight_decay', type=float, default=0.001, help='weight decay')
    parser.add_argument('-v', '--verbose', type=int, default=1, help='verbose')
    parser.add_argument('-lbd', '--weight_decoder', type=float, default=1.0, help='verbose')
    
    
    args = parser.parse_args()

    # print(args)

    data_path = './dataset/synthetic dataset %d/' % args.dataset

    surro_name = ['s%d' % i for i in range(args.surrogate - 1)] + ['Y']

    cov_exp = pd.read_csv(data_path + 'exp data/cov_data.csv')
    panel_exp = []
    for i in range(args.t0):
        panel_exp.append(pd.read_csv(data_path + 'exp data/panel_data_%d.csv' % i)[surro_name])
    panel_exp = pd.concat(panel_exp, axis=1)
    treatment_exp = pd.read_csv(data_path + 'exp data/treatment.csv')
    panel_200_exp = pd.read_csv(data_path + 'exp data/panel_data_%d.csv' % (args.t_tar-1))


    cov_obs = pd.read_csv(data_path + 'obs data 2/cov_data.csv')

    panel_obs = []
    for i in range(args.t0):
        panel_obs.append(pd.read_csv(data_path + 'obs data 2/panel_data_%d.csv' % i).loc[:, surro_name])
    panel_obs = pd.concat(panel_obs, axis=1)

    treatment_obs = pd.read_csv(data_path + 'obs data 2/treatment.csv')
    panel_200_obs = pd.read_csv(data_path + 'obs data 2/panel_data_%d.csv' % (args.t_tar-1))

    dim_input = 11 + args.surrogate
    dim_output = args.surrogate
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    learned_pos=True



    for j in range(args.repeat):
        print("repeat %d" % j)
        
        if args.model =='r_transformer':
            valid_dataset = RDataset(cov_exp, panel_exp, treatment_exp, panel_200_exp['Y'], args.t0)
            train_dataset = RDataset(cov_obs, panel_obs, treatment_obs, panel_200_obs['Y'], args.t0)

            model = RTransformer(dim_input, dim_output, args.dim_embedding, args.n_layers, args.n_heads, args.dim_k, args.dim_v, args.dim_hidden, args.t0, drop=args.drop, drop_pred=args.drop, learned_pos=learned_pos).to(device)
            mae_train_all, mape_train_all, y_gt_train_all, y_pred_train_all, mae_valid_all, mape_valid_all, y_gt_valid_all, y_pred_valid_all, ate_gt_all, ate_pred_all, loss_all = \
            train_r_transformer(model, train_dataset, valid_dataset, args.epoch, weight_decay=args.weight_decay, lr=args.lr, tune_lr_every=args.tune_lr_every, gamma=args.gamma, verbose=args.verbose, decoder_weight=args.weight_decoder)

        if args.model == 'c_transformer':
            exg = UniformSplitter(panel_200_obs[['Y']], args.dim_class)

            valid_dataset = CTDataset(cov_exp, panel_exp, treatment_exp, panel_200_exp['Y'], exg, args.t0)
            train_dataset = CTDataset(cov_obs, panel_obs, treatment_obs, panel_200_obs['Y'], exg, args.t0)
            
            model = CTransformer(dim_input, dim_output, args.dim_embedding, args.n_layers, args.n_heads, args.dim_k, args.dim_v, args.dim_hidden, args.dim_class, args.t0, drop=args.drop, drop_pred=args.drop, learned_pos=learned_pos).to(device)
            mae_train_all, mape_train_all, y_gt_train_all, y_pred_train_all, mae_valid_all, mape_valid_all, y_gt_valid_all, y_pred_valid_all, ate_gt_all, ate_pred_all, loss_all = \
            train_c_transformer(model, train_dataset, valid_dataset, exg, args.epoch, weight_decay=args.weight_decay, lr=args.lr, tune_lr_every=args.tune_lr_every, gamma=args.gamma, verbose=args.verbose, weight_decoder=args.weight_decoder)