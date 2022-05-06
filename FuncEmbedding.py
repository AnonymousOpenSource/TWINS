# -*-coding:utf-8-*-
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from models.graphcnn import GraphCNN
import pickle
from pathlib import Path
from library.libs import S2VGraph, BlockEmbedding
import os
from torch.nn.functional import softmax
import random
from Tokenizer.InstructionTokenizer import InstructionTokenizer

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

P_norm = 2
criterion = nn.TripletMarginLoss(margin=20, p=P_norm)

def read_data(file_path="dataset/data_cache/white192/data.3.pkl"):
    ret = []
    file_list = [str(x) for x in Path(file_path).glob("**/*.pkl")]
    for f in file_list:
        data = pickle.load(open(f, "rb"))
        ret.extend(data)
    return ret


def train(args, model, device, train_graphs, optimizer, epoch):
    model.train()
    total_iters = args.iters_per_epoch
    pbar = tqdm(range(total_iters), unit='batch')
    loss_accum = 0
    Lambda = 0.5
    print(Lambda)
    for pos in pbar:
        selected_idx = np.random.permutation(len(train_graphs))[:args.batch_size]
        batch_graph = [train_graphs[idx] for idx in selected_idx]
        anchor_graph = [_[0] for _ in batch_graph]
        positive_graph = [_[1] for _ in batch_graph]
        negative_graph = [_[2] for _ in batch_graph]
        anchor_output = model(anchor_graph)
        positive_output = model(positive_graph)
        negative_output = model(negative_graph)

        loss = criterion(anchor_output, positive_output, negative_output, Lambda=Lambda)
        # backprop
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss = loss.detach().cpu().numpy()
        loss_accum += loss

        # report
        pbar.set_description('epoch: %d' % (epoch))

    average_loss = loss_accum / total_iters
    print("loss training: %f" % (average_loss))

    return average_loss




def test(args, model, device, input_graphs, batch_size=-1):
    model.eval()

    selected_idx = np.random.permutation(len(input_graphs))[:batch_size]
    test_graphs = [input_graphs[idx] for idx in selected_idx]
    output = pass_data_iteratively(model, test_graphs)

    Pred = torch.cat([torch.pairwise_distance(i, k, p=P_norm) - torch.pairwise_distance(i, j, p=P_norm) for i, j, k in output], 0).to('cpu')
    all_case = len(Pred)

    TP = len(Pred[torch.where(Pred > 0)])
    acc = TP / all_case
    distance_avg = Pred.sum() / all_case
    print("Distance", distance_avg)
    print("Accuracy", acc)



    return distance_avg.cpu().numpy(), acc


###pass data to model with minibatch during testing to avoid memory overflow (does not perform backpropagation)
def pass_data_iteratively(model, graphs, minibatch_size=256):
    model.eval()
    output = []
    idx = np.arange(len(graphs))
    for i in range(0, len(graphs), minibatch_size):
        sampled_idx = idx[i:i + minibatch_size]
        if len(sampled_idx) == 0:
            continue

        batch_graph = [graphs[j] for j in sampled_idx]
        anchor_graph = [_[0] for _ in batch_graph]
        positive_graph = [_[1] for _ in batch_graph]
        negative_graph = [_[2] for _ in batch_graph]
        anchor_output = model(anchor_graph).detach()
        positive_output = model(positive_graph).detach()
        negative_output = model(negative_graph).detach()

        output.append((anchor_output, positive_output, negative_output))
    return output

# Modified below code:
# https://github.com/weihua916/powerful-gnns
def main():
    # Training settings
    parser = argparse.ArgumentParser(
        description='PyTorch graph convolutional neural net for function similarity classification')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--iters_per_epoch', type=int, default=80,
                        help='number of iterations per each epoch (default: 50)')
    parser.add_argument('--epochs', type=int, default=35000,
                        help='number of epochs to train (default: 350)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed for splitting the dataset into 10 (default: 0)')
    parser.add_argument('--num_layers', type=int, default=5,
                        help='number of layers INCLUDING the input one (default: 5)')
    parser.add_argument('--num_mlp_layers', type=int, default=3,
                        help='number of layers for MLP EXCLUDING the input one (default: 2). 1 means linear model.')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='number of hidden units (default: 64)')
    parser.add_argument('--final_dropout', type=float, default=0.01,
                        help='final layer dropout (default: 0.5)')
    parser.add_argument('--save_model', type=int, default=1,
                        help='save the best model (default: 1)')
    parser.add_argument('--graph_pooling_type', type=str, default="sum", choices=["sum", "average"],
                        help='Pooling for over nodes in a graph: sum or average')
    parser.add_argument('--neighbor_pooling_type', type=str, default="sum", choices=["sum", "average", "max"],
                        help='Pooling for over neighboring nodes: sum, average or max')
    parser.add_argument('--learn_eps', action="store_true",
                        help='Whether to learn the epsilon weighting for the center nodes. Does not affect training accuracy though.')
    args = parser.parse_args()

    data_dir = "./database/func_train/"
    model_save_dir = "FunctionEmbedding/model_store/"
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    graphs = read_data(file_path=data_dir)
    num_classes = 128

    train_graphs, test_graphs = train_test_split(graphs, test_size=0.1, random_state=66, shuffle=True)
    print("Train:", len(train_graphs), "\tTest:", len(test_graphs))
    model = GraphCNN(args.num_layers, args.num_mlp_layers, train_graphs[0][0].node_features.shape[1], args.hidden_dim,
                     num_classes, args.final_dropout, args.learn_eps, args.graph_pooling_type,
                     args.neighbor_pooling_type, device).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.95)



    best_results = 0.95
    for epoch in range(1, args.epochs + 1):
        avg_loss = train(args, model, device, train_graphs, optimizer, epoch)
        scheduler.step()
        distance_test, current_results = test(args, model, device, test_graphs)


        if current_results > best_results:
            torch.save(model.state_dict(), model_save_dir+"{}_{}.ckp".format(current_results, scheduler.get_last_lr()[-1]))



if __name__ == '__main__':
    main()
