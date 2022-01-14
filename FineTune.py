import pickle
from pathlib import Path
from tqdm import tqdm
import networkx as nx
import re
import numpy as np
import os
from transformers import RobertaModel

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
import torch.utils.data as Data
from Tokenizer.InstructionTokenizer import InstructionTokenizer
import random
from sklearn.metrics import roc_curve, auc, roc_auc_score
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

class AssemblyLanguageModel:
    def __init__(self,
                 model_directory=None,
                 tokenizer=None,
                 device=torch.device('cpu')
                 ):
        self.tokenizer = pickle.load(open(tokenizer, "rb"))
        self.model = RobertaModel.from_pretrained(model_directory, add_pooling_layer=True)
        self.device = device

        self.model.to(self.device)
        self.istrain = False

    def __call__(self, *args, **kwargs):
        return self.forward(args[0])
    def parameters(self):
        return self.model.parameters()

    def train(self):
        self.istrain = True
        return self.model.train()

    def eval(self):
        self.istrain = False
        return self.model.eval()

    def save_pretrained(self, output_dir):
        self.model.save_pretrained(output_dir)

    def forward(self, inputs):
        ret = []
        if isinstance(inputs[0], (list, str)):
            tokens = self.tokenizer(inputs, return_tensors="pt", max_length=256, padding=True, truncation=True)[
                'input_ids']
        else:
            tokens = self.tokenizer.pad(inputs)['input_ids']
        tokens = tokens.to(self.device)
        data_loader = Data.DataLoader(tokens, batch_size=16)
        for batch_input in data_loader:
            output = self.model(batch_input)
            if self.istrain:
                ret.extend(list(output.last_hidden_state[:, -1, :]))
            else:
                ret.extend(list(output.last_hidden_state[:, -1, :].detach()))
        return torch.cat(ret, dim=0).to(self.device).reshape(len(ret), len(ret[0]))


P_norm = 2
criterion = nn.TripletMarginLoss(margin=9.5, p=P_norm, eps=1e-6)


def train(model, train_inputs, device, optimizer, epoch, total_iters=500, batch_size=32):
    model.train()
    pbar = tqdm(range(total_iters), unit='batch')
    loss_accum = 0
    Lambda = random.random() * 1
    print(Lambda)
    for pos in pbar:
        selected_idx = np.random.permutation(len(train_inputs))[:batch_size]
        batch_inputs = [train_inputs[idx] for idx in selected_idx]
        anchor_inputs = [_[0] for _ in batch_inputs]
        positive_inputs= [_[1] for _ in batch_inputs]
        negative_inputs = [_[2] for _ in batch_inputs]
        anchor_output = model(anchor_inputs)
        positive_output = model(positive_inputs)
        negative_output = model(negative_inputs)
        loss = criterion(anchor_output, positive_output, negative_output, Lambda=Lambda)
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss = loss.detach().cpu().numpy()
        loss_accum += loss

        pbar.set_description('epoch: %d' % (epoch))

    average_loss = loss_accum / total_iters
    print("loss training: %f" % (average_loss))

    return average_loss

def test(model, test_inputs):
    model.eval()

    anchor_inputs = [_[0] for _ in test_inputs]
    positive_inputs = [_[1] for _ in test_inputs]
    negative_inputs = [_[2] for _ in test_inputs]
    anchor_outputs = model(anchor_inputs).detach()
    positive_outputs = model(positive_inputs).detach()
    negative_outputs = model(negative_inputs).detach()

    Pred = torch.pairwise_distance(anchor_outputs, negative_outputs, p=P_norm) - torch.pairwise_distance(anchor_outputs, positive_outputs, p=P_norm)
    all_case = len(Pred)
    TP = len(Pred[torch.where(Pred > 0)])
    accuracy = TP / all_case
    distance_avg = Pred.sum() / all_case
    print("Distance", distance_avg)
    print("Accuracy", accuracy)
    return accuracy, distance_avg.cpu().numpy()



def load_dataset(data_dir):
    ret = []
    data_files = [str(x) for x in Path(data_dir).glob("**/*.blk.pkl")]
    for f in data_files:
        data_ = pickle.load(open(f, "rb"))
        ret.extend(data_)
    return ret


def main():
    model_directory = "./BlockEmbedding/checkpoint"
    tokenizer_directory = "./Tokenizer/model_save/tokenizer.model"
    data_dir = "BlockGroundTruth/"
    model_output_dir = "BlockEmbedding/ModelBlockSimilarity/"

    epochs = 10000
    lr = 1e-3
    device = torch.device('cuda:0')

    best_results = 0.95
    inputs = load_dataset(data_dir)
    train_inputs, test_inputs = train_test_split(inputs, test_size=0.2, random_state=66, shuffle=True)
    print("Train:", len(train_inputs), "\tTest:", len(test_inputs))
    model = AssemblyLanguageModel(model_directory=model_directory, tokenizer=tokenizer_directory, device=device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.9)

    for epoch in range(1, epochs + 1):
        avg_loss = train(model, train_inputs, device, optimizer, epoch, total_iters=60, batch_size=100)
        scheduler.step()
        current_result, _ = test(model, test_inputs)
        if current_result > best_results:
            output_dir = model_output_dir + str(current_result) + "_" + str(scheduler.get_last_lr()[-1])
            best_results = current_result
            os.mkdir(output_dir)
            model.save_pretrained(output_dir)




if __name__ == '__main__':
    main()