# -*-coding:utf-8-*-
import numpy as np
import pickle
import networkx as nx
import torch
import re
from transformers import RobertaModel
from transformers import RobertaTokenizerFast
import torch.utils.data as Data
from transformers import RobertaForMaskedLM
from pathlib import Path
from models.graphcnn import GraphCNN

cs_tag = "tagcs"
loc_tag = "tagloc"
locret_tag = "taglocret"
ptr_tag = "tagptr"




class S2VGraph(object):
    def __init__(self, g, label, node_tags=None, node_features=None):
        '''
            g: a networkx graph
            label: an integer graph label
            node_tags: a list of integer node tags
            node_features: a torch float tensor, one-hot representation of the tag that is used as input to neural nets
            edge_mat: a torch long tensor, contain edge list, will be used to create torch sparse tensor
            neighbors: list of neighbors (without self-loop)
        '''
        self.label = label
        self.g = g
        self.node_tags = node_tags
        self.neighbors = []
        self.node_features = 0
        self.edge_mat = 0

        self.max_neighbor = 0
        self.binary_name = ""
        self.constants = []
        self.size = 0
        self.nodes = 0
        self.edges = 0
        self.instructions = 0

    def __len__(self):
        """Returns the number of nodes in the graph. Use: 'len(G)'.

        Returns
        -------
        nnodes : int
            The number of nodes in the graph.

        See Also
        --------
        """
        return len(self.g._node)

class BlockEmbedding:
    def __init__(self,
                 model_directory=None,
                 opr_list=None,
                 tokenizer=None,
                 device=torch.device('cpu'),
                 batch_size=64
                 ):
        self.tokenizer = pickle.load(open(tokenizer, "rb"))
        self.model = RobertaModel.from_pretrained(model_directory, add_pooling_layer=True)
        self.device = device
        self.model.to(self.device)
        self.model.eval()
        self.opr_list = None
        self.batch_size=batch_size
        if opr_list != None:
            self.opr_list = opr_list


    def batch_generate(self, ins_seq_list):
        inputs = self.tokenizer.pad(ins_seq_list, pad_length=256)['input_ids']
        data_loader = Data.DataLoader(inputs, batch_size=self.batch_size)
        features = []
        for batch_input in data_loader:
            batch_input = batch_input.to(self.device)
            output = self.model(batch_input)
            features.extend(list(output.last_hidden_state[:, -1, :].detach()))
        return features

    def read_blk(self, instructions: list):
        tokens = self.tokenizer(instructions, return_tensors="pt", max_length=254, padding=True, truncation=True)
        return tokens



class ControlGraphEmbedding:
    def __init__(self,
                 args=None,
                 num_classes=128,
                 checkpoint=None,
                 device=None,
                 ):
        if device:
            self.device = device
        else:
            self.device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
        self.model = GraphCNN(args.num_layers, args.num_mlp_layers, num_classes,
                              args.hidden_dim,
                              num_classes, args.final_dropout, args.learn_eps, args.graph_pooling_type,
                              args.neighbor_pooling_type, self.device).to(self.device)
        self.model.load_state_dict(torch.load(checkpoint))
        self.model.eval()

    def generate(self, graph):
        output = self.model(graph)
        output = output.detach()
        return output


def generate_graph(graph, blkAttrs, label=0, dim=192):

    g = S2VGraph(graph, label)
    g.neighbors = [[] for _ in range(len(g.g))]
    for i, j in g.g.edges():
        g.neighbors[i].append(j)
        g.neighbors[j].append(i)

    g.node_features = torch.zeros(len(g.g), dim)
    for n, nodeAttr in zip(range(len(blkAttrs)), blkAttrs):
        g.node_features[n] = nodeAttr
    edges = [list(pair) for pair in g.g.edges()]

    if edges:
        g.edge_mat = torch.LongTensor(edges).transpose(0, 1)
    else:
        g.edge_mat = []
    return g

def read_pickle(filename, BlkEmbedding=None, min_blk=3, dim=128):

    Feat = pickle.load(open(filename, "rb"))
    Func_dict = {}
    for func in Feat:
        g = nx.Graph()
        edges = []
        blkInsSeq = []


        if len(func['blocks']) < min_blk:
            continue
        for cur, blk in zip(range(len(func['blocks'])), func['blocks']):
            g.add_node(cur)
            blkInsSeq.append(BlkEmbedding.read_blk(func['blocks'][cur]['src']))
            for nextblk in blk['succs']:
                edges.append([cur, nextblk])
        blkAttrs = BlkEmbedding.batch_generate(blkInsSeq)

        g.add_edges_from(edges)
        func_graph = generate_graph(g, blkAttrs, func['name'], dim=dim)

        func_graph.binary_name = func["filename"]
        func_graph.constants = func['constants']
        func_graph.size = func['size']
        func_graph.nodes = func['nodes']
        func_graph.edges = func['edges']
        func_graph.instructions = func['instructions']

        Func_dict[func['name']] = func_graph

    return Func_dict

def read_pickle_gen_special_functions(filename, BlkEmbedding=None, function_list=[], dim=128, min_blk=3):

    Feat = pickle.load(open(filename, "rb"))
    Func_dict = {}
    for func in Feat:
        g = nx.Graph()
        edges = []
        blkInsSeq = []
        if func['name'] not in function_list:
            continue

        if len(func['blocks']) < min_blk:
            continue
        for cur, blk in zip(range(len(func['blocks'])), func['blocks']):
            g.add_node(cur)
            blkInsSeq.append(BlkEmbedding.read_blk(func['blocks'][cur]['src']))
            for nextblk in blk['succs']:
                edges.append([cur, nextblk])
        blkAttrs = BlkEmbedding.batch_generate(blkInsSeq)

        g.add_edges_from(edges)
        func_graph = generate_graph(g, blkAttrs, func['name'], dim=dim)

        func_graph.binary_name = func["filename"]
        func_graph.constants = func['constants']
        func_graph.size = func['size']
        func_graph.nodes = func['nodes']
        func_graph.edges = func['edges']
        func_graph.instructions = func['instructions']

        Func_dict[func['name']] = func_graph

    return Func_dict

def read_data_gen_one_list(input_files=[], model_directory="./BlockEmbedding", device=torch.device('cpu')):

    BlkEmbedding = BlockEmbedding(model_directory=model_directory, device=device)
    data = []
    for fname in input_files:
        func_dict = read_pickle(fname, BlkEmbedding=BlkEmbedding, dim=128)
        data.extend(list(func_dict.values()))
    return data
