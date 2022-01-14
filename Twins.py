# -*-coding:utf-8-*-
import time
from pathlib import Path
import pickle
import torch
from models.graphcnn import GraphCNN
from tqdm import tqdm
import os
import heapq
from libs import S2VGraph, read_pickle, BlockEmbedding, ControlGraphEmbedding
import argparse
import numpy as np
from Tokenizer.InstructionTokenizer import InstructionTokenizer
from library.MatchingScheme import Markov_LSH_Matching

def gen_one_list_for_each_binary(input_files=[], min_blk=3):

    global BlkEmbedding
    data = []
    for fname in input_files:
        func_dict = read_pickle(fname, BlkEmbedding=BlkEmbedding, dim=128, min_blk=min_blk)
        data.extend(list(func_dict.values()))
    return data


def build_for_one_list(inputs=[], min_blk=3):
    FunctionName = []
    VectorTable = []
    DetailData = {}
    FunctionMap = {}
    id = 0
    global CfgEmbedding

    pbar = tqdm(inputs)
    for func in pbar:
        funcName = func.label
        if len(func) < min_blk:
            continue

        func_features = CfgEmbedding.generate(func).cpu().numpy()
        if True in np.isnan(func_features):
            continue
        FunctionName.append((id, funcName))
        DetailData[id] = {"binary_name": func.binary_name,
                          "constants": func.constants,
                          "funcname": funcName
                          }
        FunctionMap[funcName] = id
        id += 1
        VectorTable.append(func_features[0])

    return FunctionName, VectorTable, FunctionMap


def get_idb_file(binary_name):
    if not os.path.exists(binary_name):
        print("[!] {} does not exist.".format(binary_name))
        exit(-1)
    print("[+] Extracting Binary Features")
    os.system('start {} -A -S"{}" "{}"'.format(cmd64, script, binary_name))
    while True:
        if os.path.exists(binary_name+".pkl"):
            break
        else:
            time.sleep(3)
    print("[+] Extraction is complete!")


def handle_binary(binary_name, min_blk=3):
    input_name = binary_name + ".pkl"
    file_dict = gen_one_list_for_each_binary(input_files=[input_name], min_blk=min_blk)
    FunctionName, VectorTable, FunctionMap = build_for_one_list(file_dict, min_blk=min_blk)

    return FunctionName, VectorTable, FunctionMap


def compare_binaries_with_sequences(bin1, bin2):
    FN1, VT1, FM1 = handle_binary(bin1)
    FN2, VT2, FM2 = handle_binary(bin2)
    return Markov_LSH_Matching(bin1, bin2, (FN1, VT1, FM1), (FN2, VT2, FM2))


if __name__ == '__main__':

    # ida64_absolute_path
    cmd64 = "script_absolute_path/ida64.exe"

    # script_absolute_path
    script = "script_absolute_path/TWINS/ExtractFeatures/ExtractFeatures.py"

    ckpt = "FunctionEmbedding/checkpoint/model.ckp"

    CFGconfig = "config/config.args"
    tokenizer_path = "Tokenizer/model_save/tokenizer.model"
    blkEmbedding = "./BlockEmbedding/checkpoint"
    device = torch.device('cuda:0')
    dimension = 128

    parser = argparse.ArgumentParser(
        description='Compare two binaries.')
    parser.add_argument('--file1', type=str, default=".\\example\\b2sum_O3_Coreutils",
                        help='Input File 1')
    parser.add_argument('--file2', type=str, default=".\\example\\b2sum_O2_Coreutils",
                        help='Input File 2')
    args = parser.parse_args()

    file1 = args.file1
    file2 = args.file2


    get_idb_file(file1)
    get_idb_file(file2)

    BlkEmbedding = BlockEmbedding(model_directory=blkEmbedding, tokenizer=tokenizer_path, device=device)
    if CFGconfig and ckpt:
        CGEconfig = pickle.load(open(CFGconfig, "rb"))
    else:
        print("[!] Error")
        exit(-1)
    CfgEmbedding = ControlGraphEmbedding(args=CGEconfig, checkpoint=ckpt)

    compare_binaries_with_sequences(file1, file2)
