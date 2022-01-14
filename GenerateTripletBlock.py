import os
import pickle

import numpy as np
from pathlib import Path
from Tokenizer.InstructionTokenizer import InstructionTokenizer
from tqdm import tqdm
import time

def load_files(f):
    ret = pickle.load(open(f, "rb"))
    return ret

def generate_triplet(hdict1, hdict2, seed=666, tokenizer=None, input_data=None):
    data_triplet = []
    keys = list(hdict2.keys())
    randGen = np.random.RandomState(seed)

    hdict2_tokened = {}
    for hashnum in hdict2:
        hdict2_tokened[hashnum] = tokenizer(hdict2[hashnum], truncation=True)


    max_upper = len(keys) - 20
    for hashnum in hdict1:
        t1 = time.time()
        if not hashnum:
            continue
        if hashnum in hdict2:
            anchor = hdict1[hashnum]
            anchor = tokenizer(anchor, truncation=True)
            positive = hdict2_tokened[hashnum]


            negNumber = randGen.randint(max_upper)
            for _ in range(negNumber, negNumber+10):
                negSample = keys[_]

                if negSample == hashnum:
                    continue

                negative = hdict2_tokened[negSample]
                data_triplet.append([anchor, positive, negative])

    if input_data:
        data_triplet.extend(input_data)
    return data_triplet


def main():
    file1 = "database/example/O3/base64.hash.dat"
    file2 = "database/example/O1/base64.hash.dat"
    tokenizer = pickle.load(open("Tokenizer/model_save/tokenizer.model", "rb"))
    triplet_blocks = generate_triplet(load_files(file1), load_files(file2), tokenizer=tokenizer)
    return triplet_blocks

if __name__ == '__main__':
    main()
