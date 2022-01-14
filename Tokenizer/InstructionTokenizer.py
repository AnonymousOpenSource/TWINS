import glob
import os
import pickle
from pathlib import Path
import re
from tqdm import tqdm
import torch
import numpy as np
from .tokenlib import normalize_opcode, normalize_oprand

def is_torch_available():
    return True



def to_py_obj(obj):
    """
    Convert a TensorFlow tensor, PyTorch tensor, Numpy array or python list to a python list.
    """

    if isinstance(obj, (list, tuple)):
        return [to_py_obj(o) for o in obj]
    elif is_torch_available() and isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

class InstructionTokenizer:
    def __init__(self, min_blk=3, statics=True):
        self.min_blk = min_blk
        self.TokenMap = {}
        self.OpcodeMap = {}
        self.OprandMap = {}
        self.statics = statics
        self.id = len(self.TokenMap)
        self.Opcode_id = len(self.OpcodeMap)
        self.Oprand_id = len(self.OprandMap)
        self.initMap()

    def get_special_tokens_mask(self, token_ids_0, already_has_special_tokens=None):
        all_special_ids = self.all_special_ids

        special_tokens_mask = [1 if token in all_special_ids else 0 for token in token_ids_0]

        return special_tokens_mask


    def convert_tokens_to_ids(self, token):
        return self.TokenMap[token]

    def updateTokenizer(self, dataDir):
        filelist = [str(x) for x in Path(dataDir).glob("**/*.pkl")]
        pbar = tqdm(filelist)
        for fname in pbar:
            self.deal_each_file(fname)

    def deal_each_file(self, fname):
        FunctionList = pickle.load(open(fname, "rb"))
        for func in FunctionList:
            if len(func['blocks']) < self.min_blk:
                continue
            for cur, blk in zip(range(len(func['blocks'])), func['blocks']):
                instructions = func['blocks'][cur]['src']
                for each_ins in instructions:
                    opcode, oprands = self.normalize(each_ins)
                    if self.statics:
                        num = self.OpcodeMap.get(opcode, 0) + 1
                        self.OpcodeMap[opcode] = num
                        for oprand in oprands:
                            num = self.OprandMap.get(oprand, 0) + 1
                            self.OprandMap[oprand] = num
                    token = opcode+"~"+"~".join(oprands)
                    self.insert_token(token)


    def normalize(self, each_ins):
        opcode = each_ins[0]
        oprands = each_ins[1:]
        return normalize_opcode(opcode), normalize_oprand(oprands)

    def insert_token(self, token):
        if token in self.TokenMap:
            pass
        else:
            self.TokenMap[token] = self.id
            self.id += 1



    def pad(self, encoded_inputs, return_tensors=None, pad_length=None):
        encoded_inputs = {key: [example[key] for example in encoded_inputs] for key in encoded_inputs[0].keys()}
        assert "input_ids" in encoded_inputs, (
            "You should supply an encoding or a list of encodings to this method. "
            "An encoding is the output of one the encoding methods of the tokenizer, i.e. "
            "__call__/encode_plus/batch_encode_plus. "
        )
        first_element = encoded_inputs["input_ids"][0]
        for key, value in encoded_inputs.items():
            encoded_inputs[key] = to_py_obj(value)

        batch_size = len(encoded_inputs["input_ids"])
        if pad_length:
            max_length = pad_length
        else:
            max_length = max(len(inputs) for inputs in encoded_inputs["input_ids"])

        batch_outputs = {}
        for i in range(batch_size):
            inputs = dict((k, v[i]) for k, v in encoded_inputs.items())
            outputs = self._pad(
                inputs,
                max_length=max_length,
            )

            for key, value in outputs.items():
                if key not in batch_outputs:
                    batch_outputs[key] = []
                batch_outputs[key].append(value)
        batch_outputs['input_ids'] = torch.tensor(batch_outputs['input_ids'])
        return batch_outputs

    def _pad(self, encoded_inputs, max_length=None, return_attention_mask=None):
        difference = max_length - len(encoded_inputs["input_ids"])

        if return_attention_mask:
            encoded_inputs["attention_mask"] = [1] * len(encoded_inputs["input_ids"]) + [0] * difference
        if "token_type_ids" in encoded_inputs:
            encoded_inputs["token_type_ids"] = (
                    encoded_inputs["token_type_ids"] + [self.pad_token_type_id] * difference
            )
        encoded_inputs["input_ids"] = encoded_inputs["input_ids"] + [self.pad_token_id] * difference

        return encoded_inputs


    def __call__(self, insturction_seq, return_tensors=True, max_length=256, return_dict=True, padding=False, truncation=False, **kwargs):
        ret = None
        if truncation:
            insturction_seq = insturction_seq[:max_length]
        tokens = []
        tokens.append(self.TokenMap["<s>"])
        for each_ins in insturction_seq:
            opcode, oprands = self.normalize(each_ins)
            token = opcode + "~" + "~".join(oprands)
            if token in self.TokenMap:
                tokens.append(self.TokenMap[token])
            else:
                tokens.append(self.TokenMap["<unk>"])
        tokens.append(self.TokenMap["</s>"])
        if return_dict:
            ret = {'input_ids': torch.tensor(tokens)}
        else:
            ret = torch.tensor(tokens).reshape(1, len(tokens))
        return ret

    def __len__(self):
        return len(self.TokenMap)

if __name__ == '__main__':
    output_dir = "./model_save/"
    InsTokenizer = InstructionTokenizer()
    InsTokenizer.updateTokenizer("../database/binaries/Train")


