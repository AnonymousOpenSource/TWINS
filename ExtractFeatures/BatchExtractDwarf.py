# -*-coding:utf-8-*-
import os
import glob
import argparse
import random
from pathlib import Path
import zipfile
from tqdm import tqdm
from ExtractCodeLines import extract_dwarf
import time

def unzipfile(file_name):
    """unzip zip file"""
    extracted_path = file_name.replace(".zip", "")
    zip_file = zipfile.ZipFile(file_name)
    zip_file.extractall(extracted_path)
    zip_file.close()
    return [str(x) for x in Path(extracted_path).glob("**/*")]

def main(fileList:list):
    cmd32 = "D:\\Programs\\IDA7.5SP3\\ida.exe"
    cmd64 = "D:\\Programs\\IDA7.5SP3\\ida64.exe"
    script = "F:\\Code\\BinGeMarchitecTure\\ExtractFeatures\\ExtractBlocks.py"

    pbar = tqdm(fileList)
    for f in pbar:
        filter_file = (".i64", ".idb", ".pkl", ".py", ".txt", ".dat", ".a", ".sqlite",  ".sqlite-crash", ".o")
        if f.endswith(filter_file):
            continue
        if os.path.exists(f+".hash.dat"):
            continue
        if os.path.isdir(f):
            continue

        extract_dwarf(f)
        print('{} -A -S"{}" "{}"'.format(cmd64, script, f))
        os.system('{} -A -S"{}" "{}"'.format(cmd64, script, f))
        time.sleep(0.5)



if __name__ == "__main__":
    input_file = [str(x) for x in Path("database").glob("**/*")]
    main(input_file)