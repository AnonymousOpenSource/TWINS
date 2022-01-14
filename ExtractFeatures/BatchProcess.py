# -*-coding:utf-8-*-
import os
import glob
import argparse
import random
from pathlib import Path
import zipfile
from tqdm import tqdm
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
    script = "F:\\Code\\BinGeMarchitecTure\\ExtractFeatures\\ExtractFeatures_twoBinaries.py"
    pbar = tqdm(fileList)
    for f in pbar:
        filter_file = (".i64", ".idb", ".pkl", ".py", ".txt", ".dat", ".a", ".sqlite",  ".sqlite-crash", ".o", ".id2", ".id1", "id0", ".nam", ".til")
        if f.endswith(filter_file):
            continue
        if os.path.isdir(f):
            continue
        if os.path.exists(f+".cmp.dat"):
            continue
        if f.endswith(".zip"):
            unzip_filelist = unzipfile(f)
            for unzip_f in unzip_filelist:
                filter_file = (".i64", ".idb", ".pkl", ".py", ".txt", ".dat")
                if unzip_f.endswith(filter_file):
                    continue
                print('{} -A -S"{}" "{}"'.format(cmd64, script, unzip_f))
                os.system('{} -A -S"{}" "{}"'.format(cmd64, script, unzip_f))
            continue
        print('{} -A -S"{}" "{}"'.format(cmd64, script, f))
        os.system('{} -A -S"{}" "{}"'.format(cmd64, script, f))
        time.sleep(0.8)


if __name__ == "__main__":
    input_file = [str(x) for x in Path("database").glob("**/*")]
    main(input_file)
