from pathlib import Path
from PIL import Image

import argparse
import os
import re

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str)

args = parser.parse_args()

def clean_data(path: str, label_file: str):
    with open(os.path.join(path, label_file), 'r') as f:
        for raw in f:
            if re.match('.*jpg.*', raw):
                name = raw.split('.jpg')[0].strip()
                label = raw.split('.jpg')[-1].strip()       
            elif re.match('.*jpeg.*', raw):
                name = raw.split('.jpeg')[0].strip(' \t')
                label = raw.split('.jpeg')[1].strip(' \t')
            else:
                print(f"未观察到的数据: {raw}")  
                continue           
            
            if int(label) >= 192:
                print(f"非法数据: {name} {label}")
                target_file = os.path.join(path, name + '.jpg')
                if os.path.exists(target_file):
                    os.remove(target_file)

if __name__ == '__main__':
    
    base_path = args.data

    clean_data(os.path.join(base_path, 'trainset'), 'train.txt')
    
    print("valset")
    clean_data(os.path.join(base_path, 'valset'), 'val.txt')
    
            