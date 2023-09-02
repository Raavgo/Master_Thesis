from glob import glob

import pandas as pd
import pickle
import argparse
from tqdm import tqdm

def get_original_with_fakes(json_path):
    pairs = []
    path = json_path.replace("metadata.json", "")

    df = pd.read_json(json_path).T
    original = df[df["original"].isna() == True].index.values
    for val in original:
        fake = df[df["original"] == val].index.values
        fake = [path + f for f in fake]
        pairs.append({"real": path + val, "fake": fake})

    return pairs

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Compress video")
    parser.add_argument("--path", help="root directory", default='/home/ai21m034/master_project/data/compressed/')
    args = parser.parse_args()

    print("Starting")

    pairs = []
    for i in tqdm(range(50)):
        pairs.extend(get_original_with_fakes(args.path + f'dfdc_train_part_{i}/metadata.json'))

    processed_files = glob('/data/data/*.data')
    processed_files = [f.split("/")[-1].replace(".data", ".mp4") for f in processed_files]
    negativ_files = glob('/data/data/*.txt')
    negativ_files = [f.split("/")[-1].replace(".txt", ".mp4") for f in negativ_files]
    print(len(pairs))
    pairs = [p for p in pairs if p["real"].split("/")[-1] not in processed_files]
    pairs = [p for p in pairs if p["real"].split("/")[-1] not in negativ_files]
    print(len(pairs))
    with open('pairs', 'wb') as fp:
        pickle.dump(pairs, fp)
