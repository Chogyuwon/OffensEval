import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def make_dev_set(full_path, dev_set_prop):
    data = pd.read_csv(full_path, sep="\t")
    train_data, dev_data = train_test_split(data, test_size=dev_set_prop)
    return train_data, dev_data
    
def save_data(path, data):
    data.to_csv(path, index=False, sep="\t")


if __name__ == "__main__":
    file_path = "./datasets/OLIDv1.0/"
    file_name = "olid-training-v1.0.tsv"
    train_data, dev_data = make_dev_set(full_path=file_path+file_name, dev_set_prop=0.1)
    
    print("Train_data_size :", len(train_data))
    print("Dev_data_size :", len(dev_data))
    
    save_data("./datasets/OffensEval_train.tsv", train_data)
    save_data("./datasets/OffensEval_dev.tsv", dev_data)
    
    print("Make Dev set completed !")