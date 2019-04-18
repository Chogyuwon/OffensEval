import pandas as pd
from sklearn.metrics import matthews_corrcoef, f1_score

def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels, average):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds, average=average)
    return {
        "acc": acc,
        "f1": f1,
    }



if __name__ == "__main__":
    gold_data = pd.read_csv("./datasets/TRAC-1/trac-gold-set/agr_en_fb_gold.csv", header=None, sep=",")
    pred_data = pd.read_csv("./save/TRAC_1/02/test_results.csv", header=None, sep=",")

    labels = gold_data.loc[:, 2].replace(["NAG", "CAG", "OAG"], [0, 1, 2])
    preds = pred_data.loc[:, 1].replace(["NAG", "CAG", "OAG"], [0, 1, 2])

    result = acc_and_f1(preds, labels, "macro")
    print(result)


