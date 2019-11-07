import pandas as pd
import numpy as np
train = pd.read_csv('../train.csv')
print(train.shape)
print(train.loc[train["qid"]=="9fe52a3d6f4197cde6a8"])
print(np.sum(train["target"])/train.shape[0])