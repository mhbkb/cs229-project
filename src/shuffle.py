import pandas as pd
train = pd.read_csv('../train.csv')
train=train.sample(frac=1).reset_index(drop=True)
train.to_csv('train_shuffle1.csv', index=False)