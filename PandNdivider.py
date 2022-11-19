import pandas as pd
import os
import shutil

data = pd.read_csv("train_labels.csv", delimiter=',')

labels = [list(row) for row in data.values]

test_imgs = os.listdir('train_imgs')

for file, state in labels:
    if file in test_imgs:
        if state:
            shutil.move(f'train_imgs/{file}', f'positive/{file}')
        else:
            shutil.move(f'train_imgs/{file}', f'negative/{file}')