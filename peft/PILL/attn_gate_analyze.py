import os
import pandas as pd
import numpy as np

# path = './attn_gate'
paths = os.walk(r'./attn_gate')

for path, dir_lst, file_lst in paths:
    for file_name in sorted(file_lst):
        data = np.loadtxt(os.path.join(path, file_name),delimiter=None,unpack=False)
        # data = pd.read_csv(os.path.join(path, file_name))

        print(os.path.join(path, file_name), data.mean())