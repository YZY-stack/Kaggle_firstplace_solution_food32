import os
import glob
import numpy as np
import pandas as pd
from collections import Counter

label_list = []
csv_list = glob.glob(os.path.join('*.csv'))
for csv in csv_list:
    df = pd.read_csv(csv)
    label = list(df.label)
    label_list.append(label)
label_list = np.array(label_list)

res_list = []
for i in range(len(label)):
    c = Counter(label_list[:, i])
    res_list.append(c.most_common()[0][0])

df = pd.DataFrame(list(zip(list(df.Hash), np.array(res_list))), columns=['Hash', 'label'])
df.to_csv('submission_fusion_1.csv', index=False)