import os
import cv2
import shutil
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict

THERSHOLD = 0.9999
PSEUDO_NUM = 15000
NUM_CLASS = 32
NUM_EACH_CLASS = PSEUDO_NUM // NUM_CLASS
SAVE_PATH = 'pseudo_imgs_new'
TEST_PATH = 'data_copy/food32/UPMC32/test/all'

prob_file_path = '/home/zhiyuanyan/ml_project/mds5210-final/logs/mcc/default/convnext_base_1024head_50epoch_imgsize512_bs32_pesudolabel100_custom_1/submission_49_prob.csv'
label_file_path = '/home/zhiyuanyan/ml_project/mds5210-final/logs/mcc/default/convnext_base_1024head_50epoch_imgsize512_bs32_pesudolabel100_custom_1/submission_49.csv'

prob_file = pd.read_csv(prob_file_path)
label_file = pd.read_csv(label_file_path)

df = pd.concat([label_file, prob_file['prob']], 1)
# df.groupby('label').mean().sort_values(by='prob')
data_dict = defaultdict(list)
for i in range(32):
    hash_val = (
        df[df['label']==i]
        .sort_values(by='prob', ascending=False)
        .iloc[:NUM_EACH_CLASS, :]
        .Hash
        .values
        .tolist()
    )
    data_dict[i] = hash_val

# save images
for i in tqdm(range(32)):
    dest_path = os.path.join(SAVE_PATH, str(i))
    os.makedirs(
        dest_path, 
        exist_ok=True,
    )
    hash_val = data_dict[i]
    for img_name in hash_val:
        img_name = str(img_name) + '.jpg'
        test_img_path = os.path.join(TEST_PATH, img_name)
        new_img_path = os.path.join(dest_path, 'pseudo_'+img_name)
        if not os._exists(new_img_path):
            shutil.copy(test_img_path, new_img_path)

print("finish!")