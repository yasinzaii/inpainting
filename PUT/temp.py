import numpy as np

from put.PUT.image_synthesis.utils.misc import get_all_file
import os
from PIL import Image
flow_flie="/gemini/code/zhujinxian/dataset/MPI_Sintel/version2/occlusion_flow"
mask_file="/gemini/code/zhujinxian/dataset/MPI_Sintel/version2/occlusion_maskv"
flow1="/gemini/code/zhujinxian/dataset/MPI_Sintel/version2/train_data/train/flow_np"
flow2="/gemini/code/zhujinxian/dataset/MPI_Sintel/version2/train_data/val/flow_np"
mask1="/gemini/code/zhujinxian/dataset/MPI_Sintel/version2/train_data/train/maskv"
mask2="/gemini/code/zhujinxian/dataset/MPI_Sintel/version2/train_data/val/maskv"
list1="/gemini/code/zhujinxian/dataset/MPI_Sintel/version2/train_data/train/list.txt"
list2="/gemini/code/zhujinxian/dataset/MPI_Sintel/version2/train_data/val/list.txt"
import random
# Create a list of numbers from 0 to 1040
numbers = list(range(1041))
# Shuffle the list to randomize the order
random.shuffle(numbers)
# Split the list into two groups
group2 = numbers[:121]
group1 = numbers[121:1041]
print("Group 1:", group1)
print("Group 2:", group2)
#一个字符串，以最右边的'.'分割，返回一个列表
for i in range(1041):
    flow_path=flow_flie+"/"+str(i)+".npy"
    mask_path=mask_file+"/"+str(i)+".png"
    flow=np.load(flow_path)
    mask=Image.open(mask_path)
    if i in group1:
        f=flow1+"/"+str(i)+".npy"
        m =mask1+"/"+str(i)+".png"
        mask.save(m)
        np.save(f,flow)
    else:
        f=flow2+"/"+str(i)+".npy"
        m =mask2+"/"+str(i)+".png"
        mask.save(m)
        np.save(f,flow)
group1.sort()
group2.sort()
with open(list1, 'w') as file:
    for item in group1:
        file.write(f"{item}\n")

with open(list2, 'w') as file:
    for item in group2:
        file.write(f"{item}\n")