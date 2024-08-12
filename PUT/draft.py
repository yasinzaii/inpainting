from put.PUT.image_synthesis.utils.misc import get_all_file
from put.PUT.image_synthesis.data.utils.flow_image_transform import Normalize_Transform
from PIL import Image
import numpy as np
def read_flo_file(file_path):
    with open(file_path, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        w = np.fromfile(f, np.int32, count=1)
        h = np.fromfile(f, np.int32, count=1)
        data = np.fromfile(f, np.float32, count=2 * w[0] * h[0])
        data2D = np.resize(data, (h[0], w[0], 2))
    return data2D

current_flie="/gemini/code/zhujinxian/dataset/MPI_Sintel/data/MPI-Sintel-training_extras/training/flow"
mask_file_list=get_all_file(current_flie,end_with="flo", path_type="relative")
print(mask_file_list)
for i in range(len(mask_file_list)):
    img_path=current_flie+"/"+mask_file_list[i]
    #根据img_path读取图片
    img = read_flo_file(img_path)
    #删除/gemini/code/zhujinxian/dataset/MPI_Sintel/data/MPI-Sintel-training_extras/training/occlusions/下png文件
    save_path="/gemini/code/zhujinxian/code/put/PUT/data/sintel/version2/occlusion_flow/"+str(i)+".npy"
    #将numpy类型的img保存到save_path
    np.save(save_path,img)
    save_path2="/gemini/code/zhujinxian/code/put/PUT/data/sintel/version2/test/"+str(i)+".png"
    img, max_flow = Normalize_Transform().flow_to_image(img)
    img=Image.fromarray(img.astype(np.uint8))
    img.save(save_path2)