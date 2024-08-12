import numpy as np
import matplotlib.pyplot as plt
def draw_diff(ori_flow, generate_flow, save_path_u,save_path_v):
    ori_flow_u=ori_flow[:,:,0]
    ori_flow_v=ori_flow[:,:,1]
    generate_flow_u=generate_flow[:,:,0]
    generate_flow_v=generate_flow[:,:,1]
    #计算两个flow的差值
    diff_u=generate_flow_u-ori_flow_u
    diff_v=generate_flow_v-ori_flow_v
    #计算两个flow的差值的绝对值
    diff_u=np.abs(diff_u)
    diff_v=np.abs(diff_v)
    #绘制图片
    # 创建一个图形窗口
    plt.figure(figsize=(6, 6))
    # 绘制热力图
    plt.imshow(diff_u, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Difference Intensity u')
    plt.savefig(save_path_u)
    # 创建一个图形窗口
    plt.figure(figsize=(6, 6))
    # 绘制热力图
    plt.imshow(diff_v, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Difference Intensity v')
    plt.savefig(save_path_v)