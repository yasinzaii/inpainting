#创建一个工具类
import numpy as np

#使用归一化进行转化
class Normalize_Transform(object):
    def flow_to_image(self,  flow):
        # 把data2D变成3通道的numpy，其中多出来的通道为第一个通道的值
        flow = np.concatenate((flow, np.expand_dims(flow[:, :, 0], axis=2)), axis=2)
        #获取flow的最大的绝对值
        max_flow = np.max(np.abs(flow))
        #归一化
        flow = flow / max_flow
        flow = (flow+1)/2
        flow = flow.astype(np.float32)
        img = flow * 255.0
        img = img.astype(np.float32)
        return img, max_flow

    def image_to_flow(self, img, max_flow):
        #与flow_to_image相反，转化回flow
        img = img / 255.0
        img = img * 2 - 1
        flow = img[:, :, :2]
        flow = flow * max_flow
        flow=flow.astype(np.float32)
        return flow

if __name__ == "__main__":
     for i in range(1041):
          ori_image_path="/gemini/code/zhujinxian/dataset/MPI_Sintel/version2/ori_image/"+str(i)+".npy"
          flow_path="/gemini/code/zhujinxian/dataset/MPI_Sintel/version2/ori_flow/"+str(i)+".npy"
          max_flow_path="/gemini/code/zhujinxian/dataset/MPI_Sintel/version2/max_flow/"+str(i)+".npy"
          ori_image=np.load(ori_image_path)
          max_flow=np.load(max_flow_path)
          flow=Normalize_Transform().image_to_flow(ori_image,max_flow)
          np.save(flow_path,flow)