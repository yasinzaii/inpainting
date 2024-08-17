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

class Repeat_Transform(object):
    def flow_to_image(self, flow):
        # Split the input flow into two separate arrays
        flow1 = flow[:, :, 0]
        flow2 = flow[:, :, 1]

        # Repeat the single channel three times to create two arrays with shape 256*256*3
        flow1 = np.repeat(flow1[:, :, np.newaxis], 3, axis=2)
        flow2 = np.repeat(flow2[:, :, np.newaxis], 3, axis=2)

        # Calculate the maximum flow as the square root of the sum of squares of the two channels
        max_flow = np.max(np.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2))

        # Normalize each array
        flow1 = flow1 / max_flow
        flow1 = (flow1 + 1) / 2
        flow1 = flow1.astype(np.float32)

        flow2 = flow2 / max_flow
        flow2 = (flow2 + 1) / 2
        flow2 = flow2.astype(np.float32)

        # Convert the normalized arrays to images
        img1 = flow1 * 255.0
        img1 = img1.astype(np.float32)

        img2 = flow2 * 255.0
        img2 = img2.astype(np.float32)

        return img1, img2, max_flow

    def image_to_flow(self, img1, img2, max_flow):
        # Normalize the input images
        img1 = img1 / 255.0
        img1 = img1 * 2 - 1
        img2 = img2 / 255.0
        img2 = img2 * 2 - 1
        flow1 = img1[:, :, 0]
        flow2 = img2[:, :, 0]
        # Combine the two images into a single array
        flow = np.stack((flow1, flow2), axis=2)

        # Scale the flow array by the maximum flow value
        flow = flow * max_flow
        flow = flow.astype(np.float32)

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