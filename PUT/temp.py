import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def read_png_to_numpy(image_path):
    # Open the image using PIL
    image = Image.open(image_path)

    # Convert the image to a NumPy array
    image_np = np.array(image)

    return image_np


# Example usage
image_path = '/gemini/code/zhujinxian/code/put/PUT/RESULTS/resize_occlusion_image_inpainted/completed/1007.png'
image_np = read_png_to_numpy(image_path)
mask_path = '/gemini/code/zhujinxian/code/put/PUT/RESULTS/resize_occlusion_image_inpainted/mask/1007.png'
mask_np = read_png_to_numpy(mask_path)
mask_np = np.expand_dims(mask_np, axis=-1)
im_mask = image_np * mask_np
image = Image.fromarray(im_mask.astype(np.uint8))
image