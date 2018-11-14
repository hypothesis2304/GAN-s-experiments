from scipy import misc
import glob

import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec
import tensorflow as tf

plt.rcParams['figure.figsize'] = (10.0, 8.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

num_channels = 3
img_size = 32
img_size_flat = img_size * img_size * num_channels
num_classes = 10
num_files_train = 5
images_per_file = 10000
num_images_train = num_files_train * images_per_file
    
def show_images(images):
    images = np.reshape(images, [images.shape[0], -1])  # images reshape to (batch_size, D)
    sqrtn = int(np.ceil(np.sqrt(images.shape[0])))
    sqrtimg = int(np.ceil(np.sqrt(images.shape[1])))

    fig = plt.figure(figsize=(sqrtn, sqrtn))
    gs = gridspec.GridSpec(sqrtn, sqrtn)
    gs.update(wspace=0.05, hspace=0.05)

    for i, img in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(img.reshape([img_size, img_size, num_channels]))
    plt.ion()
    plt.show()
    plt.pause(20)    
    return

# for image_path in glob.glob("./im/"):
#     image = misc.imread(image_path+"")
#     image = image.reshape((1,32,32,3))
#     print(image.shape)
png = []
for image_path in glob.glob("./im/*.png"):
    png.append(misc.imread(image_path))    

im = np.asarray(png)

print(im.shape)
show_images(im)