from __future__ import print_function, division
import tensorflow as tf
import numpy as np
import scipy.misc
import os
import glob



def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    return session
#########################################
## MAKE YOUR DATA READY HERE
########################################

path = "/home/badri/Desktop/football2_igl/train/"

with get_session() as sess:
	for folder in os.listdir(path):
		newpath = os.path.join(path + "/" + str(folder))
		os.chdir(newpath)
		for file in glob.glob("*.jpg"):
			name = file
			image = scipy.misc.imread(file)
			cropped = tf.image.resize_images(image, [256, 256])
			cropped = cropped.eval() #
			scipy.misc.imsave(str(name), cropped)
				