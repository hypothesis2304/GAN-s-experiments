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

path = "/home/badri/Desktop/football2_igl/train/A"

with get_session() as sess:
	os.chdir(path)
	for i in range(128):
		file1 = "G_epoch95img" + str(i) + ".jpg"
		img1 = scipy.misc.imread(file1)
		file2 = "Z2_epoch0img" + str(i) + ".jpg"
		img2 = scipy.misc.imread(file2)
		res = tf.concat([img1, img2], axis=1)
		res1 = res.eval()
		name = str(i + 1)+".jpg"
		scipy.misc.imsave(name, res1)
				


