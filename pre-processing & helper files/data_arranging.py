import numpy as np
import scipy.misc
import os
import glob
import shutil

path = "/home/badri/Documents/lfw-deepfunneled"
destination = "/home/badri/Documents/faces_dataset"

for root, dirs, files in os.walk(path):
	for file in files:
		path_file = os.path.join(root,file)
		shutil.copy2(path_file, destination)