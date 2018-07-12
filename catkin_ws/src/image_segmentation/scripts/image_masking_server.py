#!/usr/bin/env python

import sys
import os
#from image_masking.msg import maskedImage

# Root directory of the project
ROOT_DIR = '/home/aut/s160948/rgb_new/mrcnn/'
# Weights path
WEIGHTS_DIR = "/home/aut/s160948/rgb_new/mrcnn/logs/sunrgbd20180511T1720/mask_rcnn_sunrgbd_0020.h5"
# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

sys.path.insert(0, ROOT_DIR)

import rospy
import ros_numpy
import tensorflow as tf
from sensor_msgs.msg import Image
import matplotlib.pyplot as plt
import time
import numpy as np
import skimage.io
import model as modellib
import sunrgbd

from image_segmentation.srv import *
import rospy



class InferenceConfig(sunrgbd.SunRgbdConfig):
	 GPU_COUNT = 1
	 IMAGES_PER_GPU = 1

def handle_mask_image(req):
	print("Returning response")

	# Convert image message to numpy array
	np_image = ros_numpy.numpify(req.inputImage)

	# Save original timestamp and frame id
	original_stamp = req.inputImage.header.stamp
	original_id = req.inputImage.header.frame_id
  
	with graph.as_default():
		results = model.detect([np_image], verbose=1)

	r = results[0]

	checktime = time.time()

	masks_r = np.zeros((np_image.shape[0], np_image.shape[1]), dtype=np.uint8)
	masks_g = np.zeros((np_image.shape[0], np_image.shape[1]), dtype=np.uint8)
	masks_b = np.zeros((np_image.shape[0], np_image.shape[1]), dtype=np.uint8)
   
	if r['class_ids'].shape[0] != 0:

		split_masks = np.split(r['masks'], r['masks'].shape[2], 2)

		for idx, mask_img in enumerate(split_masks):
			mask_img = np.reshape(mask_img, [mask_img.shape[0], mask_img.shape[1]])
			masks_r[mask_img==1] = colors_arr[r['class_ids'][idx]][0]
			masks_g[mask_img==1] = colors_arr[r['class_ids'][idx]][1]
			masks_b[mask_img==1] = colors_arr[r['class_ids'][idx]][2]


	masks_r = np.reshape(masks_r, [masks_r.shape[0], masks_r.shape[1], 1])
	masks_g = np.reshape(masks_g, [masks_g.shape[0], masks_g.shape[1], 1])
	masks_b = np.reshape(masks_b, [masks_b.shape[0], masks_b.shape[1], 1])

	masks_all = np.append(masks_r, masks_g, 2) 
	masks_all = np.append(masks_all, masks_b, 2) 

	print("PROCESSING %s seconds ---" % (time.time() - checktime))

	msg_image = ros_numpy.msgify(Image, masks_all, encoding='rgb8')

	msg_image.header.stamp = original_stamp
	msg_image.header.frame_id = original_id

	return maskImageResponse(msg_image)

def mask_image__server():
	rospy.init_node('mask_image_server')
	s = rospy.Service('maskImage', maskImage, handle_mask_image)
	print("Ready to mask image.")
	rospy.spin()

if __name__ == "__main__":
	# Create inference configuration
	inference_config = InferenceConfig()

	# Visualization constants
	class_names = ['bg', 'chair', 'table', 'window', 'box', 'door', 'shelves', 'sofa', 'cabinet']
	colors_arr = [[255, 255, 255],
                 	  [255, 0, 0],
	      		  [0, 255, 0],
	      		  [0, 0, 255],
				  [100, 0, 0],
				  [0, 100, 0],
				  [0, 0, 100],
				  [255, 255, 0],
				  [255, 0, 255],
              	  ]

	# Recreate the model in inference mode
	model = modellib.MaskRCNN(mode="inference", 
							  config=inference_config,
							  model_dir=MODEL_DIR)

	# Get path to saved weights
	print("Loading weights from ", WEIGHTS_DIR)

	#Load model weights
	model.load_weights(WEIGHTS_DIR, by_name=True)
	
	#Save the graph to make it accessible in the subsriber
	graph = tf.get_default_graph()

	mask_image__server()