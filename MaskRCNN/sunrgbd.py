"""
Mask R-CNN
Configurations and data loading code for the SUN-RGBD dataset
This is an sub-class created using the original matterport class.
Written by Szymon Kowalewski
"""

from config import Config
import skimage.io
import utils
import os
import glob
import numpy as np
import model as modellib
from skimage import img_as_ubyte

import warnings

# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")


# Sun-RGBD Configuration Class
# It overwrites the original values from the original class
class SunRgbdConfig(Config):
	"""Base configuration class. For custom configurations, create a
	sub-class that inherits from this one and override properties
	that need to be changed.
	"""

	# Name the configurations. For example, 'COCO', 'Experiment 3', ...etc.
	# Useful if your code needs to do things differently depending on which
	# experiment is running.
	NAME = "SUNRGBD"  # Override in sub-classes

	# NUMBER OF GPUs to use. For CPU training, use 1
	GPU_COUNT = 1

	# Number of images to train with on each GPU. A 12GB GPU can typically
	# handle 2 images of 1024x1024px.
	# Adjust based on your GPU memory and image sizes. Use the highest
	# number that your GPU can handle for best performance.
	#IMAGES_PER_GPU = 8
	IMAGES_PER_GPU = 2

	# Number of training steps per epoch
	# This doesn't need to match the size of the training set. Tensorboard
	# updates are saved at the end of each epoch, so setting this to a
	# smaller number means getting more frequent TensorBoard updates.
	# Validation stats are also calculated at each epoch end and they
	# might take a while, so don't set this too small to avoid spending
	# a lot of time on validation stats.
	#STEPS_PER_EPOCH = 3000
	STEPS_PER_EPOCH = 2350

	# Number of validation steps to run at the end of every training epoch.
	# A bigger number improves accuracy of validation stats, but slows
	# down the training.
	#VALIDATION_STEPS = 2000
	VALIDATION_STEPS = 150

	# Number of classification classes (including background)
	NUM_CLASSES = 9  # For now background + chair + table

	# Input image resizing
	# Generally, use the "square" resizing mode for training and inferencing
	# and it should work well in most cases. In this mode, images are scaled
	# up such that the small side is = IMAGE_MIN_DIM, but ensuring that the
	# scaling doesn't make the long side > IMAGE_MAX_DIM. Then the image is
	# padded with zeros to make it a square so multiple images can be put
	# in one batch.
	# Available resizing modes:
	# none:   No resizing or padding. Return the image unchanged.
	# square: Resize and pad with zeros to get a square image
	#         of size [max_dim, max_dim].
	# pad64:  Pads width and height with zeros to make them multiples of 64.
	#         If IMAGE_MIN_DIM or IMAGE_MIN_SCALE are not None, then it scales
	#         up before padding. IMAGE_MAX_DIM is ignored in this mode.
	#         The multiple of 64 is needed to ensure smooth scaling of feature
	#         maps up and down the 6 levels of the FPN pyramid (2**6=64).
	# crop:   Picks random crops from the image. First, scales the image based
	#         on IMAGE_MIN_DIM and IMAGE_MIN_SCALE, then picks a random crop of
	#         size IMAGE_MIN_DIM x IMAGE_MIN_DIM. Can be used in training only.
	#         IMAGE_MAX_DIM is not used in this mode.
	IMAGE_RESIZE_MODE = "square"
	IMAGE_MIN_DIM = 512
	IMAGE_MAX_DIM = 768
	# Minimum scaling ratio. Checked after MIN_IMAGE_DIM and can force further
	# up scaling. For example, if set to 2 then images are scaled up to double
	# the width and height, or more, even if MIN_IMAGE_DIM doesn't require it.
	# Howver, in 'square' mode, it can be overruled by IMAGE_MAX_DIM.
	IMAGE_MIN_SCALE = 0

	# Maximum number of ground truth instances to use in one image
	MAX_GT_INSTANCES = 20

	# Max number of final detections
	DETECTION_MAX_INSTANCES = 20

	# Image mean (RGB)
	MEAN_PIXEL = np.array([126.38, 117.52, 111.82])

	# Length of square anchor side in pixels
	RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)

	# Learning rate and momentum
	# The Mask RCNN paper uses lr=0.02, but on TensorFlow it causes
	# weights to explode. Likely due to differences in optimzer
	# implementation.
	LEARNING_RATE = 0.001
	LEARNING_MOMENTUM = 0.9


class SunRgbdDataset(utils.Dataset):
	"""The base class for dataset classes.
	To use it, create a new class that adds functions specific to the dataset
	you want to use. For example:

	class CatsAndDogsDataset(Dataset):
		def load_cats_and_dogs(self):
			...
		def load_mask(self, image_id):
			...
		def image_reference(self, image_id):
			...

	See COCODataset and ShapesDataset as examples.
	"""

	def __init__(self, class_map=None):
		self._image_ids = []
		self.image_info = []
		# Background is always the first class
		self.class_info = [{"source": "", "id": 0, "name": "BG"}]
		self.source_class_ids = {}

	def load_sunrgbd(self, dataset_dir, dataset_type):
		"""Generate the requested number of synthetic images.
		count: number of images to generate.
		height, width: the size of the generated images.
		"""
		# Add classes
		self.add_class("sunrgbd", 1, "chair")
		self.add_class("sunrgbd", 2, "table")
		self.add_class("sunrgbd", 3, "window")
		self.add_class("sunrgbd", 4, "door")
		#self.add_class("sunrgbd", 5, "desk")
		self.add_class("sunrgbd", 5, "box")
		self.add_class("sunrgbd", 6, "shelves")
		#self.add_class("sunrgbd", 8, "bookshelf")
		self.add_class("sunrgbd", 7, "sofa")
		self.add_class("sunrgbd", 8, "cabinet")

		# Get folders
		dataset_dir = os.path.join(dataset_dir, dataset_type)
		examples_paths = sorted([os.path.join(dataset_dir,f) for f in os.listdir(dataset_dir)])
		number_of_examples = len(examples_paths)

		# Add images
		for example, example_path in enumerate(examples_paths):
			image_path = os.path.join(example_path,'rgb.jpg')
			#Probably not neccessary anymore as naming convention was changed
			#image_path = os.path.join(example_path,'*.jpg')
			#image_path = glob.glob(image_path)[0]
			self.add_image("sunrgbd", image_id=example, path=image_path)

	def load_image(self, image_id):
		"""Load the specified image and return a [H,W,3] Numpy array.
		"""
		# Get paths
		rgb_path = os.path.join(self.image_info[image_id]['path'])

		# Load images
		image_rgb = skimage.io.imread(rgb_path)

		return image_rgb

	def load_mask(self, image_id):
		"""Load instance masks for the given image.

		Different datasets use different ways to store masks. Override this
		method to load instance masks and return them in the form of am
		array of binary masks of shape [height, width, instances].

		Returns:
			masks: A bool array of shape [height, width, instance count] with
				a binary mask per instance.
			class_ids: a 1D array of class IDs of the instance masks.
		"""
		# Override this function to load a mask from your dataset.
		# Otherwise, it returns an empty mask.

		instance_masks = []
		class_ids = [] 

		labels_path = os.path.dirname(self.image_info[image_id]['path'])
		labels_path = os.path.join(labels_path, 'labels')

		#Get all .png files in the folder
		file_paths = os.path.join(labels_path,'*.png')
		file_paths = sorted(glob.glob(file_paths))

		#Add mask to instance_masks and append the class name found in the filename
		for file_path in file_paths:
			for cat in self.class_names:
				if cat in file_path:
					mask = skimage.io.imread(file_path)
					instance_masks.append(mask)
					class_ids.append(self.class_names.index(cat))
					#print("Filename loaded: ", file_path)
					#print("Class loaded: ", cat)

		#Pack instance masks into an array
		if class_ids:
			mask = np.stack(instance_masks, axis=2)
			class_ids = np.array(class_ids, dtype=np.int32)
			return mask, class_ids
		else:
			# Call super class to return an empty mask
			return super(SunRgbdDataset, self).load_mask(image_id)



############################################################
#  Training
############################################################


if __name__ == '__main__':
	import argparse

	# Download COCO trained weights from Releases if needed
	if not os.path.exists(COCO_MODEL_PATH):
		utils.download_trained_weights(COCO_MODEL_PATH)

	# Parse command line arguments
	parser = argparse.ArgumentParser(
		description='Train Mask R-CNN on SUN-RGBD.')
	parser.add_argument('--dataset', required=True,
						metavar="/path/to/dataset",
						help='Directory of the SUNRGBD dataset')
	parser.add_argument('--model', required=True,
						metavar="/path/to/weights.h5",
						help="Path to weights .h5 file")
	parser.add_argument('--logs', required=False,
						default=MODEL_DIR,
						metavar="/path/to/logs/",
						help='Logs and checkpoints directory (default=logs/)')
	args = parser.parse_args()
	print("Model: ", args.model)
	print("Dataset: ", args.dataset)
	print("Logs: ", args.logs)

	# Configurations
	config = SunRgbdConfig()
	config.display()

	# Create model
	model = modellib.MaskRCNN(mode="training", config=config,model_dir=args.logs)

	if args.model.lower() == "imagenet":
		model.load_weights(model.get_imagenet_weights(), by_name=True)
	elif args.model.lower() == "coco":
		# Load weights trained on MS COCO, but skip layers that
		# are different due to the different number of classes
		# See README for instructions to download the COCO weights
		print("Loading weights ", COCO_MODEL_PATH)
		model.load_weights(COCO_MODEL_PATH, by_name=True,
											exclude=["conv1", "mrcnn_class_logits", "mrcnn_bbox_fc",
													 "mrcnn_bbox", "mrcnn_mask"])
	elif args.model.lower() == "last":
		# Load the last model you trained and continue training
		print("Loading weights ", model.find_last()[1])
		model.load_weights(model.find_last()[1], by_name=True)
	else:
		model.load_weights(args.model, by_name=True)

	# Training dataset. Use the training set and 35K from the
	# validation set, as as in the Mask RCNN paper.
	dataset_train = SunRgbdDataset()
	dataset_train.load_sunrgbd(args.dataset, "training")
	dataset_train.prepare()

	# Validation dataset
	dataset_val = SunRgbdDataset()
	dataset_val.load_sunrgbd(args.dataset, "validation")
	dataset_val.prepare()

	# *** This training schedule is an example. Update to your needs ***
	# Training - Stage 1
	print("Training network heads")
	model.train(dataset_train, dataset_val,
				learning_rate=config.LEARNING_RATE,
				epochs=60,
				layers='all')
