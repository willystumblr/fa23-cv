import cv2
import numpy as np


# image read
img = cv2.imread('../data/h3wb/reimages/S1_Directions.54138969_000021.jpg')
cv2.imshow('ImageWindow', img)
cv2.waitKey(0)


# convert color space
converted_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


# set parameters for superpixel segmentation
num_superpixels = 400  # desired number of superpixels
num_iterations = 4     # number of pixel level iterations. The higher, the better quality
prior = 2              # for shape smoothing term. must be [0, 5]
num_levels = 4
num_histogram_bins = 5 # number of histogram bins
height, width, channels = converted_img.shape

print("converted_img.shape",converted_img.shape)


# initialize SEEDS algorithm
seeds = cv2.ximgproc.createSuperpixelSEEDS(width, height, channels, num_superpixels, num_levels, prior, num_histogram_bins)


# run SEEDS
seeds.iterate(converted_img, num_iterations)


# get number of superpixel
num_of_superpixels_result = seeds.getNumberOfSuperpixels()
print('Final number of superpixels: %d' % num_of_superpixels_result)


# retrieve the segmentation result
labels = seeds.getLabels() # height x width matrix. Each component indicates the superpixel index of the corresponding pixel position

print("label.shaped",labels.shape)
print(labels)

labels = np.array(labels)
labels = labels[np.newaxis,:,:]
print(labels.shape)
print(labels)
# draw contour
mask = seeds.getLabelContourMask(False)
cv2.imshow('MaskWindow', mask)
cv2.waitKey(0)


# draw color coded image
color_img = np.zeros((height, width, 3), np.uint8)
color_img[:] = (0, 0, 255)
mask_inv = cv2.bitwise_not(mask)
result_bg = cv2.bitwise_and(img, img, mask=mask_inv)
result_fg = cv2.bitwise_and(color_img, color_img, mask=mask)
result = cv2.add(result_bg, result_fg)
cv2.imshow('ColorCodedWindow', result)
cv2.waitKey(0)


cv2.destroyAllWindows()

