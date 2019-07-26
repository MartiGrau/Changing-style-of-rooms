from __future__ import print_function
from filter import smooth_filter
from low_filter import filter
from mix_mask import apply_mask
import settings
import styleTransf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import os
import glob

from PIL import Image
import matplotlib.pyplot as plt

import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

import copy
import numpy
import PIL.ImageOps
import cv2


######################################################################
#contruct the argument parser as parse the arguments
desc = "Pytorch implementation of 'Style Transfer with masks'"
parser = argparse.ArgumentParser(description=desc)

parser.add_argument('--style_path',
                    default='/imatge/mgrau/work/RESULTATS_FINALS/results_presentacio/style/classic/',
                    help='Style image path (Ex:../minimalist.jpg)')

parser.add_argument('--content_path', type=str,
                        default='/imatge/mgrau/work/RESULTATS_FINALS/results_presentacio/content/rustic/',
                        help='Content image path (Ex:../rustic.jpg)')

parser.add_argument('--mask_path', type=str,
                        default='/imatge/mgrau/work/segmentation+styleTransf/mask/mask_objects/',
                        help='Mask image path (Ex:../minimalist_mask.jpg)')

# parser.add_argument('--st_output', type=str,
#                         default='/imatge/mgrau/work/RESULTATS_FINALS/Results/ST_parameters/3/50/',
#                         help='Style Transfer output image path (Ex:../minimalist_mask.jpg)')

parser.add_argument('--result_no_filter', type=str,
                        default='/imatge/mgrau/work/RESULTATS_FINALS/results_presentacio/ST_results/classic/',
                        help='Style Transfer + mask result image path (Ex:../minimalist_mask.jpg)')

parser.add_argument('--result_filter', type=str,
                        default='/imatge/mgrau/work/RESULTATS_FINALS/results_presentacio/filter_results/classic/',
                        help='Style Transfer + mask result image path (Ex:../minimalist_mask.jpg)')

args = parser.parse_args()

######################################################################
# Next, we need to choose which device to run the network on and import the
# content and style images. Running the neural transfer algorithm on large
# images takes longer and will go much faster when running on a GPU. We can
# use ``torch.cuda.is_available()`` to detect if there is a GPU available.
# Next, we set the ``torch.device`` for use throughout the tutorial. Also the ``.to(device)``
# method is used to move tensors or modules to a desired device.

print(torch.cuda.is_available())
settings.init()
#######################################
#count number images inside  directory
num_img = sum([len(files) for r, d, files in os.walk(args.style_path)])
num_cont = sum([len(files) for r, d, files in os.walk(args.content_path)])
num_masks = sum([len(files) for r, d, files in os.walk(args.mask_path)])
# assert num_img < num_masks, \
#     "we must have more style images than masks"

######################################################################
# Loading the Images
# ------------------
#
# Now we will import the style and content images. The original PIL images have values between 0 and 255, but when
# transformed into torch tensors, their values are converted to be between
# 0 and 1. The images also need to be resized to have the same dimensions.
# An important detail to note is that neural networks from the
# torch library are trained with tensor values ranging from 0 to 1. If you
# try to feed the networks with 0 to 255 tensor images, then the activated
# feature maps will be unable sense the intended content and style.
# However, pre-trained networks from the Caffe library are trained with 0
# to 255 tensor images.

#########################################################################################################################
#desired size of the output image
# imsize = 768 if torch.cuda.is_available() else 128  # use small size if no gpu
# imsize = 400 if torch.cuda.is_available() else 128  # use small size if no gpu
imsize = 400 if torch.cuda.is_available() else 128  # use small size if no gpu

loader = transforms.Compose([
    transforms.Resize(imsize),  # scale imported image
    transforms.ToTensor()])  # transform it into a torch tensor


def image_loader(image_name):
    image = Image.open(image_name)
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    return image.to(settings.device, torch.float)

#iteration of content and style transfer images
for this_content in range(num_cont):
    content_img = image_loader(args.content_path + "{}.jpg".format(this_content + 1))
    for this_style in range(num_img):
        number_of_images = len(glob.glob(args.result_no_filter + "*.jpg"))
        style_img = image_loader(args.style_path + "{}.jpg".format(this_style+1))

        assert style_img.size() == content_img.size(), \
        "we need to import style and content images of the same size"
        output = styleTransf.st(style_img, content_img)
        torchvision.utils.save_image(output, args.result_no_filter + "output_{}.jpg".format(number_of_images + 1))

########################################################################################################################
    # POSTPROCESSING
########################################################################################################################

        #photorealism filter
        out_img = smooth_filter(args.result_no_filter +  "output_{}.jpg".format(number_of_images+1), args.content_path + "{}.jpg".format(this_content+1), f_radius=15, f_edge=1e-1)
        out_img.save(args.result_filter + "filtrat_{}.jpg".format(number_of_images+1))
#########################################################################################################################
# #APPLY MASK
# #######################################################################################################################
# content = cv2.imread(args.content_path)
# # iteration of masks images
# mascara = list()
# #without filter
# for this_mask in range(num_masks):
#     mascara = cv2.imread(args.mask_path + "mask_{}.jpg".format(this_mask+1))
#     output = cv2.imread(args.result_no_filter + "output_{}.jpg".format(this_mask+1))
#     if this_mask == 0:
#         result_no_filter = apply_mask(content, output, mascara)
#     else:
#         result_no_filter = apply_mask(result_no_filter, output, mascara)
#     cv2.imwrite(args.result_no_filter + "Rmask_{}.jpg".format(this_mask+1), result_no_filter)
# #with_filter
# for this_mask in range(num_masks):
#     mascara = cv2.imread(args.mask_path + "mask_{}.jpg".format(this_mask+1))
#     filtrat = cv2.imread(args.result_filter + "filtrat_{}.jpg".format(this_mask + 1))
#     if this_mask == 0:
#         result_filter = apply_mask(content, filtrat, mascara)
#     else:
#         result_filter = apply_mask(result_filter, filtrat, mascara)
#     cv2.imwrite(args.result_filter + "Rmask_{}.jpg".format(this_mask+1), result_filter)
#
# #save result images (with and without filtrated)
# cv2.imwrite(args.result_no_filter+ "Result.jpg", result_no_filter)
# cv2.imwrite(args.result_filter+ "Result.jpg", result_filter)

#########################################################################################################################