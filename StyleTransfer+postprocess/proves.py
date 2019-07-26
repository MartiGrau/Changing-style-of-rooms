import os
import argparse

######################################################################
#contruct the argument parser as parse the arguments
desc = "Pytorch implementation of 'Style Transfer with masks'"
parser = argparse.ArgumentParser(description=desc)

parser.add_argument('--style_path',
                    default='/imatge/mgrau/work/segmentation+styleTransf/style/styles_objects/',
                    help='Style image path (Ex:../minimalist.jpg)')

args = parser.parse_args()

#count number images inside (style_images) directory
num_img = sum([len(files) for r,d, files in os.walk(args.style_path)])
#iteration of style transfer images
for this_style in range(num_img):
    print(args.style_path + "style_{}".format(this_style+1))