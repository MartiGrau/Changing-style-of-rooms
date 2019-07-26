import argparse
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import neural_art

parser = argparse.ArgumentParser()
parser.add_argument("content_image")
parser.add_argument("texture_image")
parser.add_argument("feature_type")
parser.add_argument("--content_weight", type=float, default=0.005)
parser.add_argument("--gpu", type=int, default=-1)
args = parser.parse_args()

texture_img = neural_art.utility.load_image(args.texture_image)
texture_img = neural_art.utility.resize_img(texture_img, 300)

if args.feature_type == "matrix":
    converter = neural_art.image_converters.ImageConverterMatrix(texture_img, gpu=args.gpu, content_weight=args.content_weight, texture_weight=1)
if args.feature_type == "vector":
    converter = neural_art.image_converters.ImageConverter(texture_img, gpu=args.gpu, content_weight=args.content_weight, texture_weight=1)

content_img = neural_art.utility.load_image(args.content_image)
content_img = neural_art.utility.resize_img(content_img, 300)
converter.convert(content_img, content_img, iteration=100).save("converted2.png")
