import argparse
import sys
import os
from builtins import range

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import neural_art

parser = argparse.ArgumentParser()
parser.add_argument("content_image")
parser.add_argument("texture_image")
parser.add_argument("--content_weight", type=float, default=0.005)
parser.add_argument("--gpu", type=int, default=-1)
parser.add_argument("--iteration", type=int, default=1000)
parser.add_argument("--xsplit", type=int, default=1)
parser.add_argument("--ysplit", type=int, default=1)
parser.add_argument("--resize", type=int, default=300,
                    help="[depricated] maximum size of height and width for content and texture images")
parser.add_argument("--resize_texture", type=int, default=None,
                    help="maximum size of height and width for texture images")
parser.add_argument("--resize_content", type=int, default=None,
                    help="maximum size of height and width for content images")
parser.add_argument("--out_dir", default="output")
parser.add_argument("--no_optimize", dest="optimize", action="store_false")
parser.add_argument("--output_image", default="converted.png")
parser.add_argument("--debug", action="store_true")
parser.add_argument("--debug_span", type=int, default=100)
parser.add_argument("--average_pooling", action="store_true")
parser.add_argument("--model", default="vgg")
parser.add_argument("--modelpath")
parser.add_argument("--random_init", action="store_true")
parser.add_argument("--init_image", default=None)
args = parser.parse_args()
if args.init_image is None: args.init_image = args.content_image
if args.resize_content is None: args.resize_content = args.resize
if args.resize_texture is None: args.resize_texture = args.resize
print(args)
texture_img = neural_art.utility.load_image(args.texture_image)
texture_img = neural_art.utility.resize_img(texture_img, args.resize_texture)

XSTEP = texture_img.size[0] / args.xsplit
YSTEP = texture_img.size[1] / args.ysplit
texture_imgs = []
for x_index in range(args.xsplit):
    x = x_index * XSTEP
    for y_index in range(args.ysplit):
        y = y_index * YSTEP
        texture_imgs.append(texture_img.crop([x, y, x + XSTEP, y + YSTEP]))

content_img = neural_art.utility.load_image(args.content_image)
content_img = neural_art.utility.resize_img(content_img, args.resize_content)
init_img = neural_art.utility.load_image(args.init_image)
init_img = neural_art.utility.resize_img(init_img, args.resize_content)

model = neural_art.utility.load_nn(args.model, modelpath=args.modelpath)
converter = neural_art.image_converters.MultiReferenceImageConverter(texture_imgs, gpu=args.gpu,
                                                                     content_weight=args.content_weight,
                                                                     texture_weight=1, model=model,
                                                                     average_pooling=args.average_pooling)

if args.debug:
    if not os.path.exists(args.out_dir): os.mkdir(args.out_dir)
    debug_span = args.debug_span
else:
    debug_span = args.iteration * 2
converter.convert_debug(content_img, init_img=init_img,
                        max_iteration=args.iteration, debug_span=debug_span, output_directory=args.out_dir,
                        optimize=args.optimize, random_init=args.random_init).save(args.output_image)
