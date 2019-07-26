import argparse
import sys
import os
import chainer.optimizers

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import neural_art


parser = argparse.ArgumentParser()
parser.add_argument("content_image")
parser.add_argument("texture_image")
parser.add_argument("--content_weight", type=float, default=0.005)
parser.add_argument("--gpu", type=int, default=-1)
parser.add_argument("--iteration", type=int, default=1000)
parser.add_argument("--style_xsplit", type=int, default=1)
parser.add_argument("--style_ysplit", type=int, default=1)
parser.add_argument("--content_xsplit", type=int, default=3)
parser.add_argument("--content_ysplit", type=int, default=3)
parser.add_argument("--content_overwrap", type=int, default=100)
parser.add_argument("--resize", type=int, default=300,
                    help="maximum size of height and width for content and texture images")
parser.add_argument("--out_dir", default="output")
parser.add_argument("--no_optimize", dest="optimize", action="store_false")
parser.add_argument("--output_image", default="converted.png")
parser.add_argument("--debug", action="store_true")
parser.add_argument("--debug_span", type=int, default=100)
parser.add_argument("--average_pooling", action="store_true")
parser.add_argument("--model", default="vgg")
parser.add_argument("--random_init", action="store_true")
parser.add_argument("--init_image", default=None)
args = parser.parse_args()
if args.init_image is None: args.init_image = args.content_image
print(args)
texture_img = neural_art.utility.load_image(args.texture_image)
texture_img = neural_art.utility.resize_img(texture_img, args.resize)
content_img = neural_art.utility.load_image(args.content_image)
content_img = neural_art.utility.resize_img(content_img, args.resize)
init_img = neural_art.utility.load_image(args.init_image)
init_img = neural_art.utility.resize_img(init_img, args.resize)

XSTEP = texture_img.size[0] / args.style_xsplit
YSTEP = texture_img.size[1] / args.style_ysplit
texture_imgs = []
for x_index in xrange(args.style_xsplit):
    x = x_index * XSTEP
    for y_index in xrange(args.style_ysplit):
        y = y_index * YSTEP
        texture_imgs.append(texture_img.crop([x, y, x+XSTEP, y+YSTEP]))

model = neural_art.utility.load_nn(args.model)
if not os.path.exists(args.out_dir): os.mkdir(args.out_dir)
converter = neural_art.image_converters.LargeImageConverter(
    texture_imgs, model, gpu=args.gpu, optimizer=chainer.optimizers.Adam(alpha=4.0),
    content_weight=args.content_weight, texture_weight=1)
converter.convert_debug(
    content_img, init_img=init_img,
    overwrap=args.content_overwrap,
    max_iteration=args.iteration, debug_span=args.debug_span,
    output_directory=args.out_dir, random_init=args.random_init,
    xsplit=args.content_xsplit, ysplit=args.content_ysplit).save(args.output_image)
