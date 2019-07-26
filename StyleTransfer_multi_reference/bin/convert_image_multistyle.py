import argparse
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "/imatge/mgrau/PycharmProjects/multi_reference_neural_style"))
import neural_art

parser = argparse.ArgumentParser()
parser.add_argument("--content_image", default="/imatge/mgrau/work/RESULTATS_FINALS/cont/content_classic.jpg")
parser.add_argument("--texture_image_dir", default="/imatge/mgrau/work/RESULTATS_FINALS/results_presentacio/style/classic")
parser.add_argument("--content_weight", type=float, default=0.005)
parser.add_argument("--texture_weight", type=float, default=1)
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--iteration", type=int, default=100)
parser.add_argument("--xsplit", type=int, default=1)
parser.add_argument("--ysplit", type=int, default=1)
parser.add_argument("--resize", type=int, default=600,
                    help="maximum size of height and width for content and texture images")
parser.add_argument("--out_dir", default="/imatge/mgrau/work/RESULTATS_FINALS/Results/multireference/4")
parser.add_argument("--no_optimize", dest="optimize", action="store_false")
parser.add_argument("--output_image", default="/imatge/mgrau/work/RESULTATS_FINALS/Results/multireference/4/output.jpg")
parser.add_argument("--debug", action="store_true")
parser.add_argument("--debug_span", type=int, default=200)
parser.add_argument("--average_pooling", action="store_true")
parser.add_argument("--model", default="vgg_nopad")
parser.add_argument("--random_init", action="store_true")
parser.add_argument("--init_image", default=None)
parser.add_argument("--only_layer", default=None, type=int)
args = parser.parse_args()
if args.init_image is None: args.init_image = args.content_image
print(args)

texture_imgs = []
for texture_image_filename in os.listdir(args.texture_image_dir):
    texture_image_filename = args.texture_image_dir + "/" + texture_image_filename
    texture_img = neural_art.utility.load_image(texture_image_filename)
    texture_img = neural_art.utility.resize_img(texture_img, args.resize)

    XSTEP = texture_img.size[0] / (args.xsplit)
    YSTEP = texture_img.size[1] / (args.ysplit)
    if XSTEP > 100 and YSTEP > 100:
        for x_index in xrange(args.xsplit):
            x = x_index * XSTEP
            for y_index in xrange(args.ysplit):
                y = y_index * YSTEP
                texture_imgs.append(texture_img.crop([x, y, x+XSTEP, y+YSTEP]))

content_img = neural_art.utility.load_image(args.content_image)
content_img = neural_art.utility.resize_img(content_img, args.resize)
init_img = neural_art.utility.load_image(args.init_image)
init_img = neural_art.utility.resize_img(init_img, args.resize)

model = neural_art.utility.load_nn(args.model)
if not args.only_layer is None:
    for i in range(len(model.beta)):
        if not (i == args.only_layer):
            model.beta[i] = 0
converter = neural_art.image_converters.MultiReferenceImageConverter(texture_imgs, gpu=args.gpu, content_weight=args.content_weight, texture_weight=args.texture_weight, model=model, average_pooling=args.average_pooling)

if args.debug:
    if not os.path.exists(args.out_dir): os.mkdir(args.out_dir)
    debug_span = args.debug_span
else:
    debug_span = args.iteration * 2
converter.convert_debug(content_img, init_img=init_img,
                        max_iteration=args.iteration, debug_span=debug_span, output_directory=args.out_dir,
                        optimize=args.optimize, random_init=args.random_init).save(args.output_image)
