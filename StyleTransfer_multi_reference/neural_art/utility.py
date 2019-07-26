import os
import pickle
import neural_art
import numpy
import chainer.functions
import chainer.serializers
from PIL import Image


def load_image(img_file):
    """
    :return: Image
    """
    img = Image.open(img_file)
    if len(img.size) == 2:  # gray scale
        img_rgb = Image.new("RGB", img.size)
        img_rgb.paste(img)
        img = img_rgb
    return img


def resize_img(img, max_length):
    """
    :return: Image
    """
    orig_w, orig_h = img.size[0], img.size[1]
    if orig_w < orig_h:
        new_w = max_length * orig_w // orig_h
        new_h = max_length
    else:
        new_w = max_length
        new_h = max_length * orig_h // orig_w
    return img.resize((new_w, new_h))


def load_nn(modelname, modelpath=None):
    cachepath = "{}.dump".format(modelname)
    model = None
    if os.path.exists(cachepath):
        model = pickle.load(open(cachepath, "rb"))

    if modelname == 'vgg':
        if modelpath is None: modelpath = "/imatge/mgrau/PycharmProjects/multi_reference_neural_style/VGG_ILSVRC_16_layers.caffemodel"
        nn = neural_art.models.VGG(modelpath, no_padding=False, model=model)
    elif modelname == 'vgg_nopad':
        if modelpath is None: modelpath = "/imatge/mgrau/PycharmProjects/multi_reference_neural_style/VGG_ILSVRC_16_layers.caffemodel"
        nn = neural_art.models.VGG(modelpath, no_padding=True, model=model)
    elif modelname == 'nin':
        if modelpath is None: modelpath = "nin_imagenet.caffemodel"
        nn = neural_art.models.NIN(modelpath, model=model)
    else:
        print('invalid model name.')
        exit(1)

    # if model is None:
    #     with open(cachepath, "wb+") as f:
    #         pickle.dump(nn.model, f, pickle.HIGHEST_PROTOCOL)
    return nn


def img2array(img):
    data_subtracted = numpy.asarray(img)[:, :, :3].astype(numpy.float32) - 128
    data = data_subtracted.transpose(2, 0, 1)[::-1]
    return numpy.array([data])


def array2img(array):
    def clip(a):
        return 0 if a < 0 else (255 if a > 255 else a)

    data_added = array[0][::-1].transpose(1, 2, 0) + 128
    data = numpy.vectorize(clip)(data_added).astype(numpy.uint8)
    return Image.fromarray(data)


def get_matrix(y):
    ch = y.data.shape[1]
    w = y.data.shape[2]
    h = y.data.shape[3]
    y_2d = chainer.functions.reshape(y, (ch, w * h))
    texture_matrix = chainer.functions.matmul(y_2d, y_2d, transb=True) / numpy.float32(w * h)
    return texture_matrix


def print_ltsv(raw_dict):
    items = []
    for key, value in raw_dict.items():
        items.append("{}:{}".format(key, value))
    print("\t".join(items))
