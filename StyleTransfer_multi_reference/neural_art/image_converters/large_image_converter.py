import neural_art
import chainer
import chainer.cuda
from . import multi_reference_image_converter
import numpy
import os

class LargeImageConverter(object):
    def __init__(self, texture_imgs, model, gpu, optimizer, content_weight=1, texture_weight=1):
        """
        :type converter: multi_reference_image_converter.MultiReferenceImageConverter
        """
        self.converter = neural_art.image_converters.MultiReferenceImageConverter(
            texture_imgs, gpu=gpu, content_weight=content_weight, texture_weight=1, model=model, average_pooling=True)
        self.model = model
        self.optimizer = optimizer
        self.content_weight = content_weight
        self.texture_weight = texture_weight

        if gpu >= 0:
            chainer.cuda.get_device(gpu).use()
            self.xp = chainer.cuda.cupy
            self.model.model.to_gpu()
        else:
            self.xp = numpy

    def convert_debug(self, content_img, init_img, output_directory,
                      max_iteration=1000, debug_span=100, random_init=False,
                      xsplit=3, ysplit=3, overwrap=50, average_pooling=False):
        init_array = self.xp.array(neural_art.utility.img2array(init_img))
        content_array = neural_art.utility.img2array(content_img)
        if random_init:
            init_array = self.xp.random.uniform(-20, 20, init_array.shape, dtype=init_array.dtype)

        subrects = []
        ### (step-wrap)*(split-1) = w-step
        xstep = (init_array.shape[2]+(xsplit-1)*overwrap-1) / xsplit
        ystep = (init_array.shape[3]+(ysplit-1)*overwrap-1) / ysplit
        for x in range(0, init_array.shape[2]-xstep, xstep-overwrap):
            for y in range(0, init_array.shape[3]-ystep, ystep-overwrap):
                subrects.append((x, y, x+xstep, y+ystep))

        rects_content_layers = []
        target_texture_ratios = []
        for x1, y1, x2, y2 in subrects:
            subimg = self.xp.asarray(content_array[:, :, x1:x2, y1:y2])
            layers = self.model.forward_layers(chainer.Variable(subimg, volatile=True))
            texture_feature = self.converter._to_texture_feature(layers)
            target_texture_ratio = self.converter.optimize_texture_feature(texture_feature)
            target_texture_ratios.append(target_texture_ratio)

        parameter_now = chainer.links.Parameter(init_array)
        self.optimizer.setup(parameter_now)
        for i in xrange(max_iteration+1):
            neural_art.utility.print_ltsv({"iteration": i})
            if i % debug_span == 0 and i > 0:
                print("save")
                neural_art.utility.array2img(chainer.cuda.to_cpu(parameter_now.W.data)).save(
                    os.path.join(output_directory, "{}.png".format(i)))
            parameter_now.zerograds()
            for (x1, y1, x2, y2), target_texture_ratio in zip(subrects, target_texture_ratios):
                subimg = self.xp.asarray(content_array[:, :, x1:x2, y1:y2])
                contents_layers = self.model.forward_layers(chainer.Variable(subimg, volatile=True))
                contents_layers = [
                    chainer.Variable(layer.data) for layer in contents_layers
                ]

                x = chainer.Variable(self.xp.ascontiguousarray(parameter_now.W.data[:, :, x1:x2, y1:y2]))
                layers = self.model.forward_layers(x, average_pooling=average_pooling)
                texture_feature = self.converter._to_texture_feature(layers)
                target_texture_feature = self.converter._constructed_feature(target_texture_ratio)
                loss_texture = self.converter.squared_error(
                    texture_feature,
                    target_texture_feature
                )
                loss_content = self.converter._contents_loss(layers, contents_layers)
                loss = self.texture_weight * loss_texture + self.content_weight * loss_content
                loss.backward()
                parameter_now.W.grad[:, :, x1:x2, y1:y2] += x.grad
            self.optimizer.update()
        return neural_art.utility.array2img(chainer.cuda.to_cpu(parameter_now.W.data))



