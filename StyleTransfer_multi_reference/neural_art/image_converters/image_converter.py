from __future__ import print_function
import chainer
import chainer.links
import chainer.cuda
import chainer.optimizers
import chainer.functions
import neural_art
import numpy
import os
from builtins import range


class BaseImageConverter(object):
    def __init__(self, gpu=-1, optimizer=None, model=None, content_weight=1, texture_weight=1, average_pooling=False):
        self.content_weight = content_weight
        self.texture_weight = texture_weight
        self.average_pooling = average_pooling
        if optimizer is None:
            self.optimizer = chainer.optimizers.Adam(alpha=4.0)
        else:
            self.optimizer = optimizer
        if model is None:
            self.model = neural_art.utility.load_nn("vgg")
        else:
            self.model = model

        if gpu >= 0:
            chainer.cuda.get_device(gpu).use()
            self.xp = chainer.cuda.cupy
            self.model.model.to_gpu()
        else:
            self.xp = numpy

    def convert(self, content_img, init_img, iteration=1000):
        return self.convert_debug(content_img, init_img, max_iteration=iteration, debug_span=iteration + 1,
                                  output_directory=None)

    def convert_debug(self, content_img, init_img, output_directory, max_iteration=1000, debug_span=100,
                      random_init=False):
        init_array = self.xp.array(neural_art.utility.img2array(init_img))
        if random_init:
            init_array = self.xp.array(self.xp.random.uniform(-20, 20, init_array.shape), dtype=init_array.dtype)
        content_array = self.xp.array(neural_art.utility.img2array(content_img))
        content_layers = self.model.forward_layers(chainer.Variable(content_array),
                                                   average_pooling=self.average_pooling)

        parameter_now = chainer.links.Parameter(init_array)
        self.optimizer.setup(parameter_now)
        for i in range(max_iteration + 1):
            neural_art.utility.print_ltsv({"iteration": i})
            if i % debug_span == 0 and i > 0:
                print("dump to {}".format(os.path.join(output_directory, "{}.png".format(i))))
                neural_art.utility.array2img(chainer.cuda.to_cpu(parameter_now.W.data)).save(
                    os.path.join(output_directory, "{}.png".format(i)))
            parameter_now.zerograds()
            x = parameter_now.W
            layers = self.model.forward_layers(x, average_pooling=self.average_pooling)

            loss_texture = self._texture_loss(layers)
            loss_content = self._contents_loss(layers, content_layers)
            loss = self.texture_weight * loss_texture + self.content_weight * loss_content
            loss.backward()
            parameter_now.W.grad = x.grad
            self.optimizer.update()
        return neural_art.utility.array2img(chainer.cuda.to_cpu(parameter_now.W.data))

    def _contents_loss(self, layers, content_layers):
        """
        calculate content difference between original & processing
        """
        loss_contents = chainer.Variable(self.xp.zeros((), dtype=numpy.float32))
        for layer_index in range(len(layers)):
            loss_contents += numpy.float32(self.model.alpha[layer_index]) * chainer.functions.mean_squared_error(
                layers[layer_index],
                content_layers[layer_index])
        return loss_contents

    def _texture_loss(self, layers):
        """
        :param layers: predicted value of each layer
        :type layers: List[chainer.Variable]
        """
        raise Exception("Not implemented")

    def _to_texture_feature(self, layers):
        """
        :param layers: predicted value of each layer
        :type layers: List[chainer.Variable]
        """
        subvectors = []
        for layer_index in range(len(layers)):
            layer = layers[layer_index]
            beta = numpy.sqrt(numpy.float32(self.model.beta[layer_index]) / len(layers))
            texture_matrix = float(beta) * neural_art.utility.get_matrix(layer)
            texture_matrix /= numpy.sqrt(numpy.prod(texture_matrix.data.shape))  # normalize
            subvector = chainer.functions.reshape(texture_matrix, (numpy.prod(texture_matrix.data.shape),))
            subvectors.append(subvector)
        return chainer.functions.concat(subvectors, axis=0)

    def squared_error(self, f1, f2):
        loss = chainer.functions.sum((f1 - f2) * (f1 - f2))
        return loss


class ImageConverterMatrix(BaseImageConverter):
    """
    Neural art converter (naive implementation)
    just for comparison
    """

    def __init__(self, texture_img, gpu=-1, optimizer=None, model=None, content_weight=1, texture_weight=1):
        super(ImageConverterMatrix, self).__init__(gpu=gpu, optimizer=optimizer, model=model,
                                                   content_weight=content_weight, texture_weight=texture_weight)
        texture_array = self.xp.array(neural_art.utility.img2array(texture_img))
        self.texture_matrices = [neural_art.utility.get_matrix(layer) for layer in
                                 self.model.forward_layers(chainer.Variable(texture_array),
                                                           average_pooling=self.average_pooling)]

    def _texture_loss(self, layers):
        loss_texture = chainer.Variable(self.xp.zeros((), dtype=self.xp.float32))
        for layer_index in range(len(layers)):
            matrix = neural_art.utility.get_matrix(layers[layer_index])
            loss = self.xp.float32(self.model.beta[layer_index]) * chainer.functions.mean_squared_error(
                matrix,
                self.texture_matrices[layer_index]
            ) / self.xp.float32(len(layers))
            loss_texture += loss
        print("loss_texture", loss_texture.data)
        return loss_texture


class ImageConverter(BaseImageConverter):
    """
    Neural art converter with large texture feature vector
    """

    def __init__(self, texture_img, gpu=-1, optimizer=None, model=None, content_weight=1, texture_weight=1):
        super(ImageConverter, self).__init__(gpu=gpu, optimizer=optimizer, model=model, content_weight=content_weight,
                                             texture_weight=texture_weight)
        texture_array = self.xp.array(neural_art.utility.img2array(texture_img))
        self.texture_feature = self._to_texture_feature(
            self.model.forward_layers(chainer.Variable(texture_array), average_pooling=self.average_pooling))

    def _texture_loss(self, layers):
        original_feature = self._to_texture_feature(layers)
        loss_texture = self.squared_error(
            original_feature,
            self.texture_feature
        )
        print("loss_texture_feature", loss_texture.data)
        return loss_texture
