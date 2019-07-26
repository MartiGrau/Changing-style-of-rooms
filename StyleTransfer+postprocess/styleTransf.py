"""
Style Transfer with masks PyTorch
=============================


**Author**: `Marti Grau <https://martigrau.github.io>`_

**Edited by**: `Marti Grau <https://github.com/martigrau>`_

Introduction
------------

This tutorial explains how to implement the `Neural-Style algorithm <https://arxiv.org/abs/1508.06576>`__
developed by Leon A. Gatys, Alexander S. Ecker and Matthias Bethge.
Neural-Style, or Neural-Transfer, allows you to take an image and
reproduce it with a new artistic style. The algorithm takes three images,
an input image, a content-image, and a style-image, and changes the input
to resemble the content of the content-image and the artistic style of the style-image.


.. figure:: /_static/img/neural-style/neuralstyle.png
   :alt: content1
"""

# content-image and its style-distance with the style-image. Now we can
# import the necessary packages and begin the neural transfer.
#
# Importing Packages and Selecting a settings.device
# -----------------------------------------
# Below is a  list of the packages needed to implement the neural transfer.
#
# -  ``torch``, ``torch.nn``, ``numpy`` (indispensables packages for
#    neural networks with PyTorch)
# -  ``torch.optim`` (efficient gradient descents)
# -  ``PIL``, ``PIL.Image``, ``matplotlib.pyplot`` (load and display
#    images)
# -  ``torchvision.transforms`` (transform PIL images into tensors)
# -  ``torchvision.models`` (train or load pre-trained models)
# -  ``copy`` (to deep copy the models; system package)

from __future__ import print_function
import settings
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models

import copy





######################################################################
# Loss Functions
# --------------
# Content Loss
# ~~~~~~~~~~~~
#
# The content loss is a function that represents a weighted version of the
# content distance for an individual layer. The function takes the feature
# maps :math:`F_{XL}` of a layer :math:`L` in a network processing input :math:`X` and returns the
# weighted content distance :math:`w_{CL}.D_C^L(X,C)` between the image :math:`X` and the
# content image :math:`C`. The feature maps of the content image(:math:`F_{CL}`) must be
# known by the function in order to calculate the content distance. We
# implement this function as a torch module with a constructor that takes
# :math:`F_{CL}` as an input. The distance :math:`\|F_{XL} - F_{CL}\|^2` is the mean square error
# between the two sets of feature maps, and can be computed using ``nn.MSELoss``.
#
# We will add this content loss module directly after the convolution
# layer(s) that are being used to compute the content distance. This way
# each time the network is fed an input image the content losses will be
# computed at the desired layers and because of auto grad, all the
# gradients will be computed. Now, in order to make the content loss layer
# transparent we must define a ``forward`` method that computes the content
# loss and then returns the layers input. The computed loss is saved as a
# parameter of the module.
#

class ContentLoss(nn.Module):
    def __init__(self, target, ):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input


######################################################################
# .. Note::
#    **Important detail**: although this module is named ``ContentLoss``, it
#    is not a true PyTorch Loss function. If you want to define your content
#    loss as a PyTorch Loss function, you have to create a PyTorch autograd function
#    to recompute/implement the gradient manually in the ``backward``
#    method.

######################################################################
# Style Loss
# ~~~~~~~~~~
#
# The style loss module is implemented similarly to the content loss
# module. It will act as a transparent layer in a
# network that computes the style loss of that layer. In order to
# calculate the style loss, we need to compute the gram matrix :math:`G_{XL}`. A gram
# matrix is the result of multiplying a given matrix by its transposed
# matrix. In this application the given matrix is a reshaped version of
# the feature maps :math:`F_{XL}` of a layer :math:`L`. :math:`F_{XL}` is reshaped to form :math:`\hat{F}_{XL}`, a :math:`K`\ x\ :math:`N`
# matrix, where :math:`K` is the number of feature maps at layer :math:`L` and :math:`N` is the
# length of any vectorized feature map :math:`F_{XL}^k`. For example, the first line
# of :math:`\hat{F}_{XL}` corresponds to the first vectorized feature map :math:`F_{XL}^1`.
#
# Finally, the gram matrix must be normalized by dividing each element by
# the total number of elements in the matrix. This normalization is to
# counteract the fact that :math:`\hat{F}_{XL}` matrices with a large :math:`N` dimension yield
# larger values in the Gram matrix. These larger values will cause the
# first layers (before pooling layers) to have a larger impact during the
# gradient descent. Style features tend to be in the deeper layers of the
# network so this normalization step is crucial.
#

def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)


######################################################################
# Now the style loss module looks almost exactly like the content loss
# module. The style distance is also computed using the mean square
# error between :math:`G_{XL}` and :math:`G_{SL}`.
#

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input




# create a module to normalize input image so we can easily put it in a
# nn.Sequential
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std



def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img,
                               content_layers,
                               style_layers):
    cnn = copy.deepcopy(cnn)

    # normalization module
    normalization = Normalization(normalization_mean, normalization_std).to(settings.device)

    # just in order to have an iterable access to or list of content/syle
    # losses
    content_losses = []
    style_losses = []

    # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
    # to put in modules that are supposed to be activated sequentially
    model = nn.Sequential(normalization)

    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # The in-place version doesn't play very nicely with the ContentLoss
            # and StyleLoss we insert below. So we replace with out-of-place
            # ones here.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            # add content loss:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # add style loss:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses



######################################################################
# Gradient Descent
# ----------------
#
# As Leon Gatys, the author of the algorithm, suggested `here <https://discuss.pytorch.org/t/pytorch-tutorial-for-neural-transfert-of-artistic-style/336/20?u=alexis-jacq>`__, we will use
# L-BFGS algorithm to run our gradient descent. Unlike training a network,
# we want to train the input image in order to minimise the content/style
# losses. We will create a PyTorch L-BFGS optimizer ``optim.LBFGS`` and pass
# our image to it as the tensor to optimize.
#

def get_input_optimizer(input_img):
    # this line to show that input is a parameter that requires a gradient
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer


######################################################################
# Finally, we must define a function that performs the neural transfer. For
# each iteration of the networks, it is fed an updated input and computes

# function, which reevaluates the modul and returns the loss.
#
# We still have one final constraint to address. The network may try to
# optimize the input with values that exceed the 0 to 1 tensor range for
# the image. We can address this by correcting the input values to be
# between 0 to 1 each time the network is run.
#

def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, num_steps,
                       style_weight, content_weight=1):
    """Run the style transfer."""
    print('Building the style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses(cnn,
                                                                     normalization_mean, normalization_std, style_img,
                                                                     content_img, settings.content_layers_default, settings.style_layers_default)
    optimizer = get_input_optimizer(input_img)

    print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:

        def closure():
            # correct the values of updated input image
            input_img.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                print()

            return style_score + content_score

        optimizer.step(closure)

    # a last correction...
    input_img.data.clamp_(0, 1)

    return input_img




def st(style_img, content_img):
    ######################################################################
    # Importing the Model
    # -------------------
    cnn = models.vgg19(pretrained=True).features.to(settings.device).eval()

    ######################################################################
    # Additionally, VGG networks are trained on images with each channel
    # normalized by mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225].
    # We will use them to normalize the image before sending it into the network.
    #

    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(settings.device)
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(settings.device)


    ######################################################################
    # Next, we select the input image. You can use a copy of the content image
    # or white noise.
    #
    input_img = content_img.clone()

    output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                       content_img, style_img, input_img, settings.num_steps, settings.style_weight)
    return output
