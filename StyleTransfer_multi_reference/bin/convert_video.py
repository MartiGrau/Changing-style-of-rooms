# -*- coding: utf-8 -*-
from __future__ import print_function

import cv2
import PIL.Image
import argparse
import os
import sys
import numpy

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import neural_art

class VideoConverter(object):
    def __init__(self, frame_converter, iteration):
        """
        :type frame_converter: neural_art.image_converters.BaseImageConverter
        """
        self.frame_converter = frame_converter
        self.iteration = iteration

    def convert_video(self, video_path, output_directory, skip=0, resize=400):
        video = cv2.VideoCapture(video_path)
        video_output = None
        i = 0
        img_init = None
        while video.get(cv2.cv.CV_CAP_PROP_POS_AVI_RATIO) < 1.0:
            i += 1
            for _ in range(skip+1):
                status, bgr_img = video.read()
            img = PIL.Image.fromarray(cv2.cvtColor(
                bgr_img,
                cv2.COLOR_BGR2RGB
            ))
            img = neural_art.utility.resize_img(img, resize)
            if video_output is None:
                video_output = cv2.VideoWriter(
                    "{}/out.avi".format(output_directory),
                    fourcc=0, #raw
                    fps=video.get(cv2.cv.CV_CAP_PROP_FPS) / (skip + 1),
                    frameSize=img.size,
                    isColor=True
                )
                if(not video_output.isOpened()):
                    raise(Exception("Cannot Open VideoWriter"))
            if img_init is None:
                img_init = img
            converted_img = self.frame_converter.convert(img, init_img=img_init, iteration=self.iteration)
            converted_img.save("{}/converted_{:05d}.png".format(output_directory, i))
            img_init = converted_img
            video_output.write(cv2.cvtColor(
                numpy.asarray(converted_img),
                cv2.COLOR_RGB2BGR
            ))
        video_output.release()


parser = argparse.ArgumentParser()
parser.add_argument("video")
parser.add_argument("texture_image")
parser.add_argument("output_directory")
parser.add_argument("--gpu", type=int, default=-1)
parser.add_argument("--model", default="vgg")
parser.add_argument("--content_weight", type=float, default=0.005)
parser.add_argument("--texture_weight", type=float, default=1)
parser.add_argument("--iteration", type=int, default=1000)
parser.add_argument("--resize", type=int, default=400)
args = parser.parse_args()
print("arguments")
print(args)

try:
    os.mkdir(args.output_directory)
except:
    pass
model = neural_art.utility.load_nn(args.model)
texture_img = neural_art.utility.load_image(args.texture_image)
texture_img = neural_art.utility.resize_img(texture_img, args.resize)
frame_converter = neural_art.image_converters.MultiReferenceImageConverter(
    texture_imgs=[texture_img], gpu=args.gpu, model=model,
    content_weight=args.content_weight, texture_weight=args.texture_weight, average_pooling=True)
converter = VideoConverter(frame_converter, iteration=args.iteration)
converter.convert_video(args.video, args.output_directory, resize=args.resize)
