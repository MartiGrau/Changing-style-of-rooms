import cv2
import numpy as np
from matplotlib import pyplot as plt

def filter(initImg):
    img = cv2.imread(initImg)
    blur = cv2.GaussianBlur(img, (5,5),0)
    return blur