from __future__ import print_function
import cv2


def apply_mask (input_img,output,mascara):
    img2gray = cv2.cvtColor(mascara,cv2.COLOR_BGR2GRAY)
    ret,mask = cv2.threshold(img2gray,10,255,cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
#-------input amb mascara i output amb mascara-1
    out_mascara = cv2.bitwise_and(output,output,mask = mask) #object with new style
    in_mascara = cv2.bitwise_and(input_img,input_img,mask = mask_inv) #other image part with real img
#------
    result = cv2.bitwise_or(in_mascara,out_mascara) #blenfing object style with real background
    cv2.imshow('result',result)
##---SAVE IMAGE
    k = cv2.waitKey(10000)
    return result
