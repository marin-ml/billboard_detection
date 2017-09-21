
import cv2
import numpy as np


def grab_cut(img, rect):

    mask = np.zeros(img.shape[:2], np.uint8)

    bgdModel = np.zeros((1,65), np.float64)
    fgdModel = np.zeros((1,65), np.float64)

    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 40, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
    img2 = img*mask2[:,:,np.newaxis]

    return img2


if __name__ == '__main__':
    img_file = '1.jpg'
    img = cv2.imread(img_file)

    height, width = img.shape[:2]
    rect = (0, 0, width - 5, height - 5)

    img = grab_cut(img, rect)
    cv2.imshow('Image2', img)

    k = cv2.waitKey(0)