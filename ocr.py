import pytesseract
import requests
from PIL import Image
from PIL import ImageFilter
from io import StringIO
import cv2
import numpy as np

def process_image(url):
    image = _get_image(url)
    custom_config = r'--oem 4 --psm 6'

    image = cv2.imread(image)
    image = imutils.resize(image, height=400)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (28, 4))
    sq_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (12,12))

    #smooth img
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rect_kernel)
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rect_kernel)

    #Compute Scharr Gradient
    gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradX = np.absolute(gradX)
    (minVal, maxVal) = (np.min(gradX), np.max(gradX))
    gradX = (255 * ((gradX - minVal) / (maxVal - minVal))).astype("uint8")

    #apply closing operation
    gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rect_kernel)
    thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    #perform another closing operation
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, sq_kernel)
    thresh2 = cv2.erode(thresh, None, iterations=1)
    thresh3 = cv2.dilate(thresh, None, iterations=11)

    #set borders
    p = int(img.shape[1] * 0.001)
    thresh3[:, 0:p] = 0
    thresh3[:, img.shape[1] - p:] = 0

    #find contours
    cnts = cv2.findContours(thresh3.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    #loop over contours
    for c in cnts:
        #compute bbox
        (x, y, w, h) = cv2.boundingRect(c)
        aspect_ratio = w / float(h)
        crWidth = w / float(gray.shape[1])
        maxAR = np.max(aspect_ratio)
        maxCrWidth = np.max(crWidth)

        #check aspect ratio
        if aspect_ratio > 3 and crWidth > 0.75:
            pX = int((x + w) * 0.03)
            pY = int((y + h) * 0.03)
            (x, y) = (x - pX, y - pY)
            (w, h) = (w + (pX * 2), h + (pY * 2))

        #extract ROI
        roi = img[y:y + h, x:x + w].copy()
        rs = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        result = pytesseract.image_to_string(roi, config=custom_config)

    return result


def _get_image(url):
    return Image.open(StringIO(requests.get(url).content))