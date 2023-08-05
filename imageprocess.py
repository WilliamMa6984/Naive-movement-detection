import cv2
from matplotlib import pyplot as plt
import numpy as np
import time

def detect():
    # read images
    base = cv2.imread("base1.png", cv2.IMREAD_GRAYSCALE)
    move = cv2.imread("move1.png", cv2.IMREAD_GRAYSCALE)
    
    # calculate difference image
    move = move.astype('int8')
    base = base.astype('int8')
    diff = move - base

    notClosetozeroPos = diff > 20
    notClosetozeroNeg = diff < -20
    notClosetozero = np.logical_or(notClosetozeroPos, notClosetozeroNeg)
    diff = diff * notClosetozero

    # normalise between 0 - 255
    diff = diff.astype('float')
    diff_norm = (diff - np.min(diff)) / (np.max(diff) - np.min(diff)) * 255
    diff_norm = diff_norm.astype('uint8')

    # clean the img
    clean = notClosetozero.astype("uint8")

    kernel = np.ones((10,10),np.uint8)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel)
    kernel = np.ones((20,20),np.uint8)
    clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN, kernel)

    # normalise between 0 - 255
    clean = clean.astype('float')
    clean = (clean - np.min(clean)) / (np.max(clean) - np.min(clean)) * 255
    clean = clean.astype('uint8')

    # get bounding box
    img_boxes = clean
    detected_imgs = []

    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        [x,y,w,h] = cv2.boundingRect(cnt)

        # get images within the bounding box
        im = diff_norm[y:y+h, x:x+w]
        detected_imgs.append(im)
        # draw the bounding rectangle
        img_boxes = cv2.rectangle(diff_norm, (x, y), (x+w, y+h), (255), 2)

    return (img_boxes, detected_imgs)


if __name__ == "__main__":
    avg = 0

    for i in range(1,10):
        start = time.process_time_ns()

        # execute process
        (img_boxes, detected_imgs) = detect()

        # record process time
        end = time.process_time_ns()
        duration = (end - start) / 1000000000
        print("Duration: " + str(duration))

        avg += duration
    
    avg /= i

    print("Duration (avg): " + str(avg))

    # # show output
    # cv2.imshow("Full", img_boxes)
    # cv2.waitKey(0)

    # for im in detected_imgs:
    #     cv2.imshow("Detected", im)
    #     cv2.waitKey(0)
    
