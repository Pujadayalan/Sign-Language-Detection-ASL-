import cv2
from cvzone.HandTrackingModule import HandDetector
import math
import numpy as np

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset = 20
imgSize = 300
folder = "./Data/P"
counter = 0

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Safe cropping boundaries
        x1 = max(0, x - offset)
        y1 = max(0, y - offset)
        x2 = min(img.shape[1], x + w + offset)
        y2 = min(img.shape[0], y + h + offset)

        imgCrop = img[y1:y2, x1:x2]

        # If crop is empty → skip
        if imgCrop is None or imgCrop.size == 0:
            continue

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wGap + wCal] = imgResize
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hGap + hCal, :] = imgResize

        cv2.imshow("Image Crop", imgCrop)
        cv2.imshow("Image White", imgWhite)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)

    if key == ord('s'):   # save image
        counter += 1
        cv2.imwrite(f"{folder}/{counter}.jpg", imgWhite)
        print(counter)
