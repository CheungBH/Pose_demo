import cv2 as cv

cap = cv.VideoCapture('../demo_assets/cut_xjl_fall.mp4')
mog = cv.createBackgroundSubtractorMOG2()
se = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
while True:
    ret, image = cap.read()
    if ret is True:
        fgmask = mog.apply(image)
        ret, binary = cv.threshold(fgmask, 220, 255, cv.THRESH_BINARY)
        binary = cv.morphologyEx(binary, cv.MORPH_OPEN, se)
        bgimage = mog.getBackgroundImage()
        cv.imshow("bgimage", bgimage)
        cv.imshow("frame", image)
        cv.imshow("fgmask", binary)
        c = cv.waitKey(50)
        if c == 27:
            break
    else:
        break

cv.destroyAllWindows()