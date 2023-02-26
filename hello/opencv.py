import cv2

im = cv2.imread("Lenna.png")
bgr = im

clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
l, a, b = cv2.split(cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB))
l2 = clahe.apply(l)
bgr = cv2.cvtColor(cv2.merge((l2, a, b)), cv2.COLOR_LAB2BGR)

# ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
# y, cr, cb = cv2.split(ycrcb)
# yequ = cv2.equalizeHist(y)
# ycrcb = cv2.merge((yequ, cr, cb))
# bgr = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

# hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
# h, s, v = cv2.split(hsv)
# vequ = cv2.equalizeHist(v)
# hsv = cv2.merge((h, s, vequ))
# bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

# cv2.normalize(bgr, bgr, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

# rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

hIM = cv2.hconcat([im, bgr])

# cv2.imshow("Lenna", im)
# cv2.imshow("Clahe", bgr)
cv2.imshow("Lenna - Original / CLAHE", hIM)

cv2.waitKey(0)
cv2.destroyWindow("Lenna")
