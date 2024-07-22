# chapter2

```python
import cv2
import numpy as np


img = cv2.imread('Resources/lena.png')

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur_img = cv2.GaussianBlur(gray_img, (7,7), 0)

kernel = np.ones((5,5), np.uint8)

canny_img = cv2.Canny(img, 150, 200)
dilation_img = cv2.dilate(canny_img, kernel, iterations=1)
erode_img = cv2.erode(dilation_img, kernel, iterations=1)


cv2.imshow('gray', gray_img)
cv2.imshow('blur', blur_img)
cv2.imshow('canny', canny_img)
cv2.imshow('dilation', dilation_img)
cv2.imshow('erode', erode_img)

cv2.waitKey(0)
```
