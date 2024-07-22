# chapter9

```python
import cv2

faceCascade = cv2.CascadeClassifier('Resources/haarcascade_frontalface_default.xml')

img = cv2.imread('Resources/lena.png')

faces = faceCascade.detectMultiScale(img)

for x,y,w,h in faces:
    cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)

cv2.imshow('img', img)
cv2.waitKey(0)
```
