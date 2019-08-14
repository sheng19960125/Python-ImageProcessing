# Python-ImageProcessing  
## Python-Opencv  
* openCV主要提供三種邊緣檢測方式來處理Edge detection分為:Laplacian,Sobel,Canny，依技術的分類大致可以分為兩種:Laplacian原稱為Laplacian method，透過計算零交越點上光度的二階導數（detect zero crossings of the second derivative on intensity changes），而Sobel和Canny使用的則是Gradient methods（梯度原理），它是透過計算像素光度的一階導數差異（detect changes in the first derivative of intensity）來進行邊緣檢測，而這次我們要利用Canny來處理我們的影像。
** Laplacian:

## Example  ImageProcessing
### EX:抓取明信片
```
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

#讀取圖片並轉為灰階並顯示
image = cv2.imread("005.jpg")
cv2.imshow("Base", image)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#定義內核大小並應用高斯平滑處理
kernel_size = 7
blur_gray = cv2.GaussianBlur(gray,(kernel_size,kernel_size),0)
cv2.imshow("Gray", blur_gray);

#定義Canny的參數並應用
low_threshold = 1 
#high_threshold = 250
#特徵值抓最高值
ret2,th2 = cv2.threshold(blur_gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
plt.figure()
plt.subplot(222),plt.hist(blur_gray.ravel(),256)#.ravel方法将矩阵转化为一维
edges = cv2.Canny(blur_gray,low_threshold,ret2)
cv2.imshow("Canny", edges);

#確定輪廓
cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
print("I count{} card in this image".format(len(cnts)))
cards = image.copy()
cv2.drawContours(cards,cnts,-1,(0,255,0),2)

#取出圖檔
for (i, c) in enumerate(cnts):
    (x, y, w, h) = cv2.boundingRect(c)
    print("Card #{}".format(i + 1))
          
    card = image[y:y + h, x:x + w]
    #cv2.imshow("card", card)
    mask = np.zeros(image.shape[:2], dtype = "uint8")

    ((centerX, centerY), radius) = cv2.minEnclosingCircle(c)

    cv2.circle(mask, (int(centerX), int(centerY)), int(radius), 255, -1)

    mask = mask[y:y + h, x:x + w]
    cv2.imshow("Card #{}".format(i + 1), cv2.bitwise_and(card, card, mask = mask))
    
cv2.waitKey(0)
```
