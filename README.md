# Python-ImageProcessing  
## Python-Opencv  
### openCV主要提供三種邊緣檢測方式來處理Edge detection分為 : Laplacian, Sobel, Canny  
* 依技術的分類大致可以分為兩種 : Laplacian原稱為Laplacian method，透過計算零交越點上光度的二階導數（detect zero crossings of the second derivative on intensity changes），而Sobel和Canny使用的則是Gradient methods（梯度原理），它是透過計算像素光度的一階導數差異（detect changes in the first derivative of intensity）來進行邊緣檢測，而這次我們要利用Canny來處理我們的影像。  
    * Laplacian : 此方法對於雜訊Noise非常敏感，所以在我們使用此方法之前基本上都需要使用模糊化後再使用Laplacian處理。  
    * Sobel : 此方法與Canny使用同一種底層技術處理圖片，但Sobel是以相較簡單的方法，偵測圖像上水平及縱軸光度的變化，在加權平均方式產生個點數值來決定邊緣。  
    * Canny : 雖然她不可以單獨拿來使用，但是結合模糊化後，可以比Sobel更加能處理雜訊的問題，但想當然硬體設備的需求依定相對較高。  
    * 實例演示
```
#image = 你自己帶入的相片  
#low_threshold = 低邊緣像素值  
#high_threshold = 高邊緣向素值  
#圖形在任何一點的向素質，若此值大於high_threshold，則認定它屬於邊緣像素，反則小於low_threshold則不為邊緣像素，界於中間則由城市依像素強弱來決定。  
edges  = cv2.Canny(image,low_threshold,high_threshold)
```

* 在依照上述三種方法找完邊緣後，再來就是確定Contours輪廓，它是由一連串沒間段的點所組成的曲獻，針對識別部分，此過程是屬於很重要的一個步驟。  
在取得輪廓的過程基本流程如:依照自己所需的照片比例獲取，將召喚轉換為灰階，模糊化影像(高斯平滑模糊)，接著使用Canny方法尋找邊緣，最後極為確定輪廓，並用顏色圈起，此過程極為曲輪廓的基本流程，最後在取出圖片即可。

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
cv2.imshow("輪廓圖", cards);

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
