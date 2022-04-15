import cv2 as cv
from matplotlib import pyplot as plt
from dilation import dilate_this
from erosion import erode_this
from read import read

#read the image
origImg = read("images/rhino.jpg")

#BGR to RBG
rgbImg = cv.cvtColor(origImg, cv.COLOR_BGR2RGB)

#Grayscale image
gray = cv.cvtColor(rgbImg, cv.COLOR_RGB2GRAY) 

#Binary conversion
(thresh, binImg) = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)

#Dilated image
dilatedImg = dilate_this(binImg, dilationLevel=3)

#Eroded image
erodedImg = erode_this(binImg, erosionLevel=3)

#Plotting
titlesArray = ["original", "gray", "binary", "dilated", "eroded"]
images = [rgbImg, gray, binImg, dilatedImg, erodedImg]

for i in range(5):
    plt.subplot(1,5,i+1)
    plt.imshow(images[i], cmap="gray")
    plt.title(titlesArray[i])
    plt.xticks([])
    plt.yticks([])

plt.show()





