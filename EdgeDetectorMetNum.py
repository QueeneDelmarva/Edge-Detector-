#!/usr/bin/env python
# coding: utf-8

# In[3]:


# Edge Detector with matplotlib, numpy, and opencv 

import cv2
import numpy as np 

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Importing the image data into Numpy Arrays 
moon_image = mpimg.imread('/Users/queene/Desktop/moon.jpg')
print(moon_image)

# Show the image
imgplot = plt.imshow(moon_image)

# Grayscale the image
moon_gray = moon_image[:, :, 0]

plt.imshow(moon_gray, cmap='gray')

# Sobel Operator with OpenCV
sobel_x = cv2.Sobel(moon_gray, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(moon_gray, cv2.CV_64F, 0, 1, ksize=3)

cv2.imshow("Sobel_x", sobel_x)
cv2.imshow("Sobel_y", sobel_y)

# Laplacian Operator with OpenCV
laplacian = cv2.Laplacian(moon_gray, cv2.CV_64F, ksize=3)

cv2.imshow("Laplacian", laplacian)

# Canny Operator 
canny = cv2.Canny(moon_gray, 100, 150)

cv2.imshow("Canny", canny)

cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:




