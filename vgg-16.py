from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
import matplotlib.pyplot as plt

img_path = './301.jpg'
img = image.load_img(img_path,target_size = (256,256))
plt.imshow(img)
plt.show()
plt.ion()
model = VGG16(include_top = True,weights = 'imagenet')
print (type(model))
