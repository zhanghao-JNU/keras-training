from keras import layers
from keras.layers import  Conv2D, MaxPooling2D,ZeroPadding2D,GlobalAvgPool2D
from keras.models import Sequential

import numpy
def vgg(img_w,img_h,n_channels,weights_path=None):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(img_w,img_h,n_channels)))
    model.add(Conv2D(64, (3, 3), activation='relu',name="conv1"))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(64, (3, 3), activation='relu',name="conv2"))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128,(3, 3), activation='relu',name="conv3"))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, (3, 3), activation='relu',name="conv4"))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256,(3, 3), activation='relu',name="conv5"))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, (3, 3), activation='relu',name="conv6"))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, (3, 3), activation='relu',name="conv7"))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu',name="conv8"))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu',name="conv9"))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu',name="conv10"))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512,(3, 3), activation='relu',name="conv11"))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512,(3, 3), activation='relu',name="conv12"))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu',name="conv13"))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    #model.add(GlobalAvgPool2D(data_format="channels_last"))
    #model.load_weights('./vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')
    if weights_path:
        model.load_weights(weights_path)


    return model
def load_save_weight(img_w,img_h,n_channels):
    model = vgg(32,32,3,'./vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')
    new_model = vgg(img_w,img_h,n_channels)
    my_weights = model.get_layer('conv1').get_weights()
    final_weights = numpy.zeros((3, 3, 1500, 64))
    for i in range(64):
        a = my_weights[0][:, :, :, i]
        b = numpy.tile(a, (1, 1, 500))
        final_weights[:, :, :, i] = b
    my_weights[0] = final_weights
    new_model.get_layer('conv1').set_weights(my_weights)

    a2 = model.get_layer('conv2').get_weights()
    a3 = model.get_layer('conv3').get_weights()
    a4 = model.get_layer('conv4').get_weights()
    a5 = model.get_layer('conv5').get_weights()
    a6 = model.get_layer('conv6').get_weights()
    a7 = model.get_layer('conv7').get_weights()
    a8 = model.get_layer('conv8').get_weights()
    a9 = model.get_layer('conv9').get_weights()
    a10 = model.get_layer('conv10').get_weights()
    a11 = model.get_layer('conv11').get_weights()
    a12 = model.get_layer('conv12').get_weights()
    a13 = model.get_layer('conv13').get_weights()

    new_model.get_layer('conv2').set_weights(a2)
    new_model.get_layer('conv3').set_weights(a3)
    new_model.get_layer('conv4').set_weights(a4)
    new_model.get_layer('conv5').set_weights(a5)
    new_model.get_layer('conv6').set_weights(a6)
    new_model.get_layer('conv7').set_weights(a7)
    new_model.get_layer('conv8').set_weights(a8)
    new_model.get_layer('conv9').set_weights(a9)
    new_model.get_layer('conv10').set_weights(a10)
    new_model.get_layer('conv11').set_weights(a11)
    new_model.get_layer('conv12').set_weights(a12)
    new_model.get_layer('conv13').set_weights(a13)
    return new_model
if __name__ == "__main__":
    my_model = load_save_weight(32,32,1500)
    my_model.save_weights("./my_vgg_weights.h5")