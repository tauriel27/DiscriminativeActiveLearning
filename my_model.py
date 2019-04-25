from keras.layers import *
from keras.models import *

#定义模型
def get_my_model(shape=(64, 64, 1)):
    nclass = 2
    
    inp = Input(shape=shape)
    
    x = Convolution2D(16, (3,3), padding="same")(inp)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPool2D(strides=(2, 2))(x)
    
#     x = Convolution2D(16, (3,3), padding="same")(x)
#     x = BatchNormalization()(x)
#     x = Activation("relu")(x)
#     x = MaxPool2D(strides=(2, 2))(x)

#     x = Convolution2D(32, (3,3), padding="same")(x)
#     x = BatchNormalization()(x)
#     x = Activation("relu")(x)
# #     x = MaxPool2D(strides=(2, 2))(x)
    
    x = Convolution2D(32, (3,3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPool2D(strides=(2, 2))(x)
    
#     x = Convolution2D(64, (3,3), padding="same")(x)
#     x = BatchNormalization()(x)
#     x = Activation("relu")(x)
# #     x = MaxPool2D(strides=(2, 2))(x)
    
    x = Convolution2D(64, (3,3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPool2D(strides=(2, 2))(x)
    
    x = Convolution2D(128, (3,3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    # x = MaxPool2D(strides=(2, 2))(x)
    #
    # x = Convolution2D(256, (3,3), padding="same")(x)
    # x = BatchNormalization()(x)
    # x = Activation("relu")(x)

    
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.1)(x)
#     x = Dense(256)(x)
#     x = BatchNormalization()(x)
#     x = Dropout(0.5)(x)
#     x = Activation("relu")(x)
    out = Dense(nclass, activation='softmax', name='softmax')(x)

    model = Model(inputs=inp, outputs=out)
#    model.summary()
#    print(len(model.layers))
    return model

def get_NASNetMobile():
    from keras.applications.nasnet import NASNetMobile
    from keras.applications.mobilenetv2 import MobileNetV2

    input_tensor = Input(shape=(64, 64, 1))
    # create the base pre-trained model
    base_model = NASNetMobile(input_tensor=input_tensor, weights=None, include_top=False)
    # base_model = MobileNetV2(input_tensor=input_tensor, weights=None, include_top=False)

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # and a logistic layer -- let's say we have 200 classes
    x = Dropout(0.2)(x)
    output_tensor = Dense(2, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=input_tensor, outputs=output_tensor)

    # model.summary()
    # print(len(model.layers))
    return model

def get_MobileNetV2():
    from keras.applications.nasnet import NASNetMobile
    from keras.applications.mobilenetv2 import MobileNetV2

    input_tensor = Input(shape=(64, 64, 1))
    # create the base pre-trained model
    base_model = MobileNetV2(input_tensor=input_tensor, weights=None, include_top=False)

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # and a logistic layer -- let's say we have 200 classes
    x = Dropout(0.2)(x)
    output_tensor = Dense(2, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=input_tensor, outputs=output_tensor)

    # model.summary()
    # print(len(model.layers))
    return model

def get_my_model_1d():
#     with tf.device('/cpu:0'):
    x_in = Input(shape=(64, 1))

    x = Convolution1D(16, 3, padding='same')(x_in)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(3, strides=2)(x)

    x = Convolution1D(32, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(3, strides=2)(x)

    x = Convolution1D(64, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(3, strides=2)(x)

    x = Convolution1D(128, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling1D()(x)

    x_out = Dense(2, activation='softmax')(x)
    model = Model(inputs=x_in, outputs=x_out)

    return model