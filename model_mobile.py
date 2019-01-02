from keras.models import Sequential
from keras.layers import *
from keras.models import model_from_json


def import_weights(model):
    json_file = open('models/mobilenet.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    mobilenet = model_from_json(loaded_model_json)
    mobilenet.load_weights("weights/mobilenet.h5")

    # dw = []
    # pw = []
    # for layer in mobilenet.layers:
    #     if 'conv_dw' in layer.name:
    #         dw.append(layer.get_weights())
    #     elif 'conv_pw' in layer.name:
    #         pw.append(layer.get_weights())
    # print(len(dw))

    for i in range(1, 12):
        model.get_layer(name='conv_dw_%d' % i).set_weights(mobilenet.get_layer('conv_dw_%d' % i).get_weights())
        model.get_layer(name='conv_pw_%d' % i).set_weights(mobilenet.get_layer('conv_pw_%d' % i).get_weights())

    model.get_layer(name='conv1').set_weights(mobilenet.get_layer(name='conv1').get_weights())

    return model


# Neural network model : VGG + Conv
def CSRNet_M(shape=(224, 224, 3)):
    # Variable Input Size
    kernel = (3, 3)
    stride = (2, 2)
    # channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1
    # init = RandomNormal(stddev=0.01)
    model = Sequential()

    # Input Layer

    model.add(Conv2D(32, kernel_size=kernel, padding='same',use_bias=False,input_shape=shape,name='conv1'))
    model.add(BatchNormalization())
    model.add(ReLU(6.))

    # Depthwise1 32 & Pointwise1 64
    model.add(DepthwiseConv2D(kernel, padding='same',use_bias=False,name='conv_dw_1'))
    model.add(BatchNormalization())
    model.add(ReLU(6.))
    model.add(Conv2D(64, (1, 1), padding='same', use_bias=False,name='conv_pw_1'))
    model.add(BatchNormalization())
    model.add(ReLU(6.))

    # Depthwise2 64 & Pointwise2 128
    model.add(DepthwiseConv2D(kernel, strides=stride, padding='same',use_bias=False,name='conv_dw_2'))
    model.add(BatchNormalization())
    model.add(ReLU(6.))
    # shrink by 2*2
    model.add(Conv2D(128, (1, 1), padding='same', use_bias=False,name='conv_pw_2'))
    model.add(BatchNormalization())
    model.add(ReLU(6.))

    # Depthwise3 128 & Pointwise3 128
    model.add(DepthwiseConv2D(kernel, padding='same',use_bias=False,name='conv_dw_3'))
    model.add(BatchNormalization())
    model.add(ReLU(6.))
    model.add(Conv2D(128, (1, 1), padding='same', use_bias=False,name='conv_pw_3'))
    model.add(BatchNormalization())
    model.add(ReLU(6.))

    # Depthwise4 128 & Pointwise4 256
    model.add(DepthwiseConv2D(kernel, strides=stride, padding='same',use_bias=False,name='conv_dw_4'))
    model.add(BatchNormalization())
    model.add(ReLU(6.))
    # shrink by 2*2
    model.add(Conv2D(256, (1, 1), padding='same', use_bias=False,name='conv_pw_4'))
    model.add(BatchNormalization())
    model.add(ReLU(6.))

    # Depthwise5 256 & Pointwise5 256
    model.add(DepthwiseConv2D(kernel, padding='same',use_bias=False,name='conv_dw_5'))
    model.add(BatchNormalization())
    model.add(ReLU(6.))
    model.add(Conv2D(256, (1, 1), padding='same', use_bias=False,name='conv_pw_5'))
    model.add(BatchNormalization())
    model.add(ReLU(6.))

    # Depthwise6 256 & Pointwise6 512
    model.add(DepthwiseConv2D(kernel, strides=stride, padding='same',use_bias=False,name='conv_dw_6'))
    model.add(BatchNormalization())
    model.add(ReLU(6.))
    # shrink by 2*2
    model.add(Conv2D(512, (1, 1), padding='same', use_bias=False,name='conv_pw_6'))
    model.add(BatchNormalization())
    model.add(ReLU(6.))

    # Depthwise7 512 & Pointwise7 512
    model.add(DepthwiseConv2D(kernel, padding='same',use_bias=False,name='conv_dw_7'))
    model.add(BatchNormalization())
    model.add(ReLU(6.))
    model.add(Conv2D(512, (1, 1), padding='same', use_bias=False,name='conv_pw_7'))
    model.add(BatchNormalization())
    model.add(ReLU(6.))
    # Depthwise8 512 & Pointwise8 512
    model.add(DepthwiseConv2D(kernel, padding='same',use_bias=False,name='conv_dw_8'))
    model.add(BatchNormalization())
    model.add(ReLU(6.))
    model.add(Conv2D(512, (1, 1), padding='same', use_bias=False,name='conv_pw_8'))
    model.add(BatchNormalization())
    model.add(ReLU(6.))
    # Depthwise9 512 & Pointwise9 512
    model.add(DepthwiseConv2D(kernel, padding='same',use_bias=False,name='conv_dw_9'))
    model.add(BatchNormalization())
    model.add(ReLU(6.))
    model.add(Conv2D(512, (1, 1), padding='same', use_bias=False,name='conv_pw_9'))
    model.add(BatchNormalization())
    model.add(ReLU(6.))
    # Depthwise10 512 & Pointwise10 512
    model.add(DepthwiseConv2D(kernel, padding='same',use_bias=False,name='conv_dw_10'))
    model.add(BatchNormalization())
    model.add(ReLU(6.))
    model.add(Conv2D(512, (1, 1), padding='same', use_bias=False,name='conv_pw_10'))
    model.add(BatchNormalization())
    model.add(ReLU(6.))
    # Depthwise11 512 & Pointwise11 512
    model.add(DepthwiseConv2D(kernel, padding='same',use_bias=False,name='conv_dw_11'))
    model.add(BatchNormalization())
    model.add(ReLU(6.))
    model.add(Conv2D(512, (1, 1), padding='same', use_bias=False,name='conv_pw_11'))
    model.add(BatchNormalization())
    model.add(ReLU(6.))

    # Conv with Dilation
    model.add(Conv2D(512, (3, 3), activation='relu', dilation_rate=2, padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', dilation_rate=2, padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', dilation_rate=2, padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', dilation_rate=2, padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', dilation_rate=2, padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', dilation_rate=2, padding='same'))
    model.add(Conv2D(1, (1, 1), activation='relu', dilation_rate=1, padding='same'))

    return model
