#the purpose of this file is to create other ANN

from keras.optimizers import RMSprop
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, DepthwiseConv2D, AveragePooling2D, Activation
from keras.layers import BatchNormalization, Dropout
from keras import backend as K    

def convolution(model, alpha, n_filters, kernel_size, strides, reg, add_batch_norm, input_shape = None):
    """
    add a conv2D layer to the model
    :model : keras model to which we want to add a layer
    :alpha : shrinking parameter
    :n_filters : number of filters, the number of channels after this layer will be int(alpha*n_filters)
    :kernel_size: shape of the kernel for the convolution
    :strides : stride of the convolution
    :reg : regularization for the convolution layer, must be of type keras.regularizers
    :add_batch_norm : if True add batch normalization before the activation function
    :input_shape : shape of the input (height, width, channels) (useful only for the first convolution)
    return the model with one more convolution layer
    """
    if input_shape:
        model.add(Conv2D(int(alpha*n_filters), kernel_size, strides=strides, padding='same', activation=None,
                                input_shape=input_shape, kernel_regularizer = reg))
        if add_batch_norm:
            model.add(BatchNormalization())
        model.add(Activation('relu'))
    else:
        model.add(Conv2D(int(alpha*n_filters), kernel_size, strides=strides, padding='same', activation=None, 
                         kernel_regularizer = reg))
        if add_batch_norm:
            model.add(BatchNormalization())
        model.add(Activation('relu'))
    return model

def depthwise_convolution(model, kernel_size, strides, reg, add_batch_norm):
    """
    add a DepthwiseConv2D layer to the model
    :model : keras model to which we want to add a layer
    :kernel_size: shape of the kernel for the convolution
    :strides : stride of the convolution
    :reg : regularization for the depthwise convolution layer, must be of type keras.regularizers
    :add_batch_norm : if True add batch normalization before the activation function
    return the model with one more depthwise convolution layer
    """
    model.add(DepthwiseConv2D(kernel_size = kernel_size, padding='same', strides=strides, activation=None, 
                              depthwise_regularizer = reg))
    if add_batch_norm:
        model.add(BatchNormalization())
    model.add(Activation('relu'))
    return model

def separable_convolution(model, kernel_size_1, kernel_size_2, strides_1, strides_2, alpha, 
                          n_filters, reg_dc, reg_c, add_batch_norm):
    """
    add a Separable Convolution layer to the model (depthwise + convolution)
    :model : keras model to which we want to add a layer
    :kernel_size_1: shape of the kernel for the depthwise convolution
    :kernel_size_2: shape of the kernel for the convolution
    :strides_1 : stride of the depthwise convolution
    :strides_2 : stride of the convolution
    :alpha : shrinking parameter
    :n_filters : number of filters, the number of channels after this layer will be int(alpha*n_filters)
    :reg_dc : regularization for the depthwise convolution layer, must be of type keras.regularizers
    :reg_c : regularization for the convolution layer, must be of type keras.regularizers
    :add_batch_norm : if True add batch normalization before each activation function
    return the model with one more depthwise convolution layer
    """
    depthwise_convolution(model, kernel_size_1, strides_1, reg_dc, add_batch_norm)
    convolution(model, alpha, n_filters, kernel_size_2, strides_2, reg_c, add_batch_norm)
    return model

def MobileNetV2(input_shape, 
              n_classes = 60, 
              alpha = 1, 
              reg_c = None,
              reg_dc = None,
              add_batch_norm = True, 
              add_dropout = False, 
              dropout_rate = 0.5, 
              optimizer = RMSprop(lr=0.001, rho=0.9)):
    """
    :input_shape: shape of the input (height, width, channels)
    :n_classes: number of classes under study
    :alpha: width multiplier in (0,1] to shrink the number of channels of all convolutions (to decrease the number of parameters of the model)
    :reg_c: regularizer for each conv2D layer, must be of the form keras.regularizers.l1 or l2
    :reg_dc: regularizer for each DepthwiseConv2D layer (must be smaller because there are only a few parameters)
    :add_batch_norm : if True add batch normalization before each activation function
    :add_dropout: if True add dropout afte each Dense layer
    :dropout_rate: rate for the dropout
    :optimizer: optimizer for the back propagation (by default RMSprop as in the original paper)
    return the mobilNet ANN
    """
    
    #clear previous session (equivalent to tf.reset_default_graph())
    K.clear_session()
    
    #create the sequential model
    model = Sequential()
        
    #add all layers of the MobileNet ANN
    convolution(model, alpha, 32, (3,3), 1, reg_c, add_batch_norm, input_shape)
    separable_convolution(model, (3,3), (1,1), 1, 1, alpha, 64, reg_dc, reg_c, add_batch_norm)
    separable_convolution(model, (3,3), (1,1), 1, 1, alpha, 128, reg_dc, reg_c, add_batch_norm)
    separable_convolution(model, (3,3), (1,1), 1, 1, alpha, 128, reg_dc, reg_c, add_batch_norm)
    separable_convolution(model, (3,3), (1,1), 1, 1, alpha, 256, reg_dc, reg_c, add_batch_norm)
    separable_convolution(model, (3,3), (1,1), 1, 1, alpha, 256, reg_dc, reg_c, add_batch_norm)
    separable_convolution(model, (3,3), (1,1), 1, 1, alpha, 512, reg_dc, reg_c, add_batch_norm)
    for i in range(5):
        separable_convolution(model, (3,3), (1,1), 1, 1, alpha, 512, reg_dc, reg_c, add_batch_norm)
    separable_convolution(model, (3,3), (1,1), 2, 1, alpha, 1024, reg_dc, reg_c, add_batch_norm)
    separable_convolution(model, (3,3), (1,1), 1, 1, alpha, 1024, reg_dc, reg_c, add_batch_norm)

    shape = model.output_shape[1]

    model.add(AveragePooling2D(pool_size=(shape,shape), strides=1))
    if add_batch_norm:
        model.add(BatchNormalization())

    model.add(Flatten())
    if add_dropout:
        model.add(Dropout(rate = dropout_rate))
    
    model.add(Dense(1000, activation='relu'))
    if add_dropout:
        model.add(Dropout(rate = dropout_rate))
    if add_batch_norm:
        model.add(BatchNormalization())
    
    model.add(Dense(n_classes, activation='softmax'))
    
    #Compile model (add loss, optimizer and metrics)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    return model