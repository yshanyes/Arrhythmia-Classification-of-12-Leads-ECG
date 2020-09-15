from keras.models import Model
from keras.layers import Activation, Convolution1D,Convolution2D, Dropout, GlobalAveragePooling1D,GlobalAveragePooling2D, Concatenate, Dense, Input, AveragePooling1D,AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.layers import Conv1D, BatchNormalization, Activation, AveragePooling1D, Dense,MaxPooling1D
from keras.layers import Dropout, Concatenate, Flatten, Lambda,concatenate
from keras import regularizers
from keras.layers import Reshape, CuDNNLSTM, Bidirectional
import keras

def ResNet_model(input1,resBlock_num=15):
    # Add CNN layers left branch (higher frequencies)
    # Parameters from paper
    INPUT_FEAT = 12
    OUTPUT_CLASS = 9    # output classes

    k = 1    # increment every 4th residual block
    p = True # pool toggle every other residual block (end with 2^8)
    convfilt = 32
    convstr = 1
    ksize = 16
    poolsize = 2
    poolstr  = 2
    drop = 0.5
    
    # Modelling with Functional API
    #input1 = Input(shape=(None,1), name='input')

    #input1 = Input(shape=input_shape, name='input')
    
    ## First convolutional block (conv,BN, relu)
    x = Conv1D(filters=convfilt,
               kernel_size=ksize,
               padding='same',
               strides=convstr,
               kernel_initializer='he_normal')(input1)                
    x = BatchNormalization()(x)        
    x = Activation('relu')(x)  
    
    ## Second convolutional block (conv, BN, relu, dropout, conv) with residual net
    # Left branch (convolutions)
    x1 =  Conv1D(filters=convfilt,
               kernel_size=ksize,
               padding='same',
               strides=convstr,
               kernel_initializer='he_normal')(x)      
    x1 = BatchNormalization()(x1)    
    x1 = Activation('relu')(x1)
    x1 = Dropout(drop)(x1)
    x1 =  Conv1D(filters=convfilt,
               kernel_size=ksize,
               padding='same',
               strides=convstr,
               kernel_initializer='he_normal')(x1)
    x1 = MaxPooling1D(pool_size=poolsize,
                      strides=poolstr)(x1)
    # Right branch, shortcut branch pooling
    x2 = MaxPooling1D(pool_size=poolsize,
                      strides=poolstr)(x)
    # Merge both branches
    x = keras.layers.add([x1, x2])
    del x1,x2
    
    ## Main loop
    p = not p 
    for l in range(resBlock_num):#15
        
        if (l%4 == 0) and (l>0): # increment k on every fourth residual block
            k += 1
             # increase depth by 1x1 Convolution case dimension shall change
            xshort = Conv1D(filters=convfilt*k,kernel_size=1)(x)
        else:
            xshort = x        
        # Left branch (convolutions)
        # notice the ordering of the operations has changed        
        x1 = BatchNormalization()(x)
        x1 = Activation('relu')(x1)
        x1 = Dropout(drop)(x1)
        x1 =  Conv1D(filters=convfilt*k,
               kernel_size=ksize,
               padding='same',
               strides=convstr,
               kernel_initializer='he_normal')(x1)        
        x1 = BatchNormalization()(x1)
        x1 = Activation('relu')(x1)
        x1 = Dropout(drop)(x1)
        x1 =  Conv1D(filters=convfilt*k,
               kernel_size=ksize,
               padding='same',
               strides=convstr,
               kernel_initializer='he_normal')(x1)        
        if p:
            x1 = MaxPooling1D(pool_size=poolsize,strides=poolstr)(x1)                

        # Right branch: shortcut connection
        if p:
            x2 = MaxPooling1D(pool_size=poolsize,strides=poolstr)(xshort)
        else:
            x2 = xshort  # pool or identity            
        # Merging branches
        x = keras.layers.add([x1, x2])
        # change parameters
        p = not p # toggle pooling

    
    # Final bit    
    x = BatchNormalization()(x)
    x = Activation('relu')(x) 

    x = AveragePooling1D(int(x.shape[1]), int(x.shape[1]))(x)

    #x = Flatten()(x)
    #x = Dense(1000)(x)
    #x = Dense(1000)(x)

    #out = Dense(OUTPUT_CLASS, activation='softmax')(x)

    #model = Model(inputs=input1, outputs=out)


    #model.compile(optimizer='adam',
    #              loss='categorical_crossentropy',
    #              metrics=['accuracy'])
    #model.summary()
    #sequential_model_to_ascii_printout(model)
    #plot_model(model, to_file='model.png')
    return x

def build_network(inputs,resBlock_num,keep_prob,num_classes):

    branches = []
    
    for i in range(int(len(inputs))):
        ld = inputs[i]
        bch = ResNet_model(ld,resBlock_num)
        #ld = Reshape((int(2560), 12))(ld)
        #bch = add_resnet_layers(ld, **params)
        #bch = add_output_layer(bch,**params)

        branches.append(bch)

    features = Concatenate(axis=1)(branches)
    features = Dropout(keep_prob, [1,len(inputs),1])(features)
    features = Flatten()(features)
    net = Dense(units=num_classes, activation='sigmoid')(features)

    return net
