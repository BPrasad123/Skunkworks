from __future__ import division
from keras.models import Model
from keras.layers import *
from keras.layers.core import *
from keras.layers.convolutional import *
from keras import backend as K
from keras.optimizers import RMSprop as rmsprop
import tensorflow as tf

def one_obj(frame_l=16, joint_n=15, joint_d=3):
    '''
    frame_l: 16 number of frames in sequence. 16 is considered instead of 32 because of small dataset.
    joint_n: 15 number of joints in a frame
    joint_d: 3D co-ordinates of a joint

    Network architecture for SBU dataset as per the paper:
    The SBU Kinect Interaction dataset [Yun et al., 2012] is a
    Kinect captured human activity recognition dataset depicting
    two person interaction. It contains 282 skeleton sequences
    and 6822 frames of 8 classes. There are 15 joints for each
    skeleton. For evaluation we perform subject-independent 5-
    fold cross validation as suggested in [Yun et al., 2012].
    Considering the small size of the dataset, we simplify the
    network architecture in Figure 3 accordingly. Specifically, the
    output channels of conv1, conv2, conv3, conv5, conv6 and fc7
    are reduced to 32, 16, 16, 32, 64 and 64 respectively. And the
    conv4 layer is removed. Besides, all the input sequences are
    normalized to a length of 16 frames rather than 32.
    '''

    input_joints = Input(name='joints', shape=(frame_l, joint_n, joint_d))
    input_joints_diff = Input(name='joints_diff', shape=(frame_l, joint_n, joint_d))

    ########## START of branch 1 ##############
    # Purpose of 1x1 conv:
    # It does not involve neighboring values, hence makes the model learn point wise features
    x = Conv2D(filters = 32, kernel_size=(1,1),padding='same')(input_joints) #conv1
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Purpose of 3x1 conv:
    # Currently 3 is the channel in input shape of frames x joints x co-ordinates
    # 3x1 conv makes the model convolute over all 3 co-ordinates of a join without involving neighboring joints.
    x = Conv2D(filters = 16, kernel_size=(3,1),padding='same')(x) #conv2
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Purpose of permute:
    # During convoution local features are learnt in spatial domain by height and width of a kernel.
    # However, the global features are learnt when the outputs of those kernels are summed up element wise across the channel dimension
    # Right now channels are along coordinates of the joints rather than along the joints itself
    # In order to get the summation across all the joints, we need to push the joints into last dimension, that is channel.
    # Below, we are using permute simply swap between last two dimensions thereby making joints are channels
    x = Permute((1,3,2))(x)

    # Purpose of 3x3 conv:
    # Now that joints are channels, normal 3x3 conv can be done to learn more complex features involving neighboring joints
    x = Conv2D(filters = 16, kernel_size=(3,3),padding='same')(x) #conv3
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # conv4 has been deleted as per the instruction in the paper for SBU dataset

    ########## END of branch 1 ##############

    ########## START of branch 2 ############
    # Purpose of branch 2:
    # To learn temporal features explicitly and independently

    # Like in branch 1, point wise features are learnt without involving neighboring values
    x_d = Conv2D(filters = 32, kernel_size=(1,1),padding='same')(input_joints_diff) # branch2 conv1
    x_d = BatchNormalization()(x_d)
    x_d = ReLU()(x_d)

    # Convolution on 3 coordinates at each joint independently.
    x_d = Conv2D(filters = 16, kernel_size=(3,1),padding='same')(x_d) # branch conv2
    x_d = BatchNormalization()(x_d)
    x_d = ReLU()(x_d)

    # Moving the joints to channel dimension, to learn global features of temporal features across joints simultaneously
    x_d = Permute((1,3,2))(x_d)

    # 3x3 conv to learn more features involving neighboring values
    x_d = Conv2D(filters = 16, kernel_size=(3,3),padding='same')(x_d)
    x_d = BatchNormalization()(x_d)
    x_d = ReLU()(x_d)
    ########## END of branch 2 ##############

    # Outputs from both branch 1 and 2 are concatenated for the next step. Temporal features are explicitly learnts in branch 2.
    x = concatenate([x,x_d],axis=-1)

    # 3x3 conv to learn more features involving neighboring values
    x = Conv2D(filters = 32, kernel_size=(3,3),padding='same')(x) #conv5
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPool2D(pool_size=(2, 2))(x)
    x = Dropout(0.1)(x)

    # 3x3 conv to learn more features involving neighboring values
    x = Conv2D(filters = 64, kernel_size=(3,3),padding='same')(x) #conv6
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPool2D(pool_size=(2, 2))(x)
    x = Dropout(0.1)(x)

    model = Model([input_joints,input_joints_diff],x)

    return model

def multi_obj(frame_l=16, joint_n=15, joint_d=3):
    inp_j_0 = Input(name='inp_j_0', shape=(frame_l, joint_n, joint_d))
    inp_j_diff_0 = Input(name='inp_j_diff_0', shape=(frame_l, joint_n, joint_d))

    inp_j_1 = Input(name='inp_j_1', shape=(frame_l, joint_n, joint_d))
    inp_j_diff_1 = Input(name='inp_j_diff_1', shape=(frame_l, joint_n, joint_d))

    single = one_obj()
    x_0 = single([inp_j_0,inp_j_diff_0]) # one_obj network learns the features of person 1
    x_1 = single([inp_j_1,inp_j_diff_1]) # Similary features are learnt for person 2

    x = Maximum()([x_0,x_1]) # Fusion of features from boths the persons are done. Maximum option gave best result when compared against average and concat

    x = Flatten()(x) # converting to 1D array
    x = Dropout(0.1)(x)

    x = Dense(64)(x) # as per paper number of output channels from fully connected layer is 64
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Dropout(0.1)(x)

    x = Dense(8, activation='sigmoid')(x)

    model = Model([inp_j_0,inp_j_diff_0,inp_j_1,inp_j_diff_1],x)

    return model
