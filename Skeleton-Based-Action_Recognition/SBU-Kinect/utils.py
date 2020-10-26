import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
import random
import os
import glob
import scipy.ndimage.interpolation as inter

def get_datasets(dir):
    '''
    The function recursively reads all the text files from all the subfolders of a particular directory.
    What is this text file is:
        - Thousands of video clips were taken of 8 different activity classes such as approaching, departing, pushing, kicking, punching,
          exchanging an object, hugging, and shaking hands
        - In each video two persons are engaged in one action together
        - Each video is split into a sequence of multiple frames
        - Each person in a frame is represented by a skeletion that is again represented by 15 joints
        - Each joint is again represented by 3 co-ordinates
        - Hence, each video is represented by a T x N x D tensor where T is number of frames, N is number of joints of two persons and D is dimension of each joint
        - A text file simply contains this tensor data corresponding to a video clip
    '''
    pose_paths = glob.glob(os.path.join(dir, 's*', '*','*','*.txt'))  # recursively read text files from folders inside folders
    # pose_paths.sort()

    def read_txt(pose_path):
        a = pd.read_csv(pose_path,header=None).T # Each row starts with the frame number followed by N x D values. Hence Tranpose is done to keep all the frames in first row
        a = a[1:] # Removing the first row that has frames numbers in it
        return a.values

    train = {i:[] for i in range(1,9)} # Initializing for each class
    for pose_path in pose_paths:
        pose = read_txt(pose_path)
        train[int(pose_path.split('\\')[-3])].append(pose) # 4th last folder name denotes the class name

    test = {}
    for k in train:
        test[k] = [train[k].pop(random.choice(range(len(train[k]))))] # randomly get one observation for each class for test set creation and rest for train set

    return train, test


def display_sample(p):
    '''
    Details about the data: http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/README.txt
    '''
    for frame in range(p.shape[1]): # displays the frames in sequence so as to visually recognize the action
#         frame = random.choice(range(p.shape[1])) # each given observation is a sequence of frames. Hence get a random frame to display.
        cords = p[:, frame] # Get all the co-ordinates of the frame. Column number is same as frame number and each column with 90 coordinates in total.

        joint_info = {}
        for i, n in enumerate(range(0, len(cords), 3)): # coordinats are in sequence of x, y, z values, hence step of 3 is used
            joint_info[i+1] = [1280 - float(cords[n])*2560, 960 - (float(cords[n+1])*1920), (float(cords[n+2])*10000)/7.8125] # Getting original coordinates as per official description

        person_1 = {k:joint_info[k] for k in range(1,16,1)} # first 15 joints are for first person in the frame
        person_2 = {k-15:joint_info[k] for k in range(16,31,1)} # second set of 15 joints is for second person in the frame

        connect_map = [[1,2,2,2,3,3,3,3,4,5,7,8,10,11,13,14],[2,3,4,7,4,7,10,13,5,6,8,9,11,12,14,15]] # manual mapping for skeleton view of the coordinates

        plt.figure(figsize=((6,4)))

        for key, value in person_1.items():
            plt.plot(value[0], value[1], 'bo') # plotting the joints of person 1

        for m, n in zip(connect_map[0], connect_map[1]):
            plt.plot((person_1[m][0], person_1[n][0]), (person_1[m][1], person_1[n][1]), 'b--') # connecting persons 1 joins as per mapping for skeleton view

        for key, value in person_2.items():
            plt.plot(value[0], value[1], 'go') # plotting the joints of person 2

        for m, n in zip(connect_map[0], connect_map[1]):
            plt.plot((person_2[m][0], person_2[n][0]), (person_2[m][1], person_2[n][1]), 'g--') # skeleton view of person 2

        plt.xlim(-1280, 1280)
        plt.ylim(-960, 960)
        plt.pause(0.1)
        plt.clf()
        plt.show()

def auto_crop_pad(p):
    f = p.shape[0]
    if f > 16:
        r = random.randint(0, (f - 16))
        return p[r:r+16, :, :]
    elif f < 16:
        p_new = np.zeros([16,15,3], dtype=p.dtype)
        r = random.randint(0, (16 - f))
        p_new[r:r+f, :, :] = p
        return p_new
    else:
        return p

def separate_persons(p):
    p0 = np.copy(p.T[:,:45])
    p0 = p0.reshape([-1,15,3])
    p0 = auto_crop_pad(p0)

    p1 = np.copy(p.T[:,45:])
    p1 = p1.reshape([-1,15,3])
    p1 = auto_crop_pad(p1)

    return p0, p1

def mirror(p0, p1):
    p0_new = np.copy(p0)
    p1_new = np.copy(p1)
    p0_new[:,:,0] = abs(p0_new[:,:,0]-1) # Co-ordinates are normalized. Hence taking differnce with 1 on x axis will mirror the image
    p1_new[:,:,0] = abs(p1_new[:,:,0]-1) # mirroring the second person
    return p0_new, p1_new

def temporal_diff(p):
    p_diff = p[1:,:,:] - p[:-1,:,:] # Calculates difference between each consecutive frames. Output shape: 15 x 15 x 3
    p_diff = np.concatenate((p_diff, np.expand_dims(p_diff[-1,:,:], axis=0))) # Expands last array along 1st dimension from 15x3 to 1x15x3 and concatenates with above output to make it 16x15x3
    return p_diff
