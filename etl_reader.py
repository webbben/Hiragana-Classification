#The pre-processing function that applies gaussian blur was taken from the following blog
#Blog reference:
#https://medium.com/@tomernahshon/computer-made-japanese-letters-through-variational-autoencoder-2fdb5b6a0990

#ETL-4
import bitstring

#if error with PIL module, install image module with pip:
#this will install pillow too, which is pil
#then you can get image from PIL, but you can't import image by itself
from PIL import Image, ImageEnhance
from PIL import ImageOps, ImageMath
from matplotlib import pyplot as plt

#for pre-processing and removing noise from data (gaussian blur)
import cv2

import numpy as np
import random

#needed for reading ETL1/7
import struct

#taken from ETL4 specifications - used to decode character codes
t56s = '0123456789[#@:>? ABCDEFGHI&.](<  JKLMNOPQR-$*);\'|/STUVWXYZ ,%="!'

#set info to True to print data info
def read_record_ETL4(f, pos=0, info=False):
    "gets an individual record from the file (info=True for print info)"
    f = bitstring.ConstBitStream(filename=f)
    f.bytepos = pos * 2952
    r = f.readlist('2*uint:36,uint:8,pad:28,uint:8,pad:28,4*uint:6,pad:12,15*uint:36,pad:1008,bytes:21888')
    if info:
        print('Serial Data Number:', r[0])
        print('Serial Sheet Number:', r[1])
        print('JIS Code:', r[2])
        print('EBCDIC Code:', r[3])
        print('4 Character Code:', ''.join([t56s[c] for c in r[4:8]]))
        print('Evaluation of Individual Character Image:', r[8])
        print('Evaluation of Character Group:', r[9])
        print('Sample Position Y on Sheet:', r[10])
        print('Sample Position X on Sheet:', r[11])
        print('Male-Female Code:', r[12])
        print('Age of Writer:', r[13])
        print('Industry Classification Code:', r[14])
        print('Occupation Classifiaction Code:', r[15])
        print('Sheet Gatherring Date:', r[16])
        print('Scanning Date:', r[17])
        print('Number of X-Axis Sampling Points:', r[18])
        print('Number of Y-Axis Sampling Points:', r[19])
        print('Number of Levels of Pixel:', r[20])
        print('Magnification of Scanning Lens:', r[21])
        print('Serial Data Number (old):', r[22])
    return r


#Turns images into numpy arrays
#ETL4: There are 6113 pictures (grey scale) with resolution of 76x72 pixels.
#ETL1: There are 9200 pictures, res. changed to same as ETL4
def create_data(filename, etl=4):
    "turns data into numpy arrays"
    unwanted = ['YI', 'YE', 'WI', 'WU', 'WE'] #records with these labels will be excluded from data
    deleteIndex = [] #indices of values to be deleted (for having unwanted label type)
    if etl == 4:
        data =  np.zeros((6113,76,72))
        for i in range(6113):
            r = read_record_ETL4(filename,pos=i)
            if getLabel(r) in unwanted:
                deleteIndex.append(i)
            iF = Image.frombytes('F', (r[18], r[19]), r[-1], 'bit', 4)
            iP = iF.convert('L')
            enhancer = ImageEnhance.Brightness(iP)
            iE = enhancer.enhance(r[20])
            temp=np.array(iE)
            data[i,:,:] = temp
        data = np.delete(data, deleteIndex, 0) #removes index of unwanted data
    else: #for ETL7
        data =  np.zeros((9200,76,72))
        with open(filename, 'rb') as f:
            for i in range(9200):
                f.seek(i * 2052)
                s = f.read(2052)
                r = struct.unpack('>H2sH6BI4H4B4x2016s4x', s)
                iF = Image.frombytes('F', (64, 63), r[18], 'bit', 4)
                iP = iF.convert('L') #'P'
                iP = iP.resize((72, 76)) #resize to same dim as ETL4 images
                enhancer = ImageEnhance.Brightness(iP)
                iE = enhancer.enhance(16)
                temp=np.array(iE)
                data[i,:,:] = temp
    return data


def getLabel(r):
    "gets label from file record"
    label = ''.join([t56s[c] for c in r[4:8]]).split()
    L = label[-1]
    return L

#data1 = cleaned data
#data = raw data
#preprocessing function - copied from blog
def preprocessing_data(data1, data):
    "data1 = data you process, data = raw data (unprocessed)"
    # this function cleans the images and binarize them
    kernel = np.ones((3,3),np.float32)/9
    crop_template = np.zeros((data.shape[0],data1.shape[2],data1.shape[2])) # cropping template
    for i in range(data1.shape[0]):
          dst = cv2.GaussianBlur(data1[i,:,:],(3,3),0) # smoothing
          ret,data1[i,:,:] = cv2.threshold(dst,0,255,cv2.THRESH_TOZERO+cv2.THRESH_OTSU) #binarizing
          crop_template[i,:,:] = data1[i,:72,:] # cropping
    return crop_template


def scaleData(dataset):
    "turns preprocessed pixel data into scaled values from 0 to 1"
    dataset = dataset / 160.0
    return dataset

#because ETL4 has some unwanted label types, I deleted them from the data itself.
#But the mechanism for getting the labels here refers to the original ETL data (not numpy arrays)
#so I have to exclude them here too
def getLabelInfo(file, etl=4):
    "returns (labels, classes) for given dataset, f = filename"
    unwanted = ['YI', 'YE', 'WI', 'WU', 'WE'] #ignore labels from this list
    labels = list() #holds every images label in the dataset
    classes = list() #stores and lists the number of classes found in the dataset
    if etl == 4:
        for i in range(6113): #6113 records in ETL4
            r = read_record_ETL4(file, pos=i)
            label = ''.join([t56s[c] for c in r[4:8]]).split()
            L = label[-1]
            if L not in unwanted: #exclude labels in unwanted (they are deleted from the data)
                labels.append(L)
                if L not in classes:
                    classes.append(L)
    else: #for ETL7
        with open(file, 'rb') as f:
            for i in range(9200): #we only want the first 9200 records from ETL7
                f.seek(i * 2052)
                s = f.read(2052)
                r = struct.unpack('>H2sH6BI4H4B4x2016s4x', s)
                label = r[1] #this is in bits, so will pull string out of it
                label = str(label)[2:-1].strip()
                labels.append(label)
                if label not in classes:
                    classes.append(label)
    return (labels, classes)

def shuffleData(images, labels):
    "shuffles the images and labels. return: (images, labels)"
    shuffle_list = list()
    if len(images) != len(labels):
        print("Shuffle: images and labels aren't the same size!")
    for i in range(len(labels)):
        shuffle_list.append((images[i], labels[i])) #puts into tuples
    random.shuffle(shuffle_list)
    new_images = list()
    new_labels = list()
    for i in range(len(shuffle_list)): #unpacks tuples into their lists
        new_images.append(shuffle_list[i][0])
        new_labels.append(shuffle_list[i][1])
    new_images = np.array(new_images)
    return (new_images, new_labels)
    

def printImage(img):
    "prints the given hiragana character from numpy array data"
    plt.imshow(img)
    plt.show()


