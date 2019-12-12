#This file presents all functions to download and load the Stanford dogs dataset

import _pickle as pickle
import os
import tarfile
import glob
import urllib.request as url
import numpy as np
from PIL import Image

def download_data():
    """
    Download the Standford dogs data (120 classes) from the website, which is approximately 757MB.
    The data (a .tar file) will be store in the ./data/ folder.
    :return: None
    """
    if not os.path.exists('./data'):
        os.mkdir('./data')
        print('Start downloading data...')
        url.urlretrieve("http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar",
                        "./data/dog_images.tar")
        print('Download complete.')
    else:
        if os.path.exists('./data/dog_images.tar'):
            print('Stanford dogs dataset package already exists.')
        
def load_data(width=224, height=224, shrinking_method = 'nearest', n_classes = 120):
    """
    Unpack the Standford dogs dataset and load the datasets.
    All images in the datasets do not have the same shape
    :param height: height of the images (if None then original height is returned)
    :param width: widtht of the images (if None then original width is returned)
    :param shrinking_method: method used to change the shape of the images, it can be 'nearest', 'bilinear', 'bicubic' 
    or  'antialias', see the documentation of pillow (resize) for more information.
    :return: x[mask,:,:,:], y[mask,], label_to_breed, breed_to_label 
    an array with shuffled data (images are arrays), corresponding labels, a dictionary to match label to breed and 
    a dictionary to match breed to label
    """
    # If the data hasn't been downloaded yet, download it first.
    if not os.path.exists('./data/dog_images.tar'):
        download_data()
    else:
        print('./data/dog_images.tar already exists. Begin extracting...')
        
    # Check if the package has been unpacked, otherwise unpack the package
    if not os.path.exists('./data/Images/'):
        package = tarfile.open('./data/dog_images.tar')
        package.extractall('./data')
        package.close()
    
    print('Standford dogs data were extracted. Begin creating dataset...')    
    # Go to the location where the files are unpacked
    root_dir = os.getcwd()
    path = './data/Images'
    os.chdir(path)
    #get all folders name (corresponding to all breed of dogs)
    folders = glob.glob('*')
    #to store all images (arrays) and their labels
    data = []
    label = []
    #dictionaries to match labels to breed
    breed_to_label = {}
    label_to_breed = {}
    #counter for classes (first breed is 1 - last breed is 120, there are only 120 classes)
    i=0
    for folder in folders:
        if i<n_classes:
            #get the breed from the name of the folder
            breed = folder.split('-')[1]
            #fill both dictionaries
            breed_to_label[breed] = i
            label_to_breed[i]=breed
            #each folder contains pictures about a specific breed
            #get the names of all those images
            images = glob.glob(folder+'/*')
            #fill data and labels with the images and their label
            #different method to change the shape of an image exists
            if shrinking_method == 'nearest':
                for image in images:
                    data.append(np.asarray(Image.open(image).resize((width, height), Image.NEAREST)))
                    label.append(i)
            if shrinking_method == 'bilinear':
                for image in images:
                    data.append(np.asarray(Image.open(image).resize((width, height), Image.BILINEAR)))
                    label.append(i)
            if shrinking_method == 'bicubic':
                for image in images:
                    data.append(np.asarray(Image.open(image).resize((width, height), Image.BICUBIC)))
                    label.append(i)
            if shrinking_method == 'antialias':
                for image in images:
                    data.append(np.asarray(Image.open(image).resize((width, height), Image.ANTIALIAS)))
                    label.append(i)
            i+=1
    #remove weird images (with depth of 4)
    data = [d for d in data if d.shape[2]==3]
    label = [label[i] for i in range(len(data)) if data[i].shape[2]==3]
    
    #cast to array
    x = np.asarray(data)
    y = np.asarray(label)
        
    print('Dataset, labels and dictionaries are loaded')
    os.chdir(root_dir)
    return x, y, label_to_breed, breed_to_label