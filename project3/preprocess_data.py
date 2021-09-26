''' preprocess_data.py
    Preprocessing data in STL-10 image dataset
    Benji Andrews and Azalea Yunus
    September 17, 2020
    CS343: Neural Networks
    Project 2: Multilayer Perceptrons
'''
import numpy as np
import load_stl10_dataset

def preprocess_stl(imgs, labels):
    '''Preprocesses stl image data for training by a MLP neural network

    Parameters:
    ----------
    imgs: unint8 ndarray  [0, 255]. shape=(Num imgs, height, width, RGB color chans)

    Returns:
    ----------
    imgs: float64 ndarray [0, 1]. shape=(Num imgs N,)
    Labels: int ndarray. shape=(Num imgs N,). Contains int-coded class values 0,1,...,9

    TODO:
    1) Cast imgs to float64, normalize to the absolute range [0,1] (255 always maps to 1.0)
    2) Flatten height, width, color chan dims. New shape will be (num imgs, height*width*chans)
    3) Compute the mean of each image in the dataset, subtract it from each image
    4) Fix class labeling. Should span 0, 1, ..., 9 NOT 1,2,...10
    '''
    norm_imgs = (imgs.astype('float64') - imgs.astype('float64').min(axis=0)) / (255)
    #flat_imgs = np.reshape(norm_imgs, [norm_imgs.shape[0], np.prod(norm_imgs.shape[1:])])
    #new_imgs = flat_imgs - flat_imgs.mean(axis=0)
    
    
    new_imgs = np.swapaxes(np.swapaxes(norm_imgs, 1, 3), 2, 3)  

    new_labels = labels - 1
    return new_imgs, new_labels

def create_splits(data, y, n_train_samps=3500, n_test_samps=500, n_valid_samps=500, n_dev_samps=500):
    '''Divides the dataset up into train/test/validation/development "splits" (disjoint partitions)
    Parameters:
    ----------
    data: float64 ndarray. Image data. shape=(Num imgs, height*width*chans)
    y: ndarray. int-coded labels. shape? 

    Returns:
    ----------
    None if error
    x_train (training samples),
    y_train (training labels),
    x_test (test samples),
    y_test (test labels),
    x_val (validation samples),
    y_val (validation labels),
    x_dev (development samples),
    y_dev (development labels)

    TODO:
    1) Divvy up the images into train/test/validation/development non-overlapping subsets (see return vars)
    '''

    if n_train_samps + n_test_samps + n_valid_samps + n_dev_samps != len(data):
        samps = n_train_samps + n_test_samps + n_valid_samps + n_dev_samps
        print(f'Error! Num samples {samps} does not equal num images {len(data)}!')
        return

    # shuffle the data, set seed so data and labels stay together, set cursor
    '''
    np.random.seed(0)
    np.random.shuffle(data)
    np.random.shuffle(y)
    '''
    cursor = 0
    # get training set and update cursor
    x_train = data[0 : cursor + n_train_samps, :]
    y_train = y[0 : cursor + n_train_samps]
    cursor += n_train_samps
    # print('Cursor after training split:', cursor)

    # get testing set and update cursor
    x_test = data[cursor : cursor + n_test_samps, :]
    y_test = y[cursor : cursor + n_test_samps]
    cursor += n_test_samps
    # print('Cursor after testing split:', cursor)

    # get validation set and update cursor
    x_val = data[cursor : cursor + n_valid_samps, :]
    y_val = y[cursor : cursor + n_valid_samps]
    cursor += n_valid_samps
    # print('Cursor after validation split:', cursor)


    # get development set and update cursor
    x_dev = data[cursor : cursor + (n_dev_samps), :]
    y_dev = y[cursor : cursor + (n_dev_samps)]
    cursor += n_dev_samps
    # print('Cursor after development split:', cursor)

    return x_train, y_train, x_test, y_test, x_val, y_val, x_dev, y_dev

def load_stl10(n_train_samps=3500, n_test_samps=500, n_valid_samps=500, n_dev_samps=500, scale_fact=3):
    '''Automates the process of loading in the STL-10 dataset and labels, preprocessing, and creating
    the train/test/validation/dev splits.

    Returns:
    ----------
    None if error
    x_train (training samples),
    y_train (training labels),
    x_test (test samples),
    y_test (test labels),
    x_val (validation samples),
    y_val (validation labels),
    x_dev (development samples),
    y_dev (development labels)
    '''
    stl_imgs, stl_labels = load_stl10_dataset.load(scale_fact=scale_fact)
    stl_imgs_pp, stl_labels_pp = preprocess_stl(stl_imgs, stl_labels)
    stl_x_train, stl_y_train, stl_x_test, stl_y_test, stl_x_val, stl_y_val, stl_x_dev, stl_y_dev = create_splits(stl_imgs_pp, stl_labels_pp, n_train_samps, n_test_samps, n_valid_samps, n_dev_samps)
    return stl_x_train, stl_y_train, stl_x_test, stl_y_test, stl_x_val, stl_y_val, stl_x_dev, stl_y_dev
