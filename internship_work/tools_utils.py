import pandas as pd
import numpy as np

from sklearn.metrics import confusion_matrix

from vneuron.config import BATCH_SIZE

from tqdm.notebook import tqdm

import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
import  matplotlib.pyplot as plt


import warnings 
warnings.simplefilter('ignore')



def create_train(x_train,y_train) :
    train_dataset = (
        tf.data.Dataset
        .from_tensor_slices((x_train, y_train))
        .repeat()
        .shuffle(2048)
        .batch(BATCH_SIZE)
        .prefetch(AUTO)
    )
    return  train_dataset

def create_valid(x_valid,y_valid) :
    valid_dataset = (
        tf.data.Dataset
        .from_tensor_slices((x_valid, y_valid))
        .batch(BATCH_SIZE)
        .cache()
        .prefetch(AUTO)
    )
    
    return valid_dataset

def create_test(x_test) :
    test_dataset = (
        tf.data.Dataset
        .from_tensor_slices(x_test)
        .batch(BATCH_SIZE)
    )
    return test_dataset

def performance(y_true,y_preds) :
    print(confusion_matrix(y_true, y_preds))
    print(classification_report(y_true, y_preds))

