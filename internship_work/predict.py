import json
import pandas as pd
import numpy as np
from vneuron.tools_utils import create_test , performance
from vneuron.config import FOLDS ,SEED , BATCH_SIZE , EPOCHS , AUTO , MAX_LEN

import sys, os
from sklearn.metrics import confusion_matrix

import transformers
from transformers import TFAutoModel, AutoTokenizer
from tqdm.notebook import tqdm
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, processors

import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
import  matplotlib.pyplot as plt

from tokenizers import BertWordPieceTokenizer


def build_model(transformer, max_len=512):

    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    sequence_output = transformer(input_word_ids)[0]
    cls_token = sequence_output[:, 0, :]
    out = Dense(1, activation='sigmoid')(cls_token)
    
    model = Model(inputs=input_word_ids, outputs=out)
    model.compile(Adam(lr=6e-6), loss='binary_crossentropy', metrics=['accuracy','AUC'])
    
    return model



def get_preds(list_of_texts) :
    transformer_layer = (
            transformers.TFDistilBertModel
            .from_pretrained('distilbert-base-multilingual-cased')
        )

    model = build_model(transformer_layer, max_len=MAX_LEN)
    model.load_weights('model/weights')


    
    #model = tf.keras.models.load_model('model')
    
    print('weights loaded')

    tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased')
    tokenizer.save_pretrained('.')
    # Reload it with the huggingface tokenizers library
    fast_tokenizer = BertWordPieceTokenizer('vocab.txt', lowercase=False)

    fast_tokenizer.enable_truncation(max_length=MAX_LEN)
    fast_tokenizer.enable_padding(length=MAX_LEN)

    all_ids = []
    encs = fast_tokenizer.encode_batch(list_of_texts)
    all_ids.extend([enc.ids for enc in encs])

    all_ids = np.array(all_ids).astype(np.float32)

    to_predict= create_test(all_ids)

    predictions = model.predict(to_predict)
    #print(predictions*10)

    for prediction in predictions :
        print(prediction)
        
    dic = {'predictions' : predictions}

    parsed = []
    #response = pd.DataFrame(dic)
    #parsed = response.to_json(orient = 'columns') #not sure if works
    #json.dumps(parsed)           #to be reviewed 
    return  parsed , predictions



def test_performance(list_of_texts , y_true) :
    _ , predictions = get_preds(list_of_texts)
    performance(y_true,predictions)



if __name__ == "__main__":

    data = pd.read_csv(sys.argv[1])
    get_preds(data.text.values.astype(str))    
