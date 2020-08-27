import pandas as pd
import numpy as np
from vneuro.tools_utils import create_train , create_valid
from vneuron.config import FOLDS ,SEED , BATCH_SIZE , EPOCHS , AUTO ,MAX_LEN

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

import warnings 
warnings.simplefilter('ignore')

#MODEL = 'jplu/tf-xlm-roberta-large'

def fast_encode(texts, tokenizer, chunk_size=256, maxlen=512):

    tokenizer.enable_truncation(max_length=maxlen)
    tokenizer.enable_padding(max_length=maxlen)
    all_ids = []
    
    for i in tqdm(range(0, len(texts), chunk_size)):
        text_chunk = texts[i:i+chunk_size].tolist()
        encs = tokenizer.encode_batch(text_chunk)
        all_ids.extend([enc.ids for enc in encs])
    
    return np.array(all_ids)

def regular_encode(texts, tokenizer, maxlen=512):
    enc_di = tokenizer.batch_encode_plus(
        texts, 
        return_attention_masks=False, 
        return_token_type_ids=False,
        pad_to_max_length=True,
        max_length=maxlen
    )
    
    return np.array(enc_di['input_ids'])

def build_model(transformer, max_len=512):

    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    sequence_output = transformer(input_word_ids)[0]
    cls_token = sequence_output[:, 0, :]
    out = Dense(1, activation='sigmoid')(cls_token)
    
    model = Model(inputs=input_word_ids, outputs=out)
    model.compile(Adam(lr=6e-6), loss='binary_crossentropy', metrics=['accuracy','AUC'])
    
    return model
def train(): 
    data = pd.read_csv('../input/vneuron/extra_train_data.csv')

    #tokenizer = AutoTokenizer.from_pretrained(MODEL)
    tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased')
    # Save the loaded tokenizer locally
    tokenizer.save_pretrained('.')
    # Reload it with the huggingface tokenizers library
    fast_tokenizer = BertWordPieceTokenizer('vocab.txt', lowercase=False)

    #texts = regular_encode(data.text.values, tokenizer, maxlen=MAX_LEN)
    texts = fast_encode(data.text.values.astype(str), fast_tokenizer, maxlen=MAX_LEN)

    ys = data.intent.values

    skf = KFold(n_splits=FOLDS,shuffle=True,random_state=SEED)

    for fold,(train_indices,valid_indices) in enumerate(skf.split(texts,ys)) :
        print() ; print('#'*25)
        print('Fold' , fold+1)
        print('#'*25)
        
        # Calling the transformer layer using distilbert-base-multilingual-cased model

        #transformer_layer = TFAutoModel.from_pretrained(MODEL)
        transformer_layer = (
            transformers.TFDistilBertModel
            .from_pretrained('distilbert-base-multilingual-cased')
        )
        model = build_model(transformer_layer, max_len=MAX_LEN)
        
        #Checkpoint to save the best weights for that fold. 

        sv = tf.keras.callbacks.ModelCheckpoint(
            'fold-%i.h5'%fold, monitor='val_loss', verbose=0, save_best_only=True,
            save_weights_only=True, mode='min', save_freq='epoch')
        
        n_steps = train_indices.shape[0] // BATCH_SIZE
        history = model.fit(
        create_train(texts[train_indices],ys[train_indices]),
        steps_per_epoch=n_steps,
        validation_data=create_valid(texts[valid_indices],ys[valid_indices]),
        epochs=EPOCHS,
        callbacks =  [sv]    
        )
        
        # Plot training and validation loss and AUC : 
        plt.figure(figsize=(15,5))
        plt.plot(np.arange(EPOCHS),history.history['auc'],'-o',label='Train AUC',color='#ff7f0e')
        plt.plot(np.arange(EPOCHS),history.history['val_auc'],'-o',label='Val AUC',color='#1f77b4')
        x = np.argmax( history.history['val_auc'] ); y = np.max( history.history['val_auc'] )
        xdist = plt.xlim()[1] - plt.xlim()[0]; ydist = plt.ylim()[1] - plt.ylim()[0]
        plt.scatter(x,y,s=200,color='#1f77b4'); plt.text(x-0.03*xdist,y-0.13*ydist,'max auc\n%.2f'%y,size=14)
        plt.ylabel('AUC',size=14); plt.xlabel('Epoch',size=14)
        plt.legend(loc=2)
        plt2 = plt.gca().twinx()
        plt2.plot(np.arange(EPOCHS),history.history['loss'],'-o',label='Train Loss',color='#2ca02c')
        plt2.plot(np.arange(EPOCHS),history.history['val_loss'],'-o',label='Val Loss',color='#d62728')
        x = np.argmin( history.history['val_loss'] ); y = np.min( history.history['val_loss'] )
        ydist = plt.ylim()[1] - plt.ylim()[0]
        plt.scatter(x,y,s=200,color='#d62728'); plt.text(x-0.03*xdist,y+0.05*ydist,'min loss',size=14)
        plt.ylabel('Loss',size=14)
        plt.title('FOLD %i Distilbert-base-multilingual-cased'%
                    (fold+1),size=18)
        plt.legend(loc=3)
        plt.show()  

if __name__ == "__main__":
    train()