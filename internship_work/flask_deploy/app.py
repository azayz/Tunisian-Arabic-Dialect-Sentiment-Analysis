import sys
sys.path.append('../')


import numpy as np
from flask  import Flask , redirect , render_template , request
from vneuron.predict import get_preds
from vneuron.tools_utils import create_test
from vneuron.config import MAX_LEN , BATCH_SIZE

import transformers
from transformers import TFAutoModel, AutoTokenizer
from tqdm.notebook import tqdm
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, processors
from tokenizers import BertWordPieceTokenizer

import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint


def build_model(transformer, max_len=512):

    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    sequence_output = transformer(input_word_ids)[0]
    cls_token = sequence_output[:, 0, :]
    out = Dense(1, activation='sigmoid')(cls_token)
    
    model = Model(inputs=input_word_ids, outputs=out)
    model.compile(Adam(lr=6e-6), loss='binary_crossentropy', metrics=['accuracy','AUC'])
    
    return model
transformer_layer = (
            transformers.TFDistilBertModel
            .from_pretrained('distilbert-base-multilingual-cased')
        )

model = build_model(transformer_layer, max_len=MAX_LEN)
model.load_weights('/home/aziz/vneuron/model/weights')

fast_tokenizer = BertWordPieceTokenizer('vocab.txt', lowercase=False)
fast_tokenizer.enable_truncation(max_length=MAX_LEN)
fast_tokenizer.enable_padding(length=MAX_LEN)

app = Flask(__name__)


@app.route('/') 

def index () :

    return  render_template('index.html')


@app.route('/predict' , methods = ['POST'])
def predict() :
    text = request.form['content']
    text = [str(text)]
    all_ids = []
    encs = fast_tokenizer.encode_batch(text)
    all_ids.extend([enc.ids for enc in encs])

    all_ids = np.array(all_ids).astype(np.float32)

    to_predict= create_test(all_ids)

    predictions = model.predict(to_predict)

    #_ , preds   = get_preds(text)    
    
    return render_template('pediction.html' , preds = predictions)

if  __name__== '__main__' :
    app.run()