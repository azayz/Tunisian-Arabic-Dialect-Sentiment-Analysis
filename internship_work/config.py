import tensorflow as tf


FOLDS = 5 #number of folds for cross validation
SEED  = 42 #seed for  reproducible results
EPOCHS = 5 #number of traning epochs for each fold 
BATCH_SIZE = 32 #number of training instances in a single batch
MAX_LEN = 192 #len to padd sequences
AUTO = tf.data.experimental.AUTOTUNE