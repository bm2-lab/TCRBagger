import os
import joblib
import random
import argparse
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# set up the input paramaters
parser = argparse.ArgumentParser(description="Training TCRBagger from scratch")
parser.add_argument("-i1","--training",nargs="?",help = "the path to constructed training bags (*.pkl)",type=str,required=True)
parser.add_argument("-i2","--testing",nargs="?",help = "the path to constructed testing bags (*.pkl)",type=str,required=True)
parser.add_argument("-l1","--traininglabels",nargs="?",help = "the path to constructed training labels (*.pkl)",type=str,required=True)
parser.add_argument("-l2","--testinglabels",nargs="?",help = "the path to constructed testing labels (*.pkl)",type=str,required=True)
parser.add_argument("-o","--output",nargs="?",help = "the directory path to output new TCRBagger model",required=True,type=str)
parser.add_argument("-c","--cthread",nargs="?",help = "the number of threads used to bag embedding (default = 1)",type=int,const=1, default=1)
args = parser.parse_args()

import numpy as np
import tensorflow as tf
from random import shuffle
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.engine import training
from tensorflow.python.keras.backend_config import floatx

# gate attention mechanism based block
class attr_block(Layer):
    def __init__(self, **kwargs):
        super(attr_block, self).__init__(**kwargs)
        self.linear_V = layers.Dense(64,activation='tanh')
        self.linear_U = layers.Dense(64,activation='sigmoid')
        self.linear_weight = layers.Dense(1)
        self.supports_masking = True
    def call(self, x, mask = None):
        h1 = self.linear_V(x)
        h2 = self.linear_U(x)
        w = self.linear_weight(h1*h2)
        if mask is not None:
            w += (1-K.cast(K.expand_dims(mask,-1),K.floatx()))*-1e9
        w = keras.activations.softmax(tf.transpose(w,[0,2,1]))
        return tf.matmul(w,x)
        
    def compute_output_shape(self, input_shape):
        shape = list(input_shape)
        return tuple(shape)
    def get_config(self):
        config = super(attr_block, self).get_config()
        return config
    def compute_mask(self, inputs, mask=None):
        return None



def bag_padding(x):
    if x.shape[0] < num_instances:
        return np.r_[x,np.array([[0]*199]*(num_instances-x.shape[0]))]
    else:
        return x

# paramaters setting for the TCRBagger
d_model = 199
num_instances = 100

# create output directory path
if not os.path.exists(f"{args.output}"):
    os.system(f"mkdir {args.output}")
    
# load training and testing data
training_bags = joblib.load(args.training)
testing_bags = joblib.load(args.testing)
training_labels = np.array(joblib.load(args.traininglabels))
testing_labels = np.array(joblib.load(args.testinglabels))

# bag embedding command
os.system(r"python ./Scripts/BagEmbedding.py -i "+args.training+" -o "+args.output+" -c "+str(args.cthread))
os.system(r"python ./Scripts/BagEmbedding.py -i "+args.testing+" -o "+args.output+" -c "+str(args.cthread))

# bag embedding loading
training = joblib.load(f"{args.output}/{args.training.split('/')[-1].split('.')[0]}Embedding.pkl")
testing = joblib.load(f"{args.output}/{args.testing.split('/')[-1].split('.')[0]}Embedding.pkl")

# bag padding
training = np.array(list(map(bag_padding, training)))
testing = np.array(list(map(bag_padding, testing)))

# TCRBagger construction
Input = keras.Input(shape=(num_instances,d_model,1),dtype=float)
mask_layer = layers.Masking(mask_value=[0]*d_model)
x = mask_layer(Input)
conv1 = layers.Conv2D(32,(1,4),activation='relu')(x)
conv2 = layers.Conv2D(32,(1,4),activation = 'relu')(conv1)
maxpool1 = layers.MaxPooling2D(pool_size=(1,2))(conv2)
maxpool2 = layers.Dropout(0.50)(maxpool1)
f1 = tf.reshape(maxpool2,(-1,100,maxpool2.shape[2]*maxpool2.shape[3]))
f2 = layers.Dense(128,activation='relu')(f1)
at2 = attr_block(name='at2')(f2)
f3 = layers.Dense(1,activation = 'sigmoid')(at2)
model = keras.Model(inputs=Input,outputs=f3)
model.compile(loss='binary_crossentropy', optimizer=Adam(0.001),metrics=['accuracy'])

# model training
history = model.fit(training,training_labels,batch_size=64,epochs=100,validation_data=(testing,testing_labels),callbacks=[keras.callbacks.EarlyStopping(monitor = 'val_loss',patience = 10, restore_best_weights=True)])

# model save
model.save(f"{args.output}/New_TCRBagger.h5")  
joblib.dump(history.history,f"{args.output}/History.pkl")
