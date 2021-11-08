import os
import joblib
import random
import argparse
# set up the input paramaters
parser = argparse.ArgumentParser(description="TCR attention weight calculation")
parser.add_argument("-i","--input",nargs="?",help = "the path to one constructed bag (*.pkl)",type=str,required=True)
parser.add_argument("-o","--output",nargs="?",help = "the directory path to output files",required=True,type=str)
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
import pandas as pd
import tensorflow as tf
from random import shuffle
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
from tensorflow.python.keras.backend_config import floatx


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
        # output = tf.reshape(tf.matmul(w,x),[tf.shape(x)[0],128])
        return tf.matmul(w,x)
        
    def compute_output_shape(self, input_shape):
        shape = list(input_shape)
        return tuple(shape)
    def get_config(self):
        config = super(attr_block, self).get_config()
        return config
    def compute_mask(self, inputs, mask=None):
        return None


# Bag embedding splitting and padding operation to make each bag having 100 instances
def bag_padding(x,num_instances):
        if x.shape[0] < num_instances:
            return [np.r_[x,np.array([[0]*199]*(num_instances-x.shape[0]))]]
        else:
            return [x]

def bag_spliting(x,num_instances):
    if x.shape[0] > num_instances:
        tmp = [] 
        for i in range(int(np.ceil(x.shape[0]/num_instances))):
            x_tmp=x[i*num_instances:(i+1)*num_instances]
            tmp.append(bag_padding(x_tmp,num_instances)[0])
        return tmp
    else:
        return bag_padding(x,num_instances)

d_model = 199
# set 100 as the max number limit of instances in every bag concept
num_instances = 100
# load bag data
bag = [joblib.load(f"{args.input}")]
# get bag embedding
os.system(f"python ./Scripts/BagEmbedding.py -i {args.input} -o {args.output}")
# load embedding of bag
bagembedding = joblib.load(f"{args.output}/{args.input.split('/')[-1].split('.')[0]}Embedding.pkl")
# bag splitting and padding
data = []
for i in bagembedding:
    data += bag_spliting(i,num_instances)
data = np.array(data)

# TCRBagger model loading
TCRBagger = keras.models.load_model("./Models/TCRBaggerModel.h5",custom_objects={'attr_block':attr_block})
MiddleModel = keras.Model(inputs=TCRBagger.input,outputs=TCRBagger.layers[-3].output)
mask_layer = layers.Masking(mask_value=[0]*d_model)
tmp = MiddleModel.predict(data)
h1 = TCRBagger.layers[-2].linear_V(tmp)
h2 = TCRBagger.layers[-2].linear_U(tmp)
w = TCRBagger.layers[-2].linear_weight(h1*h2)
mask = mask_layer(data)._keras_mask
w += (1-K.cast(K.expand_dims(mask,-1),K.floatx()))*-1e9
w = keras.activations.softmax(tf.transpose(w,[0,2,1]))

tcrs = bag[0][0][1]
weights = []
for i in range(w.shape[0]):
    weights += list(w[i][0].numpy())


# write the result to the csv file
Result = pd.DataFrame({'TCRs':tcrs,'AttentionWeights':weights[:len(tcrs)]})
Result.to_csv(f"{args.output}/{args.input.split('/')[-1].split('.')[0]}_TCRAttention.csv",sep=',',index=None)
print('Interpretation Accomplished.\n')
