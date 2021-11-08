import os
import joblib
import argparse

# set up the input paramaters
parser = argparse.ArgumentParser(description="The prediction of peptide immunogenicity score by TCRBagger")
parser.add_argument("-b","--bags",nargs="?",help = "the path to constructed bags (*.pkl)",type=str)
parser.add_argument("-p","--peplist",nargs="?",help = "the path to input peptide list file (*.txt)",type=str)
parser.add_argument("-t","--tcrlist",nargs="?",help = "the path to input tcr list file (*.txt)",type=str)
parser.add_argument("-r1","--rnaseq1",nargs="?",help = "the path to RNAseq data 1 (like *.fastq.gz)",type=str)
parser.add_argument("-r2","--rnaseq2",nargs="?",help = "the path to RNAseq data 2 (like *.fastq.gz)",type=str)
parser.add_argument("-v","--vcf",nargs="?",help = "the path to vcf data (*.vcf)",type=str)
parser.add_argument("-a","--alleles",nargs="?",help = "HLA alleles, comma separated",type=str)
parser.add_argument("-c","--cthread",nargs="?",help = "the number of threads used to bag embedding (default = 1)",type=int,const=1, default=1)
parser.add_argument("-o","--output",nargs="?",help = "the directory path to output files",required=True,type=str)
args = parser.parse_args()

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
from tensorflow.python.keras.backend_config import floatx
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# function for converting the txt file into list
def read_txt(inputs_path):
    tmp = []
    f = open(inputs_path)
    lines = f.readlines()
    f.close()
    for line in lines:
        tmp.append(line.strip()) 
    return tmp

# condition 1
if str(args.bags) != 'None':
    # load the bag.pkl data
    Inputdata = joblib.load(args.bags)
    # get the peptide list from bag.pkl
    Peptidelist = [i[0] for i in Inputdata]
    # bag embedding command
    os.system(r"python ./Scripts/BagEmbedding.py -i "+args.bags+" -o "+args.output+" -c "+str(args.cthread))
    # load the bag embedding file
    InputdataEmbedding = np.array(joblib.load(args.output+'/'+args.bags.split('/')[-1].split('.')[0]+'Embedding.pkl'))

# condition 2
if str(args.peplist) != 'None' and str(args.tcrlist) != 'None':
    # load the peptide list data
    Peptidelist = read_txt(args.peplist)
    # load the tcr list data
    Tcrlist = read_txt(args.tcrlist)
    # Bag construction
    Bags = []
    for i in Peptidelist:
        Bags.append([i,Tcrlist])
    joblib.dump(Bags,args.output+"/Bags.pkl")
    # Bag embedding command
    os.system(r"python ./Scripts/BagEmbedding.py -i "+args.output+"/Bags.pkl"+" -o "+args.output+" -c "+str(args.cthread))
    InputdataEmbedding = np.array(joblib.load(args.output+'/BagsEmbedding.pkl'))

# condition 3
if str(args.peplist) != 'None' and str(args.rnaseq1) != 'None' and str(args.rnaseq2) != 'None':
    # load the peptide list data
    Peptidelist = read_txt(args.peplist)
    # using mixcr to predict the tcr profile
    cmd = f"python ./Scripts/tcr_pipeline.py -f1 {args.rnaseq1} -f2 {args.rnaseq2} -o {args.output}"
    os.system(cmd)
    # load the tcr list data
    Tcrlist = read_txt(f"{args.output}/TcrList.txt")
    # Bag construction
    Bags = []
    for i in Peptidelist:
        Bags.append([i,Tcrlist])
    joblib.dump(Bags,args.output+"/Bags.pkl")
    # Bag embedding command
    os.system(r"python ./Scripts/BagEmbedding.py -i "+args.output+"/Bags.pkl"+" -o "+args.output+" -c "+str(args.cthread))
    InputdataEmbedding = np.array(joblib.load(args.output+'/BagsEmbedding.pkl'))

# condition 4
if str(args.vcf) != 'None' and str(args.rnaseq1) != 'None' and str(args.rnaseq2) != 'None':
    # predict peptide list data
    os.system(f"python ./Scripts/MupeXI_pipline.py -i {args.vcf} -o {args.output} -a {args.alleles}")
    # load peptide list
    Peptidelist = read_txt(f"{args.output}/MutPeptideList.txt")
    # using mixcr to predict the tcr profile
    cmd = f"python ./Scripts/tcr_pipeline.py -f1 {args.rnaseq1} -f2 {args.rnaseq2} -o {args.output}"
    os.system(cmd)
    # load the tcr list data
    Tcrlist = read_txt(f"{args.output}/TcrList.txt")
    # Bag construction
    Bags = []
    for i in Peptidelist:
        Bags.append([i,Tcrlist])
    joblib.dump(Bags,args.output+"/Bags.pkl")
    # Bag embedding command
    os.system(r"python ./Scripts/BagEmbedding.py -i "+args.output+"/Bags.pkl"+" -o "+args.output+" -c "+str(args.cthread))
    InputdataEmbedding = np.array(joblib.load(args.output+'/BagsEmbedding.pkl'))
    

# Gate attention mechanism block
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


# set 100 as the max number limit of instances in every bag concept
num_instances = 100
# TCRBagger model loading
model = keras.models.load_model("./Models/TCRBaggerModel.h5",custom_objects={'attr_block':attr_block})
# calculate the immunogenicity for each peptide in the list
PeptideScores = [np.max(model.predict(np.array(bag_spliting(i,100)))) for i in InputdataEmbedding]
# write the result to the csv file
Result = pd.DataFrame({'Peptides':Peptidelist,'TCRBagger_score':PeptideScores})
Result.to_csv(args.output+"/BagsResult.csv",sep=',',index=None)
print('Prediction Accomplished.\n')