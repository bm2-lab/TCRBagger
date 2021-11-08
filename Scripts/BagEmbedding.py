import joblib
import argparse
import numpy as np
from multiprocessing import Pool
from scipy.stats import skew,kurtosis

# set up the input paramaters
parser = argparse.ArgumentParser(description="The bag embedding calculation")
parser.add_argument("-i","--input",nargs="?",help = "the path to constructed bags (*.pkl)",required=True,type=str)
parser.add_argument("-o","--output",nargs="?",help = "the directory path of output files",required=True,type=str)
parser.add_argument("-c","--cthread",nargs="?",help = "the number of threads used to bag embedding (default = 1)",type=int,const=1, default=1)
args = parser.parse_args()

# Bag embedding construction
dic_aascore_scale=joblib.load("./Requirements/dic_aascore_scale.pkl")
aa_dict=joblib.load("./Requirements/dic_Atchley_factors.pkl")
def get_features(bag_list):
    # Bag construction
    def aamapping(peptideSeq,encode_dim):
        #the longest sting will probably be shorter than 80 nt
        peptideArray = []
        if len(peptideSeq)>encode_dim:
            print('Length: '+str(len(peptideSeq))+' over bound!')
            peptideSeq=peptideSeq[0:encode_dim]
        for aa_single in peptideSeq:
            try:
                peptideArray.append(aa_dict[aa_single])
            except KeyError:
                print('Not proper aaSeqs: '+peptideSeq)
                peptideArray.append(np.zeros(5,dtype='float64'))
        for i in range(0,encode_dim-len(peptideSeq)):
            peptideArray.append(np.zeros(5,dtype='float64'))
        return np.asarray(peptideArray)

    def align_max_score(pep,single_tcr_frag):
        tcr_length=len(single_tcr_frag)
        score_list=[]
        for i in range(0,len(pep)-tcr_length+1):
            pep_tmp=pep[i:i+tcr_length]
            # choosing one of indicator in the dic_aascore_scale
            sum_tmp=np.array([dic_aascore_scale["BETM990101"][f"{pep_tmp[i]}{single_tcr_frag[i]}"] for i in range(tcr_length)]).sum()
            score_list.append(round(sum_tmp/len(single_tcr_frag),2))
        return np.array(score_list).max()

    # construct the combination feature between peptide and tcr
    def construct_feature(pep,TCR):
        #aa_list = ["A", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"]
        feature_combine=[]
        # delete the top and tail 3 AAs
        TCR_frag=TCR[3:-3]
        for frag_length in range(3,6):
            TCR_frag_list=[]
            for i in range(0,len(TCR_frag)-frag_length+1):
                TCR_frag_list.append(TCR_frag[i:i+frag_length])
            #feature_combine.append(frag_length)
            if len(TCR_frag_list)==0:
                if frag_length == 4:
                    feature_combine += feature_combine[0:8]
                if frag_length == 5:
                    three = feature_combine[0:8]
                    four = feature_combine[8:16]
                    feature_combine += [(three[i]+four[i])/2 for i in range(len(three))]
            else:
                feature=np.array([align_max_score(pep,TCR_frag_list[i]) for i in range(len(TCR_frag_list))])
                feature_combine+=[feature.max(),feature.min(),feature.mean(),feature.std()]
                # calculate the absolute median difference asfeature
                median1=np.median(feature)
                med_ab=np.absolute(feature-median1)
                feature_combine.append(np.median(med_ab))
                # the quartile difference
                feature_combine.append(np.percentile(feature, 75)-np.percentile(feature, 25))
                # skewness of data as feature
                feature_combine.append(skew(feature))
                # kurtosis of data as feature
                feature_combine.append(kurtosis(feature))
        
        tcrfeature=aamapping(TCR,20)
        tcrfeature=tcrfeature.flatten()
        feature_combine.extend(tcrfeature.tolist())
    
        
        pepfeature=aamapping(pep,15)
        pepfeature=pepfeature.flatten()
        feature_combine.extend(pepfeature.tolist())
        
        return feature_combine

    def construct_bags_feature(single_bag):
        single_ag=single_bag[0]
        cdr3_list=single_bag[1]
        inst_features = np.array([construct_feature(single_ag, cdr3_list[z]) for z in range(len(cdr3_list))])
        return inst_features

    tmp=construct_bags_feature(bag_list)
    return tmp

# load data
Inputdata = joblib.load(args.input)
p = Pool(args.cthread)
InputdataEmbedding=p.map(get_features,Inputdata)
p.close()
p.join()
# save the bag embedding data as (*.pkl)
joblib.dump(InputdataEmbedding,args.output+'/'+args.input.split('/')[-1].split('.')[0]+"Embedding.pkl")
