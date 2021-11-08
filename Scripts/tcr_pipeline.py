import os 
import argparse
import pandas as pd

# set up the input paramaters
parser = argparse.ArgumentParser(description="The TCR list predicted by the MIXCR")
parser.add_argument("-f1","--fastq1",nargs="?",help = "one of the two pair end fastq files",required=True,type=str)
parser.add_argument("-f2","--fastq2",nargs="?",help = "the other one of the two pair end fastq files",required=True,type=str)
parser.add_argument("-o","--output",nargs="?",help = "the directory path to output files",required=True,type=str)
args = parser.parse_args()

def filter_cdr3(cdr3_list):
    cdr3_last=[]
    for cdr3 in cdr3_list:
        if ("_" in cdr3 or "*" in cdr3):
            continue
        else:
            if len(cdr3)>=10 and len(cdr3)<=20:
                cdr3_last.append(cdr3)
    return cdr3_last

def findCDR3(path):
    df=pd.read_csv(path,sep="\t")
    df=df[df["allVHitsWithScore"].apply(lambda x:"TRB" in x)]
    if len(df) !=0:
        cdr3List=list(df["aaSeqCDR3"])
        cdr3List=filter_cdr3(cdr3List)
        return cdr3List
    else:
        print("There is no cdr3 in this clone file")

# comparative analysis
out_path=args.output
cmd1=f"./mixcr-2.1.5/mixcr align -p rna-seq -OallowPartialAlignments=true -f {args.fastq1} {args.fastq1} {out_path}/alignments.vdjca"
os.system(cmd1)
# Assenble contig analysis was performed twice
cmd2=f"./mixcr-2.1.5/mixcr assemblePartial -f {out_path}/alignments.vdjca {out_path}/alignments_rescued_1.vdjca"
os.system(cmd2)
cmd3=f"./mixcr-2.1.5/mixcr assemblePartial -f {out_path}/alignments_rescued_1.vdjca {out_path}/alignments_rescued_2.vdjca"
os.system(cmd3)
# Contig sequence extending
cmd4=f"./mixcr-2.1.5/mixcr extendAlignments -f {out_path}/alignments_rescued_2.vdjca {out_path}/alignments_rescued_2_extended.vdjca"
os.system(cmd4)
# Assemble for the clone subtype
cmd5=f"./mixcr-2.1.5/mixcr assemble -f {out_path}/alignments_rescued_2_extended.vdjca {out_path}/clones.clns"
os.system(cmd5)
# Output the clone subtype
cmd6=f"./mixcr-2.1.5/mixcr exportClones -c TCR -f {out_path}/clones.clns {out_path}/clones.txt"
os.system(cmd6)
#os.system("rm alignments*")
#os.system("rm clones.clns")

# write the TcrList data
cdr3s = findCDR3(f"{out_path}/clones.txt")
f = open(f"{out_path}/TcrList.txt",'a')
for i in cdr3s:
    f.write(i+'\n')
f.close()
