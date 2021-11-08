import os
import joblib
import argparse
import pandas as pd
from multiprocessing import Pool

# set up the input paramaters
parser=argparse.ArgumentParser(description="the pipeline for MupeXI")
parser.add_argument("-i","--input",nargs="?",help = "the path to vcf file",required=True,type=str)
parser.add_argument("-a","--hla",help="the path of hla alleles",required=True,type=str)
parser.add_argument("-o","--output",nargs="?",help = "the directory path to output files",required=True,type=str)
args = parser.parse_args()


vcf=args.input
outPath = args.output

# Filter VCF files
# Screening for tumor-specific mutations
cmd1=f"bcftools filter -i '(FMT/AF[1:0] >= 0.05  &&  FMT/AD[1:1] >=10)' {vcf} -o {outPath}/filter.vcf -O v "
os.system(cmd1)
# Screening for tumor-specific mutations
cmd2=f"vcftools --vcf {outPath}/filter.vcf --remove-filtered-all --recode --recode-INFO-all --out {outPath}/filter"
os.system(cmd2)
cmd3=f"python2 ./MuPeXI/MuPeXI.py -v {outPath}/filter.recode.vcf -a {args.hla} -l 9,10,11 -d {outPath} -o CadidatePep.tsv"
os.system(cmd3)

# Filter peptide and save the mutpeptide as (*.txt)
if os.path.exists(f"{outPath}/CadidatePep.tsv"):
    df=pd.read_csv(f"{outPath}/CadidatePep.tsv",sep="\t",header=5)
    df=df[df["Mut_MHCrank_EL"].apply(lambda x: x<=2)]
    df=df[df["Proteome_Peptide_Match"]=="No"]
    df.to_csv(f"{outPath}/pepFilter.tsv",sep="\t",index=False)
    f = open(f"{outPath}/MutPeptideList.txt",'a')
    for i in df['Mut_peptide']:
        f.write(i+'\n')
    f.close()

