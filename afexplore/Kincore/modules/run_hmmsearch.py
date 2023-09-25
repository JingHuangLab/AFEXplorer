#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 08:29:49 2020

@author: vivekmodi
"""
import subprocess

def run_hmmsearch(pwd,pdbfilename,index,conf_df):
    model_id=conf_df.at[index,'Model_id']
    chain_id=conf_df.at[index,'Chain_id']
    
    cmd=(f'hmmsearch -o {pdbfilename[0:-4]}_{model_id}_{chain_id}_AGC.hmmer.txt {pwd}/AGC.hmm {pdbfilename[0:-4]}_{chain_id}.fasta;\
         hmmsearch -o {pdbfilename[0:-4]}_{model_id}_{chain_id}_CAMK.hmmer.txt {pwd}/CAMK.hmm {pdbfilename[0:-4]}_{chain_id}.fasta;\
         hmmsearch -o {pdbfilename[0:-4]}_{model_id}_{chain_id}_CK1.hmmer.txt {pwd}/CK1.hmm {pdbfilename[0:-4]}_{chain_id}.fasta;\
         hmmsearch -o {pdbfilename[0:-4]}_{model_id}_{chain_id}_CMGC.hmmer.txt {pwd}/CMGC.hmm {pdbfilename[0:-4]}_{chain_id}.fasta;\
         hmmsearch -o {pdbfilename[0:-4]}_{model_id}_{chain_id}_NEK.hmmer.txt {pwd}/NEK.hmm {pdbfilename[0:-4]}_{chain_id}.fasta;\
         hmmsearch -o {pdbfilename[0:-4]}_{model_id}_{chain_id}_RGC.hmmer.txt {pwd}/RGC.hmm {pdbfilename[0:-4]}_{chain_id}.fasta;\
         hmmsearch -o {pdbfilename[0:-4]}_{model_id}_{chain_id}_STE.hmmer.txt {pwd}/STE.hmm {pdbfilename[0:-4]}_{chain_id}.fasta;\
         hmmsearch -o {pdbfilename[0:-4]}_{model_id}_{chain_id}_TKL.hmmer.txt {pwd}/TKL.hmm {pdbfilename[0:-4]}_{chain_id}.fasta;\
         hmmsearch -o {pdbfilename[0:-4]}_{model_id}_{chain_id}_TYR.hmmer.txt {pwd}/TYR.hmm {pdbfilename[0:-4]}_{chain_id}.fasta;\
         hmmsearch -o {pdbfilename[0:-4]}_{model_id}_{chain_id}_HASP.hmmer.txt {pwd}/HASP.hmm {pdbfilename[0:-4]}_{chain_id}.fasta;\
         hmmsearch -o {pdbfilename[0:-4]}_{model_id}_{chain_id}_WNK.hmmer.txt {pwd}/WNK.hmm {pdbfilename[0:-4]}_{chain_id}.fasta;\
         hmmsearch -o {pdbfilename[0:-4]}_{model_id}_{chain_id}_BUB.hmmer.txt {pwd}/BUB.hmm {pdbfilename[0:-4]}_{chain_id}.fasta;\
         hmmsearch -o {pdbfilename[0:-4]}_{model_id}_{chain_id}_ULK.hmmer.txt {pwd}/ULK.hmm {pdbfilename[0:-4]}_{chain_id}.fasta;')
    
    process=subprocess.Popen(cmd,shell=True)
    process.communicate()
    process.wait()
    #subprocess.call(cmd,shell=True)
    return
