#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 08:43:59 2020

@author: vivekmodi
"""
from Bio import SearchIO

def identify_group(pdbfilename,index,conf_df):
    model_id=str(conf_df.at[index,'Model_id'])
    chain_id=conf_df.at[index,'Chain_id']
    
    hmm_result_AGC=SearchIO.read(f'{pdbfilename[0:-4]}_{model_id}_{chain_id}_AGC.hmmer.txt',format='hmmer3-text')
    hmm_result_CAMK=SearchIO.read(f'{pdbfilename[0:-4]}_{model_id}_{chain_id}_CAMK.hmmer.txt',format='hmmer3-text')
    hmm_result_CK1=SearchIO.read(f'{pdbfilename[0:-4]}_{model_id}_{chain_id}_CK1.hmmer.txt',format='hmmer3-text')
    hmm_result_CMGC=SearchIO.read(f'{pdbfilename[0:-4]}_{model_id}_{chain_id}_CMGC.hmmer.txt',format='hmmer3-text')
    hmm_result_NEK=SearchIO.read(f'{pdbfilename[0:-4]}_{model_id}_{chain_id}_NEK.hmmer.txt',format='hmmer3-text')
    hmm_result_RGC=SearchIO.read(f'{pdbfilename[0:-4]}_{model_id}_{chain_id}_RGC.hmmer.txt',format='hmmer3-text')
    hmm_result_STE=SearchIO.read(f'{pdbfilename[0:-4]}_{model_id}_{chain_id}_STE.hmmer.txt',format='hmmer3-text')
    hmm_result_TKL=SearchIO.read(f'{pdbfilename[0:-4]}_{model_id}_{chain_id}_TKL.hmmer.txt',format='hmmer3-text')
    hmm_result_TYR=SearchIO.read(f'{pdbfilename[0:-4]}_{model_id}_{chain_id}_TYR.hmmer.txt',format='hmmer3-text')
    hmm_result_HASP=SearchIO.read(f'{pdbfilename[0:-4]}_{model_id}_{chain_id}_HASP.hmmer.txt',format='hmmer3-text')
    hmm_result_WNK=SearchIO.read(f'{pdbfilename[0:-4]}_{model_id}_{chain_id}_WNK.hmmer.txt',format='hmmer3-text')
    hmm_result_BUB=SearchIO.read(f'{pdbfilename[0:-4]}_{model_id}_{chain_id}_BUB.hmmer.txt',format='hmmer3-text')
    hmm_result_ULK=SearchIO.read(f'{pdbfilename[0:-4]}_{model_id}_{chain_id}_ULK.hmmer.txt',format='hmmer3-text')

    eval=dict()
    eval['AGC']=100;eval['CAMK']=100;eval['CK1']=100;eval['CMGC']=100;eval['NEK']=100;eval['RGC']=100;eval['STE']=100;eval['TKL']=100;eval['TYR']=100;
    eval['HASP']=100;eval['WNK']=100;eval['BUB']=100;eval['ULK']=100
    for hits in hmm_result_AGC:
        for hsp in hits:
            eval['AGC']=hsp.evalue
    for hits in hmm_result_CAMK:
        for hsp in hits:
            eval['CAMK']=hsp.evalue
    for hits in hmm_result_CK1:
        for hsp in hits:
            eval['CK1']=hsp.evalue
    for hits in hmm_result_CMGC:
        for hsp in hits:
            eval['CMGC']=hsp.evalue
    for hits in hmm_result_NEK:
        for hsp in hits:
            eval['NEK']=hsp.evalue
    for hits in hmm_result_RGC:
        for hsp in hits:
            eval['RGC']=hsp.evalue
    for hits in hmm_result_STE:
        for hsp in hits:
            eval['STE']=hsp.evalue
    for hits in hmm_result_TKL:
        for hsp in hits:
            eval['TKL']=hsp.evalue
    for hits in hmm_result_TYR:
        for hsp in hits:
            eval['TYR']=hsp.evalue
    for hits in hmm_result_HASP:
        for hsp in hits:
            eval['HASP']=hsp.evalue
    for hits in hmm_result_WNK:
        for hsp in hits:
            eval['WNK']=hsp.evalue
    for hits in hmm_result_BUB:
        for hsp in hits:
            eval['BUB']=hsp.evalue
    for hits in hmm_result_ULK:
        for hsp in hits:
            eval['ULK']=hsp.evalue

    minEval=100;
    for groups in ('AGC','CAMK','CK1','CMGC','NEK','RGC','STE','TKL','TYR','HASP','WNK','BUB','ULK'):
        if eval[groups]<minEval:
            minEval=eval[groups]
            conf_df.at[index,'Group']=groups

    return conf_df