#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 12:41:19 2020

@author: vivekmodi
"""
from Bio import PDB
from Bio.SeqUtils import seq1

def identify_restypes(pdbfilename,conf_df,index,structure):
    
    for model in structure:
        for chain in model:
            if int(model.id)==int(conf_df.at[index,'Model_id']) and str(chain.id)==str(conf_df.at[index,'Chain_id']):
                for residue in chain:
                    if int(residue.id[1])==int(conf_df.at[index,'Lys_num']) and residue.id[0]==' ':
                        conf_df.at[index,'Lys_restype']=seq1(residue.resname)
                    if int(residue.id[1])==int(conf_df.at[index,'Glu_num']) and residue.id[0]==' ':
                        conf_df.at[index,'Glu_restype']=seq1(residue.resname)
                    if int(residue.id[1])==int(conf_df.at[index,'Glu4_num']) and residue.id[0]==' ':
                        conf_df.at[index,'Glu4_restype']=seq1(residue.resname)
                    #print(conf_df.at[index,'Phe_num'])
                    if int(residue.id[1])==conf_df.at[index,'Phe_num'] and residue.id[0]==' ':
                        conf_df.at[index,'Phe_restype']=seq1(residue.resname)
                    if int(residue.id[1])==conf_df.at[index,'XDFG_num'] and residue.id[0]==' ':
                        conf_df.at[index,'XDFG_restype']=seq1(residue.resname)
                    if int(residue.id[1])==conf_df.at[index,'Asp_num'] and residue.id[0]==' ':
                        conf_df.at[index,'Asp_restype']=seq1(residue.resname)
    return conf_df