#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 11:57:13 2020

@author: vivekmodi
"""
import os, subprocess

def delete_files(pdbfilename,model_id,chain_id,identified_group):
    
    for groups in ('AGC','CAMK','CK1','CMGC','NEK','RGC','STE','TKL','TYR','HASP','WNK','BUB','ULK'):
        
        if groups==identified_group:
            
            continue
        else:
            if os.path.isfile(f'{pdbfilename[0:-4]}_{model_id}_{chain_id}_{groups}.hmmer.txt'):
                cmd=f'rm {pdbfilename[0:-4]}*{groups}.hmmer.txt;'
                subprocess.call(cmd,shell=True)
