#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 09:14:31 2020

@author: vivekmodi
"""
from Bio import SearchIO

def identify_residues(pdbfilename,index,conf_df):
    model_id=conf_df.at[index,'Model_id']
    chain_id=conf_df.at[index,'Chain_id']
    group=conf_df.at[index,'Group']
    first_res=conf_df.at[index,'First_res']
    lys={'AGC':30,'CAMK':30,'CK1':30,'CMGC':30,'NEK':30,'RGC':29,'STE':30,'TKL':28,'TYR':34,'HASP':28,'WNK':31,'BUB':30,'ULK':31}
    glu={'AGC':49,'CAMK':47,'CK1':44,'CMGC':45,'NEK':48,'RGC':45,'STE':47,'TKL':46,'TYR':51,'HASP':52,'WNK':49,'BUB':39,'ULK':48}
    glu4={'AGC':53,'CAMK':51,'CK1':48,'CMGC':49,'NEK':52,'RGC':49,'STE':51,'TKL':50,'TYR':55,'HASP':56,'WNK':53,'BUB':43,'ULK':52}
    xdfg={'AGC':141,'CAMK':141,'CK1':140,'CMGC':149,'NEK':144,'RGC':139,'STE':140,'TKL':141,'TYR':145,'HASP':203,'WNK':148,'BUB':154,'ULK':142}
    asp={'AGC':142,'CAMK':142,'CK1':141,'CMGC':150,'NEK':145,'RGC':140,'STE':141,'TKL':142,'TYR':146,'HASP':204,'WNK':149,'BUB':155,'ULK':143}
    phe={'AGC':143,'CAMK':143,'CK1':142,'CMGC':151,'NEK':146,'RGC':141,'STE':142,'TKL':143,'TYR':147,'HASP':205,'WNK':150,'BUB':156,'ULK':144}
    
    hinge1={'AGC':79,'CAMK':77,'CK1':75,'CMGC':84,'NEK':78,'RGC':75,'STE':77,'TKL':76,'TYR':81,'HASP':123,'WNK':83,'BUB':76,'ULK':78}   #Align col 426
    
    type2_resi={'AGC':(52,56,114,122),'CAMK':(50,54,112,120),'CK1':(47,51,110,118),'CMGC':(48,52,120,128),'NEK':(51,55,117,125),'RGC':(48,52,111,120),'STE':(50,54,113,121),'TKL':(49,53,112,122),'TYR':(54,58,118,126),\
                'HASP':(55,59,156,164),'WNK':(52,56,118,128),'BUB':(42,46,117,124),'ULK':(51,55,113,121)}  #Align col 149
   
    back_pocket1={'AGC':range(34,62),'CAMK':range(34,60),'CK1':range(32,58),'CMGC':range(34,63),'NEK':range(35,61),'RGC':range(33,58),'STE':range(34,60),'TKL':range(32,59),'TYR':range(38,64),'HASP':range(32,74),'WNK':range(37,62),'BUB':range(33,59),'ULK':range(35,61)}   #Align col 106-185
    back_pocket2={'AGC':range(64,73),'CAMK':range(62,71),'CK1':range(60,69),'CMGC':range(65,74),'NEK':range(63,72),'RGC':range(60,69),'STE':range(62,71),'TKL':range(61,70),'TYR':range(66,75),'HASP':range(76,85),'WNK':range(64,75),'BUB':range(61,70),'ULK':range(63,72)}   #Align col 187-196
    back_pocket3={'AGC':range(73,76),'CAMK':range(71,74),'CK1':range(69,72),'CMGC':range(78,81),'NEK':range(72,75),'RGC':range(69,72),'STE':range(71,74),'TKL':range(70,73),'TYR':range(75,78),'HASP':range(117,120),'WNK':range(76,79),'BUB':range(70,74),'ULK':range(72,75)}  #Align col 420-423
    front_pocket={'AGC':range(34,47),'CAMK':range(34,45),'CK1':range(32,42),'CMGC':range(34,43),'NEK':range(35,46),'RGC':range(33,43),'STE':range(34,45),'TKL':range(32,44),'TYR':range(38,49),'HASP':range(32,50),'WNK':range(37,47),'BUB':range(33,37),'ULK':range(35,46)}  #Align col 106-144
    xdf_residues={'AGC':range(141,144),'CAMK':range(141,144),'CK1':range(140,143),'CMGC':range(149,152),'NEK':range(144,147),'RGC':range(139,142),'STE':range(140,143),'TKL':range(141,144),'TYR':range(145,148),'HASP':range(203,206),'WNK':range(148,151),'BUB':range(154,157),'ULK':range(142,145)}   #Align col 1337-1340  X-D-F
    
    type2_resi_num=set();back_pocket1_num=set();back_pocket2_num=set();back_pocket3_num=set()
    front_pocket_num=set();xdf_residues_num=set()
    conf_df.at[index,'Type2_resi_num']=[type2_resi_num]   #this is the way list is assigned to pandas, need to do it only for the first index where the column is defined
    conf_df.at[index,'Back_pocket1_num']=[back_pocket1_num]
    conf_df.at[index,'Back_pocket2_num']=[back_pocket2_num]
    conf_df.at[index,'Back_pocket3_num']=[back_pocket3_num]
    conf_df.at[index,'Front_pocket_num']=[front_pocket_num]
    conf_df.at[index,'XDF_residues_num']=[xdf_residues_num]

    hmm_result=SearchIO.read(f'{pdbfilename[0:-4]}_{model_id}_{chain_id}_{group}.hmmer.txt',format='hmmer3-text')
    for hits in hmm_result:     #extract hit from alignment in HMM output file
        for hsps in hits:
            col_num=0;hmm_index=hsps.query_start;hit_index=hsps.hit_start+first_res-1
            for hmm_res in hsps.aln[0]:
                col_num=col_num+1
                if hmm_res!='.':
                    hmm_index=hmm_index+1
                if hsps.aln[1][col_num-1]!='-':
                    hit_index=hit_index+1
                if hmm_index==lys[group] and hmm_res!='.':
                    conf_df.at[index,'Lys_num']=hit_index
                    conf_df.at[index,'Lys_restype']=list({hsps.aln[1][col_num-1]})[0]       #The residue is extracted using list otherwise it is printed in ''
                if hmm_index==glu[group] and hmm_res!='.':
                    conf_df.at[index,'Glu_num']=hit_index
                    conf_df.at[index,'Glu_restype']=list({hsps.aln[1][col_num-1]})[0]
                if hmm_index==glu4[group] and hmm_res!='.':
                    conf_df.at[index,'Glu4_num']=hit_index
                    conf_df.at[index,'Glu4_restype']=list({hsps.aln[1][col_num-1]})[0]
                if hmm_index==xdfg[group] and hmm_res!='.':
                    conf_df.at[index,'XDFG_num']=hit_index
                    conf_df.at[index,'XDFG_restype']=list({hsps.aln[1][col_num-1]})[0]
                if hmm_index==asp[group] and hmm_res!='.':
                    conf_df.at[index,'Asp_num']=hit_index
                    conf_df.at[index,'Asp_restype']=list({hsps.aln[1][col_num-1]})[0]
                if hmm_index==phe[group] and hmm_res!='.':
                    conf_df.at[index,'Phe_num']=hit_index
                    conf_df.at[index,'Phe_restype']=list({hsps.aln[1][col_num-1]})[0]
                if hmm_index==hinge1[group] and hmm_res!='.':
                    conf_df.at[index,'Hinge1_num']=int(hit_index)
                if hmm_index in type2_resi[group] and hmm_res!='.':
                    type2_resi_num.add(int(hit_index))
                if hmm_index in back_pocket1[group] and hmm_res!='.':
                    back_pocket1_num.add(int(hit_index))
                if hmm_index in back_pocket2[group] and hmm_res!='.':
                    back_pocket2_num.add(int(hit_index))
                if hmm_index in back_pocket3[group] and hmm_res!='.':
                    back_pocket3_num.add(int(hit_index))
                if hmm_index in xdf_residues[group] and hmm_res!='.':
                    xdf_residues_num.add(int(hit_index))
                if hmm_index in front_pocket[group] and hmm_res!='.':
                    front_pocket_num.add(int(hit_index))
               
    conf_df.at[index,'Type2_resi_num']=type2_resi_num   
    conf_df.at[index,'Back_pocket1_num']=back_pocket1_num
    conf_df.at[index,'Back_pocket2_num']=back_pocket2_num
    conf_df.at[index,'Back_pocket3_num']=back_pocket3_num
    conf_df.at[index,'Front_pocket_num']=front_pocket_num
    conf_df.at[index,'XDF_residues_num']=xdf_residues_num
    return conf_df
