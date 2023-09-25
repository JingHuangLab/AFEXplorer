#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 09:58:48 2020

@author: vivekmodi
"""

import numpy as np
from Bio import PDB

def compute_distance(pdbfilename,index,conf_df,structure):
    restype_atom_dict={'F':'CZ','R':'CZ','L':'CG','P':'CG','N':'CG','M':'CE','S':'OG','H':'NE2','V':'CB','A':'CB','W':'CZ3','Y':'OH','G':'CA'}
    try:
        phe_atom_type=restype_atom_dict[conf_df.at[index,'Phe_restype']]
    except:
        conf_df.at[index,'Glu4-Phe-dis']=999
        conf_df.at[index,'Lys-Phe-dis']=999
        conf_df.at[index,'Lys-Glu-dis']=999
        return conf_df

    #Distance Glu4-Phe
    conf_df.at[index,'Glu4-Phe-dis']=distance_atoms(pdbfilename,conf_df.at[index,'Model_id'],conf_df.at[index,'Chain_id'],\
                                     conf_df.at[index,'Glu4_num'],conf_df.at[index,'Phe_num'],'CA',phe_atom_type,structure)      #Change atom names for other residue types

    #Distance Lys-Phe
    conf_df.at[index,'Lys-Phe-dis']=distance_atoms(pdbfilename,conf_df.at[index,'Model_id'],conf_df.at[index,'Chain_id'],\
                                    conf_df.at[index,'Lys_num'],conf_df.at[index,'Phe_num'],'CA',phe_atom_type,structure)      #Change atom names for other residue types

    #Distance Lys-Glu
    conf_df.at[index,'Lys-Glu-dis']=distance_atoms(pdbfilename,conf_df.at[index,'Model_id'],conf_df.at[index,'Chain_id'],\
                                    conf_df.at[index,'Lys_num'],conf_df.at[index,'Glu_num'],'CB','CB',structure)      #Change atom names for other residue types

    return conf_df


def distance_atoms(pdbfilename,model_id,chain_id,res1,res2,atm1,atm2,structure):
    atom_present=0; res1=int(res1); res2=int(res2)

    for model in structure:
        for chain in model:
            insertion_num=0    #Count residues with insertion codes and skip them; Not the best way to solve the problem. The fasta file should not have residues with insert code.
            if int(model.id)==int(model_id) and chain.id==chain_id:
                for residue in chain:
                    if residue.get_id()[0]==' ' and residue.get_id()[2]!=' ':      #Insertion code present
                        insertion_num+=1

                    if int(residue.id[1])==(int(res1)-insertion_num) and residue.get_id()[0]==' ':
                        if residue.has_id(atm1):
                            residue1=chain[res1-insertion_num]
                            atom_present=atom_present+1

                    if int(residue.id[1])==(int(res2)-insertion_num) and residue.get_id()[0]==' ':
                        if residue.has_id(atm2):
                            residue2=chain[res2-insertion_num]
                            atom_present=atom_present+1


    if atom_present==2:
        distance=np.round((residue1[atm1]-residue2[atm2]),2)
        return distance
    else:
        return 999
