#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 12:27:13 2020

@author: vivekmodi
"""

def assign_default_values(index,conf_df):
    conf_df.at[index,'Sequence']='X'
    conf_df.at[index,'First_res']=999
    conf_df.at[index,'Group']='None'
    conf_df.at[index,'Lys_num']=999
    conf_df.at[index,'Lys_restype']='X'
    conf_df.at[index,'Glu_num']=999
    conf_df.at[index,'Glu_restype']='X'
    conf_df.at[index,'Glu4_num']=999
    conf_df.at[index,'Glu4_restype']='X'
    conf_df.at[index,'XDFG_num']=999
    conf_df.at[index,'XDFG_restype']='X'
    conf_df.at[index,'Asp_num']=999
    conf_df.at[index,'Asp_restype']='X'
    conf_df.at[index,'Phe_num']=999
    conf_df.at[index,'Phe_restype']='X'
    conf_df.at[index,'Glu4-Phe-dis']=999
    conf_df.at[index,'Lys-Phe-dis']=999
    conf_df.at[index,'Lys-Glu-dis']=999
    conf_df.at[index,'XDFG_Phi']=999
    conf_df.at[index,'XDFG_Psi']=999
    conf_df.at[index,'Asp_Phi']=999
    conf_df.at[index,'Asp_Psi']=999
    conf_df.at[index,'Phe_Phi']=999
    conf_df.at[index,'Phe_Psi']=999
    conf_df.at[index,'Phe_Chi1']=999
    conf_df.at[index,'Chelix']='Unassigned'
    conf_df.at[index,'Spatial_label']='Unassigned'
    conf_df.at[index,'Dihedral_label']='Unassigned'
    conf_df.at[index,'Ligand']='None'
    conf_df.at[index,'Ligand_label']='None'

    return conf_df
