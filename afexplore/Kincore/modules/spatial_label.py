#!/usr/bin/env python3
# -*- coding: utf-8 -*-

def spatial_label(index,conf_df):
    dis_phe_rre4=conf_df.at[index,'Glu4-Phe-dis']
    dis_phe_lys=conf_df.at[index,'Lys-Phe-dis']
    
    if dis_phe_rre4<=11 and dis_phe_lys>=11 and dis_phe_rre4!=999 and dis_phe_lys!=999:
        conf_df.at[index,'Spatial_label']='DFGin'
        
    if dis_phe_rre4>11 and dis_phe_lys<=14 and dis_phe_rre4!=999 and dis_phe_lys!=999:
        conf_df.at[index,'Spatial_label']='DFGout'
        
    if dis_phe_rre4<=11 and dis_phe_lys<=11 and dis_phe_rre4!=999 and dis_phe_lys!=999:
        conf_df.at[index,'Spatial_label']='DFGinter'
        
    return conf_df