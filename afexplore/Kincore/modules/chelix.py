#!/usr/bin/env python3
# -*- coding: utf-8 -*-


def chelix_conformation(index,conf_df):
    dis_sb=conf_df.at[index,'Lys-Glu-dis']
    
    if dis_sb<=10:
        conf_df.at[index,'Chelix']='Chelix-in'
    else:
        conf_df.at[index,'Chelix']='Chelix-out'

    return conf_df