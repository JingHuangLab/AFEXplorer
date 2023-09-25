#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri July 31 17:35:32 2020

@author: vivekmodi
"""

import pandas as pd, sys, os, gzip, argparse
from Bio import PDB

from Kincore.modules.assign_default_values import assign_default_values
from Kincore.modules.extract_sequence import extract_seq
from Kincore.modules.run_hmmsearch import run_hmmsearch
from Kincore.modules.identify_restypes import identify_restypes
from Kincore.modules.identify_group import identify_group
from Kincore.modules.identify_residues import identify_residues
from Kincore.modules.compute_distance import compute_distance
from Kincore.modules.compute_dihedrals import compute_dihedrals
from Kincore.modules.chelix import chelix_conformation
from Kincore.modules.spatial_label import spatial_label
from Kincore.modules.dihedral_label import dihedral_label
from Kincore.modules.delete_files import delete_files
from Kincore.modules.extract_ligands import extract_ligands
from Kincore.modules.classify_ligands import classify_ligands

def read_inputlist(hmm_loc,pdbfilename,align,user_chain,user_lys,user_glu,user_phe,header):     #Function to read input file list
    fhandle_files=open(pdbfilename,'r')
    for names in fhandle_files:
        name=names.strip();names=names.split()
        identify_state(hmm_loc,name,align,user_chain,user_lys,user_glu,user_phe,header)
        header=1

def print_header_withoutgroup(pdbfilename,index,conf_df):       #Print Header when align=False
    print('Input'.rjust(len(pdbfilename)+1)+'Model'.rjust(6)+'Chain'.rjust(6)+'Spatial_label'.rjust(14)+\
          'Dihedral_label'.rjust(15)+'C-helix_label'.rjust(14)+'X-DFG Φ Ψ'.center(22)+'DFG-Asp Φ Ψ'.center(22)+'DFG-Phe Φ Ψ χ1'.center(30))

def print_header_withgroup(pdbfilename,index,conf_df):    #Print Header when align=True
    print('Input'.rjust(len(pdbfilename)+1)+'Group'.rjust(6)+'Model'.rjust(6)+'Chain'.rjust(6)+'Spatial_label'.rjust(14)+\
    'Dihedral_label'.rjust(15)+'C-helix_label'.rjust(14)+'Ligand'.rjust(len(conf_df.at[index,'Ligand'])+1)+'Ligand_label'.rjust(len(conf_df.at[index,'Ligand_label'])+9)+\
    'X-DFG Φ Ψ'.center(22)+'DFG-Asp Φ Ψ'.center(22)+'DFG-Phe Φ Ψ χ1'.center(30))

def identify_state(pwd,pdbfilename,align,user_chain,user_lys,user_glu,user_phe,header):
    conf_df=pd.DataFrame()
    chain_list=list()

    if '.gz' in pdbfilename.lower():      #Check if the file is compressed
        handle=gzip.open(pdbfilename,'rt')
    else:
        handle=open(pdbfilename,'r')

    if '.cif' in pdbfilename.lower():
            parser=PDB.MMCIFParser(QUIET=True)
    if '.pdb' in pdbfilename.lower():
            parser=PDB.PDBParser(QUIET=True)

    structure=parser.get_structure(pdbfilename, handle)

    if align.upper()=='FALSE':
        for model in structure:
            index=int(model.id)
            conf_df.at[index,'Model_id']=int(model.id)
            for chain in model:
                for chains in list(user_chain.split(',')):
                    if chain.id==chains:       #Match user chain with chain in structure
                        conf_df.at[index,'Chain_id']=chain.id
                        conf_df=assign_default_values(index,conf_df)
                        conf_df.at[index,'Lys_num']=int(user_lys)
                        conf_df.at[index,'Glu_num']=int(user_glu)
                        conf_df.at[index,'Glu4_num']=int(user_glu)+4
                        conf_df.at[index,'Phe_num']=int(user_phe)
                        conf_df.at[index,'XDFG_num']=int(user_phe)-2
                        conf_df.at[index,'Asp_num']=int(user_phe)-1
                        conf_df=identify_restypes(pdbfilename,conf_df,index,structure)
                        conf_df=compute_distance(pdbfilename,index,conf_df,structure)
                        conf_df=compute_dihedrals(pdbfilename,index,conf_df,structure)
                        conf_df=spatial_label(index,conf_df)
                        conf_df=dihedral_label(index,conf_df,0.45)
                        conf_df=chelix_conformation(index,conf_df)
                        if header==0:
                            print_header_withoutgroup(pdbfilename,index,conf_df)
                            header=1
                        xdfg_string=str(str(int(conf_df.at[index,'XDFG_num']))+conf_df.at[index,'XDFG_restype'])
                        asp_string=str(str(int(conf_df.at[index,'Asp_num']))+conf_df.at[index,'Asp_restype'])
                        phe_string=str(str(int(conf_df.at[index,'Phe_num']))+conf_df.at[index,'Phe_restype'])
                        print(pdbfilename.rjust(len(pdbfilename)+1)+str(int(conf_df.at[index,'Model_id'])).rjust(6)+conf_df.at[index,'Chain_id'].rjust(6)+\
                              conf_df.at[index,'Spatial_label'].rjust(14)+conf_df.at[index,'Dihedral_label'].rjust(15)+conf_df.at[index,'Chelix'].rjust(14)+\
                              xdfg_string.rjust(6)+str('%.2f' %conf_df.at[index,'XDFG_Phi']).rjust(8)+str('%.2f' %conf_df.at[index,'XDFG_Psi']).rjust(8)+\
                              asp_string.rjust(6)+str('%.2f' %conf_df.at[index,'Asp_Phi']).rjust(8)+str('%.2f' %conf_df.at[index,'Asp_Psi']).rjust(8)+phe_string.rjust(6)+str('%.2f' %conf_df.at[index,'Phe_Phi']).rjust(8)+\
                              str('%.2f' %conf_df.at[index,'Phe_Psi']).rjust(8)+str('%.2f' %conf_df.at[index,'Phe_Chi1']).rjust(8))

    elif align.upper()=='TRUE':
        index=-1
        for model in structure:
            for chain in model:
                if len(chain.get_list())<=30:
                    continue
                chain_list.append(chain.id)
                index+=1

                for residue in chain:
                    if residue.id[2]!=' ':
                        print(f'{pdbfilename} Please enter a structure file without insertion codes.')
                        sys.exit()
                conf_df.at[index,'Model_id']=str(model.id)
                conf_df.at[index,'Chain_id']=chain.id
                conf_df=assign_default_values(index,conf_df)

                conf_df=extract_seq(pdbfilename,index,conf_df)
                run_hmmsearch(pwd,pdbfilename,index,conf_df)
                conf_df=identify_group(pdbfilename,index,conf_df)

                if conf_df.at[index,'Group']=='None':
                    delete_files(pdbfilename,conf_df.at[index,'Model_id'],conf_df.at[index,'Chain_id'], conf_df.at[index,'Group'])


                else:
                    conf_df=identify_residues(pdbfilename,index,conf_df)
                    if conf_df.at[index,'Lys_restype']=='X' or conf_df.at[index,'Glu_restype']=='X' or conf_df.at[index,'Phe_restype']=='X' or conf_df.at[index,'XDFG_restype']=='X' or conf_df.at[index,'Asp_restype']=='X':
                        #print('#'+pdbfilename.rjust(13)+conf_df.at[index,'Group'].rjust(6)+str(int(conf_df.at[index,'Model_id'])).rjust(6)+conf_df.at[index,'Chain_id'].rjust(6)+'       Conserved residues missing in the structure.')
                        delete_files(pdbfilename,conf_df.at[index,'Model_id'],conf_df.at[index,'Chain_id'], conf_df.at[index,'Group'])

                    else:
                        conf_df=compute_distance(pdbfilename,index,conf_df,structure)
                        conf_df=compute_dihedrals(pdbfilename,index,conf_df,structure)
                        conf_df=chelix_conformation(index,conf_df)
                        conf_df=spatial_label(index,conf_df)
                        conf_df=dihedral_label(index,conf_df,0.45)
                        conf_df=chelix_conformation(index,conf_df)
                        conf_df=extract_ligands(pdbfilename,index,conf_df,structure)
                        try:
                            conf_df=classify_ligands(pdbfilename,index,conf_df,structure)
                        except:
                            conf_df.at[index,'Ligand_label']='Error'

                        if header==0:
                            print_header_withgroup(pdbfilename,index,conf_df)
                            header=1

                        xdfg_string=str(str(int(conf_df.at[index,'XDFG_num']))+conf_df.at[index,'XDFG_restype'])
                        asp_string=str(str(int(conf_df.at[index,'Asp_num']))+conf_df.at[index,'Asp_restype'])
                        phe_string=str(str(int(conf_df.at[index,'Phe_num']))+conf_df.at[index,'Phe_restype'])
                        print(pdbfilename.rjust(len(pdbfilename)+1)+conf_df.at[index,'Group'].rjust(6)+conf_df.at[index,'Model_id'].rjust(6)+conf_df.at[index,'Chain_id'].rjust(6)+\
                        # conf_df.at[index,'Spatial_label'].rjust(14)+conf_df.at[index,'Dihedral_label'].rjust(15)+conf_df.at[index,'Chelix'].rjust(14)+conf_df.at[index,'Ligand'].rjust(len(conf_df.at[index,'Ligand'])+1)+\
                        # conf_df.at[index,'Ligand_label'].rjust(len(conf_df.at[index,'Ligand_label'])+9)+xdfg_string.rjust(6)+str('%.2f' %conf_df.at[index,'XDFG_Phi']).rjust(8)+str('%.2f' %conf_df.at[index,'XDFG_Psi']).rjust(8)+\
                        # asp_string.rjust(6)+str('%.2f' %conf_df.at[index,'Asp_Phi']).rjust(8)+str('%.2f' %conf_df.at[index,'Asp_Psi']).rjust(8)+phe_string.rjust(6)+str('%.2f' %conf_df.at[index,'Phe_Phi']).rjust(8)+\
                        # str('%.2f' %conf_df.at[index,'Phe_Psi']).rjust(8)+str('%.2f' %conf_df.at[index,'Phe_Chi1']).rjust(8))
                        str('%.3f' %conf_df.at[index,'Glu4-Phe-dis']).rjust(8)+str('%.3f' %conf_df.at[index,'Lys-Phe-dis']).rjust(8))

                        delete_files(pdbfilename,conf_df.at[index,'Model_id'],conf_df.at[index,'Chain_id'],conf_df.at[index,'Group'])

        for i in conf_df.index:
            if conf_df.at[i,'Group']=='None':
                print(f'# {pdbfilename} Model {conf_df.at[i,"Model_id"]} Chain {conf_df.at[i,"Chain_id"]} is probably not a protein kinase.')
            if conf_df.at[i,'Group']!='None' and (conf_df.at[i,'Lys_restype']=='X' or conf_df.at[i,'Glu_restype']=='X' or conf_df.at[i,'Phe_restype']=='X' or conf_df.at[i,'XDFG_restype']=='X' or conf_df.at[i,'Asp_restype']=='X'):
                print(f'# {pdbfilename} Model {conf_df.at[i,"Model_id"]} Chain {conf_df.at[i,"Chain_id"]} is missing the required residues.')

    return conf_df


if __name__ == '__main__':
    hmm_loc=os.path.dirname(os.path.realpath(__file__))+'/HMMs'    #gets the original location of the file
    parser=argparse.ArgumentParser()
    parser.add_argument('PDB',help='PDB file in .cif or .pdb format. You can also provide compressed file as file.cif.gz or file.pdb.gz or list of filenames in a .txt file.')
    parser.add_argument('Align',help='True/False; If "True" (Slower) then the program will align the input protein sequence to pre-computed HMMs and identify conserved \
                        residues to determine kinase conformational labels and ligand type (Example $python kinase_state.py 1GAG.pdb True).\
                        If "False" (Faster) then please provide the optional arguments (Example $python kinase_state.py -L 1030 -G 1047 -P 1151 -C A,B,C 1GAG.pdb False). Please note that \
                        the program can provide ligand information only when this argument is True.')

    parser.add_argument('-L','--Lys',help='Residue number of the conserved B3-Lys in the input PDB file')
    parser.add_argument('-G','--Glu',help='Residue number of the conserved C-helix-Glu in the input PDB file')
    parser.add_argument('-P','--Phe',help='Residue number of the conserved DFG-Phe in the input PDB file')
    parser.add_argument('-C','--Chain',type=str,help='Chain id of the protein kinase chain in the input file')
    args=parser.parse_args()

    pdbfilename=args.PDB
    if args.Align.upper()=='TRUE':
        user_chain=user_lys=user_glu=user_phe=align=''
    else:
        if not args.Lys or not args.Glu or not args.Phe or not args.Chain:
            print('Please provide residue numbers and chain id.')
            exit()
        else:
            user_chain=str(args.Chain);user_lys=int(args.Lys);user_glu=int(args.Glu);user_phe=int(args.Phe)


    header=0
    if '.txt' in pdbfilename:
        read_inputlist(hmm_loc,pdbfilename,args.Align,user_chain,user_lys,user_glu,user_phe,header)
    else:
        identify_state(hmm_loc,pdbfilename,args.Align,user_chain,user_lys,user_glu,user_phe,header)
