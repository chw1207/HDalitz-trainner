#!/usr/bin/env python
# coding: utf-8

import sys
import os
import uproot as uproot4
import pandas as pd
import numpy as np
import ROOT
import matplotlib.pyplot as plt
import json
import pickle
import seaborn as sns
sns.set() 
verbose = True

from glob import glob

list_feature = [
    'category',
    'rho',
    'nVtx',
    'instwei',
    'mcwei',
    'genwei',
    'elePresel_lep1', 'elePresel_lep2',
    'eleSCEta_lep1', 'eleSCEta_lep2',
    'eleEn_lep1','eleEn_lep2',
    'eleSCEn_lep1','eleSCEn_lep2',
    'eleEcalEn_lep1','eleEcalEn_lep2',
    'eleESEnP1_lep1','eleESEnP1_lep2',
    'eleESEnP2_lep1','eleESEnP2_lep2',
    'eleD0_lep1','eleD0_lep2',
    'eleDz_lep1','eleDz_lep2',
    'eleSIP_lep1','eleSIP_lep2',
    'elePtError_lep1','elePtError_lep2',
    'eleEta_lep1','eleEta_lep2',
    'eleR9_lep1','eleR9_lep2',
    'eleSCEtaWidth_lep1','eleSCEtaWidth_lep2',
    'eleSCPhiWidth_lep1','eleSCPhiWidth_lep2',
    'eleHoverE_lep1','eleHoverE_lep2',
    'eleEoverP_lep1','eleEoverP_lep2',
    'eleEoverPout_lep1','eleEoverPout_lep2',
    'eleEoverPInv_lep1','eleEoverPInv_lep2',
    'eleBrem_lep1','eleBrem_lep2',
    'eledEtaAtVtx_lep1','eledEtaAtVtx_lep2',
    'eledPhiAtVtx_lep1','eledPhiAtVtx_lep2',
    'eleSigmaIEtaIEtaFull5x5_lep1','eleSigmaIEtaIEtaFull5x5_lep2',
    'eleSigmaIPhiIPhiFull5x5_lep1','eleSigmaIPhiIPhiFull5x5_lep2',
    'elePFChIso_lep1','elePFChIso_lep2',
    'elePFPhoIso_lep1','elePFPhoIso_lep2',
    'elePFNeuIso_lep1','elePFNeuIso_lep2',
    'elePFPUIso_lep1','elePFPUIso_lep2',
    'eleIDMVAIso_lep1','eleIDMVAIso_lep2',
    'eleIDMVANoIso_lep1','eleIDMVANoIso_lep2',
    'eleR9Full5x5_lep1','eleR9Full5x5_lep2',
    'eleTrkdxy_lep1','eleTrkdxy_lep2',
    'eleMissHits_lep1','eleMissHits_lep2',
    'nGsfMatchToReco_lep1','nGsfMatchToReco_lep2',
    'gsfPtSum_lep1', 'gsfPtRatio_lep1', 'gsfDeltaR_lep1', 'gsfMissHitsSum_lep1',
    'eleConvVeto_lep1', 'eleConvVeto_lep2',
    'circularity_lep1', 'circularity_lep2',
    'elePt_lep1', 'elePt_lep2',
    'eleCalibPt_lep1', 'eleCalibPt_lep2',
    'eleDiffCalibOriPt_lep1', 'eleDiffCalibOriPt_lep2',
    'eleSCRawEn_lep1', 'eleSCRawEn_lep2'

] 

treedir = '/home/chenghan/Analysis/Dalitz/electron/Misc/RECO/minitree/2017'
HDalitz_files = glob("{}/Minitree_HDalitz_*.root".format(treedir))
DYJets_files = glob("{}/Minitree_DYJetsToLL_*.root".format(treedir))
allfiles = HDalitz_files + DYJets_files

# reference: https://root.cern/doc/master/df026__AsNumpyArrays_8py.html
print ('RDataFrame is loading minitrees...')
ROOT.EnableImplicitMT(len(allfiles)) # Tell ROOT you want to go parallel

# use the merged electrons from HDalitz and resolved electrons from DY+Jets
df_HDalitz = ROOT.RDataFrame("outTree", HDalitz_files)
df_tmp_merged = df_HDalitz.Filter("category == 2 || category == 3", "merged")
df_DYJets = ROOT.RDataFrame("outTree", DYJets_files)
df_tmp_resolved = df_DYJets.Filter("category == 1", "resolved")

# df = ROOT.RDataFrame("outTree", allfiles)
# df_tmp_resolved = df.Filter("category == 1", "resolved")
# df_tmp_merged = df.Filter("category == 2 || category == 3", "merged")

print ("Convert RDataFrame to pandas dataframe...")
df_resolved = pd.DataFrame(df_tmp_resolved.AsNumpy(columns = list_feature))
df_merged = pd.DataFrame(df_tmp_merged.AsNumpy(columns = list_feature))

# ------ HR's version ------ #
# df_DYJets = pd.DataFrame()
# df_HDalitz = pd.DataFrame()

# print ('uproot loading HDalitz minitrees...')
# for df_tmp in uproot4.iterate("{}/Minitree_HDalitz*.root:outTree".format(treedir), list_feature, library="pd"):
#     df_HDalitz = df_HDalitz.append(df_tmp)
# print ('uproot loading DYJetsToLL minitrees...')
# for df_tmp in uproot4.iterate("{}/Minitree_DYJetsToLL*.root:outTree".format(treedir), list_feature, library="pd"):
#     df_DYJets = df_DYJets.append(df_tmp)
# df_test = pd.concat([df_DYJets, df_HDalitz], ignore_index = True, sort = False) # concatenate the dataframe
# df_resolved = df_test.query("(category == 1)")
# df_merged = df_test.query("(category == 2) or (category == 3)")
# del df_test

import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

print ("Deal with pandas dataframe...")
df_resolved_lep1 = df_resolved.filter(regex = 'lep1') # variables of lep1 
df_resolved_lep2 = df_resolved.filter(regex = 'lep2') # variables of lep2
df_resolved_lep1.columns = df_resolved_lep1.columns.str.replace('_lep1', '') # rename the columns
df_resolved_lep2.columns = df_resolved_lep2.columns.str.replace('_lep2', '')
df_resolved_lep1[['rho', 'nVtx', 'instwei', 'mcwei', 'genwei']] = df_resolved[['rho', 'nVtx', 'instwei', 'mcwei', 'genwei']].to_numpy()
df_resolved_lep2[['rho', 'nVtx', 'instwei', 'mcwei', 'genwei']] = df_resolved[['rho', 'nVtx', 'instwei', 'mcwei', 'genwei']].to_numpy()
df_resolved_new = pd.concat([df_resolved_lep1, df_resolved_lep2], ignore_index = True, sort = False) # concatenate the dataframe 

del df_resolved_lep1, df_resolved_lep2
df_resolved_new = df_resolved_new.sample(frac = 1).reset_index(drop = True) # shuffle DataFrame rows
df_resolved_new = df_resolved_new.dropna()

df_merged_new = df_merged.filter(regex = 'lep1') # variables of lep1 
df_merged_new.columns = df_merged_new.columns.str.replace('_lep1', '') # rename the columns
df_merged_new[['rho', 'nVtx', 'instwei', 'mcwei', 'genwei']] = df_merged[['rho', 'nVtx', 'instwei', 'mcwei', 'genwei']].to_numpy()
df_merged_new = df_merged_new.sample(frac = 1).reset_index(drop = True) # shuffle DataFrame rows
df_merged_new = df_merged_new.dropna()

print("Save the pkl files...")
datadir = './data/Resolved-Merged-Classifier'
os.makedirs(datadir, exist_ok = True)

category = ['EB','EE']
sel = ['(abs(eleSCEta) <= 1.479)', '(abs(eleSCEta) > 1.479 and abs(eleSCEta) <= 2.5)']

for i, cat in enumerate(category):
    df1_tmp = df_resolved_new.query(sel[i])
    df2_tmp = df_merged_new.query(sel[i])
    
    if verbose:
        print('[Count] # of valid entries in {} of the resolved dataframe ='.format(cat), len(df1_tmp.index))
        print('[Count] # of valid entries in {} of the merged dataframe ='.format(cat), len(df2_tmp.index))

    file = open('{}/Dataframe_ResolvedMerged_HDalitz_allprod_eeg_allMass_{}_2017.pkl'.format(datadir,cat),'wb')
    print('[Save] {}/Dataframe_ResolvedMerged_HDalitz_allprod_eeg_allMass_{}_2017.pkl'.format(datadir,cat))
    pickle.dump(df1_tmp, file)
    pickle.dump(df2_tmp, file)
    file.close()
    
if verbose:
    print('[Count] # of valid entries of the resolved dataframe =',len(df_resolved_new.index))
    print('[Count] # of valid entries of the merged dataframe =',len(df_merged_new.index))

file = open('{}/Dataframe_ResolvedMerged_HDalitz_allprod_eeg_allMass_2017.pkl'.format(datadir),'wb')
print('[Save] {}/Dataframe_ResolvedMerged_HDalitz_allprod_eeg_allMass_2017.pkl'.format(datadir))
pickle.dump(df_resolved_new, file)
pickle.dump(df_merged_new, file)
file.close()





