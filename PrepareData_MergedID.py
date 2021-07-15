#!/usr/bin/env python
# coding: utf-8

import sys
from datetime import datetime
import os
import uproot as uproot4
import glob
import pandas as pd
import numpy as np
import ROOT
import matplotlib.pyplot as plt
import json
import pickle
import seaborn as sns
sns.set() 

from Tools.PlotTools import *

#### ----- import ResolvedMergedClassifier feature list ----- ####
configfname = 'TrainConfig_ResolvedMergedClassifier_EB'
print (color.BOLD + color.BLUE + '---import configuration: {}.py---'.format(configfname) + color.END)
exec('import Tools.{} as Conf'.format(configfname)) # used to get the features list
verbose = True
print (color.BOLD + color.BLUE + '---import done!---'.format(Conf.OutputDirName)+ color.END)

#### ----- feature list: Merged ID ----- ####
feature_HDalitz = [
    'category',
    'rho',
    'nVtx',
    'nGoodVtx',
    'isPVGood',
    'instwei',
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
    'circularity_lep1', 'circularity_lep2',
    'eleConvVeto_lep1', 'eleConvVeto_lep2',
    'elePt_lep1', 'elePt_lep2'
] 

feature_GJets = [
    'rho',
    'nVtx',
    'nGoodVtx',
    'isPVGood',
    'instwei',
    'elePresel',
    'eleSCEta',
    'eleEn',
    'eleSCEn',
    'eleEcalEn',
    'eleESEnP1',
    'eleESEnP2',
    'eleD0',
    'eleDz',
    'eleSIP',
    'elePtError',
    'eleEta',
    'eleR9',
    'eleSCEtaWidth',
    'eleSCPhiWidth',
    'eleHoverE',
    'eleEoverP',
    'eleEoverPout',
    'eleEoverPInv',
    'eleBrem',
    'eledEtaAtVtx',
    'eledPhiAtVtx',
    'eleSigmaIEtaIEtaFull5x5',
    'eleSigmaIPhiIPhiFull5x5',
    'elePFChIso',
    'elePFPhoIso',
    'elePFNeuIso',
    'elePFPUIso',
    'eleIDMVAIso',
    'eleIDMVANoIso',
    'eleR9Full5x5',
    'eleTrkdxy',
    'eleMissHits',
    'nGsfMatchToReco',
    'gsfPtSum', 'gsfPtRatio', 'gsfDeltaR', 'gsfMissHitsSum',
    'circularity', 
    'eleConvVeto',
    'elePt'
] 

#### ----- Load miniTree ----- ####
treedir = '/home/chenghan/Analysis/Dalitz/electron/Misc/RECO/minitree/2017'

ROOT.EnableImplicitMT(5) # Tell ROOT you want to go parallel

print ('RDataFrame is loading HDalitz minitrees to pandas dataframe...')
df_tmp_HDalitz = ROOT.RDataFrame("outTree", "{}/Minitree_HDalitz_*.root".format(treedir))
df_HDalitz = pd.DataFrame(df_tmp_HDalitz.AsNumpy(columns = feature_HDalitz))

print ('RDataFrame is loading GJets minitrees to pandas dataframe...')
df_tmp_GJets = ROOT.RDataFrame("outTree", "{}/Minitree_gjet_*_MGG_*.root".format(treedir))
df_GJets = pd.DataFrame(df_tmp_GJets.AsNumpy(columns = feature_GJets))

# ------ Load miniTree: HR's version ------ #
# df_HDalitz = pd.DataFrame() # Merged-ID signal
# df_GJets = pd.DataFrame() # Merged-ID background
# start = datetime.now()
# if verbose:
#     print('Loading HDalitz minitrees to dataframe')
# for df_tmp in uproot4.iterate("{}/Minitree_HDalitz_*.root:outTree".format(treedir), feature_HDalitz, library="pd"):
#     df_HDalitz = df_HDalitz.append(df_tmp) 
# if verbose:
#     print('Loading Gamma+jets minitrees to dataframe')
# for df_tmp in uproot4.iterate("{}/Minitree_gjet_*_MGG_*.root:outTree".format(treedir), feature_GJets, library="pd"):
#     df_GJets = df_GJets.append(df_tmp)
# end = datetime.now()
# if verbose:
#     print ('uproot4 takes', end - start, 'to load minitrees in', treedir)


#### ----- get the classifier model ----- ####
clf_pickle = [
    pickle.load(open('./Results/Output_ResolvedMergedClassifier_EB_2017_EGM/XGB/XGB_modelXGB.pkl', 'rb')),
    pickle.load(open('./Results/Output_ResolvedMergedClassifier_EE_2017_EGM/XGB/XGB_modelXGB.pkl', 'rb'))
]

listclf = [model for model in clf_pickle]

#### ----- HDalitz dataframe ----- ####
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

print("Deal with HDalitz pandas dataframe ...")
# merged
df_merged_HDalitz = df_HDalitz.query("(category == 2) or (category == 3)")
df_merged_HDalitz_lep1 = df_merged_HDalitz.filter(regex = 'lep1') # variables of lep1
df_merged_HDalitz_lep1.columns = df_merged_HDalitz_lep1.columns.str.replace('_lep1', '') # rename the columns
df_merged_HDalitz_lep1[['rho', 'nVtx', 'nGoodVtx', 'isPVGood', 'instwei']] = df_merged_HDalitz[['rho', 'nVtx', 'nGoodVtx', 'isPVGood', 'instwei']].to_numpy()
df_merged_HDalitz_new = df_merged_HDalitz_lep1.sample(frac = 1).reset_index(drop = True) # shuffle DataFrame rows
df_merged_HDalitz_new = df_merged_HDalitz_new.dropna()

# resolved
df_resolved_HDalitz = df_HDalitz.query("(category == 1)")
df_resolved_HDalitz_lep1 = df_resolved_HDalitz.filter(regex = 'lep1') # variables of lep1 
df_resolved_HDalitz_lep2 = df_resolved_HDalitz.filter(regex = 'lep2') # variables of lep2
df_resolved_HDalitz_lep1.columns = df_resolved_HDalitz_lep1.columns.str.replace('_lep1', '') # rename the columns
df_resolved_HDalitz_lep2.columns = df_resolved_HDalitz_lep2.columns.str.replace('_lep2', '')
df_resolved_HDalitz_lep1[['rho', 'nVtx', 'nGoodVtx', 'isPVGood', 'instwei']] = df_resolved_HDalitz[['rho', 'nVtx', 'nGoodVtx', 'isPVGood', 'instwei']].to_numpy()
df_resolved_HDalitz_lep2[['rho', 'nVtx', 'nGoodVtx', 'isPVGood', 'instwei']] = df_resolved_HDalitz[['rho', 'nVtx', 'nGoodVtx', 'isPVGood', 'instwei']].to_numpy()
df_resolved_HDalitz_new = pd.concat([df_resolved_HDalitz_lep1, df_resolved_HDalitz_lep2], ignore_index = True, sort = False) # concatenate the dataframe
df_resolved_HDalitz_new = df_resolved_HDalitz_new.sample(frac = 1).reset_index(drop = True) # shuffle DataFrame rows
df_resolved_HDalitz_new = df_resolved_HDalitz_new.dropna()

# EB EE (combine resolved and merged)
df_HDalitz_new = pd.concat([df_merged_HDalitz_new, df_resolved_HDalitz_new], ignore_index = True, sort = False) # concatenate the dataframe 
df_HDalitz_new = df_HDalitz_new.sample(frac = 1).reset_index(drop = True) # shuffle DataFrame rows
df_HDalitz_new = df_HDalitz_new.dropna()
df_HDalitz_EB_new = df_HDalitz_new.query('(abs(eleSCEta) <= 1.479)')
df_HDalitz_EE_new = df_HDalitz_new.query('(abs(eleSCEta) > 1.479 and abs(eleSCEta) <= 2.5)')

# add classifier information
X_EB_HDalitz = df_HDalitz_EB_new.loc[:,Conf.features['XGB']] # extract the features columns
df_HDalitz_EB_new['XGB_pred'] = listclf[0].predict_proba(X_EB_HDalitz)[:,1]
X_EE_HDalitz = df_HDalitz_EE_new.loc[:,Conf.features['XGB']] # extract the features columns
df_HDalitz_EE_new['XGB_pred'] = listclf[1].predict_proba(X_EE_HDalitz)[:,1]

df_HDalitz_EBEE_new = pd.concat([df_HDalitz_EB_new, df_HDalitz_EE_new], ignore_index = True, sort = False) # concatenate the dataframe 
df_HDalitz_EBEE_new = df_HDalitz_EBEE_new.dropna()
print (color.RED + '[INFO] Does HDalitz dataframe have NAN value(s)? {}'.format(df_HDalitz_EBEE_new.isnull().values.any()) + color.END)

#### ----- GJets dataframe ----- ####
print("Deal with GJets pandas dataframe ...")
df_GJets_EB = df_GJets.query('(abs(eleSCEta) <= 1.479)')
df_GJets_EE = df_GJets.query('(abs(eleSCEta) > 1.479 and abs(eleSCEta) <= 2.5)')

X_EB_GJets = df_GJets_EB.loc[:,Conf.features['XGB']] 
df_GJets_EB['XGB_pred'] = listclf[0].predict_proba(X_EB_GJets)[:,1]
X_EE_GJets = df_GJets_EE.loc[:,Conf.features['XGB']] 
df_GJets_EE['XGB_pred'] = listclf[1].predict_proba(X_EE_GJets)[:,1]

df_GJets_EBEE_new = pd.concat([df_GJets_EB, df_GJets_EE], ignore_index = True, sort = False) # concatenate the dataframe
df_GJets_EBEE_new = df_GJets_EBEE_new.dropna()
print (color.RED + '[INFO] Does GJets dataframe have NAN value(s)? {}'.format(df_GJets_EBEE_new.isnull().values.any()) + color.END)

#### ----- Save the dataframe to pkl files ----- ####
def df2pickle(df, datasetname):
    datadir = './data/Merged-ID'
    os.makedirs(datadir, exist_ok=True)

    category = ['EB','EE']
    sel = ['(abs(eleSCEta) <= 1.479)', '(abs(eleSCEta) > 1.479 and abs(eleSCEta) <= 2.5)']

    for i, cat in enumerate(category):
        df_tmp = df.query(sel[i])

        file = open('{}/Dataframe_MergedID_{}_{}_2017.pkl'.format(datadir, datasetname, cat),'wb')
        pickle.dump(df_tmp, file)
        file.close()

    file = open('{}/Dataframe_MergedID_{}_2017.pkl'.format(datadir, datasetname),'wb')
    pickle.dump(df, file)
    file.close()
    
print("Save HDalitz pkl file ...")
df2pickle(df_HDalitz_EBEE_new, 'HDalitz_allprod_eeg_allMass')

print("Save GJets pkl file ...")
df2pickle(df_GJets_EBEE_new, 'GJets')

print (color.BOLD + color.BLUE + '---Data preparation done!---'.format(Conf.OutputDirName)+ color.END)

# print("Draw the correlation plot ...")
# corrdir = './plots/correlations/MergedID'
# os.makedirs(corrdir, exist_ok=True)
# corr_HDalitz = df_HDalitz_EBEE_new.corr()
# corr_GJets = df_GJets_EBEE_new.corr()
# corrplot(corr_HDalitz, (11,11), size_scale=125, marker='s', plotname='{}/correlation_HDalitz'.format(corrdir))
# corrplot(corr_GJets, (11,11), size_scale=125, marker='s', plotname='{}/correlation_GJets'.format(corrdir))