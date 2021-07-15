#!/usr/bin/env python
# coding: utf-8
import sys
import os
import uproot
import glob
import pandas as pd
import numpy as np
from numpy import sort
import ROOT
import matplotlib.pyplot as plt
import mplhep as hep
import json
import pickle
import multiprocessing 
os.system("")

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_curve, auc, confusion_matrix, classification_report, recall_score, roc_auc_score, average_precision_score
from sklearn.feature_selection import SelectFromModel

from Tools.PlotTools import *
import seaborn as sns

def in_ipynb():
    try:
        cfg = get_ipython().config
        # print(cfg)
        if 'jupyter' in cfg['IPKernelApp']['connection_file']:
            return True
        else:
            return False
    except NameError:
        return False

# configfname = 'TrainConfig_ResolvedMergedClassifier_EB'
# print (color.BOLD + color.BLUE + '---Load configuration: {}.py---'.format(configfname) + color.END)
# importConfig = ""
# if in_ipynb(): 
#     print (color.BOLD + color.BLUE + '---Run in IPython notebook---'.format(configfname) + color.END)
#     exec('import Tools.{} as Conf'.format(configfname))
#     TrainConfig='Tools/{}'.format(configfname)
# else:
TrainConfig = sys.argv[1]
prGreen("Importing settings from "+ TrainConfig.replace("/", "."))
importConfig = TrainConfig.replace("/", ".")
exec("import "+importConfig+" as Conf")
    
print (color.BOLD + color.BLUE + '---Whether to use debug mode: {}---'.format(Conf.Debug)+ color.END)
if Conf.Debug==True:
    prGreen("Running in debug mode : Only every 10th event will be used")
    
print (color.BOLD + color.BLUE + '---Making output directory: {}---'.format(Conf.OutputDirName)+ color.END)
if len(Conf.MVAs)>0:
    for MVAd in Conf.MVAs:
        os.system("mkdir -p " + Conf.OutputDirName+"/"+MVAd)
os.system("mkdir -p " + Conf.OutputDirName)
os.system("mkdir -p " + Conf.OutputDirName+"/CodeANDConfig")
os.system("mkdir -p " + Conf.OutputDirName+"/Thresholds")
os.system("mkdir -p " + Conf.OutputDirName+"/FeaturePlot")
os.system("cp "+TrainConfig+".py ./"+ Conf.OutputDirName+"/CodeANDConfig/")
os.system("cp Trainer.py ./"+ Conf.OutputDirName+"/CodeANDConfig/")

print (color.BOLD + color.BLUE + '---Preparation done!---'.format(Conf.OutputDirName)+ color.END)


## ----- Load the pkl files (type: pandas Dataframe) -----##
cat = 'EleType'
weight = 'NewWt'
label = Conf.ClfLabel
Clfname = Conf.Clfname

Bkgdf = pd.DataFrame()
Sigdf = pd.DataFrame()

if Clfname == 'RMClf':
    # For Resolved-Merged classifier, data are stored in a single pickle, but two pandas dataframes
    # https://stackoverflow.com/a/15463472
    picklefname = Conf.Picklefname
    print (color.BOLD + color.BLUE + '---Loading dataframes from: {}---'.format(picklefname) + color.END)
    with open(picklefname, 'rb') as file:
        if Conf.CommonCut:
            prGreen('Select the events/objects with commen cuts: {}'.format(Conf.CommonCut))
            Bkgdf = pickle.load(file).query(Conf.CommonCut) # Bkgdf -> Resolved 
            Sigdf = pickle.load(file).query(Conf.CommonCut) # Sigdf -> Merged
        else:
            Bkgdf = pickle.load(file) # Bkgdf -> Resolved 
            Sigdf = pickle.load(file) # Sigdf -> Merged

elif Clfname == 'MergedID':
    print (color.BLUE + '---Loading background dataframes from: {}---'.format(Conf.Pickle_bkg) + color.END)
    with open(Conf.Pickle_bkg, "rb") as f:
        if Conf.CommonCut:
            prGreen('Select the background events/objects with commen cuts: {}'.format(Conf.CommonCut))
            Bkgdf = pickle.load(f).query(Conf.CommonCut)
        else:
            Bkgdf = pickle.load(f)
    
    print (color.BLUE + 'Loading signal dataframes from: {}'.format(Conf.Pickle_signal) + color.END)
    with open(Conf.Pickle_signal, "rb") as f:
        if Conf.CommonCut:
            prGreen ('Select the signal events/objects with commen cuts: {}'.format(Conf.CommonCut))
            Sigdf = pickle.load(f).query(Conf.CommonCut)
        else:
            Sigdf = pickle.load(f)
else:
    print (color.BOLD + color.RED + '[ERROR] Classifier name {} is not correct. Please check!'.format(Clfname) + color.END)
    sys.exit(0)
    
print (color.BLUE + '---Dataframes loading done!---' + color.END)

## ----- Split the training and test Indices -----##
# Reference:
# [1] scale_pos_weight: https://xgboost.readthedocs.io/en/latest/parameter.html
# [2] https://github.com/dmlc/xgboost/blob/master/demo/kaggle-higgs/higgs-numpy.py

Sigdf[cat] = 1
Bkgdf[cat] = 0

Sigdf['Type'] = 'Signal'
Bkgdf['Type'] = 'Background'

if weight == 'NewWt':
    Sigdf[weight] = 1
    Bkgdf[weight] = 1
    # Bkgdf = Bkgdf.sample(n = len(Sigdf.index)) ## Use the same number of resolved electrons as merged ones in the training
if Conf.Reweighing != 'Nothing':
    print ('Before pT eta reweghting: ')
print('Number of entries of the background dataframe =', len(Bkgdf.index))
print('Number of entries of the signal dataframe =', len(Sigdf.index))
df_final = pd.concat([Sigdf, Bkgdf], ignore_index = True, sort = False)
print('Number of entries of the final dataframe =', len(df_final.index))

print (color.RED + '[INFO] If the dataframe has NAN value(s)? {}'.format(df_final.isna().any().any()) + color.END)
if (df_final.isna().any().any()):
    print (df_final.isna().any()) # to see which column(s) has(have) NAN values
    df_final = df_final.dropna().reset_index(drop = True)
    print (color.BOLD + color.BLUE + '---Rows with NAN values are deleted---'.format(Conf.OutputDirName)+ color.END)

TrainIndices, TestIndices = train_test_split(df_final.index.values.tolist(), test_size = Conf.testsize, random_state = Conf.RandomState, shuffle = True)

df_final.loc[TrainIndices, 'Dataset'] = "Train"
df_final.loc[TestIndices, 'Dataset'] = "Test"

df_final.loc[TrainIndices, 'TrainDataset'] = 1
df_final.loc[TestIndices, 'TrainDataset'] = 0

## ----- Deal with the pT eta reweighting -----##
def df_pteta_rwt(Mdf, label, returnOnlyPosWeights = 0, ptw = [10,30,40,50,200,10000], etaw = [-1.5,-1.0,1.0,1.5], eta = '', pt = '',SumWeightCol = "wt", NewWeightCol = "NewWt", target = 1, cand = 0):
    
    Mdf["rwt"] = 1
    Mdf[NewWeightCol] = 1

    ptwt = [1.0] * len(ptw)
    etawt = [1.0] * len(etaw)

    for k in range(len(etaw)):
        if k == len(etaw) - 1:
            continue
        for i in range(len(ptw)):
            if i == len(ptw) - 1:
                continue
            targetSum = Mdf.loc[(Mdf[pt] < ptw[i+1]) & (Mdf[pt] >= ptw[i]) & (Mdf[eta] < etaw[k+1]) & (Mdf[eta] >= etaw[k]) & (Mdf[label] == target), SumWeightCol].sum()

            candSum = Mdf.loc[(Mdf[pt] < ptw[i+1]) & (Mdf[pt] >= ptw[i]) & (Mdf[eta] < etaw[k+1]) & (Mdf[eta] >= etaw[k]) & (Mdf[label] == cand), SumWeightCol].sum()

            if candSum > 0 and targetSum > 0:
                ptwt[i] = candSum / (targetSum)
            else:
                ptwt[i] = 0
    
            Mdf.loc[(Mdf[pt] < ptw[i+1]) & (Mdf[pt] >= ptw[i]) & (Mdf[eta] < etaw[k+1]) & (Mdf[eta] >= etaw[k]) & (Mdf[label] == cand), "rwt"] = 1.0
            Mdf.loc[(Mdf[pt] < ptw[i+1]) & (Mdf[pt] >= ptw[i]) & (Mdf[eta] < etaw[k+1]) & (Mdf[eta] >= etaw[k]) & (Mdf[label] == target), "rwt"] = ptwt[i]
    
            Mdf.loc[:, NewWeightCol] = Mdf.loc[:, "rwt"] * Mdf.loc[:, SumWeightCol]

    MtargetSum = Mdf.loc[Mdf[label] == target, NewWeightCol].sum()
    McandSum = Mdf.loc[Mdf[label] == cand, NewWeightCol].sum()
    print('Number of entries in signal after  weighing = ' + str(MtargetSum))
    print('Number of entries in background after  weighing = ' + str(McandSum))

    return Mdf[NewWeightCol]

if Conf.Reweighing != 'Nothing':
    print ('After pT eta reweghting: ')
df_final[weight] = 1
if Conf.Reweighing != 'Nothing':
    print("In Training:")
if Conf.Reweighing == 'ptetaSig':
    df_final.loc[TrainIndices, weight] = df_pteta_rwt(df_final.loc[TrainIndices], cat, ptw = Conf.ptbins, etaw = Conf.etabins, pt = Conf.ptwtvar, eta = Conf.etawtvar, SumWeightCol = 'instwei', NewWeightCol = weight, target = 0, cand = 1)
if Conf.Reweighing == 'ptetaBkg':
    df_final.loc[TrainIndices, weight] = df_pteta_rwt(df_final.loc[TrainIndices], cat, ptw = Conf.ptbins, etaw = Conf.etabins, pt = Conf.ptwtvar, eta = Conf.etawtvar, SumWeightCol = 'instwei', NewWeightCol = weight, target = 1, cand = 0)

if Conf.Reweighing != 'Nothing':
    print("In Testing:")
if Conf.Reweighing == 'ptetaSig':
    df_final.loc[TestIndices, weight] = df_pteta_rwt(df_final.loc[TestIndices], cat, ptw = Conf.ptbins, etaw = Conf.etabins, pt = Conf.ptwtvar, eta = Conf.etawtvar, SumWeightCol = 'instwei', NewWeightCol = weight, target = 0, cand = 1)
if Conf.Reweighing == 'ptetaBkg':
    df_final.loc[TestIndices, weight] = df_pteta_rwt(df_final.loc[TestIndices], cat, ptw = Conf.ptbins, etaw = Conf.etabins, pt = Conf.ptwtvar, eta = Conf.etawtvar, SumWeightCol = 'instwei', NewWeightCol = weight, target = 1, cand = 0)

if Conf.Reweighing != 'Nothing':
    df_final["ele_pt_bin"] = pd.cut(df_final[Conf.ptwtvar], bins = Conf.ptbins, labels = list(range(len(Conf.ptbins)-1)))
    df_final["ele_eta_bin"] = pd.cut(df_final[Conf.etawtvar], bins = Conf.etabins, labels = list(range(len(Conf.etabins)-1)))

    plot_ptetaKin(
        df_final, category = cat, label = label,
        ptName = Conf.ptwtvar, etaName = Conf.etawtvar,
        ptBin = Conf.ptbins, etaBin = Conf.etabins,
        outdir = Conf.OutputDirName + "/pt_eta_rewei"
    )

## ----- Deal with scale_pos_weight -----##
# Reference:
# [1] scale_pos_weight: https://xgboost.readthedocs.io/en/latest/parameter.html
# [2] https://github.com/dmlc/xgboost/blob/master/demo/kaggle-higgs/higgs-numpy.py
if Conf.Reweighing == 'Nothing':
    weight = 'instwei'

sum_wpos = df_final[weight][df_final.Type == "Signal"].sum()
sum_wneg = df_final[weight][df_final.Type == "Background"].sum()
scaleposweight = sum_wneg / sum_wpos
print ('XGBoost parameter [scale_pos_weight] will be set to {}'.format(scaleposweight))

print(color.BLUE + 'Make training feature plots' + color.END)
with open('./Tools/{}'.format(Conf.featureplotparam_json), 'r') as fp:
        myTags = json.load(fp)
        
        for tagName, par in myTags.items():
            setlogy = False
            if par["logy"] == 'true':
                setlogy = True

            drawoverflow = False
            if par["xoverflow"] == 'true':
                drawoverflow = True

            drawunderflow = False
            if par["xunderflow"] == 'true':
                drawunderflow = True
                
            DrawFeatureHist(
                df_final, tagName, par['bininfo'], par['xAxisLabel'], par['yAxisUnit'], 
                histcolor = ['#0061a8', '#e84545'], 
                logy = setlogy, y_axisscale = par['y_axisscale'], drawxoverflow = drawoverflow, drawxunderflow = drawunderflow, 
                outname='{}/FeaturePlot/TrainFeature_{}'.format(Conf.OutputDirName, tagName),
                wei = weight
            )


# def PrepDataset(df_final, TrainIndices, TestIndices, features, cat, weight):
#     X_train = df_final.loc[TrainIndices, features]
#     Y_train = df_final.loc[TrainIndices, cat]
#     Wt_train = df_final.loc[TrainIndices, weight]
    
#     X_test = df_final.loc[TestIndices, features]
#     Y_test = df_final.loc[TestIndices, cat]
#     Wt_test = df_final.loc[TestIndices, weight]

#     # X_train = xgb.DMatrix(X_train, feature_names=Conf.features[MVA])
#     X_train = np.asarray(X_train)
#     Y_train = np.asarray(Y_train)
#     Wt_train = np.asarray(Wt_train)
    
#     # X_train = xgb.DMatrix(X_train, feature_names=Conf.features[MVA])
#     X_test = np.asarray(X_test)
#     Y_test = np.asarray(Y_test)
#     Wt_test = np.asarray(Wt_test)
#     return X_train, Y_train, Wt_train, X_test, Y_test, Wt_test

# # Reference:
# # [1] Use of GPU: https://www.kaggle.com/vinhnguyen/accelerating-hyper-parameter-searching-with-gpu
# # [2] deterministic_histogram: https://xgboost.readthedocs.io/en/latest/parameter.html # additional-parameters-for-gpu-hist-tree-method
# roc_XGB = []
# for MVA in Conf.MVAs:
#     if 'XGB' in MVA:
#         ## ----- Prepare data to train the model -----##
#         print (color.BLUE + 'Using method: {}'.format(MVA) + color.END)
#         X_train, Y_train, Wt_train, X_test, Y_test, Wt_test = PrepDataset(df_final, TrainIndices, TestIndices, Conf.features[MVA], cat, weight)
        
#         ## ----- Start to train the model -----##
#         print(color.GREEN + MVA + " Training starting" + color.END)
#         xgb_model = xgb.XGBClassifier(
#             objective = "binary:logistic", 
#             random_state = Conf.RandomState, 
#             use_label_encoder = False, 
#             scale_pos_weight = scaleposweight, 
#             tree_method = 'gpu_hist', 
#             # deterministic_histogram = False
#         )
#         print(color.GREEN + "Performing XGB grid search" + color.END)

#         if Conf.Multicore:
#             NCPUs = 12
#             print ('Using multicore, with NCPUs =', NCPUs)
#             cv = GridSearchCV(
#                 xgb_model, 
#                 Conf.XGBGridSearch[MVA],
#                 scoring = 'neg_log_loss',
#                 cv = 3,
#                 verbose = 1,
#                 n_jobs = NCPUs,
#             )
#         else:
#             print ('Using single core')
#             cv = GridSearchCV(
#                 xgb_model,
#                 Conf.XGBGridSearch[MVA],
#                 scoring = 'neg_log_loss',
#                 cv = 3,
#                 verbose = 1
#             )
        
#         search = cv.fit(X_train, Y_train, sample_weight = Wt_train, verbose = 1)

#         ##----- Save the training model -----##
#         pickle.dump(cv, open(Conf.OutputDirName + "/" + MVA + "/" + MVA + "_" + "modelXGB.pkl", "wb"))
#         ROOT.TMVA.Experimental.SaveXGBoost(search.best_estimator_, "myBDT", Conf.OutputDirName+'/'+Conf.xgbmodeloutname, num_inputs = len(Conf.features[MVA]))
        
#         ##----- Print the training results -----##
#         print(color.BOLD + color.GREEN + '-------- Results for XGBClassifier --------' + color.END)
#         print(color.GREEN + "XGB Best Parameters:" + color.END)
#         print(color.GREEN + str(search.best_params_) + color.END)
#         print(color.GREEN + "Expected neg log loss of XGB model = " + str((np.round(np.average(search.best_score_),3))*100) + '%' + color.END)
#         pred = search.predict(X_test)
#         print(color.GREEN + "XGB Accuracy: " + str(round(accuracy_score(Y_test, pred, sample_weight = Wt_test) * 100., 2)) + "%" + color.END)
#         # print(color.GREEN + "XGB ROC AUC: " + str(round(roc_auc_score(Y_test, pred, sample_weight = Wt_test) * 100., 2)) + "%" + color.END)
#         # precision, recall, thresholds = precision_recall_curve(Y_test, pred, sample_weight = Wt_test)
#         # area = auc(recall, precision)
    
#         ##----- Plot the training results (MVA && ROC)-----##
#         df_final.loc[TrainIndices,MVA+"_pred"] = cv.predict_proba(X_train)[:,1]
#         df_final.loc[TestIndices,MVA+"_pred"] = cv.predict_proba(X_test)[:,1]
        
#         # MVA distribution
#         print(color.GREEN + "Plotting output response for XGB" + color.END)
#         fig, axes = plt.subplots(1, 1, figsize = (6, 6))
#         df_resolved = df_final[(df_final["TrainDataset"] == 1)]
#         plot_mva(
#             df_final[(df_final["TrainDataset"] == 1)],
#             MVA+"_pred", bins = np.linspace(0, 1, num = 40),
#             cat = cat,Wt = weight, ax = axes, sample = 'train', ls = 'dashed', alpha = 0.4,
#             logscale = Conf.MVAlogplot
#         )
#         plot_mva(
#             df_final[(df_final["TrainDataset"] == 0)],
#             MVA+"_pred", bins = np.linspace(0, 1, num = 40),
#             cat = cat, Wt = weight, ax = axes, sample = 'test',ls = 'solid', alpha = 0.6,
#             logscale = Conf.MVAlogplot
#         )
#         plt.savefig(Conf.OutputDirName+"/"+MVA+"/"+MVA+"_"+"XGBMVA.pdf", bbox_inches = 'tight')
#         plt.savefig(Conf.OutputDirName+"/"+MVA+"/"+MVA+"_"+"XGBMVA.png", bbox_inches = 'tight')
#         print("Save the fig in: {}".format(Conf.OutputDirName+"/"+MVA+"/"+MVA+"_"+"XGBMVA.pdf"))
#         print("Save the fig in: {}".format(Conf.OutputDirName+"/"+MVA+"/"+MVA+"_"+"XGBMVA.png"))
    
#         # ROC curve
#         print(color.GREEN + "Plotting ROC for XGB" + color.END)
#         fig, axes = plt.subplots(1, 1, figsize = (6, 6))
#         roc_XGB = plot_roc_curve(
#             df_final[(df_final["TrainDataset"] == 1)],
#             MVA+"_pred", tpr_threshold = 0, ax = axes, color = '#184d47', linestyle='-', 
#             label = Conf.MVALabels[MVA] + ' Training',
#             cat = cat,Wt = weight
#         )
#         plot_roc_curve(
#             df_final[(df_final["TrainDataset"] == 0)],
#             MVA+"_pred", tpr_threshold = 0, ax = axes, color = '#e2703a', linestyle = '--', 
#             label = Conf.MVALabels[MVA] + ' Testing',
#             cat = cat, Wt = weight
#         )
#         if len(Conf.OverlayWP) > 0:
#             for c,OverlayWpi in zip(Conf.OverlayWPColors,Conf.OverlayWP):
#                 plot_single_roc_point(df_final[(df_final["TrainDataset"] == 0)], var = OverlayWpi, ax = axes, color = c, marker = 'o', markersize = 6, label = OverlayWpi+" Test dataset", cat=cat,Wt=weight)
#         axes.set_ylabel("Background rejection", fontsize = 15, loc = 'top')
#         axes.set_xlabel("Signal efficiency", fontsize = 15, loc = 'right')
#         # axes.set_title("")
#         axes.text(1.05, 0.5, 'CMS EGamma ID-Trainer', horizontalalignment = 'center', verticalalignment = 'center', rotation = 'vertical', transform = axes.transAxes, fontsize = 13)
#         axes.text(0, 1, "CMS", horizontalalignment = 'left', verticalalignment = 'bottom', transform=axes.transAxes, fontsize = 15, fontweight = 'bold')
#         axes.text(0.13, 1, "$\it{Simulation}$", horizontalalignment = 'left', verticalalignment = 'bottom', transform = axes.transAxes, fontsize = 13)
#         axes.text(1, 1, "41.7 $fb^{-1} (13TeV,\ 2017)$", horizontalalignment = 'right', verticalalignment = 'bottom', transform = axes.transAxes, fontsize = 13)
#         plt.savefig(Conf.OutputDirName+"/"+MVA+"/"+MVA+"_"+"XGBROC.pdf", bbox_inches = 'tight')
#         plt.savefig(Conf.OutputDirName+"/"+MVA+"/"+MVA+"_"+"XGBROC.png", bbox_inches = 'tight')
#         plt.close('all')
#         print("Save the fig in: {}".format(Conf.OutputDirName+"/"+MVA+"/"+MVA+"_"+"XGBROC.pdf"))
#         print("Save the fig in: {}".format(Conf.OutputDirName+"/"+MVA+"/"+MVA+"_"+"XGBROC.png"))
        
#         print (color.BLUE + MVA + ' training done!' + color.END)

# ##----- Feature selection -----##
# # Reference:
# # [1] Selection: https://machinelearningmastery.com/feature-importance-and-feature-selection-with-xgboost-in-python/
# import warnings
# if not sys.warnoptions:
#     warnings.simplefilter("ignore")

# print (color.BLUE + 'XGB feature selection starting' + color.END)

# # Fit model using each importance as a threshold
# thresholds = sort(cv.best_estimator_.feature_importances_)
# # print(thresholds)
# score_list = []
# for thresh in thresholds:
#     # select features using threshold
#     selection = SelectFromModel(cv.best_estimator_, threshold = thresh, prefit = True)
#     select_X_train = selection.transform(X_train)
    
#     # train model
#     selection_model = xgb.XGBClassifier(
#         # n_jobs = 12,
#         objective = "binary:logistic", 
#         random_state = Conf.RandomState, 
#         use_label_encoder = False, 
#         scale_pos_weight = scaleposweight,
#         tree_method = 'gpu_hist',
#         # deterministic_histogram = False,
#         **search.best_params_
#     )
#     selection_model.fit(select_X_train, Y_train, sample_weight = Wt_train, verbose = 1)

#     # eval model
#     select_X_test = selection.transform(X_test)
#     Y_pred = selection_model.predict(select_X_test)
#     # predictions = [round(value) for value in Y_pred]
#     accuracy = accuracy_score(Y_test, Y_pred, sample_weight = Wt_test)
#     score_list.append(accuracy)
#     print(
#         "Thresh = " + str(round(thresh, 3)) + ", n = " + str(select_X_train.shape[1]) +
#         ", Accuracy = " + str(round(accuracy * 100., 2)) + "%"
#     )

# df_featimp = pd.DataFrame({'feature': Conf.features[MVA],'importance':cv.best_estimator_.feature_importances_})
# df_featimp.sort_values('importance', inplace=True, ascending=True)
# df_featimp = df_featimp.reset_index(drop=True)

# score_df = pd.DataFrame(
#     list(map(lambda t: [t[0]*100, t[1]], zip(score_list, range(len(score_list), 0, -1)))) , 
#     columns=['accuracy', 'number_of_feature']
# )
# score_df['feature'] = df_featimp['feature']
# score_df['importance'] = df_featimp['importance']
# score_df = score_df.set_index('feature')
# score_df['accu_diff'] = score_df['accuracy'].diff()

# plot_FeatureImpSel(score_df, MVA='XGB', OutputDirName=Conf.OutputDirName)
# df2table(score_df, MVA='XGB', OutputDirName=Conf.OutputDirName)

# print (color.BLUE + 'XGB feature selection done!' + color.END)


# # roc_DNN = []

# # for MVA in Conf.MVAs:
# #     if 'DNN' in MVA:
# #         print (color.BLUE + 'Using method: {}'.format(MVA) + color.END)
        
# #         X_train, Y_train, Wt_train, X_test, Y_test, Wt_test = PrepDataset(df_final,TrainIndices,TestIndices,Conf.features[MVA],cat,weight)
        
# #         print(color.GREEN + MVA + " fitting running" + color.END)
        
# #         es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
        
# #         modelDNN=Conf.DNNDict[MVA]['model']
# #         modelDNN.compile(loss='binary_crossentropy', 
# #                          optimizer=Adam(learning_rate=Conf.DNNDict[MVA]['learning_rate']), 
# #                          metrics=['accuracy'])
# #         train_history = modelDNN.fit(X_train,Y_train,
# #                                      epochs=Conf.DNNDict[MVA]['epochs'],
# #                                      batch_size=Conf.DNNDict[MVA]['batchsize'],
# #                                      validation_data=(X_test,Y_test, Wt_test),
# #                                      verbose=1,
# #                                      callbacks=[es], sample_weight=Wt_train)
# #         modelDNN.save(Conf.OutputDirName+"/"+MVA+"/"+MVA+"_"+"modelDNN.h5")
        
# #         df_final.loc[TrainIndices,MVA+"_pred"]=modelDNN.predict(X_train)
# #         df_final.loc[TestIndices,MVA+"_pred"]=modelDNN.predict(X_test)
    
# #         print(color.GREEN + "Plotting output response for DNN" + color.END)
        
# #         fig, axes = plt.subplots(1, 1, figsize=(5, 5))
# #         plot_mva(df_final[(df_final["TrainDataset"] == 1)],MVA+"_pred",bins=[0+i*(1/50) for i in range(51)],cat=cat,Wt=weight,ax=axes,sample='train',ls='dashed', alpha=0.4, logscale=Conf.MVAlogplot)
# #         plot_mva(df_final[(df_final["TrainDataset"] == 0)],MVA+"_pred",bins=[0+i*(1/50) for i in range(51)],cat=cat,Wt=weight,ax=axes,sample='test',ls='solid', alpha=0.8, logscale=Conf.MVAlogplot)
# #         plt.savefig(Conf.OutputDirName+"/"+MVA+"/"+MVA+"_"+"DNNMVA.pdf", bbox_inches='tight')
# #         plt.savefig(Conf.OutputDirName+"/"+MVA+"/"+MVA+"_"+"DNNMVA.png", bbox_inches='tight')
    
# #         print(color.GREEN + "Plotting ROC for DNN" + color.END)
        
# #         fig, axes = plt.subplots(1, 1, figsize=(5, 5))
# #         roc_DNN =  plot_roc_curve(df_final[(df_final["TrainDataset"] == 1)],MVA+"_pred", tpr_threshold=0, ax=axes, color='#184d47', linestyle='-', label=Conf.MVALabels[MVA]+' Training',cat=cat,Wt=weight)
# #         plot_roc_curve(df_final[(df_final["TrainDataset"] == 0)],MVA+"_pred", tpr_threshold=0, ax=axes, color='#e2703a', linestyle='--', label=Conf.MVALabels[MVA]+' Testing',cat=cat,Wt=weight)
# #         if len(Conf.OverlayWP)>0:
# #             for c,OverlayWpi in zip(Conf.OverlayWPColors,Conf.OverlayWP):
# #                 plot_single_roc_point(df_final[(df_final["TrainDataset"] == 0)], var=OverlayWpi, ax=axes, color=c, marker='o', markersize=6, label=OverlayWpi+" Test dataset", cat=cat,Wt=weight)
# #         axes.set_ylabel("Background rejection")
# #         axes.set_xlabel("Signal efficiency")
# #         axes.set_title("DNN")
# #         axes.text(1.05, 0.5, 'CMS EGamma ID-Trainer',
# #             horizontalalignment='center',
# #             verticalalignment='center',
# #             rotation='vertical',
# #             transform=axes.transAxes)
# #         axes.tick_params(direction='in', which='both', bottom=True, top=True, left=True, right=True)
# #         plt.savefig(Conf.OutputDirName+"/"+MVA+"/"+MVA+"_"+"DNNROC.pdf", bbox_inches='tight')
# #         plt.savefig(Conf.OutputDirName+"/"+MVA+"/"+MVA+"_"+"DNNROC.png", bbox_inches='tight')
# #         plt.close('all')
        
# #         print (color.BLUE + MVA + 'training done!' + color.END)



# ##PlotFinalROC
# # print(color.GREEN+"Plotting Final ROC"+color.END)
# # print("Plotting Final ROC")

# # fig, axes = plt.subplots(1, 1, figsize=(4.5, 4.5))
# # if len(Conf.OverlayWP)>0:
# #     for c,OverlayWpi in zip(Conf.OverlayWPColors,Conf.OverlayWP):
# #         plot_single_roc_point(df_final[(df_final["TrainDataset"] == 0)], var=OverlayWpi, ax=axes, color=c, marker='o', markersize=8, label=OverlayWpi+" Test dataset", cat=cat,Wt=weight)
# # if len(Conf.MVAs)>0:
# #     for c,MVAi in zip(Conf.MVAColors,Conf.MVAs):
# #         plot_roc_curve(df_final[(df_final["TrainDataset"] == 1)],MVAi+"_pred", tpr_threshold=0.7, ax=axes, color=c, linestyle='-', label=Conf.MVALabels[MVAi]+' Training',cat=cat,Wt=weight)
# #         plot_roc_curve(df_final[(df_final["TrainDataset"] == 0)],MVAi+"_pred", tpr_threshold=0.7, ax=axes, color=c, linestyle='--', label=Conf.MVALabels[MVAi]+' Testing',cat=cat,Wt=weight)
        
# #     axes.set_ylabel("Background rejection")
# #     axes.set_xlabel("Signal efficiency")
# #     # axes.set_title("Final")
# #     axes.text(1.05, 0.5, 'CMS EGamma ID-Trainer',
# #         horizontalalignment='center',
# #         verticalalignment='center',
# #         rotation='vertical',
# #         transform=axes.transAxes)
# # plt.savefig(Conf.OutputDirName+"/ROCFinal.pdf", bbox_inches='tight')
# # plt.savefig(Conf.OutputDirName+"/ROCFinal.png", bbox_inches='tight')

# ##----- Save the threashold to txt file -----##
# import math 
# def ROCinfo(list_roc, clfname='XGBoost bdt'):
#     df_roc = pd.DataFrame(list(zip(list_roc[1], list_roc[2], list_roc[3])),
#                           columns =['sig_eff', 'bag_rej', 'threshold'])
#     df_roc['disto1'] = df_roc.apply(lambda row: math.sqrt((1 - row.sig_eff) + (1 - row.bag_rej)), axis = 1) 

#     idx_closetto1 = df_roc[['disto1']].idxmin()[0]
#     idx_90wp = df_roc.iloc[(df_roc['sig_eff']-0.9).abs().argsort()[:1]].index.tolist()[0]
#     idx_80wp = df_roc.iloc[(df_roc['sig_eff']-0.8).abs().argsort()[:1]].index.tolist()[0]
    
#     print("Save the WP file in: {}".format(Conf.OutputDirName+'/{}_WP.txt'.format(clfname)))
#     with open(Conf.OutputDirName+'/{}_WP.txt'.format(clfname), "w") as text_file:
#         print ('{} score threshold that results in the closet point to (1, 1), corresponding to {:.3g}% of signal efficiency and {:.3g} of background rejection, is {:.3g}'.format(clfname, df_roc.at[idx_closetto1,'sig_eff']*100.,df_roc.at[idx_closetto1,'bag_rej']*100., df_roc.at[idx_closetto1,'threshold']), file = text_file)
#         print ('{} score for 90% working point (signal efficiency), {:.3g} of background rejection, is {:.3g}'.format(clfname, df_roc.at[idx_90wp,'bag_rej']*100., df_roc.at[idx_90wp,'threshold']), file = text_file)
#         print ('{} score for 80% working point (signal efficiency), {:.3g} of background rejection, is {:.3g}'.format(clfname, df_roc.at[idx_80wp,'bag_rej']*100., df_roc.at[idx_80wp,'threshold']), file = text_file)
    
#     return df_roc.at[idx_closetto1,'threshold']

# xgb_threshold = ROCinfo(roc_XGB, clfname = 'XGBoost_bdt')
# # ROCinfo(roc_DNN, clfname='Tensorflow_DNN')

# ##----- efficiency vs pT, nVtx -----##
# print (color.BLUE + 'Efficiency caculation starting!' + color.END)
# bin_pt = [19., 26., 31., 35., 40., 45., 50, 55., 60., 65., 70., 75., 80., 87., 97., 117., 140.]
# # bin = np.linspace(0, 140, num = 70)
# plot_eff(
#     df_final, "elePt", xaxixs = "P^{e}_{T} [GeV]", bin = bin_pt, 
#     TrainModel = Conf.OutputDirName + "/" + MVA + "/" + MVA + "_" + "modelXGB.pkl",
#     features = Conf.features['XGB'],
#     importConfig = importConfig, score_cut = xgb_threshold, wei = "instwei", 
#     outdir = "{}/eff".format(Conf.OutputDirName)
# )

# bin_nVtx = [2., 10., 15., 20, 25, 30, 35, 40, 45, 55, 70, 90]
# # bin = np.linspace(0, 100, num = 50)
# plot_eff(
#     df_final, "nVtx", xaxixs = "# of vertices", bin = bin_nVtx, 
#     TrainModel = Conf.OutputDirName + "/" + MVA + "/" + MVA + "_" + "modelXGB.pkl",
#     features = Conf.features['XGB'],
#     importConfig = importConfig, score_cut = xgb_threshold, wei = "instwei", 
#     outdir = "{}/eff".format(Conf.OutputDirName)
# )

# print (color.BLUE + 'All done!' + color.END)




