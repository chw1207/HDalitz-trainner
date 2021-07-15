# In this file you can specify the training configuration

#####################################################################
# Do not touch this
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout
#####################################################################
# Start here
#####################################################################
# All plots, models, config file will be stored here
OutputDirName = './Output_ResolvedMergedClassifier_EE'
Debug = False  # If True, only a 10% of events/objects are used for either Signal or background
Picklefname = './data/Resolved-Merged-Classifier/Dataframe_ResolvedMerged_HDalitz_allprod_eeg_m125_EE_2017.pkl'
#####################################################################
# Not Used
#####################################################################
# #Files, Cuts and XsecWts should have the same number of elements
# SigFiles = [
#     '/eos/user/h/hajheng/minitree_hdalitz/Minitree_HDalitz_ggF_eeg_m125_2017_RECO.root',
#     '/eos/user/h/hajheng/minitree_hdalitz/Minitree_HDalitz_VBF_eeg_m125_2017_RECO.root',
#     '/eos/user/h/hajheng/minitree_hdalitz/Minitree_HDalitz_WH_eeg_m125_2017_RECO.root',
#     '/eos/user/h/hajheng/minitree_hdalitz/Minitree_HDalitz_ZH_eeg_m125_2017_RECO.root'
# ]  # Add as many files you like

# #Cuts to select appropriate signal
# SigCuts= [
#     "(category == 1)"
# ] #Cuts same as number of files (Kept like this because it may be different for different files)

# #Any extra xsec weight : Useful when stitching samples for either signal or background : '1' means no weight
# SigXsecWts=[
#     1
# ] #Weights same as number of files (Kept like this because it may be different for different files)

# #Files, Cuts and XsecWts should have the same number of elements
# BkgFiles = [
#     '/afs/cern.ch/work/h/hajheng/private/HDalitz/Misc/RECO/minitree/Minitree_HDalitz_ggF_eeg_m125_2017_RECO.root'
# ] # Add as many files you like

# #Cuts to select appropriate background
# BkgCuts= [
#     "(category == 2) || (category == 3)"]#Cuts same as number of files (Kept like this because it may be different for different files)

# #Any extra xsec weight : Useful when stitching samples for either signal or background : '1' means no weight
# BkgXsecWts=[
#     1
# ] #Weights same as number of files (Kept like this because it maybe different for different files)

#####################################################################

testsize = 0.2  # (0.2 means 20%)

# Common cuts for both signal and background (Would generally correspond to the training region)
CommonCut = ""

Tree = "outTree"  # Location/Name of tree inside Root files

################################

# MVAs to use as a list, e.g : ["XGB","DNN", "Genetic"]
MVAs = ["XGB_1", "DNN_1"]
# XGB and DNN are keywords so names can be XGB_new, DNN_old etc. But keep XGB and DNN in the names (That is how the framework identifies which algo to run

MVAColors = ["#1a508b", "#c70039"]  # Plot colors for MVAs

MVALabels = {"XGB_1": "XGB",
             "DNN_1": "DNN"
             }  # These labels can be anything (this is how you will identify them on plot legends)

################################
features = {
    "XGB_1": ["eleEn", "eleSCEn", "eleEcalEn",
              "eleESEnP1", "eleESEnP2",
              "eleD0", "eleDz", "eleSIP",
              "elePtError", "eleEta",
              "eleR9", "eleSCEtaWidth", "eleSCPhiWidth",
              "eleHoverE", "eleEoverP", "eleEoverPout", "eleEoverPInv", "eleBrem",
              "eledEtaAtVtx", "eledPhiAtVtx",
              "eleSigmaIEtaIEtaFull5x5", "eleSigmaIPhiIPhiFull5x5",
              "elePFChIso", "elePFPhoIso", "elePFNeuIso", "elePFPUIso",
              "eleIDMVAIso", "eleIDMVANoIso",
              "eleR9Full5x5", "eleTrkdxy"],
    "DNN_1": ["eleEn", "eleSCEn", "eleEcalEn",
              "eleESEnP1", "eleESEnP2",
              "eleD0", "eleDz", "eleSIP",
              "elePtError", "eleEta",
              "eleR9", "eleSCEtaWidth", "eleSCPhiWidth",
              "eleHoverE", "eleEoverP", "eleEoverPout", "eleEoverPInv", "eleBrem",
              "eledEtaAtVtx", "eledPhiAtVtx",
              "eleSigmaIEtaIEtaFull5x5", "eleSigmaIPhiIPhiFull5x5",
              "elePFChIso", "elePFPhoIso", "elePFNeuIso", "elePFPUIso",
              "eleIDMVAIso", "eleIDMVANoIso",
              "eleR9Full5x5", "eleTrkdxy"],
}  # Input features to MVA #Should be in your ntuples

feature_bins = {
    "XGB_1": [np.linspace(0, 200, 51), np.linspace(0, 200, 51), np.linspace(0, 200, 51),
              np.linspace(0, 60, 31), np.linspace(0, 60, 31),
              np.linspace(0, 3, 61), np.linspace(
        0, 5, 51), np.linspace(0, 10, 51),
        np.linspace(0, 20, 51), np.linspace(-3, 3, 61),
        np.linspace(0, 1, 51), np.linspace(
        0, 0.2, 51), np.linspace(0, 0.5, 51),
        np.linspace(0, 5, 51), np.linspace(0, 5, 51), np.linspace(
        0, 2, 51), np.linspace(-1, 1, 51), np.linspace(0, 1, 51),
        np.linspace(-0.2, 0.2,
                    41), np.linspace(-0.3, 0.3, 121),
        np.linspace(0, 0.1, 51), np.linspace(0, 0.1, 51),
        np.linspace(0, 20, 81), np.linspace(0, 20, 81), np.linspace(
        0, 20, 81), np.linspace(0, 20, 81),
        np.linspace(-1, 1, 51), np.linspace(-1, 1, 51),
        np.linspace(0, 1, 51), np.linspace(0, 1, 51)],
    "DNN_1": [np.linspace(0, 200, 51), np.linspace(0, 200, 51), np.linspace(0, 200, 51),
              np.linspace(0, 60, 31), np.linspace(0, 60, 31),
              np.linspace(0, 3, 61), np.linspace(
        0, 5, 51), np.linspace(0, 10, 51),
        np.linspace(0, 20, 51), np.linspace(-3, 3, 61),
        np.linspace(0, 1, 51), np.linspace(
        0, 0.2, 51), np.linspace(0, 0.5, 51),
        np.linspace(0, 5, 51), np.linspace(0, 5, 51), np.linspace(
        0, 2, 51), np.linspace(-1, 1, 51), np.linspace(0, 1, 51),
        np.linspace(-0.2, 0.2,
                    41), np.linspace(-0.3, 0.3, 121),
        np.linspace(0, 0.1, 51), np.linspace(0, 0.1, 51),
        np.linspace(0, 20, 81), np.linspace(0, 20, 81), np.linspace(
        0, 20, 81), np.linspace(0, 20, 81),
        np.linspace(-1, 1, 51), np.linspace(-1, 1, 51),
        np.linspace(0, 1, 51), np.linspace(0, 1, 51)],
}  # Binning used only for plotting features (should be in the same order as features), does not affect training
# template
# np.linspace(lower boundary, upper boundary, totalbins+1)

# when not sure about the binning, you can just specify numbers, which will then correspond to total bins
# You can even specify lists like [10,20,30,100]

# EGamma WPs to compare to (Should be in your ntuple)
OverlayWP = []
OverlayWPColors = []  # Colors on plots for WPs
#####################################################################

# Grid Search parameters for XGB (will only be used if MVAs contains "XGB"
# XGBGridSearch = { # EB
#     "XGB_1": {'learning_rate': [0.1],
#               'max_depth': [6],
#               'min_child_weight': [500],
#               'gamma': [0],
#               'scale_pos_weight': [1],
#               'n_estimator': [500],
#               'early_stopping_rounds': [5]}
# }
XGBGridSearch = { #EE
    "XGB_1": {'learning_rate': [0.1],
              'max_depth': [4],
              'min_child_weight': [500],
              'gamma': [0],
              'scale_pos_weight': [1],
              'n_estimator': [100],
              'early_stopping_rounds': [5]}
}
#
# To choose just one value for a parameter you can just specify value but in a list
# Like "XGB_1":{'gamma':[0.5],'learning_rate':[0.1, 0.01]}
# Here gamma is fixed but learning_rate will be varied

# The above are just one/two paramter grids
# All other parameters are XGB default values
# But you give any number you want:
# example:
# XGBGridSearch= {'learning_rate':[0.1, 0.01, 0.001],
#                'min_child_weight': [1, 5, 10],
#                'gamma': [0.5, 1, 1.5, 2, 5],
#                'subsample': [0.6, 0.8, 1.0],
#                'colsample_bytree': [0.6, 0.8, 1.0],
#                'max_depth': [3, 4, 5]}
# Just rememeber the larger the grid the more time optimization takes

# DNN parameters and model (will only be used if MVAs contains "DNN"

# Example for DNN_1
modelDNN_DNN_1 = Sequential()
modelDNN_DNN_1.add(Dense(2*len(features["DNN_1"]), kernel_initializer='glorot_normal',
                         activation='relu', input_dim=len(features["DNN_1"])))
modelDNN_DNN_1.add(Dense(
    len(features["DNN_1"]), kernel_initializer='glorot_normal', activation='relu'))
modelDNN_DNN_1.add(Dropout(0.1))
modelDNN_DNN_1.add(
    Dense(1, kernel_initializer='glorot_uniform', activation='sigmoid'))
DNNDict = {
    "DNN_1": {'epochs': 100, 'batchsize': 100, 'lr': 0.001, 'model': modelDNN_DNN_1}
}
DNNGridSearch = {
    "DNN_1": {'optimizer': ('rmsprop', 'adam')}
}


#####################################################################

# Example for 80% and 90% Signal Efficiency Working Points
SigEffWPs = ["80%", "90%"]

# Reweighting scheme #Feature not available but planned
#Reweighting = 'pt-etaSig'
'''
Possible choices :
None : No reweighting
FlatpT : Binned flat in pT (default binning)
Flateta : Binned flat in eta (default binning)
pt-etaSig : To Signal pt-eta spectrum 
pt-etaBkg : To Background pt-eta spectrum
'''

# Optional Features
# SaveDataFrameCSV=False #True will save the final dataframe with all features and MAV predictions
RandomState = 42  # Choose the same number everytime for reproducibility
MVAlogplot = True  # If true, MVA outputs are plotted in log scale
Multicore = True  # If True all CPU cores available are used XGB
