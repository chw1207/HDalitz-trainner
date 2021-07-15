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
OutputDirName = './Results/Output_ResolvedMergedClassifier_EB_2017_EGM'
Debug = False  # If True, only a 10% of events/objects are used for either Signal or background
Picklefname = './data/Resolved-Merged-Classifier/Dataframe_ResolvedMerged_HDalitz_allprod_eeg_allMass_EB_2017.pkl'
Clfname = 'RMClf'
#####################################################################
testsize = 0.2  # (0.2 means 20%)

# Common cuts for both signal and background (Would generally correspond to the training region)
CommonCut = "(elePresel == True) and (nVtx > 1)"
# CommonCut = "(nVtx > 1) and (elePt < 100)"
# Tree = "outTree"  # Location/Name of tree inside Root files

################################

# MVAs to use as a list, e.g : ["XGB","DNN", "Genetic"]
# MVAs = ["XGB_1", "DNN_1"]
MVAs = ["XGB"]
# XGB and DNN are keywords so names can be XGB_new, DNN_old etc. But keep XGB and DNN in the names (That is how the framework identifies which algo to run

MVAColors = ["#1a508b", "#c70039"]  # Plot colors for MVAs

# MVALabels = {"XGB_1": "XGB",
#              "DNN_1": "DNN"
#              }  # These labels can be anything (this is how you will identify them on plot legends)
MVALabels = {"XGB": "XGB"}

################################

ClfLabel = ["Resolved", "Merged"]

################################
features = {
    "XGB": [
        "rho",
        "eleEn", 
        "eleSCEta", 
        "eleD0", 
        "eleSIP",
        "elePtError", 
        "eleSCPhiWidth",
        "eleEoverP", 
        "eleBrem",
        "eledPhiAtVtx",
        "eleSigmaIEtaIEtaFull5x5", 
        "elePFChIso", 
        "eleR9Full5x5", 
        "eleMissHits",
        "eleConvVeto",
        # "elePt"
    ]
}
# features = {
#     "XGB": [
#         "rho",
#         "eleEn", 
#         "eleSCEta", 
#         # "eleEta",
#         "eleD0", 
#         # "eleDz", 
#         "eleSIP",
#         "elePtError", 
#         # "eleSCEtaWidth", 
#         "eleSCPhiWidth",
#         "eleEoverP", 
#         # "eleEoverPout", 
#         # "eleEoverPInv", 
#         "eleBrem",
#         # "eledEtaAtVtx", 
#         "eledPhiAtVtx",
#         "eleSigmaIEtaIEtaFull5x5", 
#         # "eleSigmaIPhiIPhiFull5x5",
#         "elePFChIso", 
#         # "elePFPhoIso", 
#         # "elePFPUIso",
#         "eleR9Full5x5", 
#         "eleMissHits",
#         "eleConvVeto",
#         # "circularity",
#         # "eleIDMVAIso", 
#         # "eleIDMVANoIso",
#         # "elePt"
#     ]
# }  # Input features to MVA #Should be in your ntuples

featureplotparam_json = 'FeaturePlotParam_ResolvedMergedClf_EB.json'

xgbmodeloutname = 'XGBoostModel_EB.root'

# EGamma WPs to compare to (Should be in your ntuple)
OverlayWP = []
OverlayWPColors = []  # Colors on plots for WPs
#####################################################################

# Grid Search parameters for XGB (will only be used if MVAs contains "XGB"
# XGBGridSearch = {  # EB
#     "XGB_1": {'learning_rate': [0.05, 0.1, 0.15],
#               'max_depth': [2, 3, 4],
#               'min_child_weight': [0, 50, 100],
#               'gamma': [0, 0.1, 0.2],
#               'scale_pos_weight': [1]}
# } # ==> {'gamma': 0, 'learning_rate': 0.15, 'max_depth': 4, 'min_child_weight': 50}

XGBGridSearch = {  # EB
    "XGB": {
        'learning_rate': [0.3],
        'max_depth': [4],
        'min_child_weight': [40],
        'gamma': [0.01]
    }
}
# XGBGridSearch = {  # EB
#     "XGB": {
#         'learning_rate': [0.4],
#         'max_depth': [2],
#         'min_child_weight': [2000],
#         'gamma': [20]
#     }
# }


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
#####################################################################
# DNN parameters and model (will only be used if MVAs contains "DNN"

# Example for DNN_1 
# Use the same set of training features as XGB
# modelDNN_DNN_1 = Sequential()
# modelDNN_DNN_1.add(Dense(2*len(features["XGB"]), kernel_initializer='glorot_normal',
#                          activation='relu', input_dim=len(features["XGB"])))
# modelDNN_DNN_1.add(Dense(
#     len(features["XGB"]), kernel_initializer='glorot_normal', activation='relu'))
# modelDNN_DNN_1.add(Dropout(0.1))
# modelDNN_DNN_1.add(
#     Dense(1, kernel_initializer='glorot_uniform', activation='sigmoid'))
# DNNDict = {
#     "DNN_1": {'epochs': 100, 'batchsize': 100, 'learning_rate': 0.001, 'model': modelDNN_DNN_1}
# }
# DNNGridSearch = {
#     "DNN_1": {'optimizer': ('rmsprop', 'adam')}
# }


#####################################################################

# Example for 80% and 90% Signal Efficiency Working Points
SigEffWPs = ["80%", "90%"]

######### Reweighting scheme
# weightvar = 'instwei' # relative xs: ggF as 1, for it being the production with the largest xs and VBF as xs_VBF/xs_ggF 
Reweighing = 'ptetaBkg'
ptbins = [0, 10, 20, 30, 40, 50, 60, 100, 150] 
etabins = np.linspace(-1.5, 1.5, num = 11)
ptwtvar = 'elePt'
etawtvar = 'eleEta'

'''
Possible choices :
Nothing : No reweighting
ptetaSig : To Signal pt-eta spectrum 
ptetaBkg : To Background pt-eta spectrum
'''
######### 

# Optional Features
# SaveDataFrameCSV=False #True will save the final dataframe with all features and MAV predictions
RandomState = 42  # Choose the same number everytime for reproducibility
MVAlogplot = True  # If true, MVA outputs are plotted in log scale
Multicore = True  # If True all CPU cores available are used XGB
