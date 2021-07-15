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
OutputDirName = './Results/Output_Merged1GsfID_EB_2017_EGM'
Debug = False  # If True, only a 10% of events/objects are used for either Signal or background
Pickle_signal = './data/Merged-ID/Dataframe_MergedID_HDalitz_allprod_eeg_allMass_EB_2017.pkl'
Pickle_bkg = './data/Merged-ID/Dataframe_MergedID_GJets_EB_2017.pkl'
Clfname = 'MergedID'
#####################################################################
testsize = 0.2  # (0.2 means 20%)

# Common cuts for both signal and background (Would generally correspond to the training region)
CommonCut = "(XGB_pred >= 0.507) and (nGsfMatchToReco == 1) and (elePresel == True) and (nVtx > 1)"
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

ClfLabel = ["Background", "Signal"]

################################
features = {
    "XGB": [
        'rho',
        'eleEn',  
        'eleSCEta', 
        'eleD0', 
        'eleDz', 
        'eleSIP',
        'eleSCEtaWidth', 
        'eleHoverE', 
        'eleEoverP', 
        'eledEtaAtVtx', 
        'eleSigmaIEtaIEtaFull5x5', 
        'elePFChIso', 
        'elePFPhoIso', 
        'eleMissHits',
        'eleConvVeto'
    ]
}
# features = {
#     "XGB": [
#         'rho',
#         'eleEn', 
#         # 'eleSCEn', 
#         'eleSCEta', 
#         # 'eleEcalEn',
#         'eleD0', 
#         'eleDz', 
#         'eleSIP',
#         # 'elePtError', 
#         # 'eleEta',
#         # 'eleR9', 
#         'eleSCEtaWidth', 
#         # 'eleSCPhiWidth',
#         'eleHoverE', 
#         'eleEoverP', 
#         # 'eleEoverPout', 
#         # 'eleEoverPInv', 
#         # 'eleBrem',
#         'eledEtaAtVtx', 
#         # 'eledPhiAtVtx',
#         'eleSigmaIEtaIEtaFull5x5', 
#         # 'eleSigmaIPhiIPhiFull5x5',
#         'elePFChIso', 
#         'elePFPhoIso', 
#         # 'elePFNeuIso', 
#         # 'elePFPUIso',
#         # 'eleIDMVAIso', 
#         # 'eleIDMVANoIso',
#         # 'eleR9Full5x5', 
#         # 'eleTrkdxy',
#         'eleMissHits',
#         # 'gsfPtSum',
#         # 'gsfPtRatio',
#         # 'gsfDeltaR',
#         # 'circularity',
#         'eleConvVeto'
#     ]
# }  # Input features to MVA #Should be in your ntuples

featureplotparam_json = 'FeaturePlotParam_MergedIDEB1Gsf.json'

xgbmodeloutname = 'XGBoostModel_Merged1GsfID_EB.root'

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

XGBGridSearch = { 
    "XGB": {
        'learning_rate': [0.3],
        'max_depth': [2],
        'min_child_weight': [600],
        'gamma': [0.2],
        # 'max_delta_step': [4]
        'subsample': [0.5]
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


# # In this file you can specify the training configuration

# #####################################################################
# # Do not touch this
# import numpy as np
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Activation
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.layers import Dropout
# #####################################################################
# # Start here
# #####################################################################
# # All plots, models, config file will be stored here
# OutputDirName = './Results/Output_Merged1GsfID_EB_2017_llg'
# Debug = False  # If True, only a 10% of events/objects are used for either Signal or background
# Pickle_signal = './data/Merged-ID/Dataframe_MergedID_HDalitz_allprod_eeg_allMass_EB_2017.pkl'
# Pickle_bkg = './data/Merged-ID/Dataframe_MergedID_GJets_EB_2017.pkl'
# Clfname = 'MergedID'
# #####################################################################
# testsize = 0.2  # (0.2 means 20%)

# # Common cuts for both signal and background (Would generally correspond to the training region)
# # TODO: The cut value of Resolved-Merged classifier should be determined
# CommonCut = "(XGB_pred >= 0.513) and (nGsfMatchToReco == 1) and (elePresel == True) and (nVtx > 1)"
# # CommonCut = "(XGB_pred >= 0.0)" 

# # Tree = "outTree"  # Location/Name of tree inside Root files

# ################################

# # MVAs to use as a list, e.g : ["XGB","DNN", "Genetic"]
# MVAs = ["XGB"]
# # XGB and DNN are keywords so names can be XGB_new, DNN_old etc. But keep XGB and DNN in the names (That is how the framework identifies which algo to run

# MVAColors = ["#1a508b", "#c70039"]  # Plot colors for MVAs

# # MVALabels = {"XGB_1": "XGB",
# #              "DNN_1": "DNN"}  # These labels can be anything (this is how you will identify them on plot legends)
# MVALabels = {"XGB": "XGB"}
# ################################
# weightvar = 'instwei'
# ClfLabel = ["Signal","Background"]
# ################################
# features = {
#     "XGB": [
#         'rho',
#         'eleEn', 
#         # 'eleSCEn', 
#         'eleSCEta', 
#         # 'eleEcalEn',
#         'eleD0', 
#         'eleDz', 
#         'eleSIP',
#         # 'elePtError', 
#         # 'eleEta',
#         # 'eleR9', 
#         'eleSCEtaWidth', 
#         # 'eleSCPhiWidth',
#         'eleHoverE', 
#         'eleEoverP', 
#         # 'eleEoverPout', 
#         # 'eleEoverPInv', 
#         # 'eleBrem',
#         'eledEtaAtVtx', 
#         # 'eledPhiAtVtx',
#         'eleSigmaIEtaIEtaFull5x5', 
#         # 'eleSigmaIPhiIPhiFull5x5',
#         'elePFChIso', 
#         'elePFPhoIso', 
#         # 'elePFNeuIso', 
#         # 'elePFPUIso',
#         # 'eleIDMVAIso', 
#         # 'eleIDMVANoIso',
#         # 'eleR9Full5x5', 
#         # 'eleTrkdxy',
#         'eleMissHits',
#         # 'gsfPtSum',
#         # 'gsfPtRatio',
#         # 'gsfDeltaR',
#         # 'circularity',
#         'eleConvVeto'
#     ]
# }  # Input features to MVA #Should be in your ntuples

# featureplotparam_json = 'FeaturePlotParam_MergedIDEB1Gsf.json'

# xgbmodeloutname = 'XGBoostModel_Merged1GsfID_EB.root'

# # when not sure about the binning, you can just specify numbers, which will then correspond to total bins
# # You can even specify lists like [10,20,30,100]

# # EGamma WPs to compare to (Should be in your ntuple)
# OverlayWP = []
# OverlayWPColors = []  # Colors on plots for WPs
# #####################################################################

# # Grid Search parameters for XGB (will only be used if MVAs contains "XGB"
# # XGBGridSearch = {  # EB
# #     "XGB_1": {'learning_rate': [0.1],
# #               'max_depth': [6],
# #               'min_child_weight': [500],
# #               'gamma': [0],
# #               'scale_pos_weight': [1],
# #               'n_estimator': [500],
# #               'early_stopping_rounds': [5]}
# # }
# XGBGridSearch = {  # EB
#     "XGB": {
#         'learning_rate': [0.3],
#         'max_depth': [3],
#         'min_child_weight': [100],
#         'gamma': [0.2],
#     }
# }

# # The above are just one/two paramter grids
# # All other parameters are XGB default values
# # But you give any number you want:
# # example:
# # XGBGridSearch= {'learning_rate':[0.1, 0.01, 0.001],
# #                'min_child_weight': [1, 5, 10],
# #                'gamma': [0.5, 1, 1.5, 2, 5],
# #                'subsample': [0.6, 0.8, 1.0],
# #                'colsample_bytree': [0.6, 0.8, 1.0],
# #                'max_depth': [3, 4, 5]}
# # Just rememeber the larger the grid the more time optimization takes

# # DNN parameters and model (will only be used if MVAs contains "DNN"

# # Example for DNN_1
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


# #####################################################################

# # Example for 80% and 90% Signal Efficiency Working Points
# SigEffWPs = ["80%", "90%"]

# # Reweighting scheme #Feature not available but planned
# #Reweighting = 'pt-etaSig'
# '''
# Possible choices :
# None : No reweighting
# FlatpT : Binned flat in pT (default binning)
# Flateta : Binned flat in eta (default binning)
# pt-etaSig : To Signal pt-eta spectrum 
# pt-etaBkg : To Background pt-eta spectrum
# '''

# # Optional Features
# # SaveDataFrameCSV=False #True will save the final dataframe with all features and MAV predictions
# RandomState = 42  # Choose the same number everytime for reproducibility
# MVAlogplot = True  # If true, MVA outputs are plotted in log scale
# Multicore = True  # If True all CPU cores available are used XGB
