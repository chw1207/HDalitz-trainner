import ROOT 
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
# from argparse import ArgumentParser

def plottable(df):
    # plot the table
    # ax = plt.gca() #get current axis
    fig, ax = plt.subplots(1, 1, figsize = (9, 6))
    ax.axis('off')

    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    rcolors = np.full(len(fl), '#bedcfa')
    ccolors = np.full(2, '#bedcfa')
    Table = pd.plotting.table(
        ax, df, 
        rowLoc = 'left', loc = 'center', cellLoc = 'center', 
        rowColours = rcolors, colColours = ccolors,
    )
    
    Table.auto_set_column_width(col = list(range(len(df.columns))))
    
    Table.set_fontsize(10)
    Table.scale(1, 1.4)
    # plt.subplots_adjust(left = 0.22)
    outdir = "./plots/FeatureTable"
    os.makedirs(outdir, exist_ok = True) # mkdir the ouput directory of the plots
    outName = importConfig.replace("Tools.TrainConfig_", "")
    plt.savefig('{}/Feature_table_{}.pdf'.format(outdir, outName), bbox_inches = 'tight')
    print('Save table as {}/Feature_table_{}.pdf'.format(outdir, outName))


def main():
    # description = {
    #     "eleEn":            "energy",
    #     "eleSCEn":          "superCluster.energy",
    #     "eleSCEta":         "the super-cluster eta",
    #     "eleSCPhi":         "the super-cluster phi",
    #     "eleD0":            "the transverse impact parameter",
    #     "eleDz":            "the longitudinal impact parameter",
    #     "eleSIP":           "the 3D impact parameter significance",
    #     "eleSCEtaWidth":    "the super-cluster eta width",
    #     "eleSCPhiWidth":    "the super-cluster phi width",
        
    #     # Reference: Gsf electron variables
    #     # [1] https://cmssdt.cern.ch/SDT/doxygen/CMSSW_7_2_2/doc/html/d8/dac/GsfElectron_8h_source.html
    #     # PCA : the point of closest approach, brem: bremsstrahlung
    #     "eleBrem":          "the fraction of energy lost through bremsstrahlung[(track momentum in - track momentum out) / track momentum in]",
    #     "eleEoverP":        "the super-cluster energy / track momentum at the PCA to the beam spot.",
    #     "eleEoverPout":     "the seed cluster energy / track momentum at the PCA to the seed cluster, extrapolated from the outermost track state.",
    #     "eledEtaAtVtx":     "the super-cluster eta - track eta position at calo extrapolated from innermost track state",
    #     "eledPhiAtVtx":     "the super-cluster phi - track phi position at calo extrapolated from the innermost track state",
    #     "elePtError":       "The error of the tansverse momentum", 

    #     # SS variables
    #     "eleSigmaIEtaIEtaFull5x5":  "The energy weighted standard deviation of single crystal eta within the 5x5 crystals",
    #     "eleSigmaIPhiIPhiFull5x5":  "The energy weighted standard deviation of single crystal phi within the 5x5 crystals",
    #     "elePFChIso":       "PF charged isolation",
    #     "elePFPhoIso":      "PF photon isolation",
    #     "eleR9Full5x5":     "Full 5x5 R9",

    #     # Reference: 
    #     # [1] https://github.com/cms-sw/cmssw/blob/dfd0d1dc4786721ed41d7500305a075998a49479/PhysicsTools/PatAlgos/plugins/PATElectronProducer.cc#L537-L546
    #     "eleConvVeto":      "electron conversion veto (vertex fit || missHit < 1)",
    #     "eleMissHits":      "the number of the missing hit in the pixel detector"
    # }

    description = {
        "rho":              "The average energy density in the event",
        "eleEn":            "energy",
        "eleEta":           "eta",
        "eleSCEn":          "superCluster.energy",
        "eleSCEta":         "superCluster.eta",
        "eleSCPhi":         "superCluster.phi",
        "eleD0":            "gsfTrack.dxy(pv)",
        "eleDz":            "gsfTrack.dz(pv)",
        "eleSIP":           "fabs(dB(PV3D)) / edB(PV3D)",
        "eleSCEtaWidth":    "superCluster.etaWidth",
        "eleSCPhiWidth":    "superCluster.phiWidth",
        
        "eleBrem":          "fbrem",
        "eleHoverE":        "hcalOverEcal",
        "eleEoverP":        "eSuperClusterOverP",
        "eleEoverPout":     "eEleClusterOverPout",
        "eledEtaAtVtx":     "deltaEtaSuperClusterTrackAtVtx",
        "eledPhiAtVtx":     "deltaPhiSuperClusterTrackAtVtx",
        "elePtError":       "ecalTrkEnergyErrPostCorr * pt/p",

        "eleSigmaIEtaIEtaFull5x5":  "full5x5_sigmaIetaIeta",
        "eleSigmaIPhiIPhiFull5x5":  "full5x5_sigmaIphiIphi",
        "elePFChIso":       "pfIsolationVariables.sumChargedHadronPt",
        "elePFPhoIso":      "pfIsolationVariables.sumPhotonEt",
        "elePFNeuIso":      "pfIsolationVariables.sumNeutralHadronEt",
        "eleR9Full5x5":     "full5x5_r9",

        "eleConvVeto":      "passConversionVeto",
        "eleMissHits":      "gsfTrack.hitPattern.numberOfAllHits(MISSING_INNER_HITS)",

        # GSF track info
        "gsfPtSum":         "scalar sum of lead and sub-lead Gsf track's pT matched to the electron",
        "gsfDeltaR":        "delta R between lead and sub-lead Gsf track matched to the electron"
    }
    
    des = []
    for i in fl:
        des.append(description[i])
    
    content = {"Features": fl, "Description of the features": des}
    table = pd.DataFrame(content)
    table.index = [i for i in range(1, len(fl)+1)]
    plottable(table)

if __name__ == "__main__" :
    TrainConfig = sys.argv[1]
    importConfig = TrainConfig.replace("/", ".")
    print("Importing settings from %s" %(importConfig))
    exec("import "+importConfig+" as Conf")
    fl = Conf.features["XGB"]

    state = main()
    sys.exit(state)