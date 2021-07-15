# HDalitz-trainner

### Original package: https://github.com/cms-egamma/ID-Trainer 

This trainner is used to train the following models in the <img src="https://render.githubusercontent.com/render/math?math=H\rightarrow\gamma^*\gamma\rightarrow ee\gamma"> analysis.
- **Resolved-Merged electrons classifier:** To differentiate the resolved and merged electrons.
- **Merged ID:** To discriminate merged <img src="https://render.githubusercontent.com/render/math?math=\gamma^*"> and the associated background.

---
### Prepare training samples 
- PrepareData_ResolvedMergedClassifier.py
- PrepareData_MergedID.py

One could exucute the python scripts via
```bash
$ python3 PrepareData_ResolvedMergedClassifier.py
$ python3 PrepareData_MergedID.py
```
---

### Train the model
- Trainer-HDalitz.py

One could exucute the python scripts via
```bash
$ python3 Trainer-HDalitz.py Tools/[Training config python file]

# eq. 
# python3 Trainer-HDalitz.py Tools/TrainConfig_ResolvedMergedClassifier_EB
```

- There are currently 6 training configuration files
    - TrainConfig_ResolvedMergedClassifier_EB.py
    - TrainConfig_ResolvedMergedClassifier_EE.py
    - TrainConfig_Merged2GsfID_EB.py
    - TrainConfig_Merged2GsfID_EE.py
    - TrainConfig_Merged1GsfID_EB.py
    - TrainConfig_Merged1GsfID_EE.py
---

### plot the results
The setting of the plots can be modified in the ```Tools/PlotTools.py```. The plotting style of the distributions of the features can be modified in the json files in Tools directory.
