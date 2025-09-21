The path to the CNC Folder needs to be updated in each python notebook. 

Generate pseudo-labels
Run create-pseudo-labels.ipynb first. 
This notebook creates pseudo-labels for the dataset and saves them to the specified path. It is created so that each dataset needs to be observed prior to setting the percentile bounds for the labels. (by default it will get saved under /Path/datasets_pseudo/Threshold/)

After generating pseudo-labels, use the saved data path as input for the remaining notebooks.
1. moving-average.ipynb
2. isolation-tree.ipynb
3. autoencoder-model.ipynb
4. autoencoder-LSTM.ipynb

Results and images would be saved to the paths 'path/results/' and 'path/Plots/' 

For 3 and 4, the numerical_features variable needs to be changed as required (by default considers all features) due to computational limitations. 


