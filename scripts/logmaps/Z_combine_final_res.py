import pandas as pd
import sys
sys.path.append('./')
sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../../')

from config_logmapres import interim_res_path, final_res_path
import os

os.makedirs(final_res_path, exist_ok=True)

csv_files = interim_res_path.glob('*.csv')

dfs = [pd.read_csv(f, index_col=0) for f in csv_files]


df = pd.concat(dfs,
               axis=0,
               ignore_index=False)

# change som method names
# df[df.method=="RecurrenceManifold"]["method"] = "shRec"
# df[df.method=="FastICA"].method = "ICA"
# df[df.method=="DynamicalComponentsAnalysis"].method = "DCA"
df.method = df.method.map({"RecurrenceManifold": "shRec",
                               "FastICA": "ICA",
                               "DynamicalComponentsAnalysis": "DCA",
                               "AniSOM": "ASOM",
                               "random": "Random",
                               "CCA": "CCA",
                               "PCA": "PCA",
                               "DCCA": "DCCA",
                               "SFA": "SFA"})


df.to_csv(final_res_path / 'logmaps_res.csv')

print(df.head())