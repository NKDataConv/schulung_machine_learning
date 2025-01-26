import pandas as pd
import numpy as np
import random
from os import path

from sklearn.model_selection import train_test_split
from sklearn import preprocessing


# --- Setze globale pandas optionen
pd.set_option('display.max_columns',20)

# --- Ensure reproducability
random.seed(42)
np.random.seed(42)

# --- Read data
dat = pd.read_csv('daten/soccerdata/mainfact_bundesliga_full.csv')

# --- Datamanagement
# with pd.option_context('display.max_columns', 100):
#     print(dat.describe())

cols2remove = ['Heim_Schuesse_aufs_Tor','IW_H','IW_D','IW_A','LB_H','LB_D','LB_A',
               'Mean_A','Mean_AsiaHC_A','Mean_AsiaHC_H','Mean_D','Mean_H',
               'SB_H','SB_D','SB_A','SJ_H','SJ_D','SJ_A','Top_A','Top_AsiaHC','Top_AsiaHC_H','Top_AsiaHC_A',
               'Top_AsiaHC_Hbz','Top_D','Top_H','Top_H_D_A','Top_unter_ueber','BS_A','BS_D','BS_H',
               'GB_A','GB_D','GB_H','Ausw_Schuesse_aufs_Tor']
for i in cols2remove:
    del dat[i]

# Just required for neural networks - you can remove this line for tree based models
dat.dropna(inplace=True)

# --- Remove first 10 matchdays assuming table-information is irrelevant
dat = dat.loc[dat['Spieltag'] > 10,:]

# --- Set Spieltag as index
dat.set_index(["Spieltag","ID"], inplace=True)

# --- Fix covariates

# --- Here all possible input variables are listed
# x = dat.loc[:,['MA_Heim', 'MA_Ausw', 'MA_Heim_Tore',
#       'MA_Ausw_Tore', 'Pos_Heim', 'Pos_Ausw', 'Pkt_Heim', 'Pkt_Ausw',
#       'Tore_Heim', 'Tore_Ausw', 'Tordiff_Heim', 'Tordiff_Ausw', 'Siege_Heim',
#       'Unent_Heim', 'Nieder_Heim', 'Siege_Ausw', 'Unent_Ausw', 'Nieder_Ausw',
#       'DIFF_POS', 'DIFF_POS_GERICHTET', 'DIFF_PKT', 'DIFF_PKT_GERICHTET',
#       'FAVORITE', 'Gegentore_Heim', 'Gegentore_Ausw', 'Last3GamesHeim',
#       'Last3GamesAusw']]


# --- To make it more simple: a small set of covariates is used first
covariates = ['Pos_Heim', 'Pos_Ausw', 'Pkt_Heim', 'Pkt_Ausw', 'Tore_Heim', 'Tore_Ausw', 'Tordiff_Heim',
              'Heimstaerke','Auswstaerke', 'MA_Heim_Tore', 'MA_Ausw_Tore',
              'Tordiff_Ausw', 'Siege_Heim', 'Unent_Heim', 'Nieder_Heim', 'Siege_Ausw', 'Unent_Ausw', 'Nieder_Ausw',
              'FAVORITE', 'DIFF_POS', 'DIFF_POS_GERICHTET', 'DIFF_PKT','Gegentore_Heim', 'Gegentore_Ausw',
              'Last3GamesHeim','Last3GamesAusw','BW_H','BW_D','BW_A']
x = dat.loc[:,covariates]


# --- Extract quotas for model evaluation
quota_vars = ['BW_H','BW_D','BW_A']
quotas = dat.loc[:,quota_vars]

# --- Make data numeric
# -- Target (no label encoder is used to have control over classes and number assignment
custom_mapping = {
    'H': 1,
    'D': 0,
    'A': 2,
    np.nan: -9
}

y = [np.nan]*len(dat)
for i in range(0, len(dat)):
    y[i] = custom_mapping[dat['Endergebnis'].iloc[i]]

# - Label-Encoder for dichotomous nominal data
if "FAVORITE" in covariates:
    for row in x.iterrows():
        x.loc[row[0], 'FAVORITE'] = custom_mapping[row[1]['FAVORITE']]

#    x['FAVORITE'] = x['FAVORITE'].astype('int8')

# - get_dummies for binarization of nominal data
x = pd.get_dummies(x)
