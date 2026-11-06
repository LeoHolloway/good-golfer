import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

df = pd.read_csv('pga_tour_data.csv')

print(df.head())

# checking how finishes are recorded
finishes = df['Finish']
print(finishes.head())


# I initially tried this with my sworn enemy Bryson DeChambeau, but for some reason he had really thin SG data
sg_measurements_scheffler = df[df['player id'] == 9478][['sg_putt', 'sg_arg', 'sg_app', 'sg_ott', 'sg_t2g', 'sg_total']]
print(sg_measurements_scheffler.head())

print(df['purse'].head(n=20))

# It's very clear that these are SG measurements for each tournament
# This could answer the question of what needs to be performing well to do well at a tournament,
# but we are more interested in what needs to be performing well on average to do well over the course of a season
# We will compute averages


scaler = StandardScaler()

pca = PCA(n_components=3)
