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
# finishes = df['Finish']
# print(finishes.head())

# checking if placement data is complete. Seems pretty good and it will always be more accurate than just pretending ties make the non tie amount
# print((df[(df['tournament id'] == 401353224) & (df['season'] == 2022)])['Finish'].sort_values().to_list())

# I initially tried this with my sworn enemy Bryson DeChambeau, but for some reason he had really thin SG data
# sg_measurements_scheffler = df[df['player id'] == 9478][['sg_putt', 'sg_arg', 'sg_app', 'sg_ott', 'sg_t2g', 'sg_total']]
# print(sg_measurements_scheffler.head())

# print(df['purse'].head(n=20))

# It's very clear that these are SG measurements for each tournament
# This could answer the question of what needs to be performing well to do well at a tournament,
# but we are more interested in what needs to be performing well on average to do well over the course of a season

# I will come back to this problem for more detailed stats
# In particular, it's traditional wisdom that high variance players do better in terms of money and wins, would be fun to check it out

# Making a winnings column

    

# Make holdout dataset for testing the model
holdout = df[df['season'] == 2022]
# Drop holdout data
train = df[df['season'] != 2022]

player_stats = train.groupby('player id')[['sg_putt', 'sg_arg', 'sg_app', 'sg_ott', 'sg_t2g']].mean()
print(player_stats.head())

# Heatmap of correlation between SG averages
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(player_stats.corr(), annot=True, cmap='coolwarm', ax=ax)
plt.title('Correlation Heatmap of SG Measurements')
plt.show()

# SG putting distribution. I anticipate this will be the most important single factor
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(player_stats['sg_putt'], bins=30, kde=True, ax=ax)
plt.title('Distribution of SG Putting')
plt.show()

# 

pca = PCA(n_components=3)