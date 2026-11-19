# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json

from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# %%
with open("approx_winnings_percentage.json", "r", encoding="utf-8") as f:
    winnings_rate = json.load(f)

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

# TODO: Make an actual winnings column
def approximate_winnings(df):
    proportion = df['Finish'].str.lstrip('T').map(winnings_rate).fillna(0)
    df['winnings'] = df['purse'] * proportion

approximate_winnings(df)

# Check winnings actually worked
print(df[['Finish', 'purse', 'winnings']].head(n=20))

# Make holdout dataset for testing the model
holdout = df[df['season'] == 2022]
# Drop holdout data
train = df[df['season'] != 2022]

player_stats = train.groupby('player id')[['sg_putt', 'sg_arg', 'sg_app', 'sg_ott', 'winnings']].mean().dropna()
holdout_stats = holdout.groupby('player id')[['sg_putt', 'sg_arg', 'sg_app', 'sg_ott', 'winnings']].mean().dropna()
print(player_stats.head())

# Heatmap of correlation between SG averages
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(player_stats.corr(), annot=True, cmap='coolwarm', ax=ax)
plt.title('Correlation Heatmap of SG Measurements')
plt.show()

# SG distributions. I anticipate putting will be the most important single factor
sg_titles = {
    'sg_putt': 'SG Putting',
    'sg_arg': 'SG ARG',
    'sg_app': 'SG App',
    'sg_ott': 'SG OTT',
}
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, (col, title) in zip(axes.flat, sg_titles.items()):
    sns.histplot(player_stats[col], bins=30, kde=True, ax=ax)
    ax.set_title(f'Distribution of {title}')
fig.tight_layout()
plt.savefig("plots/sg_histograms.png", dpi=300, bbox_inches="tight")
plt.show()

# Variances of SG
print(f"putt {np.var(player_stats['sg_putt'])}")
print(f"arg {np.var(player_stats['sg_arg'])}")
print(f"app {np.var(player_stats['sg_app'])}")
print(f"ott {np.var(player_stats['sg_ott'])}")

# Means of SG. Population mean is 0 by definition
print(f"putt {np.mean(player_stats['sg_putt'])}")
print(f"arg {np.mean(player_stats['sg_arg'])}")
print(f"app {np.mean(player_stats['sg_app'])}")
print(f"ott {np.mean(player_stats['sg_ott'])}")
# They're all below 0. Explanation in writeup

pca_full = PCA(n_components=4)
sg_cols = ['sg_putt', 'sg_arg', 'sg_app', 'sg_ott']
pca_full.fit(player_stats[sg_cols])
print(pca_full.explained_variance_ratio_)
loadings = pd.DataFrame(pca_full.components_.T, columns=['PC1', 'PC2', 'PC3', 'PC4'], index=sg_cols)
print(f"pct explained by pca: {np.sum(pca_full.explained_variance_ratio_)}")

# This is the section we are actually after.

# plot components
fig, ax = plt.subplots(figsize=(10, 6))
for i, (pc, loading) in enumerate(loadings.items()):
    ax.plot(loadings.index, loading, 'o-', markersize=10,
            label=f'{pc} ({pca_full.explained_variance_ratio_[i]:.0%})')

ax.axhline(0, color='gray', linewidth=0.8)
ax.set_ylabel('Loading')
ax.legend()
plt.title('Unscaled PCA Loadings by SG Category')
plt.savefig("plots/unscaled_pca_loadings.png", dpi=300, bbox_inches="tight")
plt.show()



# Scaled version
scaled_sg = StandardScaler().fit_transform(player_stats[sg_cols])
pca_scaled = PCA(n_components=4)
pca_scaled.fit(scaled_sg)
print(pca_scaled.explained_variance_ratio_)
scaled_loadings = pd.DataFrame(pca_scaled.components_.T, columns=['PC1', 'PC2', 'PC3', 'PC4'], index=sg_cols)

fig, ax = plt.subplots(figsize=(10, 6))
for i, (pc, loading) in enumerate(scaled_loadings.items()):
    ax.plot(scaled_loadings.index, loading, 'o-', markersize=10,
            label=f'{pc} ({pca_scaled.explained_variance_ratio_[i]:.0%})')

ax.axhline(0, color='gray', linewidth=0.8)
ax.set_ylabel('Loading')
ax.legend()
plt.title('Scaled PCA Loadings by SG Category')
plt.savefig("plots/scaled_pca_loadings.png", dpi=300, bbox_inches="tight")
plt.show()


# Now with only 3
pca = PCA(n_components=3)
pca.fit(player_stats[sg_cols])
print(pca.explained_variance_ratio_)
loadings = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2', 'PC3'], index=sg_cols)
print(f"pct explained by pca: {np.sum(pca.explained_variance_ratio_)}")

# ------------------------------
# Linear regression, see readme for notes on viability
# ------------------------------

X_transformed = pca.transform(player_stats[sg_cols])
Y = player_stats['winnings'] * 1e6
logY = np.log1p(Y)

# This is for non pca reg
X_filtered = player_stats[sg_cols][logY > 0]

# gonna drop 0 earners first, I'll try it with undropped too
X_filtered_transformed = X_transformed[logY > 0]
x_train, y_train = X_filtered_transformed, logY[logY > 0]

# first, non pca reg(log winnings against sg)
non_pca_reg = LinearRegression().fit(X_filtered, y_train)
print('Non PCA coefficients:', non_pca_reg.coef_)
print('Non PCA Train R^2:', non_pca_reg.score(X_filtered, y_train))
# 0.6324644393760144

# linreg of log winnings against PCs 
reg = LinearRegression().fit(x_train, y_train)
print('PC coefficients:', reg.coef_)
print('PC Train R^2:', reg.score(x_train, y_train))
# 0.594688097599677

# translate PC coeffs to SG coeffs
stat_effects = pd.Series(reg.coef_ @ pca.components_, index=sg_cols)
print("SG effects on log winnings:")
print(stat_effects)

holdout_transformed = pca.transform(holdout_stats[sg_cols])
holdout_Y = holdout_stats['winnings'] * 1e6
log_holdout_Y = np.log1p(holdout_Y)

X_holdout__filtered_transformed = holdout_transformed[log_holdout_Y > 0]
log_Y_holdout_filtered = log_holdout_Y[log_holdout_Y > 0]

print('Holdout R^2:', reg.score(X_holdout__filtered_transformed, log_Y_holdout_filtered))
# 0.5198839537258719, not as good but not by a crazy amount. To be expected.

# residuals 
fitted = reg.predict(x_train)
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(fitted, y_train - fitted, alpha=0.6)
ax.axhline(0, color='gray', linewidth=0.8)
ax.set_xlabel('Fitted log winnings')
ax.set_ylabel('Residual')
plt.title('Residuals vs Fitted')
plt.show()

# clear heteroskedasticity, definitely seen worse though