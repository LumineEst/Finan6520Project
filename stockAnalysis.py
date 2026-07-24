import os
import warnings
os.environ['PYTHONWARNINGS'] = 'ignore'
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import xgboost as xgb
import umap
import shap
import optuna
import re
import matplotlib.patheffects as patheffects
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness
from sklearn.metrics import silhouette_score, matthews_corrcoef, normalized_mutual_info_score, precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay
from sklearn.cluster import DBSCAN, KMeans
from sklearn.covariance import MinCovDet, EmpiricalCovariance
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.linear_model import LogisticRegression

# ============================================
# GLOBAL CONFIGURATION CONSTANTS
# ============================================
randSeed = 37         # Random seed for all stochastic components (Optuna, UMAP, Clustering, Modeling)
corrThres = 0.85      # Spearman correlation threshold for pruning highly correlated features
varMIThres = 0.03     # Minimum mutual information score required for a feature to be retained
trustThres = 0.75     # Minimum average trustworthiness score for manifold embeddings
priorWeight = 5       # Bayesian prior weight used in sector-level fundamental smoothing
numDimStudies = 15    # Number of Optuna trials for dimensionality reduction / clustering selection
numModelTrials = 100  # Number of Optuna trials per local classifier (cluster-level models)
tradeTrials = 250     # Number of Optuna trials to determine optimal trading thresholds

def expanding_preprocess(X, scalerType='standard'):
    """
    Apply time-safe expanding window imputation and scaling.
    """
    # Expanding median imputation expanding().median().shift(1) -> past median for each time step
    groupObj = X.groupby(level='Ticker') if 'Ticker' in X.index.names else X.groupby('Ticker')
    imputed = X.fillna(groupObj.transform(lambda x: x.expanding().median().shift(1)))
    imputed = imputed.fillna(0)
    if scalerType is None: return imputed

    # Expanding scaling (all statistics based only on past data)
    if scalerType == 'standard':
        # Expanding mean and standard deviation per ticker (shifted by one step)
        mean = groupObj.transform(lambda x: x.expanding().mean().shift(1))
        std = groupObj.transform(lambda x: x.expanding().std().shift(1).replace(0, 1))
        return ((imputed - mean) / std).fillna(0)

    elif scalerType == 'minmax':
        # Expanding minimum and maximum per ticker (shifted by one step)
        minVal = groupObj.transform(lambda x: x.expanding().min().shift(1))
        maxVal = groupObj.transform(lambda x: x.expanding().max().shift(1))
        rangeVal = (maxVal - minVal).replace(0, 1)
        return ((imputed - minVal) / rangeVal).fillna(0)

    else:
        # Expanding median and interquartile range per ticker (shifted by one step)
        median = groupObj.transform(lambda x: x.expanding().median().shift(1))
        q75 = groupObj.transform(lambda x: x.expanding().quantile(0.75).shift(1))
        q25 = groupObj.transform(lambda x: x.expanding().quantile(0.25).shift(1))
        return ((imputed - median) / (q75 - q25).replace(0, 1)).fillna(0)

# ===============================================
# PHASE 1: TARGET DEFINITION & VARIABLE SELECTION
# ===============================================
def variable_selection(df, featureCols, priceCol='Adj Close', forwardWindow=20, valuationCol='EV/EBITDA', sectorCol='Sector'):
    """
    Define forward-looking alpha targets and select a de-correlated feature subset.
      1. Enforce chronological ordering by Ticker/Date.
      2. Compute forward log-return ("Forward Alpha") over a specified horizon.
      3. Standardize forward alpha per date and define a trinary Target State.
      4. Compute mutual information per feature against the Target State.
      5. Prune highly correlated features using Spearman correlation + MI.
      6. Visualize information gain and post-pruning correlations.
      7. Build a sector-level clustermap for the selected features.
    """
    dfEval = df.copy()

    # Enforce strict time-series order (needed for time-series CV and expanding statistics)
    if 'Date' in dfEval.index.names: dfEval = dfEval.sort_index(level=['Ticker', 'Date'])
    elif 'Date' in dfEval.columns: dfEval = dfEval.sort_values(['Ticker', 'Date'])

    # Compute forward log alpha per ticker over the specified forward window
    if 'Ticker' in dfEval.index.names:
        dfEval['Forward Alpha'] = np.log(dfEval.groupby(level='Ticker')[priceCol].shift(-forwardWindow) / dfEval[priceCol])
    else:
        dfEval['Forward Alpha'] = np.log(dfEval.groupby('Ticker')[priceCol].shift(-forwardWindow) / dfEval[priceCol])

    # Restrict to rows where the forward alpha is defined (non-NaN) then extract dates in a unified way (index vs column)
    trainMask = dfEval['Forward Alpha'].notna()
    dates = (dfEval.loc[trainMask].index.get_level_values('Date') if 'Date' in dfEval.index.names  else dfEval.loc[trainMask, 'Date'])

    # Compute per-date mean and std of forward alpha for cross-sectional standardization
    alphaMean = dfEval.loc[trainMask].groupby(dates)['Forward Alpha'].transform('mean')
    alphaStd = (dfEval.loc[trainMask].groupby(dates)['Forward Alpha'].transform('std').replace(0, 1).fillna(1))

    # Z-score of forward alpha within each date cross-section
    alphaZ = (dfEval.loc[trainMask, 'Forward Alpha'] - alphaMean) / alphaStd

    # Initialize trinary target state: -1 (down), 0 (neutral), 1 (up)
    dfEval['Target State'] = 0
    conditions = [(alphaZ < -1), (alphaZ > 1)]
    dfEval.loc[trainMask, 'Target State'] = np.select(conditions, [-1, 1], default=0)

    # Auxiliary target for manifold supervision (neutral = -1, down = 0, up = 1)
    dfEval['UMAP Target'] = -1
    dfEval.loc[trainMask, 'UMAP Target'] = np.select(conditions, [0, 1], default=-1)

    # Apply time-safe preprocessing (imputation only) to training features
    XTrain = expanding_preprocess(dfEval.loc[trainMask, featureCols], scalerType=None)
    yTrinary = dfEval.loc[trainMask, 'Target State'].values.astype(int)

    # Compute mutual information scores of features against the trinary target
    miScores = mutual_info_classif(XTrain, yTrinary, random_state=randSeed)
    miSeries = pd.Series(miScores, index=featureCols).sort_values(ascending=False)

    # Compute absolute Spearman correlation matrix between features
    corrMatrix = XTrain.corr(method='spearman').abs()
    dropped = set()

    # Prune highly correlated feature pairs, keeping the one with higher MI
    for i in range(len(corrMatrix.columns)):
        for j in range(i + 1, len(corrMatrix.columns)):
            colA, colB = corrMatrix.columns[i], corrMatrix.columns[j]
            corrVal = corrMatrix.iloc[i, j]

            # If correlation between variables is above the threshold, drop the feature with the lower mutual information score
            if corrVal > corrThres:
                if miSeries[colA] > miSeries[colB]: dropCol, keepCol = colB, colA
                else: dropCol, keepCol = colA, colB
                if dropCol not in dropped:
                    print(
                        f"[Prune] Dropping '{dropCol}' due to high correlation with "
                        f"'{keepCol}' (|Spearman|={corrVal:.3f}) and lower MI "
                        f"({miSeries[dropCol]:.4f} < {miSeries[keepCol]:.4f})."
                    )
                    dropped.add(dropCol)
                
    # Keep features that are not pruned and have MI above the relevance threshold
    selectedFeatures = [f for f in featureCols if f not in dropped and miSeries[f] > varMIThres]
    if len(selectedFeatures) < 2: selectedFeatures = miSeries[~miSeries.index.isin(dropped)].head(5).index.tolist()

    # -------------------------------
    # VISUALIZATION: Information Gain
    # -------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))

    # Bar plot: MI each selected feature contributes (top 10)
    topFeatures = miSeries[selectedFeatures].sort_values(ascending=False).head(10)
    sns.barplot(x=topFeatures.values, y=topFeatures.index, palette='viridis', ax=axes[0])
    axes[0].set_title("Information Gain (Surviving Features)")
    axes[0].set_xlabel("Mutual Information Score")

    # Heatmap: post-pruning Spearman correlation structure among selected features
    mask = np.triu(np.ones_like(corrMatrix.loc[selectedFeatures, selectedFeatures], dtype=bool))
    sns.heatmap( corrMatrix.loc[selectedFeatures, selectedFeatures], mask=mask, annot=True, cmap='RdBu_r', fmt=".2f", vmin=-1, vmax=1, ax=axes[1])
    axes[1].set_title("Post-Pruning Spearman Correlation")
    plt.setp(axes[1].get_xticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    # ----------------------------------------------
    # VISUALIZATION: Sector / Feature Clustermap Map
    # ----------------------------------------------
    if sectorCol not in dfEval.columns: return dfEval, selectedFeatures

    # Safely extract dates depending on whether they are stored in the index or a column
    dateGroup = (dfEval.index.get_level_values('Date') if 'Date' in dfEval.index.names else dfEval['Date'])

    # Filter to keep only features that vary across tickers on the same day
    uniques = dfEval[selectedFeatures].groupby(dateGroup).nunique().max()
    companyFeatures = uniques[uniques > 1].index.tolist()
    if not companyFeatures: return dfEval, selectedFeatures

    # Group by sector using only company-specific features (median cross-section)
    grouped = dfEval.groupby(sectorCol)[companyFeatures].median().dropna(axis=1, how='all')
    if grouped.empty or len(grouped) < 2: return dfEval, selectedFeatures

    # Standardize sector-level feature medians for clustering
    scaled = StandardScaler().fit_transform(grouped)
    scaledDf = pd.DataFrame(scaled, index=grouped.index, columns=grouped.columns)

    # Generates a hierarchical clustermap based on sector-level median values
    cg = sns.clustermap(scaledDf.T, cmap='coolwarm', metric='euclidean', method='ward', figsize=(12, 10))
    cg.fig.suptitle("Hierarchical Clustermap of Features by Sector", fontsize=16, fontweight='bold', y=1.02)
    plt.setp(cg.ax_heatmap.get_xticklabels(), rotation=45, ha='right')
    plt.show()

    return dfEval, selectedFeatures

# ==============================================
# PHASE 2: DIMENSIONALITY REDUCTION & CLUSTERING
# ==============================================
def get_multi_trust(XScaled, emb, maxSamples=10000):
    """
    Compute an average trustworthiness score over multiple neighborhood sizes.
    """
    # Optionally downsample if the dataset is very large
    if len(XScaled) > maxSamples:
        np.random.seed(randSeed)
        idx = np.random.choice(len(XScaled), maxSamples, replace=False)
        xEval, embEval = XScaled[idx], emb[idx]
    else: xEval, embEval = XScaled, emb

    # Neighborhood sizes to probe for local structure preservation
    neighbors = [10, 15, 20, 50, 100, 250]
    validNeighbors = [n for n in neighbors if n < len(xEval)]
    if not validNeighbors: return 0.0

    # Average trustworthiness across all valid k values
    return np.mean([trustworthiness(xEval, embEval, n_neighbors=n) for n in validNeighbors])

def manifold_clustering(df, features, sectorCol):
    """
    Build manifold embeddings and cluster assignments for the dataset.
      1. Extract raw feature matrix and targets (sector and alpha-based).
      2. For each target type ("Alpha" and "Peer"), run Optuna to:
         a. Choose a scaling strategy.
         b. Tune UMAP, PCA, t-SNE.
         c. Tune clustering (KMeans / DBSCAN).
         d. Score each combination using trustworthiness + silhouette + penalties.
      3. Refit the best configuration and store:
         - Embedding coordinates (DimOne/Two/Three).
         - Cluster IDs (per target).
      4. Compute directional feature loadings for the optimal manifold.
      5. Visualize feature loadings and sector composition by cluster.
    """
    XRaw = df[features].copy()     # Raw feature matrix for manifold learning

    # Encoded sector labels (for peer manifold supervision / analysis)
    ySector = LabelEncoder().fit_transform(df[sectorCol].astype(str)) if sectorCol in df.columns else None

    # Alpha-based manifold target (-1/0/1 mapped earlier into UMAP Target)
    yAlpha = df['UMAP Target'].values.astype(int)    
    nComp = 3   # Number of embedding dimensions to use

    def optimize_and_build(targetName, yTarget):
        """
        Run Optuna to find an optimal manifold + clustering configuration for a given target.
        """
        def objective(trial):
            # Choose a scaling strategy for the raw features
            scalerChoice = trial.suggest_categorical('scaler', ['standard', 'minmax', 'robust'])
            try:
                XScaled = expanding_preprocess(XRaw, scalerType=scalerChoice).values
                # Choose dimensionality reduction algorithm and supervision level
                dimAlgo = trial.suggest_categorical('dimAlgo', ['UMAP', 'PCA'])  #'t-SNE'
                supervision = trial.suggest_categorical('supervision', ['Unsupervised', 'Targeted'])

                # ----- Dimensionality Reduction Block -----
                # Linear PCA
                if dimAlgo == 'PCA':
                    # Tune PCA-specific hyperparameters
                    pcaComponents = trial.suggest_int( 'pcaComponents', 3, min(10, XScaled.shape[1]))
                    pcaWhiten = trial.suggest_categorical('pcaWhiten', [False, True])
                
                    # Fit PCA with the chosen number of components and whitening option
                    embFull = PCA(n_components=pcaComponents, whiten=pcaWhiten, random_state=randSeed).fit_transform(XScaled)
                    emb = embFull[:, :nComp]
    
                # t-SNE (placeholder—disabled in current search space--due to increasing runtime with little impact)
                elif dimAlgo == 't-SNE':
                    perp = trial.suggest_int('perplexity', 10, min(50, len(XScaled) - 1))
                    emb = TSNE(n_components=nComp, perplexity=perp, init='pca', random_state=randSeed).fit_transform(XScaled)

                # UMAP: tune neighbors, min_dist, and metric; optionally use a supervision target
                else:
                    numNeighbors = trial.suggest_int('numNeighbors', 30, min(200, len(XScaled) - 1))
                    minDist = trial.suggest_float('minDist', 0.0, 0.8)
                    uMetric = trial.suggest_categorical('uMetric', ['euclidean', 'cosine', 'correlation', 'braycurtis'])

                    if targetName == 'Alpha': 
                        targetArr = None
                        supervision = 'Unsupervised'
                    else: targetArr = yTarget if supervision == 'Targeted' and yTarget is not None else None

                    # Target weight only matters when a supervision target exists
                    tWeight = trial.suggest_float('tWeight', 0.1, 0.9) if yTarget is not None else 0.5
                    emb = umap.UMAP( n_components=nComp, n_neighbors=numNeighbors, min_dist=minDist, metric=uMetric, 
                        target_weight=tWeight, random_state=randSeed).fit_transform(XScaled, y=targetArr)

                # Reject manifolds that do not preserve local topology well enough
                trust = get_multi_trust(XScaled, emb)
                if trust < trustThres: return -1.0

                # ----- Clustering Block -----
                clusterAlgo = trial.suggest_categorical('clusterAlgo', ['KMeans']) #'DBSCAN'
                if clusterAlgo == 'KMeans':
                    k = trial.suggest_int('k', 3, 7)
                    labels = KMeans(n_clusters=k, random_state=randSeed, n_init='auto').fit_predict(emb)
                else: labels = DBSCAN(eps=trial.suggest_float('eps', 0.3, 2.0),
                        min_samples=trial.suggest_int('minSamples', 3, 25)).fit_predict(StandardScaler().fit_transform(emb))
                
                # Mask out noise labels for scoring and viability checks
                validMask = labels != -1
                numClusters = len(set(labels[validMask]))
                
                # Enforce a stricter minimum cluster count for Alpha
                if targetName == 'Alpha':
                    if numClusters < 3: return -1.0
                else:
                    if numClusters < 2: return -1.0
                
                # Silhouette score on the embedding, using only valid (non-noise) labels
                sampleSize = min(15000, validMask.sum())
                try: baseScore = silhouette_score(emb[validMask], labels[validMask], sample_size=sampleSize, random_state=randSeed)
                except ValueError: return -1.0
                
                # Penalize noisy solutions (lots of -1 labels)
                noiseRatio = 1.0 - (validMask.sum() / len(labels))
                
                # Penalize highly fragmented solutions (too many clusters)
                fragPenalty = 0.1 * max(0, numClusters - 4)
                
                # Penalize solutions where only a small subset of points are in "usable" clusters
                clusterCounts = pd.Series(labels[validMask]).value_counts()
                usableSamples = clusterCounts[clusterCounts >= 100].sum()
                viabilityRatio = usableSamples / len(labels)
                viabilityPenalty = (1.0 - viabilityRatio) * 0.5
                
                # Cluster-size distribution (for balance / entropy)
                clusterProbs = clusterCounts / clusterCounts.sum()
                entropy = -np.sum(clusterProbs * np.log(clusterProbs)) / np.log(numClusters)
                
                # Require at least 3 "big" clusters for Alpha (where possible)
                bigClusterPenalty = 0.0
                if targetName == 'Alpha':
                    bigClusterCounts = clusterCounts[clusterCounts >= 500]
                    numBigClusters = len(bigClusterCounts)
                    if numBigClusters < 3:  bigClusterPenalty = 0.30 * (3 - numBigClusters)
                
                # Label alignment only for Peer (as you had)
                labelScore = 0.0
                if (targetName == 'Peer') and (yTarget is not None):
                    targetValid = np.asarray(yTarget)[validMask]
                    if len(np.unique(targetValid)) > 1: labelScore = normalized_mutual_info_score(targetValid, labels[validMask])
                    bigClusterCounts = clusterCounts[clusterCounts >= 500]
                    numBigClusters = len(bigClusterCounts)
                    if numBigClusters < 3:  bigClusterPenalty = 0.30 * (5 - numBigClusters)
                
                # Slightly higher entropy weight for Alpha to encourage balanced cluster sizes
                if targetName == 'Alpha': entropyWeight = 0.3
                else: entropyWeight = 0.25
                
                # Final objective: silhouette adjusted for noise, fragmentation, viability, cluster balance, and (for Peer) label alignment
                return ((baseScore * (1.0 - (1.2 * noiseRatio))) - fragPenalty - viabilityPenalty - bigClusterPenalty + (entropyWeight * entropy) + labelScore)
            except ValueError: raise optuna.TrialPruned()
            except Exception: raise optuna.TrialPruned()

        # ----- Optuna Hyperparameter Tuning -----
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=randSeed))
        study.optimize(objective, n_trials=numDimStudies, show_progress_bar=False)

        # ----- Rebuild the Optimal Manifold and Clusters -----
        best = study.best_params
        XScaled = expanding_preprocess(XRaw, scalerType=best['scaler']).values

        # Recreate the best embedding using the chosen algorithm and hyperparameters
        if best['dimAlgo'] == 'PCA':
            pcaComponents = best.get('pcaComponents', nComp)
            pcaWhiten = best.get('pcaWhiten', False)
            embFull = PCA(n_components=pcaComponents, whiten=pcaWhiten, random_state=randSeed).fit_transform(XScaled)
        
            # Keep a fixed 3-D manifold representation for the rest of the pipeline
            emb = embFull[:, :nComp]
        elif best['dimAlgo'] == 't-SNE':
            emb = TSNE(n_components=nComp, perplexity=best['perplexity'], init='pca', random_state=randSeed).fit_transform(XScaled)
        else:
            if targetName == 'Alpha': targetArr = None
            else: targetArr = yTarget if best['supervision'] == 'Targeted' and yTarget is not None else None
            tWeight = best.get('tWeight', 0.5)
            emb = umap.UMAP(n_components=nComp, n_neighbors=best['numNeighbors'], min_dist=best['minDist'],
                metric=best['uMetric'], target_weight=tWeight, random_state=randSeed).fit_transform(XScaled, y=targetArr)

        # Rebuild clustering using the best algorithm and tuned parameters
        if best['clusterAlgo'] == 'KMeans': labels = KMeans(n_clusters=best['k'], random_state=randSeed, n_init=10).fit_predict(emb)
        else: labels = DBSCAN(eps=best['eps'],  min_samples=best['minSamples']).fit_predict(StandardScaler().fit_transform(emb))

        # Shift cluster IDs so that -1 remains noise, but valid clusters start at 1
        labels = np.where(labels != -1, labels + 1, -1)
        # Store embedding coordinates and cluster IDs back into the main DataFrame
        df[f'{targetName} DimOne'] = emb[:, 0]
        df[f'{targetName} DimTwo'] = emb[:, 1]
        df[f'{targetName} DimThree'] = emb[:, 2]
        df[f'{targetName} ClusterID'] = labels

        def get_directional_mi(xMatrix, umapDim, featureNames):
            """
            Compute signed mutual information loadings between features and a manifold dimension.
            - MI magnitude comes from mutual_info_regression.
            - Sign is given by Spearman correlation between each feature and the embedding dimension.
            """
            miScores = mutual_info_regression(xMatrix, umapDim, random_state=randSeed)
            directionalLoadings = []
            for i in range(xMatrix.shape[1]):
                corr, _ = spearmanr(xMatrix[:, i], umapDim)
                sign = np.sign(corr) if corr != 0 else 1
                directionalLoadings.append(miScores[i] * sign)
            return pd.Series(directionalLoadings, index=featureNames)

        # -------------------------------------------
        # VISUALIZATION: Feature Loadings on Manifold
        # -------------------------------------------
        miDimOne = get_directional_mi(XScaled, emb[:, 0], features)
        miDimTwo = get_directional_mi(XScaled, emb[:, 1], features)
        miDimThree = get_directional_mi(XScaled, emb[:, 2], features)

        loadingsDf = pd.DataFrame({'DimOne': miDimOne, 'DimTwo': miDimTwo, 'DimThree': miDimThree}, index=features,
        ).sort_values('DimOne', ascending=False)

        # Reorder by absolute contribution to DimOne (largest absolute loading first)
        loadingsDf = loadingsDf.iloc[loadingsDf['DimOne'].abs().argsort()[::-1]]

        plt.figure(figsize=(8, 6))
        sns.heatmap(loadingsDf.head(10), annot=True, cmap='viridis', fmt=".3f")
        metricStr = best.get('uMetric', 'N/A')
        supervisionStr = best.get('supervision', 'N/A')
        
        plt.title(
            f"Top 10 Feature Impacts to {targetName} Optimal Manifold "
            f"({metricStr} {best['dimAlgo']} - {supervisionStr})"
        )
        plt.tight_layout()
        plt.show()

        # ---------------------------------------------
        # VISUALIZATION: Sector Composition per Cluster
        # ---------------------------------------------
        if sectorCol in df.columns:
            validClusters = df[df[f'{targetName} ClusterID'] != -1].copy()
            validClusters['Cluster Name'] = "Cluster" + validClusters[f'{targetName} ClusterID'].astype(str)
            compCounts = pd.crosstab(validClusters['Cluster Name'], validClusters[sectorCol])
            
            # Sort clusters by total size then count (largest to smallest)
            clusterTotals = compCounts.sum(axis=1)
            compCounts = compCounts.loc[clusterTotals.sort_values(ascending=False).index]
            sectorTotals = compCounts.sum(axis=0)
            compCounts = compCounts[sectorTotals.sort_values(ascending=False).index]
            
            # Stacked bar of counts
            compCounts.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='Set3', edgecolor='black')
            plt.title(f"Sector Composition Map ({targetName} Manifold Output)")
            plt.xlabel("")
            plt.ylabel("Count")   # <-- quantity, not percentage
            plt.legend(title="Sector", bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.xticks(rotation=0)
            plt.tight_layout()
            plt.show()
    # Build the alpha-prediction manifold and peer (sector) manifold
    optimize_and_build('Alpha', yAlpha)
    optimize_and_build('Peer', ySector)
    return df

# ==========================================
# PHASE 3: MASTER PROOF & LOCAL EXPERTS
# ==========================================

def calculate_decay_weights(dates, hlDays=180, yLocal=None):
    """
    Compute time-decay weights, optionally combined with class balancing.
      - Applies exponential time decay based on a half-life (hlDays).
      - If yLocal is provided as labels in {0,1}, upweights positive class
        using a simple Bayesian ratio to counter class imbalance.
      - If yLocal is a scalar (int/float), it is treated as hlDays, and no
        class balancing is applied.
    """
    # Parse the 14-digit packed timestamp (YYYYMMDDHHMMSS) into datetime
    parsedDates = pd.to_datetime(dates.astype(str), format='%Y%m%d%H%M%S')
    latestDate = parsedDates.max()

    # Time delta (in days) between each observation and the most recent date
    deltaDays = (latestDate - parsedDates) / pd.Timedelta(days=1)

    # If yLocal is a scalar, interpret it as a custom half-life and drop labels
    if isinstance(yLocal, (int, float)): hlDays, yLocal = yLocal, None

    # Exponential decay: w(t) = exp(-lambda * age), lambda from half-life
    alphaDecay = np.log(2) / hlDays
    timeWeights = np.exp(-alphaDecay * deltaDays.to_numpy())

    # Optional class balancing when binary labels are provided
    if yLocal is not None:
        yArray = np.asarray(yLocal)
        posMask = (yArray == 1)
        nPos = max(1, posMask.sum())
        nNeg = len(yArray) - nPos

        # Bayesian adjustment factor: more weight when positives are rare
        bayesRatio = (nNeg + priorWeight) / (nPos + priorWeight)
        combinedWeights = timeWeights.copy()
        combinedWeights[posMask] *= bayesRatio
        return combinedWeights

    # If no labels provided, return pure time-decay weights
    return timeWeights


def time_series_mcc_cv(model, X, y, cv, sample_weight=None):
    """
    Compute MCC for a model under a TimeSeriesSplit CV scheme.
    - Skips folds where the training set collapses to a single class.
    - Returns the mean Matthews correlation coefficient across valid folds.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    sw = np.asarray(sample_weight) if sample_weight is not None else None
    scores = []
    
    for trainIdx, testIdx in cv.split(X):
        # Skip folds that cannot train a classifier
        if len(np.unique(y[trainIdx])) < 2: continue
        m = clone(model)
        fit_kwargs = {}
        if sw is not None: fit_kwargs['sample_weight'] = sw[trainIdx]

        # Fit on chronological training slice, evaluate on forward test slice
        m.fit(X[trainIdx], y[trainIdx], **fit_kwargs)
        yPred = m.predict(X[testIdx])
        scores.append(matthews_corrcoef(y[testIdx], yPred))

    # If no valid folds were found, return 0.0
    return np.mean(scores) if scores else 0.0


def optimize_cluster_model(X, y, datesSeries, isLinear):
    """
    Tune Local Expert hyperparameters per cluster using Optuna.
      - X: feature matrix for this cluster.
      - y: binary labels (target for this expert).
      - datesSeries: packed timestamps for time-decay weighting.
      - isLinear: True for LogisticRegression, False for XGBoost.
      - best_params: dict of tuned hyperparameters for the chosen model family.
    """
    def objective(trial):
        # Tune half-life for time decay (shorter half-life = stronger recency bias)
        hlDays = trial.suggest_int('hlDays', 60, 180)
        timeWeights = calculate_decay_weights(datesSeries, hlDays=hlDays)
        cv = TimeSeriesSplit(n_splits=3)
        try:
            if isLinear:
                # Logistic regression hyperparameters
                cValue = trial.suggest_float('cValue', 1e-2, 10.0, log=True)
                penaltyType = trial.suggest_categorical('penaltyType', ['l1', 'l2'])
                model = LogisticRegression(C=cValue, penalty=penaltyType, solver='saga',
                    max_iter=1000, random_state=randSeed, class_weight='balanced')

                # Time-series MCC with time-decay weights
                score = time_series_mcc_cv(model, X, y, cv=cv, sample_weight=timeWeights)

            else:
                # XGBoost tree capacity is bounded by data size for stability
                maxD = min(6, max(3, len(X) // 15))
                maxEst = min(150, max(50, len(X) * 2))

                # XGBoost hyperparameters
                maxDepth = trial.suggest_int('maxDepth', 3, maxD)
                learningRate = trial.suggest_float('learningRate', 1e-3, 0.1, log=True)
                nEstimators = trial.suggest_int('nEstimators', 50, maxEst)
                gammaVal = trial.suggest_float('gamma', 1e-3, 5.0, log=True)
                alphaVal = trial.suggest_float('reg_alpha', 1e-3, 5.0, log=True)
                lambdaVal = trial.suggest_float('reg_lambda', 1e-3, 5.0, log=True)
                colSample = trial.suggest_float('colsample_bytree', 0.5, 0.9)
                subSample = trial.suggest_float('subsample', 0.6, 0.9)
                minChildWeight = trial.suggest_int('min_child_weight', 1, 7)

                # Basic imbalance handling: scale positive class by neg/pos
                posMask = (y == 1)
                numPos = max(1, posMask.sum())
                negWeight = (len(y) - posMask.sum()) / numPos

                combinedWeights = timeWeights.copy()
                combinedWeights[posMask] *= negWeight

                model = xgb.XGBClassifier(max_depth=maxDepth, learning_rate=learningRate, n_estimators=nEstimators,
                    gamma=gammaVal, reg_alpha=alphaVal, reg_lambda=lambdaVal, colsample_bytree=colSample,
                    subsample=subSample, min_child_weight=minChildWeight, random_state=randSeed)

                # Time-series MCC with combined time + class weights
                score = time_series_mcc_cv(model, X, y, cv=cv, sample_weight=combinedWeights)

            # Guard against numerical issues returning NaN
            return 0.0 if np.isnan(score) else score

        except ValueError: return 0.0

    # Run Optuna study to maximize the objective (MCC under time-series CV)
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=randSeed))
    study.optimize(objective, n_trials=numModelTrials, show_progress_bar=False)
    return study.best_params

def generate_oos_probabilities(estimator, Xin, yin, cv, sampleWeight=None):
    """
    Generate out-of-sample positive-class probabilities under TimeSeriesSplit.
      - For each fold, train on all data up to that fold's train indices
        and predict probabilities on the forward test indices.
      - For the very first train fold, also fill in in-sample probabilities
        so that every row receives a model-based probability.
      - Handles both binary (two-column) and degenerate (one-column) predict_proba.
    """
    X = np.asarray(Xin)
    y = np.asarray(yin)
    sw = np.asarray(sampleWeight) if sampleWeight is not None else None

    # Initialize all probabilities to 0.5 as a neutral baseline
    oosProbs = np.full(len(y), 0.5)
    splits = list(cv.split(X))

    def get_pos_probs(model, xEval):
        """Extract probabilities for the positive class (label 1) safely."""
        probs = model.predict_proba(xEval)

        # If only one column, that single class may or may not be the positive one
        if probs.shape[1] == 1: return np.ones(len(xEval)) if model.classes_[0] == 1 else np.zeros(len(xEval))

        # Otherwise, find the column index corresponding to class label 1
        classIdx = np.where(model.classes_ == 1)[0]
        return probs[:, classIdx[0]] if len(classIdx) > 0 else np.zeros(len(xEval))

    # In-sample predictions for the initial chronological training fold
    firstTrain = splits[0][0]
    if len(np.unique(y[firstTrain])) > 1:
        firstModel = clone(estimator)
        if sampleWeight is not None: firstModel.fit(X[firstTrain], y[firstTrain], **{'sample_weight': sw[firstTrain]})
        else: firstModel.fit(X[firstTrain], y[firstTrain])
        oosProbs[firstTrain] = get_pos_probs(firstModel, X[firstTrain])

    # Strict out-of-sample predictions for the forward-walking timeline
    for trainIdx, testIdx in splits:
        if len(np.unique(y[trainIdx])) > 1:
            foldModel = clone(estimator)
            if sampleWeight is not None: foldModel.fit(X[trainIdx], y[trainIdx], **{'sample_weight': sw[trainIdx]})
            else: foldModel.fit(X[trainIdx], y[trainIdx])
            oosProbs[testIdx] = get_pos_probs(foldModel, X[testIdx])

    return oosProbs

def multi_classification(df, features): 
    """
    Train cluster-specific 'local expert' classifiers for Buy/Sell signals.
      1. Build a local dataset (features + alpha manifold coordinates).
      2. For each direction (Buy=+1, Sell=-1):
         a. Tune a linear (LogisticRegression) and a nonlinear (XGBoost) model.
         b. Compare their MCC under time-series CV.
         c. Select the dominant model and generate:
            - Out-of-sample probabilities for training rows.
            - Full-probability surface over all rows in the cluster.
         d. Compute SHAP values for interpretability.
      3. Aggregate cluster-level performance and SHAP summaries for later plots.
    """
    clusterResults = []   # Stores MCC comparison and dominance per (cluster, direction)
    localExperts = {}     # Mapping: "Cluster <id> Buy/Sell" -> trained model
    shapVisuals = []      # SHAP summaries for the global feature-importance matrix
    alphaRegressors = {}  # clusterId -> regression model for Forward Alpha

    # Use both raw features and alpha manifold coordinates as predictors
    modelFeatures = features + ['Alpha DimOne', 'Alpha DimTwo', 'Alpha DimThree']

    # Initialize prediction/probability columns with neutral defaults
    df['Buy Prob'] = 0.5
    df['Sell Prob'] = 0.5
    df['Predicted Buy'] = 0
    df['Predicted Sell'] = 0
    df['Pred Alpha'] = np.nan
    df['Buy Score'] = np.nan
    df['Sell Score'] = np.nan

    # Loop over each alpha cluster to train local experts
    for clusterId in df['Alpha ClusterID'].unique():
        # Skip noise / unassigned cluster and require a minimum sample size per cluster
        if clusterId == -1: continue
        mask = df['Alpha ClusterID'] == clusterId
        if mask.sum() < max(50, int(0.01 * len(df))): continue

        localDf = df[mask].copy()
        # Enforce strict chronology within the cluster before TimeSeriesSplit
        if 'Date' in localDf.columns: localDf = localDf.sort_values('Date')
        else: localDf = localDf.sort_index(level='Date')

        # Time-safe imputation and standardized feature matrix
        XLocal = expanding_preprocess(localDf[modelFeatures], scalerType=None)
        XScaled = expanding_preprocess(XLocal, scalerType='standard').values

        # Extract dates in a consistent way for decay-weighting
        localDates = (localDf.index.get_level_values('Date') if 'Date' in localDf.index.names else localDf['Date'])

        # Use only rows with defined forward alpha for training
        trainMaskLocal = localDf['Forward Alpha'].notna()
        xLocalTrain = XLocal[trainMaskLocal]
        xScaledTrain = XScaled[trainMaskLocal]
        localDatesTrain = localDates[trainMaskLocal]
        
        # --- Per-cluster regression model for Forward Alpha ---
        yAlphaTrain = localDf.loc[trainMaskLocal, 'Forward Alpha']
        if len(yAlphaTrain) >= 50:  # minimal sample size to fit a stable regressor
            alphaReg = RandomForestRegressor(n_estimators=100, max_depth=4, random_state=randSeed)
            alphaReg.fit(xLocalTrain, yAlphaTrain)
            alphaRegressors[clusterId] = alphaReg
        
            # Predict Forward Alpha for all rows in this cluster using the same preprocessed features
            df.loc[mask, 'Pred Alpha'] = alphaReg.predict(XLocal)

        # Train two separate classifiers: one for Upside (Buy), one for Downside (Sell)
        for targetVal, probCol, predCol in [(1, 'Buy Prob', 'Predicted Buy'), (-1, 'Sell Prob', 'Predicted Sell')]:
            # Use only strong up/down labels; drop neutral (0) from training
            targetTrain = localDf.loc[trainMaskLocal, 'Target State']
            directionMask = targetTrain.isin([targetVal, -targetVal]).values

            # Not enough labeled examples for this direction in this cluster
            if directionMask.sum() < 10: continue

            # Restrict training data to up/down only
            xScaledTrainDir = xScaledTrain[directionMask]
            xLocalTrainDir = xLocalTrain[directionMask]
            localDatesTrainDir = localDatesTrain[directionMask]
            yLocalTrain = (targetTrain.values[directionMask] == targetVal).astype(int)

            # Skip directions that are too rare or degenerate for a cluster
            if np.unique(yLocalTrain).size < 2 or yLocalTrain.sum() < 3: continue

            cv = TimeSeriesSplit(n_splits=3)

            # Tune hyperparameters for both linear and XGBoost experts on this cluster
            linParams = optimize_cluster_model(xScaledTrainDir, yLocalTrain, localDatesTrainDir, isLinear=True)
            xgbParams = optimize_cluster_model(xLocalTrainDir, yLocalTrain, localDatesTrainDir, isLinear=False)

            # Rebuild optimal time-decay weights using tuned half-lives
            linWeights = calculate_decay_weights(localDatesTrainDir, hlDays=linParams.pop('hlDays'))
            xgbBaseWeights = calculate_decay_weights(localDatesTrainDir, hlDays=xgbParams.pop('hlDays'))

            # Re-apply imbalance penalty to optimal XGBoost weights
            posMask = (yLocalTrain == 1)
            xgbRatio = (len(yLocalTrain) - posMask.sum()) / (max(1, posMask.sum()))
            xgbCombinedWeights = xgbBaseWeights.copy()
            xgbCombinedWeights[posMask] *= xgbRatio

            # Rebuild linear expert with tuned hyperparameters
            linModel = LogisticRegression(
                C=linParams['cValue'],
                penalty=linParams['penaltyType'],
                solver='saga',
                max_iter=1000,
                random_state=randSeed,
                class_weight='balanced'
            )

            # Rebuild XGBoost expert with tuned hyperparameters
            xgbModel = xgb.XGBClassifier(
                max_depth=xgbParams['maxDepth'],
                learning_rate=xgbParams['learningRate'],
                n_estimators=xgbParams['nEstimators'],
                gamma=xgbParams['gamma'],
                reg_alpha=xgbParams['reg_alpha'],
                reg_lambda=xgbParams['reg_lambda'],
                random_state=randSeed,
                colsample_bytree=xgbParams['colsample_bytree'],
                subsample=xgbParams['subsample'],
                min_child_weight=xgbParams['min_child_weight']
            )

            # Time-series MCC comparison between linear and nonlinear experts
            linMcc = time_series_mcc_cv(linModel, xScaledTrainDir, yLocalTrain, cv=cv, sample_weight=linWeights)
            xgbMcc = time_series_mcc_cv(xgbModel, xLocalTrainDir, yLocalTrain, cv=cv, sample_weight=xgbCombinedWeights)

            dominance = "Linear" if linMcc > xgbMcc else "XGBoost"
            directionLabel = 'Buy' if targetVal == 1 else 'Sell'

            # --------------------------------
            # Fit Dominant Expert & SHAP Setup
            # --------------------------------
            if dominance == "Linear":
                expert = linModel

                # Out-of-sample probabilities for training rows (time-series CV)
                oosProbs = generate_oos_probabilities(expert, xScaledTrainDir, yLocalTrain, cv=cv, sampleWeight=linWeights)

                # Final fit on full training history for this cluster/direction
                expert.fit(xScaledTrainDir, yLocalTrain, sample_weight=linWeights)

                # Full probability surface over all rows in this cluster
                fullProbs = expert.predict_proba(XScaled)[:, 1]

                # Only overwrite for rows used in training (up/down); others stay at the 0.5 default
                trainDirMask = np.zeros_like(trainMaskLocal.values, dtype=bool)
                trainDirMask[trainMaskLocal.values] = directionMask
                fullProbs[trainDirMask] = oosProbs

                # Store probabilities and signed predictions in main df
                df.loc[mask, probCol] = fullProbs
                df.loc[mask, predCol] = expert.predict(XScaled) * targetVal

                # Linear SHAP explainer on standardized features (up/down subset)
                explainer = shap.LinearExplainer(expert, xScaledTrainDir)
                shapValues = explainer.shap_values(xScaledTrainDir)

            else:
                expert = xgbModel

                # Out-of-sample probabilities for training rows (time-series CV)
                oosProbs = generate_oos_probabilities(expert, xLocalTrainDir, yLocalTrain, cv=cv, sampleWeight=xgbCombinedWeights)

                # Final fit on full training history for this cluster/direction
                expert.fit(xLocalTrainDir, yLocalTrain, sample_weight=xgbCombinedWeights)

                # Full probability surface over all rows in this cluster
                fullProbs = expert.predict_proba(XLocal)[:, 1]

                trainDirMask = np.zeros_like(trainMaskLocal.values, dtype=bool)
                trainDirMask[trainMaskLocal.values] = directionMask
                fullProbs[trainDirMask] = oosProbs

                df.loc[mask, probCol] = fullProbs
                df.loc[mask, predCol] = expert.predict(XLocal) * targetVal

                # Separate "baseTree" only for SHAP (up/down subset)
                baseTree = xgbModel.fit(xLocalTrainDir, yLocalTrain, sample_weight=xgbCombinedWeights)
                explainer = shap.TreeExplainer(baseTree)
                shapValues = explainer.shap_values(xLocalTrainDir)

            # ---------------------------------
            # Cluster-level SHAP Visualizations
            # ---------------------------------
            if mask.sum() > 300:
                plotShap = shapValues[1] if isinstance(shapValues, list) else shapValues

                plt.figure(figsize=(8, 5))

                if dominance == "Linear":
                    meanShap = plotShap.mean(axis=0)
                    topIdx = np.argsort(np.abs(meanShap))[::-1][:10]
                    topFeatures = [modelFeatures[i] for i in topIdx]
                    topValues = meanShap[topIdx]

                    penalty = getattr(expert, 'penalty', None)
                    linType = "Lasso" if penalty == 'l1' else "Ridge"

                    meanFeatVal = np.asarray(xScaledTrainDir.mean(axis=0), dtype=float)
                    topFeatVals = meanFeatVal[topIdx]
                    norm = plt.Normalize(vmin=topFeatVals.min(), vmax=topFeatVals.max())
                    cmap = shap.plots.colors.red_blue
                    colors = [cmap(norm(v)) for v in topFeatVals]

                    fig, ax = plt.subplots(figsize=(8, 5))
                    sns.barplot(x=topValues, y=topFeatures, palette=colors, ax=ax)
                    ax.axvline(0, color="black", linewidth=1)
                    ax.set_xlabel("Mean SHAP (Signed impact on log-odds)")
                    ax.set_title(f"SHAP Feature Direction (Cluster {clusterId} - {directionLabel} {linType} Linear Model)")
                    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                    sm.set_array(topFeatVals)
                    cbar = fig.colorbar(sm, ax=ax)
                    cbar.set_label("Mean Standardized Feature Value")
                    fig.tight_layout()
                    plt.show()

                else:
                    shap.summary_plot(plotShap, xLocalTrainDir, feature_names=modelFeatures, show=False)
                    plt.title(f"SHAP Directional Impact: Cluster {clusterId} ({directionLabel} XGBoost Model)")
                    plt.tight_layout()
                    plt.show()

            # Persist the chosen expert for this (cluster, direction)
            localExperts[f"Cluster {clusterId} {directionLabel}"] = expert

            # -------------------------------
            # SHAP Top-5 extraction per model
            # -------------------------------
            meanShapAbs = np.abs(shapValues[1] if isinstance(shapValues, list) else shapValues).mean(axis=0)
            topIdx = np.argsort(meanShapAbs)[-5:][::-1]
            topFeaturesList = [modelFeatures[i] for i in topIdx]
            clusterLabel = f"Cluster {clusterId}"

            clusterResults.append({
                'clusterId': clusterId,
                'direction': directionLabel,
                'linMcc': linMcc,
                'xgbMcc': xgbMcc,
                'dominance': dominance,
                'clusterLabel': clusterLabel,
            })

            shapVisuals.append({
                'Model': f"Cluster {clusterId} ({directionLabel})",
                'Top 5': list(zip(topFeaturesList, [meanShapAbs[i] for i in topIdx])),
                'Raw SHAP': meanShapAbs,
                'Weight': len(XLocal),
            })

    df['Buy Score'] = df['Pred Alpha'].fillna(0.0)
    df['Sell Score'] = -df['Pred Alpha'].fillna(0.0)
        
    # ===========================================
    # Linear vs XGBoost MCC comparison
    # ===========================================
    if clusterResults:
        # Aggregate per-cluster MCC results into a DataFrame for plotting
        resultsDf = pd.DataFrame(clusterResults)
        fig, ax = plt.subplots(figsize=(8, 6))
    
        # Build a color map: one color per alpha cluster ID
        clusterIds = sorted(resultsDf['clusterId'].unique())
        clusterPalette = sns.color_palette("tab10", n_colors=len(clusterIds))
        clusterColorMap = {cid: clusterPalette[i] for i, cid in enumerate(clusterIds)}
    
        # Scatter each (cluster, direction) point: Linear MCC vs XGBoost MCC
        for _, row in resultsDf.iterrows():
            cid = row['clusterId']
            directionLabel = row['direction']  # 'Buy' or 'Sell'
            linMcc = row['linMcc']
            xgbMcc = row['xgbMcc']
    
            # Marker shape encodes direction: circle for Buy, X for Sell
            marker = 'o' if directionLabel == 'Buy' else 'x'
            color = clusterColorMap[cid]
    
            ax.scatter(linMcc, xgbMcc, c=[color], marker=marker, s=120, edgecolor='white', linewidth=0.8)
    
        # Reference diagonal: where Linear and XGBoost MCC are equal
        mccMin = resultsDf[['linMcc', 'xgbMcc']].min().min()
        mccMax = resultsDf[['linMcc', 'xgbMcc']].max().max()
        ax.plot([mccMin, mccMax], [mccMin, mccMax], 'k--', alpha=0.5)
    
        ax.set_xlabel('Linear MCC')
        ax.set_ylabel('XGBoost MCC')
        ax.set_title('Cluster-wise Linear vs XGBoost MCC')
        ax.grid(alpha=0.3)

        legendHandles = []
        legendLabels = []
    
        # One color entry per cluster (color encodes cluster ID)
        for cid in clusterIds:
            h = ax.scatter([], [], c=[clusterColorMap[cid]], marker='o', s=100, edgecolor='white', label=f'Cluster {cid}')
            legendHandles.append(h)
            legendLabels.append(f'Cluster {cid}')
    
        # Marker-only entries for Buy vs Sell (shape encodes direction)
        buyHandle = ax.scatter([], [], c='gray', marker='o', s=100, edgecolor='white', label='Buy')
        sellHandle = ax.scatter([], [], c='gray', marker='x', s=100, edgecolor='white', label='Sell')
    
        legendHandles.extend([buyHandle, sellHandle])
        legendLabels.extend(['Buy', 'Sell'])
        ax.legend(legendHandles, legendLabels, loc='lower right', fontsize='small')
    
        plt.tight_layout()
        plt.show()
    else:
        # If no cluster results were collected, skip this comparison plot
        print("[Warning] No clusterResults available; skipping MCC comparison plot.")
        
    # ----------------------------------------------
    # VISUALIZATION: Local Expert Feature Chart Grid
    # ----------------------------------------------
    if shapVisuals:
        matrixData, labelsData, models = [], [], []
    
        def sort_key(x):
            # Extract numeric cluster ID from 'Model' label for stable ordering
            nums = re.findall(r'\d+', x['Model'])
            return int(nums[0]) if nums else 0
        
        # Build a matrix of top-5 SHAP values per (cluster, direction) model
        for item in sorted(shapVisuals, key=sort_key):
            models.append(item['Model'])
            matrixData.append([v for f, v in item['Top 5']])
            labelsData.append([f"{f}\n({v:.3f})" for f, v in item['Top 5']])
            
        plt.figure(figsize=(15, max(4, len(models) * 0.85)))
        sns.heatmap(matrixData, annot=labelsData, fmt="", yticklabels=models, xticklabels=[f"Rank {i+1}" for i in range(5)],
            cmap='Blues', cbar_kws={'label': 'Mean |SHAP| Impact'}, linewidths=1, linecolor='white')
        plt.title("Local Expert Top 5 Features per Cluster", pad=15, fontweight='bold')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
    
    # ----------------------------------------
    # VISUALIZATION: Global Feature Importance
    # ----------------------------------------
    if shapVisuals:
        # Initialize an accumulator for global SHAP weighted by sample counts
        globalShap = np.zeros(len(modelFeatures))
        totalSamples = 0
    
        # Weighted sum of mean |SHAP| across all cluster-level models
        for item in shapVisuals:
            globalShap += item['Raw SHAP'] * item['Weight']
            totalSamples += item['Weight']
            
        # Convert to weighted average and take the top 10 most impactful features and generate barplot
        globalDf = (pd.Series(globalShap / totalSamples, index=modelFeatures).sort_values(ascending=False).head(10))
        plt.figure(figsize=(10, 6))
        sns.barplot(x=globalDf.values, y=globalDf.index, palette='magma')
        plt.title("Global Feature Importance (Top 10 Weighted Across Clusters)")
        plt.xlabel("Mean |SHAP| (Impact on Market Valuation)")
        plt.ylabel("")
        plt.tight_layout()
        plt.show()
    return df, localExperts

# ======================================
# PHASE 4: SENSITIVITY GRID & VALIDATION
# ======================================
def manifold_validation(df, valuationCols, bestZ, chiCutoff):
    """
    Diagnostics & charts based on tuned trading rules.
      - Calibrates Buy Prob (for reporting).
      - Uses existing MDist/Z and Validation to build:
        * Yield curve vs Z threshold (informational).
        * Precision-Recall Curves and Confusion Matrix
        * Risk/Reward Yield Matrix (Z vs MDist).
        * Alpha manifold 3D plot.
        * Peer manifold + sector ratios.
        * Decile performance plot.
        * Rolling long-short vs benchmark performance.
    """
    # -------------------------------------------
    # 2×2 diagnostic dashboard:
    #   (1,1) Buy score deciles vs Forward Alpha
    #   (1,2) Sell score deciles vs Forward Alpha
    #   (2,1) Precision–Recall curves (Buy & Sell)
    #   (2,2) Confusion matrix for last Month
    # -------------------------------------------
    dfDiag = df.reset_index()  # flattened copy with Date/Ticker as columns
    requiredCols = {'Date', 'Target State', 'Forward Alpha', 'Buy Prob', 'Sell Prob'}
    if requiredCols.issubset(dfDiag.columns):
        dfDiag = dfDiag.dropna(subset=['Forward Alpha']).copy()
        dfDiag['dateDt'] = pd.to_datetime(dfDiag['Date'], format='%Y%m%d%H%M%S', errors='coerce')
        dfDiag = dfDiag.dropna(subset=['dateDt'])
    
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axBuy, axSell, axPr, axCm = axes.flatten()
    
        # ---- (1,1) Buy Score deciles vs Forward Alpha ----
        if 'Buy Score Calibrated' in dfDiag.columns: scoreColBuy = 'Buy Score Calibrated'
        elif 'Buy Score' in dfDiag.columns: scoreColBuy = 'Buy Score'
        else: scoreColBuy = 'Buy Prob'
    
        try:
            rankedBuy = dfDiag[scoreColBuy].rank(method='first')
            dfDiag['buyDecile'] = pd.qcut(rankedBuy, 10, labels=False) + 1
            buyDecPerf = dfDiag.groupby('buyDecile')['Forward Alpha'].mean().sort_index()
            axBuy.plot(buyDecPerf.index, buyDecPerf.values, marker='o', color='tab:blue')
            axBuy.set_xlabel('Buy Score Decile')
            axBuy.set_ylabel('Mean Forward Alpha')
            axBuy.set_title('Forward Alpha by Buy-Score Decile')
            axBuy.grid(alpha=0.3)
        except ValueError:
            axBuy.text(0.5, 0.5, 'Insufficient data for Buy deciles', ha='center', va='center', transform=axBuy.transAxes)
            axBuy.set_axis_off()
    
        # ---- (1,2) Sell Score deciles vs Forward Alpha ----
        if 'Sell Score' in dfDiag.columns: scoreColSell = 'Sell Score'
        else: scoreColSell = 'Sell Prob'
    
        try:
            rankedSell = dfDiag[scoreColSell].rank(method='first')
            dfDiag['sellDecile'] = pd.qcut(rankedSell, 10, labels=False) + 1
            sellDecPerf = dfDiag.groupby('sellDecile')['Forward Alpha'].mean().sort_index()
            axSell.plot(sellDecPerf.index, sellDecPerf.values, marker='o', color='tab:red')
            axSell.set_xlabel('Sell Score Decile')
            axSell.set_ylabel('Mean Forward Alpha')
            axSell.set_title('Forward Alpha by Sell-Score Decile')
            axSell.grid(alpha=0.3)
        except ValueError:
            axSell.text(0.5, 0.5, 'Insufficient data for Sell deciles', ha='center', va='center', transform=axSell.transAxes)
            axSell.set_axis_off()
    
        # ---- (2,1) Precision–Recall curves for Buy & Sell ----
        try:
            yTrueBuy = (dfDiag['Target State'] == 1).astype(int)
            yScoreBuy = dfDiag['Buy Prob'].fillna(0.0).values
            yTrueSell = (dfDiag['Target State'] == -1).astype(int)
            yScoreSell = dfDiag['Sell Prob'].fillna(0.0).values
            precBuy, recBuy, _ = precision_recall_curve(yTrueBuy, yScoreBuy)
            precSell, recSell, _ = precision_recall_curve(yTrueSell, yScoreSell)
    
            axPr.plot(recBuy, precBuy, label='Buy', color='tab:blue')
            axPr.plot(recSell, precSell, label='Sell', color='tab:red')
            axPr.set_xlabel('Recall')
            axPr.set_ylabel('Precision')
            axPr.set_title('Precision–Recall (Buy & Sell)')
            axPr.grid(alpha=0.3)
            axPr.legend()
        except Exception:
            axPr.text(0.5, 0.5, 'Unable to compute PR curves', ha='center', va='center', transform=axPr.transAxes)
            axPr.set_axis_off()
    
        # ---- (2,2) Confusion matrix for last ~30 days ----
        try:
            if len(dfDiag) > 0:
                maxDate = dfDiag['dateDt'].max()
                startDate = maxDate - pd.Timedelta(days=30)
                recentDf = dfDiag[dfDiag['dateDt'] >= startDate].copy()
    
                if len(recentDf) > 0:
                    # Predicted label: 1 = Buy, -1 = Sell, 0 = Hold
                    predLabel = np.zeros(len(recentDf), dtype=int)
                    buyMask = (recentDf.get('Predicted Buy', 0) == 1)
                    sellMask = (recentDf.get('Predicted Sell', 0) == -1)
                    predLabel[buyMask] = 1
                    predLabel[~buyMask & sellMask] = -1
                    trueLabel = recentDf['Target State'].astype(int)
    
                    # Confusion matrix over {up, neutral, down}
                    cm = confusion_matrix(trueLabel, predLabel, labels=[1, 0, -1])
                    cmDisp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Up (Pred)', 'Neutral (Pred)', 'Down (Pred)'])
                    cmDisp.plot(ax=axCm, cmap='Blues', colorbar=False)
                    axCm.set_title('Confusion Matrix (Last ~30 Days)')
                else:
                    axCm.text(0.5, 0.5, 'No recent data', ha='center', va='center', transform=axCm.transAxes)
                    axCm.set_axis_off()
            else:
                axCm.text(0.5, 0.5, 'No data', ha='center', va='center', transform=axCm.transAxes)
                axCm.set_axis_off()
        except Exception:
            axCm.text(0.5, 0.5, 'Unable to compute confusion matrix', ha='center', va='center', transform=axCm.transAxes)
            axCm.set_axis_off()
    
        fig.tight_layout()
        plt.show()
    else:
        print("[Warning] Missing columns for 2×2 diagnostics; skipping score/PR/confusion plots.")

    # ---------------------------------------------------
    # Monotonic calibration of Buy Score vs Forward Alpha
    # ---------------------------------------------------
    if {'Buy Score', 'Forward Alpha'}.issubset(df.columns):
        dfEvalIso = df.reset_index()
        dfEvalIso = dfEvalIso.dropna(subset=['Buy Score', 'Forward Alpha']).copy()

        if not dfEvalIso.empty:
            iso = IsotonicRegression(out_of_bounds='clip')
            xFit = dfEvalIso['Buy Score'].values
            yFit = dfEvalIso['Forward Alpha'].values
            iso.fit(xFit, yFit)

            # Calibrated expected alpha given Buy Score
            df['Buy Score Calibrated'] = iso.predict(df['Buy Score'].fillna(0.0).values)
        else: print("[Warning] No data for isotonic calibration; skipping Buy Score calibration.")
    else: print("[Warning] Missing Buy Score / Forward Alpha; skipping isotonic calibration.")

    # -------------------------------------------
    # Yield curve vs Z threshold (informational)
    # -------------------------------------------
    # Restrict to assets that pass the Mahalanobis distance cutoff
    survAssets = df[df['MDist'] <= chiCutoff]
    # Determine the max absolute Z among survivors to set a reasonable Z grid
    maxZ = survAssets['Z Score'].abs().max() if not survAssets.empty else 1.75
    zGrid = np.linspace(1.0, max(1.75, maxZ), 10)

    sensitivityResults = []
    for z in zGrid:
        # Assets that are both within MDist cutoff and have |Z| above the threshold z
        maskExtreme = (df['MDist'] <= chiCutoff) & (df['Z Score'].abs() >= z)
        validAssets = int(maskExtreme.sum())
        sensitivityResults.append({'Z-Score Min': z, 'Valid Assets': validAssets})

    # Build a small sensitivity table indexed by the Z-score threshold
    sensDf = pd.DataFrame(sensitivityResults).set_index('Z-Score Min')
    if not sensDf.empty and sensDf['Valid Assets'].max() > 0:
        # Plot how many names survive as we tighten the |Z|-threshold
        plt.figure(figsize=(8, 4))
        plt.plot(sensDf.index, sensDf['Valid Assets'], marker='o', color='teal', linewidth=2, markersize=8)
        plt.fill_between(sensDf.index, sensDf['Valid Assets'], color='teal', alpha=0.2)
        plt.title(f"Portfolio Yield Curve vs Strictness (Selected Z: {bestZ:.2f})")
        plt.xlabel("Absolute Z-Score Minimum Threshold")
        plt.ylabel("Assets with |Z| >= threshold (MDist <= cutoff)")
        plt.grid(True, linestyle=':', alpha=0.6)
        # Annotate each point with the asset count for quick visual inspection
        for z, val in zip(sensDf.index, sensDf['Valid Assets']):
            plt.annotate(f"{val}", (z, val), textcoords="offset points", xytext=(0, 10), ha='center', fontweight='bold')
        plt.tight_layout()
        plt.show()
        
    # ===========================================
    # Risk/Reward Yield Matrix (Z Score vs MDist)
    # ===========================================
    plotDf = df[df['Validation'] != 'Neutral/Noise'].copy()  # focus on buys, sells, outliers
    plotDf = plotDf[plotDf['MDist'] > 0]                     # drop zero/invalid Mahalanobis distances

    if not plotDf.empty:
        mDistFloor = 1e-2                                    # lower bound to avoid log(0) issues
        mDistCap = np.percentile(plotDf['MDist'], 99)        # cap MDist at 99th percentile to limit extreme values
        plotDf['MDistClamp'] = plotDf['MDist'].clip(lower=mDistFloor, upper=mDistCap)

        # JointGrid to show MDist vs Z-Score with a marginal distribution on top
        rrym = sns.JointGrid(data=plotDf, x='Z Score', y='MDistClamp', height=8, ratio=4)

        # Top marginal: KDE of Z-score distribution
        sns.kdeplot( data=plotDf, x='Z Score', fill=True, ax=rrym.ax_marg_x, color='gray', alpha=0.3 )
        rrym.ax_marg_y.set_visible(False)                    # hide right-side marginal (unused)

        # Split into rejected outliers vs validated trade signals
        outliers = plotDf[plotDf['Validation'] == 'Rejected Outlier']
        validSignals = plotDf[plotDf['Validation'].isin(['Validated Buy', 'Validated Sell'])]

        # Plot outliers as faint black X markers
        if not outliers.empty:
            rrym.ax_joint.scatter(outliers['Z Score'], outliers['MDistClamp'], color='black', alpha=0.2, s=20, marker='x', label='Rejected' )

        # Plot validated buys/sells as larger, colored markers
        if not validSignals.empty:
            buyValid = validSignals[validSignals['Validation'] == 'Validated Buy']
            sellValid = validSignals[validSignals['Validation'] == 'Validated Sell']

            if not buyValid.empty: rrym.ax_joint.scatter(buyValid['Z Score'], buyValid['MDistClamp'], color='green',
                    alpha=0.9, s=120, marker='o', edgecolor='white', linewidth=0.8, label='Validated Buy')
            if not sellValid.empty: rrym.ax_joint.scatter(sellValid['Z Score'], sellValid['MDistClamp'], color='red',
                    alpha=0.9, s=120, marker='x', linewidth=1.5, label='Validated Sell')

        # Use log scale on MDist to emphasize relative distance while compressing extremes
        rrym.ax_joint.set_yscale('log')
        ymin = max(mDistFloor, plotDf['MDistClamp'].min())
        ymax = mDistCap
        rrym.ax_joint.set_ylim(ymin, ymax)

        # Tuned thresholds: vertical Z cutoffs and horizontal MDist cutoff
        rrym.ax_joint.axvline(-bestZ, color='green', linestyle='--', alpha=0.6)
        rrym.ax_joint.axvline(bestZ, color='red', linestyle='--', alpha=0.6)
        rrym.ax_joint.axhline(chiCutoff, color='black', linestyle='--', alpha=0.4)
        rrym.ax_joint.set_xlabel('Z Score')
        rrym.ax_joint.set_ylabel('MDist (clamped, log scale)')
        rrym.ax_joint.legend()
        rrym.fig.suptitle("Risk/Reward Yield Matrix (Z Score vs MDist)", fontsize=14, fontweight='bold', y=0.98)
        rrym.fig.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

    # ========================================
    # Final Portfolio Mapped to Alpha Manifold
    # ========================================
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Build a color map per alpha cluster for the background manifold
    alphaClusterIds = sorted(df['Alpha ClusterID'].dropna().unique())
    clusterPalette = sns.color_palette("tab10", n_colors=len(alphaClusterIds))
    clusterColorMap = {cid: clusterPalette[i] for i, cid in enumerate(alphaClusterIds)}

    # Plot the full alpha manifold as semi-transparent colored points by cluster
    for cid in alphaClusterIds:
        clusterMask = (df['Alpha ClusterID'] == cid)
        ax.scatter(df.loc[clusterMask, 'Alpha DimOne'], df.loc[clusterMask, 'Alpha DimTwo'], df.loc[clusterMask, 'Alpha DimThree'],
            c=[clusterColorMap[cid]], s=8, alpha=0.2, edgecolor='none')

    # Overlay validation categories: neutral, outliers, and validated signals
    noiseDf = df[df['Validation'] == 'Neutral/Noise']
    outliers = df[df['Validation'] == 'Rejected Outlier']
    validSignals = df[df['Validation'].isin(['Validated Buy', 'Validated Sell'])]

    # Neutral/noise points shown as faint gray background
    if not noiseDf.empty: ax.scatter(noiseDf['Alpha DimOne'], noiseDf['Alpha DimTwo'], noiseDf['Alpha DimThree'],
            c='gray', s=5, alpha=0.08, label='Neutral/Noise')
    # Rejected outliers highlighted as black X markers
    if not outliers.empty: ax.scatter( outliers['Alpha DimOne'], outliers['Alpha DimTwo'], outliers['Alpha DimThree'],
            c='black', marker='x', s=30, alpha=0.7, label='Rejected Outlier')

    # Validated buys/sells emphasized with larger colored markers
    if not validSignals.empty:
        buyValid = validSignals[validSignals['Validation'] == 'Validated Buy']
        sellValid = validSignals[validSignals['Validation'] == 'Validated Sell']
        if not buyValid.empty: ax.scatter(buyValid['Alpha DimOne'], buyValid['Alpha DimTwo'], buyValid['Alpha DimThree'],
                c='green', marker='o', s=120, edgecolor='white', linewidth=0.8, alpha=0.95, label='Validated Buy')
        if not sellValid.empty: ax.scatter(sellValid['Alpha DimOne'], sellValid['Alpha DimTwo'], sellValid['Alpha DimThree'],
                c='red', marker='x', s=120, linewidth=1.5, alpha=0.95, label='Validated Sell')

    ax.set_title("Final Evaluated Portfolio Mapped to Alpha Prediction Manifold", fontweight='bold', pad=20)
    ax.set_xlabel('Dim One')
    ax.set_ylabel('Dim Two')
    ax.set_zlabel('Dim Three')

    # Deduplicate legend entries by label
    handles, labels = ax.get_legend_handles_labels()
    legendMap = dict(zip(labels, handles))
    ax.legend(legendMap.values(), legendMap.keys(), bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.tight_layout()
    plt.show()

    # =================================================
    # Portfolio Mapped to Peer Manifold + Sector Ratios
    # =================================================
    fig = plt.figure(figsize=(23, 12))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1.0])

    ax3d = fig.add_subplot(gs[0, 0], projection='3d')
    axBar = fig.add_subplot(gs[0, 1])

    # Build sector palette and mapping if Sector column is available
    if 'Sector' in df.columns: sectors = sorted(df['Sector'].dropna().unique())
    else: sectors = []
    sectorPalette = sns.color_palette("tab10", n_colors=len(sectors)) if sectors else []
    sectorColorMap = {s: sectorPalette[i] for i, s in enumerate(sectors)}

    # Plot neutral names in peer manifold, colored by sector, as a faint background
    neutralMask = (df['Validation'] == 'Neutral/Noise')
    for sectorName in sectors:
        sectorNeutralMask = neutralMask & (df['Sector'] == sectorName)
        sectorNeutral = df[sectorNeutralMask]
        if sectorNeutral.empty: continue
        ax3d.scatter(sectorNeutral['Peer DimOne'], sectorNeutral['Peer DimTwo'], sectorNeutral['Peer DimThree'],
            c=[sectorColorMap[sectorName]], s=6, alpha=0.1, edgecolor='none', zorder=1)

    # Overlay rejected outliers in peer manifold
    outlierMask = (df['Validation'] == 'Rejected Outlier')
    outliersPeer = df[outlierMask]
    if not outliersPeer.empty: ax3d.scatter(outliersPeer['Peer DimOne'], outliersPeer['Peer DimTwo'], outliersPeer['Peer DimThree'],
            c='black', marker='x', s=15, alpha=0.3, linewidths=0.8, zorder=2, label='Rejected Outlier')

    # Overlay validated buys/sells by sector with prominent markers
    validMask = df['Validation'].isin(['Validated Buy', 'Validated Sell'])
    for sectorName in sectors:
        sectorValidMask = validMask & (df['Sector'] == sectorName)
        sectorValid = df[sectorValidMask]
        if sectorValid.empty: continue

        sectorBuy = sectorValid[sectorValid['Validation'] == 'Validated Buy']
        if not sectorBuy.empty: ax3d.scatter(sectorBuy['Peer DimOne'], sectorBuy['Peer DimTwo'], sectorBuy['Peer DimThree'],
                c=[sectorColorMap[sectorName]], marker='o', s=120, edgecolor='white', linewidth=0.8, alpha=0.95, zorder=3, depthshade=False)

        sectorSell = sectorValid[sectorValid['Validation'] == 'Validated Sell']
        if not sectorSell.empty: ax3d.scatter(sectorSell['Peer DimOne'], sectorSell['Peer DimTwo'], sectorSell['Peer DimThree'],
                c=[sectorColorMap[sectorName]], marker='x', s=120, edgecolor="white", linewidth=0.8, alpha=0.95, zorder=3, depthshade=False)

    ax3d.set_xlabel('Peer Dim One')
    ax3d.set_ylabel('Peer Dim Two')
    ax3d.set_zlabel('Peer Dim Three')
    ax3d.set_title("Portfolio Mapped to Peer (Sector) Manifold", fontweight='bold', pad=15)

    # Deduplicate legend entries for 3D peer plot
    handles3d, labels3d = ax3d.get_legend_handles_labels()
    legendMap3d = dict(zip(labels3d, handles3d))
    ax3d.legend( legendMap3d.values(), legendMap3d.keys(), bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')

    # Right-hand bar chart: validation mix by sector (buy/sell/outlier/neutral ratios)
    if 'Sector' in df.columns:
        sectorValidCounts = df.groupby('Sector')['Validation'].value_counts().unstack().fillna(0)
        # Ensure all categories exist as columns
        for cat in ['Validated Buy', 'Validated Sell', 'Rejected Outlier', 'Neutral/Noise']:
            if cat not in sectorValidCounts.columns: sectorValidCounts[cat] = 0
        # Convert counts to ratios within each sector
        sectorRatios = sectorValidCounts.div(sectorValidCounts.sum(axis=1), axis=0)
        barData = sectorRatios[['Validated Buy', 'Validated Sell', 'Rejected Outlier', 'Neutral/Noise']]
        barColors = ['green', 'red', 'black', 'gray']
        barData.plot(kind='bar', stacked=True, ax=axBar, color=barColors, edgecolor='white')
        axBar.set_ylabel("Ratio")
        axBar.set_xlabel("Sector")
        axBar.set_title("Validation Ratios by Sector")
        plt.setp(axBar.get_xticklabels(), rotation=45, ha='right')

        # Color x-tick labels by sector color and apply a stroke for readability
        for tick in axBar.get_xticklabels():
            sectorName = tick.get_text()
            if sectorName in sectorColorMap: tick.set_color(sectorColorMap[sectorName])
            else: tick.set_color('black')
            tick.set_fontweight('bold')
            tick.set_path_effects([patheffects.withStroke(linewidth=0.8, foreground='black'), patheffects.Normal()])
        axBar.legend(loc='upper right', fontsize='small')

    plt.tight_layout()
    plt.show()

    # -------------------------------------------
    # Decile performance: Buy Prob vs Forward Alpha
    # -------------------------------------------
    dfEval = df.reset_index()  # work on a flattened copy with Date/Ticker as columns
    if {'Date', 'Buy Prob', 'Forward Alpha'}.issubset(dfEval.columns):
        # Keep only rows with both a buy probability and forward alpha defined
        dfEval = dfEval.dropna(subset=['Buy Prob', 'Forward Alpha']).copy()
        # Parse Date into a true datetime index for grouping
        dfEval['dateDt'] = pd.to_datetime(dfEval['Date'], format='%Y%m%d%H%M%S', errors='coerce')
        dfEval = dfEval.dropna(subset=['dateDt'])

        scoreCol = 'Buy Score Calibrated' if 'Buy Score Calibrated' in dfEval.columns else 'Buy Score'

        # Global decile assignment across all dates based on the score ranking
        try:        # Use rank to break ties so qcut can create 10 bins
            ranked = dfEval[scoreCol].rank(method='first')
            dfEval['scoreDecile'] = pd.qcut(ranked, 10, labels=False) + 1
        except ValueError: dfEval['scoreDecile'] = np.nan

        dfEval = dfEval[dfEval['scoreDecile'].notna()]

    # --------------------------------------------
    # Rolling performance: Long-Short vs Benchmark
    # --------------------------------------------
    dfPerf = df.reset_index()  # flattened copy for time-based grouping

    if {'Date', 'Forward Alpha', 'Validation'}.issubset(dfPerf.columns):
        # Require forward alpha to compute returns; drop rows without it
        dfPerf = dfPerf.dropna(subset=['Forward Alpha']).copy()

        if not dfPerf.empty:
            # Convert packed Date string into datetime for time-series analysis
            dfPerf['dateDt'] = pd.to_datetime(dfPerf['Date'], format='%Y%m%d%H%M%S', errors='coerce')
            dfPerf = dfPerf.dropna(subset=['dateDt'])
            grouped = dfPerf.groupby('dateDt')

            # Long-only return: average forward alpha among validated buys per date
            longRet = grouped.apply(lambda g: g.loc[g['Validation'] == 'Validated Buy', 'Forward Alpha'].mean()).fillna(0.0)

            # Short-only return: average forward alpha among validated sells per date
            shortRet = grouped.apply(lambda g: g.loc[g['Validation'] == 'Validated Sell', 'Forward Alpha'].mean()).fillna(0.0)

            # Long-short spread (long minus short) per date
            lsRet = (longRet - shortRet).fillna(0.0)
            # Benchmark: cross-sectional mean forward alpha over all assets per date
            benchRet = grouped['Forward Alpha'].mean().fillna(0.0)

            # Align all series to the same date index and fill missing dates with 0
            lsRet = lsRet.reindex(benchRet.index).fillna(0.0)
            longRet = longRet.reindex(benchRet.index).fillna(0.0)
            shortRet = shortRet.reindex(benchRet.index).fillna(0.0)

            # Compute cumulative sums to show rolling performance over time
            cumLs = lsRet.cumsum()
            cumBench = benchRet.cumsum()
            cumLong = longRet.cumsum()
            cumShort = shortRet.cumsum()
            excessCum = cumLs - cumBench  # excess alpha vs benchmark

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(cumLs.index, cumLs.values, label='Long-Short Strategy', color='tab:blue')
            ax.plot(cumBench.index, cumBench.values, label='Benchmark (All Assets)', color='tab:orange', linestyle='--')
            ax.plot(excessCum.index, excessCum.values, label='Excess Alpha (Strategy - Benchmark)', color='tab:green', linestyle=':')
            ax.plot(cumLong.index, cumLong.values, label='Long-Only (Validated Buys)', color='tab:purple', alpha=0.6)
            ax.plot(cumShort.index, cumShort.values, label='Short-Only (Validated Sells)', color='tab:red', alpha=0.6, linestyle='-.')
            ax.set_xlabel('Date')
            ax.set_ylabel('Cumulative Forward Alpha')
            ax.set_title('Rolling Long-Short vs Benchmark Performance')
            ax.grid(alpha=0.3)
            ax.legend()
            if 'Financial Stress Index' in dfPerf.columns:
                stressSeries = grouped['Financial Stress Index'].mean()
                stressCutoff = stressSeries.quantile(0.8)  # top 20% stress as "high"
                highMask = stressSeries >= stressCutoff
                for i in range(len(stressSeries) - 1):
                    if highMask.iloc[i]: ax.axvspan(stressSeries.index[i], stressSeries.index[i + 1], color='lightgray', alpha=0.2)
            fig.tight_layout()
            plt.show()
        else: print("[Warning] No rows for rolling performance plot.")
    else: print("[Warning] Missing Date / Forward Alpha / Validation; skipping rolling performance plot.")
    return df

def computeMdZ(df, valuationCols):
    """
    Compute/refresh per-name valuation Z Score and Mahalanobis distance (MDist)
    over peer clusters.
    """
    # Initialize / clear Z and MDist
    df['Z Score'] = np.nan
    df['MDist'] = np.nan

    # Primary valuation column used for directional Z-score
    valCol = valuationCols[0]
    # Direction mapping: +1 means "high is expensive", -1 means "high is cheap"
    directionMap = {'EV/EBITDA': 1, 'Debt-to-Equity': 1,
        'Current Ratio': -1, 'Free Cash Flow': -1, 'ROE': -1, 'ROA': -1}
    direction = directionMap.get(valCol, 1)

    # Loop over peer clusters
    for clusterId in df['Peer ClusterID'].unique():
        # Skip noise/unassigned peer clusters
        if clusterId == -1: continue

        clusterMask = (df['Peer ClusterID'] == clusterId)
        clusterData = df[clusterMask].copy()
        if len(clusterData) < len(valuationCols): continue

        # --- Z Score (point-in-time, log-space, peer-relative) ---
        if 'Date' in clusterData.index.names: dates = clusterData.index.get_level_values('Date')
        else: dates = clusterData['Date']

        posMask = clusterData[valCol] > 0
        if posMask.sum() > 1:
            logVals = np.log(clusterData.loc[posMask, valCol])

            # Expanding mean/std over time within the peer cluster
            valMean = logVals.expanding().mean()
            valStd = logVals.expanding().std()

            # Floor std to avoid crazy Z-scores when dispersion is tiny
            valStd = valStd.clip(lower=0.05)
            zLocal = (logVals - valMean) / valStd
            df.loc[clusterMask & posMask, 'Z Score'] = (zLocal * direction).values
        else: df.loc[clusterMask, 'Z Score'] = 0.0

        # --- MDist (peer-relative multivariate distance) ---
        sectorZCols = [f"{col} Sector Z" for col in valuationCols if f"{col} Sector Z" in df.columns]
        if len(sectorZCols) >= 2: useCols = sectorZCols
        else: useCols = valuationCols

        # Chronological ordering for expanding covariance estimation
        if 'Date' in clusterData.index.names:
            clusterData = clusterData.sort_index(level='Date')
            dates = clusterData.index.get_level_values('Date')
        else:
            clusterData = clusterData.sort_values('Date')
            dates = clusterData['Date']

        metricsAll = clusterData[useCols].copy()
        metricsAll = metricsAll.fillna(metricsAll.median()).fillna(0)

        # Require enough observations relative to dimensionality
        requiredSamples = len(useCols) * 4

        for dt in np.unique(dates):
            histMask = (dates <= dt)
            histData = metricsAll[histMask]
            if len(histData) < requiredSamples: continue

            currentIdx = clusterData[dates == dt].index
            try:
                colMeans = histData.mean()
                colStds = histData.std().replace(0, 1)
                histStd = (histData - colMeans) / colStds
                currentStd = (metricsAll.loc[currentIdx] - colMeans) / colStds

                # Robust covariance estimate; fall back to empirical if needed
                try: covModel = MinCovDet(support_fraction=0.9).fit(histStd)
                except Exception: covModel = EmpiricalCovariance().fit(histStd)
                df.loc[currentIdx, 'MDist'] = covModel.mahalanobis(currentStd)
            except Exception: pass
    return df

def optimizeTradingParams(df, nTrials=tradeTrials):
    """
    Use Optuna to tune trading hyperparameters
    """
    def objective(trial):
        valChiThres = trial.suggest_float('valChiThres', 0.7, 0.95)
        highScoreThresh = trial.suggest_float('highScoreThresh', 0.65, 0.85)
        zRelaxFactor = trial.suggest_float('zRelaxFactor', 0.7, 1.0)
        mdistSoftMult = trial.suggest_float('mdistSoftMult', 1.0, 2.0)
        minNames = trial.suggest_int('minNames', 15, 30)
        minBuys = trial.suggest_int('minBuys', 3, 8)
        minSells = trial.suggest_int('minSells', 3, 8)
    
        dfEval, metrics = applyTradingRules(
            df,
            valChiThres=valChiThres,
            highScoreThresh=highScoreThresh,
            zRelaxFactor=zRelaxFactor,
            mdistSoftMult=mdistSoftMult,
            minNames=minNames,
            minBuys=minBuys,
            minSells=minSells,
        )
        
        decileSlope = metrics['decileSlope']
        cumLsAlpha = metrics['cumLsAlpha']
        cumBenchAlpha = metrics.get('cumBenchAlpha', 0.0)
        
        if np.isnan(decileSlope): decileSlope = 0.0
        # Only reward positive monotonicity and reward excess alpha
        decileSlope = max(decileSlope, 0.0)
        excessAlpha = cumLsAlpha - cumBenchAlpha
        if cumLsAlpha <= 0.0: return -1e6
        
        # Combined objective: favor monotonic deciles and strong excess alpha
        score = (decileSlope ** 2) + 0.2 * excessAlpha
        return score

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=randSeed))
    study.optimize(objective, n_trials=nTrials, show_progress_bar=False)

    print("[Info] Best trading params:", study.best_params)
    print("[Info] Best objective score:", study.best_value)

    # Apply best params once to get final Validation labels
    bestParams = study.best_params
    dfBest, metricsBest = applyTradingRules(df,
        valChiThres=bestParams['valChiThres'],
        highScoreThresh=bestParams['highScoreThresh'],
        zRelaxFactor=bestParams['zRelaxFactor'],
        mdistSoftMult=bestParams['mdistSoftMult'],
        minNames=bestParams['minNames'],
        minBuys=bestParams['minBuys'],
        minSells=bestParams['minSells'],
    )

    print("[Info] Best decileSlope:", metricsBest['decileSlope'])
    print("[Info] Best cumLsAlpha:", metricsBest['cumLsAlpha'])

    return dfBest, study.best_params, metricsBest

def applyTradingRules(df, valChiThres, highScoreThresh, zRelaxFactor, mdistSoftMult, minNames, minBuys, minSells):
    """
    Apply trading/validation rules given a set of hyperparameters.
    """
    dfCopy = df.copy()
    dof = 3  # len(valuationCols); 
    chiCutoff = stats.chi2.ppf(valChiThres, dof)

    # Hard vs soft MDist masks
    hardMask = (dfCopy['MDist'] <= chiCutoff)
    softMask = (dfCopy['MDist'] <= chiCutoff * mdistSoftMult)

    # Build Z grid and choose bestZ on this config
    survAssets = dfCopy[hardMask]
    maxZ = survAssets['Z Score'].abs().max() if not survAssets.empty else 1.75
    zGrid = np.linspace(1.0, max(1.75, maxZ), 10)
    bestScore = -np.inf
    bestZ = 1.0

    for z in zGrid:
        baseMask = softMask
        highBuy = dfCopy['Buy Prob'] >= highScoreThresh
        highSell = dfCopy['Sell Prob'] >= highScoreThresh
        buyMask = baseMask & (dfCopy['Predicted Buy'] == 1) & ((dfCopy['Z Score'] <= -z) | (highBuy & (dfCopy['Z Score'] <= -z * zRelaxFactor)))
        sellMask = baseMask & (dfCopy['Predicted Sell'] == -1) & ((dfCopy['Z Score'] >= z) | (highSell & (dfCopy['Z Score'] >= z * zRelaxFactor)))
        numBuys = int(buyMask.sum())
        numSells = int(sellMask.sum())
        validAssets = numBuys + numSells

        if validAssets < minNames or numBuys < minBuys or numSells < minSells: continue

        avgBuyAlpha = dfCopy.loc[buyMask, 'Forward Alpha'].mean()
        avgSellAlpha = dfCopy.loc[sellMask, 'Forward Alpha'].mean()
        avgLsAlpha = avgBuyAlpha - avgSellAlpha

        if avgLsAlpha > bestScore:
            bestScore = avgLsAlpha
            bestZ = z

    # If nothing passed thresholds, mark everything neutral and return bad metrics
    if bestScore == -np.inf:
        dfCopy['Validation'] = 'Neutral/Noise'
        metrics = {'decileSlope': 0.0, 'cumLsAlpha': 0.0}
        return dfCopy, metrics

    # Final masks with chosen bestZ
    highBuy = dfCopy['Buy Prob'] >= highScoreThresh
    highSell = dfCopy['Sell Prob'] >= highScoreThresh
    buyMaskFinal = softMask & (dfCopy['Predicted Buy'] == 1) & ((dfCopy['Z Score'] <= -bestZ) | (highBuy & (dfCopy['Z Score'] <= -bestZ * zRelaxFactor)))
    sellMaskFinal = softMask & (dfCopy['Predicted Sell'] == -1) & ((dfCopy['Z Score'] >= bestZ) | (highSell & (dfCopy['Z Score'] >= bestZ * zRelaxFactor)))
    dfCopy['Validation'] = 'Neutral/Noise'
    dfCopy.loc[buyMaskFinal, 'Validation'] = 'Validated Buy'
    dfCopy.loc[sellMaskFinal, 'Validation'] = 'Validated Sell'
    dfCopy.loc[(~buyMaskFinal & ~sellMaskFinal & ~hardMask), 'Validation'] = 'Rejected Outlier'

    # --- Decile Slope (score vs forward alpha) ---
    dfEval = dfCopy.reset_index().copy()
    dfEval = dfEval.dropna(subset=['Forward Alpha']).copy()
    
    if not dfEval.empty:
        # Prefer calibrated Buy Score if available, then raw Buy Score, then fall back to Buy Prob
        if 'Buy Score Calibrated' in dfEval.columns: scoreCol = 'Buy Score Calibrated'
        elif 'Buy Score' in dfEval.columns: scoreCol = 'Buy Score'
        else: scoreCol = 'Buy Prob'
    
        try:
            # Global deciles based on score ranking across the whole sample
            ranked = dfEval[scoreCol].rank(method='first')
            dfEval['scoreDecile'] = pd.qcut(ranked, 10, labels=False) + 1
        except ValueError: dfEval['scoreDecile'] = np.nan
    
        dfEval = dfEval[dfEval['scoreDecile'].notna()]
        if not dfEval.empty:
            decilePerf = dfEval.groupby('scoreDecile')['Forward Alpha'].mean().sort_index()
            x = decilePerf.index.values.astype(float)
            y = decilePerf.values
            decileSlope = np.corrcoef(x, y)[0, 1] if len(x) > 1 else 0.0
        else: decileSlope = 0.0
    else: decileSlope = 0.0

    # --- Long-short cumulative alpha and benchmark cumulative alpha ---
    dfPerf = dfCopy.reset_index().copy()
    dfPerf = dfPerf.dropna(subset=['Forward Alpha']).copy()
    
    if not dfPerf.empty:
        dfPerf['dateDt'] = pd.to_datetime(dfPerf['Date'], format='%Y%m%d%H%M%S', errors='coerce')
        dfPerf = dfPerf.dropna(subset=['dateDt'])
        grouped = dfPerf.groupby('dateDt')
    
        # Long-only and short-only series
        longRet = grouped.apply(lambda g: g.loc[g['Validation'] == 'Validated Buy', 'Forward Alpha'].mean()).fillna(0.0)
        shortRet = grouped.apply(lambda g: g.loc[g['Validation'] == 'Validated Sell', 'Forward Alpha'].mean()).fillna(0.0)
        lsRet = (longRet - shortRet).fillna(0.0)
        cumLsAlpha = lsRet.cumsum().iloc[-1] if len(lsRet) > 0 else 0.0
    
        # Equal-weight long-only benchmark (all assets)
        benchRet = grouped['Forward Alpha'].mean().fillna(0.0)
        cumBenchAlpha = benchRet.cumsum().iloc[-1] if len(benchRet) > 0 else 0.0
    else:
        cumLsAlpha = 0.0
        cumBenchAlpha = 0.0
    
    metrics = {
        'decileSlope': decileSlope,
        'cumLsAlpha': float(cumLsAlpha),
        'cumBenchAlpha': float(cumBenchAlpha),
        'bestZ': float(bestZ),
        'chiCutoff': float(chiCutoff),
    }
    return dfCopy, metrics

# ==========================================
# MASTER EXECUTION PIPELINE
# ==========================================
def execute_master_screener(rawDf, featureCols, valuationCols, priceCol='Adj Close', sectorCol='Sector'):
    dfOne, optimalFeatures = variable_selection(rawDf, featureCols, priceCol=priceCol, forwardWindow=20, sectorCol=sectorCol)
    dfTwo = manifold_clustering(dfOne, optimalFeatures, sectorCol)
    dfThree, localModels = multi_classification(dfTwo, optimalFeatures)

    # Compute MDist and Z Score once, for use by trading rules and charts
    dfThree = computeMdZ(dfThree, valuationCols)

    # Tune trading hyperparameters and get tuned Validation
    dfOptimized, bestParams, bestMetrics = optimizeTradingParams(dfThree, nTrials=tradeTrials)

    # Produce charts using tuned thresholds and Validation
    manifold_validation(dfOptimized, valuationCols, bestZ=bestMetrics['bestZ'], chiCutoff=bestMetrics['chiCutoff'])

    # Final portfolio from tuned Validation
    finalPortfolio = dfOptimized[dfOptimized['Validation'].isin(['Validated Buy', 'Validated Sell'])].copy()

    return finalPortfolio, localModels


# ==========================================
# RUNTIME TRIGGER
# ==========================================
if __name__ == "__main__":
    # Load Data (force Date as string to preserve exact YYYYMMDDHHMMSS format)
    df = pd.read_csv('market_data.csv', dtype={'Date': str})
    df.set_index(['Date', 'Ticker'], inplace=True)
    
    # Compute sector-relative Z-scores for key fundamental metrics
    fundamentals = ['Operating Margin', 'Gross Margin', 'ROE', 'ROA', 'Debt-to-Equity', 'Current Ratio', 'Free Cash Flow', 'EV/EBITDA']
    for col in fundamentals:
        if col in df.columns:
            # Global (cross-sectional) mean/std per Date
            globalMean = df.groupby('Date')[col].transform('mean')
            globalStd = df.groupby('Date')[col].transform('std').replace(0, 1)

            # Sector-level counts and means per (Date, Sector)
            sectorCount = df.groupby(['Date', 'Sector'])[col].transform('count')
            sectorMean = df.groupby(['Date', 'Sector'])[col].transform('mean')

            # Bayesian-smoothed sector mean (shrunk toward global mean)
            bayesMean = ((sectorCount * sectorMean) + (priorWeight * globalMean)) / (sectorCount + priorWeight)

            # Sector Z: deviation from shrunk sector mean, scaled by global dispersion
            df[f"{col} Sector Z"] = (df[col] - bayesMean) / globalStd

    # Define the Feature Space (macro + momentum + sector-normalized fundamentals)
    features = [
        'Real 10Y Yield', 'Yield Curve Spread', 'Month Momentum', 'Quarter Momentum', 
        'CPI YoY Inflation', 'Federal Funds Rate Delta', 'Unemployment Rate Delta', 'Relative Volume',
        'Financial Stress Index Delta', 'Volume', 'Operating Margin Sector Z', 'Gross Margin Sector Z', 
        'ROE Sector Z', 'ROA Sector Z', 'Debt-to-Equity Sector Z', 'Current Ratio Sector Z', 
        'Free Cash Flow Sector Z', 'EV/EBITDA Sector Z', 'High Yield Credit Spread', 'Month Volatility',
        "Credit Stress Exposure", "10Y Yield Beta", "FCF Risk Premium",
    ]
    
    # Define the Validation Space (valuation metrics used for Z/MDist filters)
    valuationMetrics = ['EV/EBITDA', 'Free Cash Flow', 'Debt-to-Equity']
    
    # Execute the full screening pipeline
    portfolioDf, trainedModels = execute_master_screener(rawDf=df, featureCols=features, valuationCols=valuationMetrics, priceCol='Adj Close', sectorCol='Sector')

    # Extract current market payload for D3 dashboard
    if not portfolioDf.empty:
        portfolioDf = portfolioDf.reset_index()

        # Isolate only the most recent cross-section to avoid plotting historical ghosts
        latestDate = portfolioDf['Date'].max()
        currentPortfolio = portfolioDf[portfolioDf['Date'] == latestDate].copy()
        print(f"Pipeline Complete. Validated {len(currentPortfolio)} active assets for {latestDate}.")

        # Define the exact columns the D3 JS dashboard requires
        keepCols = ['Date', 'Ticker', 'Sector', 'MDist', 'Z Score', 'Validation',
            'Alpha DimOne', 'Alpha DimTwo', 'Alpha DimThree', 'Alpha ClusterID',
            'Peer DimOne', 'Peer DimTwo', 'Peer DimThree', 'Peer ClusterID',
            'Predicted Buy', 'Predicted Sell', 'Buy Prob', 'Sell Prob'] + valuationMetrics

        # Export flat JSON for the frontend (record-wise, ISO date format)
        exportPayload = currentPortfolio[keepCols]
        exportPayload.to_json('portfolio_dashboard_payload.json', orient='records', date_format='iso')
        print("Exported interactive payload to 'portfolio_dashboard_payload.json'")
    else: print("Pipeline finished, but 0 assets passed the strict Mahalanobis and Z-Score thresholds.")
        
    # -----------------------------------------
    # Daily recommendation scoreboard (console)
    # -----------------------------------------

    currentPortfolio['signal'] = np.where(currentPortfolio['Validation'] == 'Validated Buy', 'BUY',
        np.where(currentPortfolio['Validation'] == 'Validated Sell', 'SELL', 'HOLD'))

    # Sort order: Buys first, then Sells, then Holds
    sortOrder = {'BUY': 0, 'SELL': 1, 'HOLD': 2}
    currentPortfolio['signalOrder'] = currentPortfolio['signal'].map(sortOrder)

    # Build list of columns to show, but only keep those that actually exist
    colsToShow = ['Ticker', 'Sector', 'signal', 'Buy Prob', 'Buy Prob Calibrated', 'Z Score', 'MDistClamp', 'MDist']
    colsToShow = [c for c in colsToShow if c in currentPortfolio.columns]

    # Sort keys: primary by signalOrder (BUY/SELL/HOLD), secondary by calibrated or raw Buy Prob
    sortByCols = ['signalOrder']
    if 'Buy Prob Calibrated' in currentPortfolio.columns: sortByCols.append('Buy Prob Calibrated')
    else: sortByCols.append('Buy Prob')

    print("\n=== Daily Recommended Actions ===")
    print(currentPortfolio.sort_values(sortByCols, ascending=[True, False])[colsToShow].to_string(index=False))

    # Clean up helper column to avoid polluting downstream data
    currentPortfolio.drop(columns=['signalOrder'], inplace=True)
