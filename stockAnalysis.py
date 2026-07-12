import os
import warnings
os.environ['PYTHONWARNINGS'] = 'ignore'
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import umap
import xgboost as xgb
import shap
import optuna
import re

from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN, KMeans
from sklearn.covariance import MinCovDet, EmpiricalCovariance
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import partial_dependence

randSeed = 37
corrThres = 0.85
varMIThres = 0.01
trustThres = 0.75
valChiThres = 0.7
priorWeight = 5

def plot_hierarchical_clustermap(df, featureCols, sectorCol):
    """Generates a hierarchical clustermap based on Industry/Sector median values."""
    if sectorCol not in df.columns: return
    # Safely extract dates depending on if it is an index or a column
    dateGroup = df.index.get_level_values('Date') if 'Date' in df.index.names else df['Date']
    # If a feature only ever has 1 unique value per date, it is a global/macro variable
    uniques = df[featureCols].groupby(dateGroup).nunique().max()
    # Filter to only keep features that vary across tickers on the same day
    companyFeatures = uniques[uniques > 1].index.tolist()

    if not companyFeatures: return
    # Group using only the company-specific features
    grouped = df.groupby(sectorCol)[companyFeatures].median().dropna(axis=1, how='all')
    if grouped.empty or len(grouped) < 2: return
    scaled = StandardScaler().fit_transform(grouped)
    scaledDf = pd.DataFrame(scaled, index=grouped.index, columns=grouped.columns)

    try:
        cg = sns.clustermap(scaledDf.T, cmap='coolwarm', metric='euclidean', method='ward', figsize=(12, 10))
        cg.fig.suptitle("Hierarchical Clustermap of Features by Sector", fontsize=16, fontweight='bold', y=1.02)
        plt.setp(cg.ax_heatmap.get_xticklabels(), rotation=45, ha='right')
        plt.show()
    except Exception as e: print(f"[Warning] Hierarchical Clustermap failed: {e}")

def expanding_preprocess(X, scalerType='standard'):
    """Applies expanding window imputation and scaling to prevent time-series data leakage."""
    # 1. Expanding Median Imputation (Requires chronological data)
    groupObj = X.groupby(level='Ticker') if 'Ticker' in X.index.names else X.groupby('Ticker')
    
    imputed = X.fillna(groupObj.transform(lambda x: x.expanding().median().shift(1).bfill().fillna(x.median())))
    imputed = imputed.fillna(X.median()).fillna(0)

    if scalerType is None: return imputed

    # 2. Expanding Scaling
    if scalerType == 'standard':
        mean = groupObj.transform(lambda x: x.expanding().mean().shift(1).bfill().fillna(x.mean()))
        std = groupObj.transform(lambda x: x.expanding().std().shift(1).bfill().fillna(x.std()).replace(0, 1))
        return ((imputed - mean) / std).fillna(0)
    elif scalerType == 'minmax':
        minVal = groupObj.transform(lambda x: x.expanding().min().shift(1).bfill().fillna(x.min()))
        maxVal = groupObj.transform(lambda x: x.expanding().max().shift(1).bfill().fillna(x.max()))
        rangeVal = (maxVal - minVal).replace(0, 1)
        return ((imputed - minVal) / rangeVal).fillna(0)
    else:
        median = groupObj.transform(lambda x: x.expanding().median().shift(1).bfill().fillna(x.median()))
        q75 = groupObj.transform(lambda x: x.expanding().quantile(0.75).shift(1).bfill().fillna(x.quantile(0.75)))
        q25 = groupObj.transform(lambda x: x.expanding().quantile(0.25).shift(1).bfill().fillna(x.quantile(0.25)))
        return ((imputed - median) / (q75 - q25).replace(0, 1)).fillna(0)

# ===============================================
# PHASE 1: TARGET DEFINITION & VARIABLE SELECTION
# ===============================================
def variable_selection(df, featureCols, priceCol='Adj Close', forwardWindow=20, valuationCol='EV/EBITDA', sectorCol='Sector'):
    dfEval = df.copy()
    
    # Enforce Time-Series Chronological Order for CV logic later
    if 'Date' in dfEval.index.names:
        dfEval = dfEval.sort_index(level=['Ticker','Date'])
    elif 'Date' in dfEval.columns:
        dfEval = dfEval.sort_values(['Ticker', 'Date'])
        
    if 'Ticker' in dfEval.index.names:
        dfEval['Forward Alpha'] = np.log(dfEval.groupby(level='Ticker')[priceCol].shift(-forwardWindow) / dfEval[priceCol])
    else:
        dfEval['Forward Alpha'] = np.log(dfEval.groupby('Ticker')[priceCol].shift(-forwardWindow) / dfEval[priceCol])

    trainMask = dfEval['Forward Alpha'].notna()
    dates = dfEval.loc[trainMask].index.get_level_values('Date') if 'Date' in dfEval.index.names else dfEval.loc[trainMask, 'Date']
    alphaMean = dfEval.loc[trainMask].groupby(dates)['Forward Alpha'].transform('mean')
    alphaStd = dfEval.loc[trainMask].groupby(dates)['Forward Alpha'].transform('std').replace(0,1).fillna(1)
    alphaZ = (dfEval.loc[trainMask, 'Forward Alpha'] - alphaMean) / alphaStd
    dfEval['Target State'] = 0
    conditions = [(alphaZ < -1), (alphaZ > 1)]
    dfEval.loc[trainMask, 'Target State'] = np.select(conditions, [-1, 1], default=0)
    dfEval['UMAP Target'] = -1
    dfEval.loc[trainMask, 'UMAP Target'] = np.select(conditions, [0, 1], default=-1)

    XTrain = expanding_preprocess(dfEval.loc[trainMask, featureCols], scalerType=None)
    yTrinary = dfEval.loc[trainMask, 'Target State'].values.astype(int)
    
    # Finding MI Classification values against the Target State 
    miScores = mutual_info_classif(XTrain, yTrinary, random_state=randSeed)
    miSeries = pd.Series(miScores, index=featureCols).sort_values(ascending=False)
    
    # Find Correlated variables through Spearman Correlation
    corrMatrix = XTrain.corr(method='spearman').abs()
    dropped = set()
    for i in range(len(corrMatrix.columns)):
        for j in range(i + 1, len(corrMatrix.columns)):
            colA, colB = corrMatrix.columns[i], corrMatrix.columns[j]
            # If Correlation between variables over threshold, then drop based on MI score
            if corrMatrix.iloc[i, j] > corrThres:
                dropped.add(colB if miSeries[colA] > miSeries[colB] else colA)
    
    # Drop the selected correlated variables, as well as those below the MI relevance threshold                
    selectedFeatures = [f for f in featureCols if f not in dropped and miSeries[f] > varMIThres]
    
    if len(selectedFeatures) < 2:
        selectedFeatures = miSeries[~miSeries.index.isin(dropped)].head(5).index.tolist()
    
    # VISUALIZATION: Information Gain Audit
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    
    # Bar Plot fo How much MI each included variable contributes
    topFeatures = miSeries[selectedFeatures].sort_values(ascending=False).head(10)
    sns.barplot(x=topFeatures.values, y=topFeatures.index, palette='viridis', ax=axes[0])
    axes[0].set_title("Information Gain (Surviving Features)")
    axes[0].set_xlabel("Mutual Information Score")
    
    # Heat Map showing which variables were selected or dropped
    mask = np.triu(np.ones_like(corrMatrix.loc[selectedFeatures, selectedFeatures], dtype=bool))
    sns.heatmap(corrMatrix.loc[selectedFeatures, selectedFeatures], mask=mask, annot=True, cmap='RdBu_r', fmt=".2f", vmin=-1, vmax=1, ax=axes[1])
    axes[1].set_title("Post-Pruning Spearman Correlation")
    plt.setp(axes[1].get_xticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    
    # VISUALIZATION: Generate Sector/Feature Hierarchical Clustermap
    plot_hierarchical_clustermap(dfEval, selectedFeatures, sectorCol)
    
    return dfEval, selectedFeatures

# ==============================================
# PHASE 2: DIMENSIONALITY REDUCTION & CLUSTERING
# ==============================================
def get_multi_trust(XScaled, emb, maxSamples=10000):
    """Evaluates manifold topology preservation across multiple neighborhood scales."""
    if len(XScaled) > maxSamples:
        np.random.seed(randSeed)
        idx = np.random.choice(len(XScaled), maxSamples, replace=False)
        xEval, embEval = XScaled[idx], emb[idx]
    else: xEval, embEval = XScaled, emb
    neighbors = [10, 15, 20, 50, 100]
    validNeighbors = [n for n in neighbors if n < len(xEval)]
    if not validNeighbors: return 0.0
    return np.mean([trustworthiness(xEval, embEval, n_neighbors=n) for n in validNeighbors])

def manifold_clustering(df, features, sectorCol):
    XRaw = df[features].copy()
    ySector = LabelEncoder().fit_transform(df[sectorCol].astype(str)) if sectorCol in df.columns else None
    yAlpha = df['UMAP Target'].values.astype(int)
    # Select the number of Components of the manifold
    nComp = 3
    def optimize_and_build(targetName, yTarget):
    # Dynamic Manifold Hyperparamater Selection
        def objective(trial):
            # Select a Scaler and perform a fit to the median
            scalerChoice = trial.suggest_categorical('scaler', ['standard', 'minmax', 'robust'])
            try:
                XScaled = expanding_preprocess(XRaw, scalerType=scalerChoice).values
                # Select a Dimensionality Reduction Algorithm
                dimAlgo = trial.suggest_categorical('dimAlgo', ['UMAP']) # PCA', 't-SNE'
                # Set Supervision mode
                supervision = trial.suggest_categorical('supervision', ['Unsupervised', 'Targeted'])
                
                # Linear Primary Component Analysis
                if dimAlgo == 'PCA':
                    emb = PCA(n_components=nComp, random_state=randSeed).fit_transform(XScaled)
                # t-Distributed Stochastic Neighbor with different perplexity neighbor parameters
                elif dimAlgo == 't-SNE':
                    perp = trial.suggest_int('perplexity', 10, min(50, len(XScaled) - 1))
                    emb = TSNE(n_components=nComp, perplexity=perp, init='pca', random_state=randSeed).fit_transform(XScaled)
                # Uniform Manifold Approximation and Projection, checking different comginations of neighbors
                else:
                    numNeighbors = trial.suggest_int('numNeighbors', 30, min(200, len(XScaled) - 1))
                    minDist = trial.suggest_float('minDist', 0.0, 0.8)
                    uMetric = trial.suggest_categorical('uMetric', ['euclidean', 'cosine', 'correlation', 'braycurtis'])
                    targetArr = yTarget if supervision == 'Targeted' and yTarget is not None else None
                    tWeight = trial.suggest_float('tWeight', 0.1, 0.9) if yTarget is not None else 0.5
                    emb = umap.UMAP(n_components=nComp, n_neighbors=numNeighbors, min_dist=minDist, metric=uMetric, target_weight=tWeight, random_state=randSeed).fit_transform(XScaled, y=targetArr)
                      
                # Multi-Scale Trust Metric Evaluation
                trust = get_multi_trust(XScaled, emb)
                if trust < trustThres: return -1.0 
                
                # Evaluating clustering across both KMeans and DBSCAN
                clusterAlgo = trial.suggest_categorical('clusterAlgo', ['KMeans']) # , 'DBSCAN'
                # Evaluate different clustering sizes of k for KMeans
                if clusterAlgo == 'KMeans':
                    labels = KMeans(n_clusters=trial.suggest_int('k', 3, 15), random_state=randSeed, n_init='auto').fit_predict(emb)
                # Evaluate different minimum sample sizes and epsilon search radius
                else:
                    labels = DBSCAN(eps=trial.suggest_float('eps', 0.1, 1.5), min_samples=trial.suggest_int('minSamples', 3, 25)).fit_predict(StandardScaler().fit_transform(emb))
                
                # Filtering out combinations which are below the trustworthiness threshold
                validMask = labels != -1
                numClusters = len(set(labels[validMask]))
                
                if numClusters < 2: return -1.0
                
                try: baseScore = silhouette_score(XScaled[validMask], labels[validMask], sample_size=15000, random_state=randSeed)
                except ValueError: return -1.0
    
                noiseRatio = 1.0 - (validMask.sum() / len(labels))
                fragPenalty = 0.05 * max(0, numClusters - 10)
                
                clusterCounts = pd.Series(labels[validMask]).value_counts()
                usableSamples = clusterCounts[clusterCounts >= 100].sum()
                viabilityRatio = usableSamples / len(labels)
                viabilityPenalty = (1.0 - viabilityRatio) * 0.5
                
                clusterProbs = clusterCounts / clusterCounts.sum()
                entropy = -np.sum(clusterProbs * np.log(clusterProbs)) / np.log(numClusters)
                return (baseScore * (1.0 - noiseRatio)) - fragPenalty - viabilityPenalty + (0.1 * entropy)
            except ValueError: raise optuna.TrialPruned()   
            except Exception as e: 
                print(f"Trial failed with unexpected error: {e}")
                raise optuna.TrialPruned()
    
        numStudies = 25
        # Trigger Optuna Hyperparameter tuning for Manifold clustering
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=randSeed))
        study.optimize(objective, n_trials=numStudies, show_progress_bar=False)
        
        # Rebuild Optimal State
        best = study.best_params
        XScaled = expanding_preprocess(XRaw, scalerType=best['scaler']).values
        
        if best['dimAlgo'] == 'PCA':
            #emb = PCA(n_components=2, random_state=randSeed).fit_transform(XScaled)
            emb = PCA(n_components=nComp, random_state=randSeed).fit_transform(XScaled)
        elif best['dimAlgo'] == 't-SNE':
            #emb = TSNE(n_components=2, perplexity=best['perplexity'], init='pca', random_state=randSeed).fit_transform(XScaled)
            emb = TSNE(n_components=nComp, perplexity=best['perplexity'], init='pca', random_state=randSeed).fit_transform(XScaled)
        else:
            targetArr = yTarget if best['supervision'] == 'Targeted' and yTarget is not None else None
            tWeight = best.get('tWeight', 0.5)
            #emb = umap.UMAP(n_components=2, n_neighbors=best['numNeighbors'], min_dist=0.1, random_state=randSeed).fit_transform(XScaled, y=y_target)
            emb = umap.UMAP(n_components=nComp, n_neighbors=best['numNeighbors'], min_dist=best['minDist'], metric=best['uMetric'], target_weight=tWeight, random_state=randSeed).fit_transform(XScaled, y=targetArr)

        if best['clusterAlgo'] == 'KMeans':
            labels = KMeans(n_clusters=best['k'], random_state=randSeed, n_init=10).fit_predict(emb)
        else:
            labels = DBSCAN(eps=best['eps'], min_samples=best['minSamples']).fit_predict(StandardScaler().fit_transform(emb))
        labels = np.where(labels != -1, labels + 1, -1)
        df[f'{targetName} DimOne'], df[f'{targetName} DimTwo'], df[f'{targetName} DimThree'], df[f'{targetName} ClusterID'] = emb[:, 0], emb[:, 1], emb[:,2], labels
      
        # VISUALIZATION: Manifold Optuna Convergence tracking
        trialsDf = study.trials_dataframe()
        plt.figure(figsize=(10, 4))
        plt.plot(trialsDf['number'] + 1, trialsDf['value'].cummax(), marker='o', linestyle='--', color='b', alpha=0.5, label='Best Score')
        plt.plot(trialsDf['number'] + 1, trialsDf['value'], marker='o', linestyle='', color='r', alpha=0.7, label='Trial Score')
        plt.locator_params(axis='x', integer=True)
        plt.title(f"{targetName} Optuna Convergence: Best Silhouette Score ({study.best_value:.3f})")
        plt.xlabel("Trial Number")
        plt.ylabel("Penalized Silhouette Score")
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.show()

        def get_directional_mi(xMatrix, umapDim, featureNames):
            miScores = mutual_info_regression(xMatrix, umapDim, random_state=randSeed)
            directionalLoadings = []
            for i in range(xMatrix.shape[1]):
                corr, _ = spearmanr(xMatrix[:, i], umapDim)
                sign = np.sign(corr) if corr != 0 else 1
                directionalLoadings.append(miScores[i] * sign)
            return pd.Series(directionalLoadings, index=featureNames)
                
        # VISUALIZATION: Embedding Feature Loadings (Mutual Information)
        miDimOne = get_directional_mi(XScaled, emb[:, 0], features)
        miDimTwo = get_directional_mi(XScaled, emb[:, 1], features)
        miDimThree = get_directional_mi(XScaled, emb[:, 2], features)
        loadingsDf = pd.DataFrame({'DimOne': miDimOne, 'DimTwo': miDimTwo, 'DimThree': miDimThree}, index=features).sort_values('DimOne', ascending=False)
        loadingsDf = loadingsDf.iloc[loadingsDf['DimOne'].abs().argsort()[::-1]]
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(loadingsDf.head(10), annot=True, cmap='viridis', fmt=".3f")
        plt.title(f"Top 10 Feature Loadings to {targetName} Optimal Manifold ({best['uMetric']} {best['dimAlgo']} - {best['supervision']})")
        plt.tight_layout()
        plt.show()
        
        if sectorCol in df.columns:
            validClusters = df[df[f'{targetName} ClusterID'] != -1]
            validClusters['Cluster Name'] = "Cluster" + validClusters[f'{targetName} ClusterID'].astype(str)
            compTable = pd.crosstab(validClusters['Cluster Name'], validClusters[sectorCol], normalize='index') * 100
            compTable.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='Set3', edgecolor='black')
            plt.title("Sector Composition Map (Manifold Output)")
            plt.xlabel("")
            plt.ylabel("Percentage (%)")
            plt.legend(title="Sector", bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.xticks(rotation=0)
            plt.tight_layout()
            plt.show()
    
    optimize_and_build('Alpha', yAlpha)
    optimize_and_build('Peer', ySector)
    
    return df


# ==========================================
# PHASE 3: MASTER PROOF & LOCAL EXPERTS
# ==========================================

def calculate_decay_weights(dates, yLocal=None, hlDays=180):
    """Calculates exponential decay weights combined with class balancing."""
    # Parse the 14-digit packed timestamp (YYYYMMDDHHMMSS)
    parsedDates = pd.to_datetime(dates.astype(str), format='%Y%m%d%H%M%S')
    latestDate = parsedDates.max()
    deltaDays = (latestDate - parsedDates) / pd.Timedelta(days=1)
    
    if isinstance(yLocal, (int, float)): hlDays, yLocal = yLocal, None

    # Calculate exponential decay (lambda derived from half-life)
    alphaDecay = np.log(2) / hlDays
    timeWeights = np.exp(-alphaDecay * deltaDays.to_numpy())
    
    if yLocal is not None:
        yArray = np.asarray(yLocal)
        posMask = (yArray == 1)
        nPos = max(1, posMask.sum())
        nNeg = len(yArray) - nPos
        bayesRatio = (nNeg + priorWeight) / (nPos + priorWeight)
        combinedWeights = timeWeights.copy()
        combinedWeights[posMask] *= bayesRatio
        return combinedWeights
    
    return timeWeights

def optimize_cluster_model(X, y, datesSeries, isLinear):
    """Sub-routine to tune Local Expert hyperparameters per cluster."""
    def objective(trial):
        hlDays = trial.suggest_int('hlDays', 30, 365)
        timeWeights = calculate_decay_weights(datesSeries, hlDays)
        cv = TimeSeriesSplit(n_splits=3)
        try: 
            if isLinear:
                cValue = trial.suggest_float('cValue', 1e-3, 1e2, log=True)
                penaltyType = trial.suggest_categorical('penaltyType', ['l1', 'l2'])
                model = LogisticRegression(C=cValue, penalty=penaltyType, solver='saga', max_iter=1000, random_state=randSeed, class_weight='balanced')
                score = cross_val_score(model, X, y, cv=cv, scoring='matthews_corrcoef', error_score=0, params={'sample_weight': timeWeights}).mean()
            else:
                maxD = min(8, max(3, len(X) // 15))
                maxEst = min(150, max(30, len(X) * 2))
                maxDepth = trial.suggest_int('maxDepth', 3, maxD)
                learningRate = trial.suggest_float('learningRate', 1e-3, 0.1, log=True)
                nEstimators = trial.suggest_int('nEstimators', 30, maxEst)
                gammaVal = trial.suggest_float('gamma', 1e-3, 5.0, log=True)
                alphaVal = trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True)
                lambdaVal = trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True)
                colSample = trial.suggest_float('colsample_bytree', 0.4, 0.9)
                subSample = trial.suggest_float('subsample', 0.5, 0.9)
                minChildWeight = trial.suggest_int('min_child_weight', 1, 7)
    
                posMask = (y == 1)
                numPos = max(1, posMask.sum())
                negWeight = (len(y) - posMask.sum()) / numPos
    
                combinedWeights = timeWeights.copy()
                combinedWeights[posMask] *= negWeight
    
                model = xgb.XGBClassifier(max_depth=maxDepth, learning_rate=learningRate, n_estimators=nEstimators, gamma=gammaVal, reg_alpha=alphaVal, 
                                          reg_lambda=lambdaVal, colsample_bytree=colSample, subsample=subSample, min_child_weight=minChildWeight, random_state=randSeed)
                score = cross_val_score(model, X, y, cv=cv, scoring='matthews_corrcoef', error_score=0, params={'sample_weight': combinedWeights}).mean()
            return 0.0 if np.isnan(score) else score
        except ValueError: return 0.0

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=randSeed))
    study.optimize(objective, n_trials=500, show_progress_bar=False)
    return study.best_params

def generate_oos_probabilities(estimator, Xin, yin, cv, sampleWeight=None):
    """Generates out-of-sample probabilities for expanding TimeSeriesSplit."""
    X = np.asarray(Xin)
    y = np.asarray(yin)
    sw = np.asarray(sampleWeight) if sampleWeight is not None else None
    oosProbs = np.full(len(y), 0.5)
    splits = list(cv.split(X))

    def get_pos_probs(model, xEval):
        probs = model.predict_proba(xEval)
        if probs.shape[1] == 1: return np.ones(len(xEval)) if model.classes_[0] == 1 else np.zeros(len(xEval))
        classIdx = np.where(model.classes_ == 1)[0]
        return probs[:, classIdx[0]] if len(classIdx) > 0 else np.zeros(len(xEval))

    # In-sample predictions for the initial chronological training fold 
    firstTrain = splits[0][0]
    if len(np.unique(y[firstTrain])) > 1:
        firstModel = clone(estimator)
        if sampleWeight is not None: firstModel.fit(X[firstTrain], y[firstTrain], **{'sample_weight': sw[firstTrain]})
        else: firstModel.fit(X[firstTrain], y[firstTrain])
        oosProbs[firstTrain] = get_pos_probs(firstModel, X[firstTrain])
    # Strict Out-Of-Sample predictions for the forward-walking timeline
    for trainIdx, testIdx in splits:
        if len(np.unique(y[trainIdx])) > 1:
            foldModel = clone(estimator)
            if sampleWeight is not None: foldModel.fit(X[trainIdx], y[trainIdx], **{'sample_weight': sw[trainIdx]})
            else: foldModel.fit(X[trainIdx], y[trainIdx])
            oosProbs[testIdx] = get_pos_probs(foldModel, X[testIdx])
    return oosProbs

def multi_classification(df, features): 
    clusterResults = []
    localExperts = {}
    shapVisuals = []
    
    # Adding Manifold "Loadings" as classification variables
    modelFeatures = features + ['Alpha DimOne', 'Alpha DimTwo', 'Alpha DimThree']
    # Setup separate probability/prediction columns
    df['Buy Prob'] = 0.5
    df['Sell Prob'] = 0.5
    df['Predicted Buy'] = 0
    df['Predicted Sell'] = 0

    for clusterId in df['Alpha ClusterID'].unique():
        if clusterId == -1: continue

        mask = df['Alpha ClusterID'] == clusterId
        if mask.sum() < 100: continue
        localDf = df[mask].copy()

        # Enforce strict chronology before TimeSeriesSplit
        if 'Date' in localDf.columns: localDf = localDf.sort_values('Date')
        else: localDf = localDf.sort_index(level='Date')

        XLocal = expanding_preprocess(localDf[modelFeatures], scalerType=None)
        XScaled = expanding_preprocess(XLocal, scalerType='standard').values
        localDates = localDf.index.get_level_values('Date') if 'Date' in localDf.index.names else localDf['Date']
        trainMaskLocal = localDf['Forward Alpha'].notna()
        xLocalTrain = XLocal[trainMaskLocal]
        xScaledTrain = XScaled[trainMaskLocal]
        localDatesTrain = localDates[trainMaskLocal]

        # Train two separate classifiers (1 for Upside, -1 for Downside)
        for targetVal, probCol, predCol in [(1, 'Buy Prob', 'Predicted Buy'), (-1, 'Sell Prob', 'Predicted Sell')]:
            yLocalTrain = (localDf.loc[trainMaskLocal, 'Target State'] == targetVal).astype(int)

            if yLocalTrain.nunique() < 2 or yLocalTrain.sum() < 5: continue
            cv = TimeSeriesSplit(n_splits=3)

            # Tune Hyperparameters with nested Optuna
            linParams = optimize_cluster_model(xScaledTrain, yLocalTrain, localDatesTrain, isLinear=True)
            xgbParams = optimize_cluster_model(xLocalTrain, yLocalTrain, localDatesTrain, isLinear=False)

            # Pop the half-life out of the dictionary to rebuild optimal weights
            #linWeights = calculate_decay_weights(localDatesTrain, hlDays=180)
            linWeights = calculate_decay_weights(localDatesTrain, linParams.pop('hlDays'))
            #xgbBaseWeights = calculate_decay_weights(localDatesTrain, yLocal=yLocalTrain, hlDays=180)
            xgbBaseWeights = calculate_decay_weights(localDatesTrain, xgbParams.pop('hlDays'))
    
            # Re-apply imbalance penalty to optimal XGBoost weights
            posMask = (yLocalTrain == 1)
            xgbRatio = (len(yLocalTrain) - posMask.sum()) / (max(1, posMask.sum()))
            xgbCombinedWeights = xgbBaseWeights.copy()
            xgbCombinedWeights[posMask] *= xgbRatio
    
            linModel = LogisticRegression(
                C=linParams['cValue'], 
                penalty=linParams['penaltyType'],
                solver='saga', 
                max_iter=1000, 
                random_state=randSeed, 
                class_weight='balanced')
            xgbModel = xgb.XGBClassifier(
                max_depth=xgbParams['maxDepth'], 
                learning_rate=xgbParams['learningRate'], 
                n_estimators=xgbParams['nEstimators'],
                random_state=randSeed)
            
            linMcc = cross_val_score(linModel, xScaledTrain, yLocalTrain, cv=cv, scoring='matthews_corrcoef', error_score=0, params={'sample_weight': linWeights}).mean()
            xgbMcc = cross_val_score(xgbModel, xLocalTrain, yLocalTrain, cv=cv, scoring='matthews_corrcoef', error_score=0, params={'sample_weight': xgbCombinedWeights}).mean()
            
            dominance = "Linear" if linMcc > xgbMcc else "XGBoost"
            directionLabel = 'Buy' if targetVal == 1 else 'Sell'
            
            if dominance == "Linear":
                expert = linModel
                oosProbs = generate_oos_probabilities(expert, xScaledTrain, yLocalTrain, cv=cv, sampleWeight=linWeights)
                expert.fit(xScaledTrain, yLocalTrain, sample_weight=linWeights)
                fullProbs = expert.predict_proba(XScaled)[:, 1]
                fullProbs[trainMaskLocal.values] = oosProbs
                df.loc[mask, probCol] = fullProbs
                df.loc[mask, predCol] = expert.predict(XScaled) * targetVal
                explainer = shap.LinearExplainer(expert, xScaledTrain)
                shapValues = explainer.shap_values(xScaledTrain)
            else:
                # Extract tree solely for SHAP, use Calibrated output for signal
                expert = xgbModel
                oosProbs = generate_oos_probabilities(expert, xLocalTrain, yLocalTrain, cv=cv, sampleWeight=xgbCombinedWeights)
                expert.fit(xLocalTrain, yLocalTrain, sample_weight=xgbCombinedWeights)
                fullProbs = expert.predict_proba(XLocal)[:, 1]
                fullProbs[trainMaskLocal.values] = oosProbs
                df.loc[mask, probCol] = fullProbs
                df.loc[mask, predCol] = expert.predict(XLocal) * targetVal
                baseTree = xgbModel.fit(xLocalTrain, yLocalTrain, sample_weight=xgbCombinedWeights)
                explainer = shap.TreeExplainer(baseTree)
                shapValues = explainer.shap_values(xLocalTrain)
                
            if mask.sum() > 300:
                plt.figure(figsize=(8,5))
                plotShap = shapValues[1] if isinstance(shapValues, list) else shapValues
                shap.summary_plot(plotShap, xLocalTrain, feature_names=modelFeatures, show=False)
                plt.title(f"SHAP Directional Impact: Cluster {clusterId} ({directionLabel} Model)")
                plt.tight_layout()
                plt.show()
    
            localExperts[f"Cluster {clusterId} {directionLabel}"] = expert
            
            # Extract Top 5 absolute SHAP values for the Matrix Grid
            meanShap = np.abs(shapValues[1] if isinstance(shapValues, list) else shapValues).mean(axis=0)
            topIdx = np.argsort(meanShap)[-5:][::-1]
            topFeaturesList = [modelFeatures[i] for i in topIdx]
            clusterLabel = f"Cluster {clusterId})"
            
            clusterResults.append({
                'ClusterID': clusterId,
                'Direction': directionLabel,
                'Lasso MCC': linMcc,
                'XGB MCC': xgbMcc,
                'Dominance': dominance,
                'Cluster Label': clusterLabel
            })
            
            shapVisuals.append({
                'Model': f"Cluster {clusterId} ({directionLabel})",
                'Top 5': list(zip(topFeaturesList, [meanShap[i] for i in topIdx])),
                'Raw SHAP': meanShap,
                'Weight': len(XLocal)
            })
            
    # ==========================================
    # PHASE 3: VISUALIZATIONS
    # ==========================================
    
    # VISUALIZATION: Algorithm Dominance (Split Subplots & Sorted)
    if clusterResults:
        proofDf = pd.DataFrame(clusterResults)
        proofDf['ClusterID'] = pd.to_numeric(proofDf['ClusterID'])
        proofDf = proofDf.sort_values('ClusterID')
        
        fig, axes = plt.subplots(1, 2, figsize=(19, 7), sharey=True)
        
        for idx, direction in enumerate(['Buy', 'Sell']):
            subset = proofDf[proofDf['Direction'] == direction]
            ax = axes[idx]
            if not subset.empty:
                sns.scatterplot(data=subset, x='Lasso MCC', y='XGB MCC', hue='Cluster Label', 
                                palette='tab10', s=200, ax=ax, edgecolor='black', alpha=0.9)
                minVal = min(subset['Lasso MCC'].min(), subset['XGB MCC'].min(), 0) - 0.05
                maxVal = max(subset['Lasso MCC'].max(), subset['XGB MCC'].max(), 0) + 0.05
                ax.set_xlim(minVal, maxVal)
                ax.set_ylim(minVal, maxVal)
                ax.plot([minVal, maxVal], [minVal, maxVal], 'k--', alpha=0.5, label='Linear Boundary')
                ax.fill_between([minVal, maxVal], [minVal, maxVal], maxVal, color='blue', alpha=0.05)
                ax.set_title(f"{direction} Models: Linear vs. Non-Linear Dominance")
                ax.set_xlabel("Linear Baseline (Lasso MCC)")
                ax.set_ylabel("Non-Linear Signal (XGBoost MCC)")
                ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, fontsize=9, title="Cluster Profile")
                ax.grid(True, linestyle=':', alpha=0.5)
                
        plt.tight_layout()
        plt.show()
        
    # VISUALIZATION: Local Expert Feature Chart (Matrix Grid)
    if shapVisuals:
        matrixData, labelsData, models = [], [], []
        def sort_key(x):
            nums = re.findall(r'\d+', x['Model'])
            return int(nums[0]) if nums else 0
        
        for item in sorted(shapVisuals, key=sort_key):
            models.append(item['Model'])
            matrixData.append([v for f, v in item['Top 5']])
            labelsData.append([f"{f}\n({v:.3f})" for f, v in item['Top 5']])
            
        plt.figure(figsize=(12, max(4, len(models) * 0.85)))
        sns.heatmap(matrixData, annot=labelsData, fmt="", cmap='Blues', 
                    yticklabels=models, xticklabels=[f"Rank {i+1}" for i in range(5)],
                    cbar_kws={'label': 'Mean |SHAP| Impact'}, linewidths=1, linecolor='white')
        plt.title("Local Expert Top 5 Features per Cluster", pad=15, fontweight='bold')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
        
    # VISUALIZATION: 3D Partial Dependence Plot (Spatial Sensitivity)
    largestCluster = df[df['Alpha ClusterID'] != -1]['Alpha ClusterID'].value_counts().index[0]
    buyKey = f"Cluster_{largestCluster}_Buy"
    
    if buyKey in localExperts:
        expert = localExperts[buyKey]
        expertCore = expert.estimator if hasattr(expert, 'estimator') else expert
        
        localMask = df['Alpha ClusterID'] == largestCluster
        XPdp = expanding_preprocess(df[localMask][modelFeatures], scalerType=None)
        
        # Calculate 2-way PDP across the manifold components
        pdResults = partial_dependence(expertCore, XPdp, features=['Alpha DimOne', 'Alpha DimTwo'], grid_resolution=30)
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        XX, YY = np.meshgrid(pdResults['values'][0], pdResults['values'][1])
        Z = pdResults['average'][0].T 
        
        surf = ax.plot_surface(XX, YY, Z, cmap='magma', edgecolor='none', alpha=0.85)
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Partial Dependence (Buy Probability)')
        ax.set_xlabel('DimOne (Spatial)')
        ax.set_ylabel('DimTwo (Spatial)')
        ax.set_zlabel('Signal Probability')
        plt.title(f"3D Manifold Sensitivity (Cluster {largestCluster} - Buy Model)")
        plt.tight_layout()
        plt.show()
    
    # VISUALIZATION: Global Feature Importance (Sorted Top 10)
    if shapVisuals:
        globalShap = np.zeros(len(modelFeatures))
        totalSamples = 0
        for item in shapVisuals:
            globalShap += item['Raw SHAP'] * item['Weight']
            totalSamples += item['Weight']
            
        globalDf = pd.Series(globalShap / totalSamples, index=modelFeatures).sort_values(ascending=False).head(10)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x=globalDf.values, y=globalDf.index, palette='magma')
        plt.title("Global Feature Importance (Top 10 weighted across clusters)")
        plt.xlabel("Mean |SHAP| (Impact on Market Valuation)")
        plt.tight_layout()
        plt.show()        
    
    return df, localExperts

# ==========================================
# PHASE 4: SENSITIVITY GRID & VALIDATION
# ==========================================
def manifold_validation(df, valuationCols): 
    df['MDist'] = np.nan
    df['Z Score'] = np.nan
    df['Validation'] = 'Neutral/Noise'
    sensitivityResults = []
    
    dof = len(valuationCols)
    chiCutoff = stats.chi2.ppf(1 - valChiThres, dof) 

    valCol = valuationCols[0]
    directionMap = {'EV/EBITDA': 1, 'Debt-to-Equity': 1, 'Current Ratio': -1, 'Free Cash Flow': -1, 'ROE': -1, 'ROA': -1}
    direction = directionMap.get(valCol, 1)

    # 1. Point-In-Time Evaluation over Structural Peers
    for clusterId in df['Peer ClusterID'].unique():
        if clusterId == -1: continue
        
        mask = df['Peer ClusterID'] == clusterId
        clusterData = df[mask].copy()
        if len(clusterData) < len(valuationCols): continue
        
        dates = clusterData.index.get_level_values('Date') if 'Date' in clusterData.index.names else clusterData['Date']
        
        # Point-In-Time Valuation Z-Score (Vectorized)
        posMask = clusterData[valCol] > 0
        if posMask.sum() > 1:
            logVals = np.log(clusterData.loc[posMask, valCol]) * 100

            valMean = logVals.expanding().mean()
            valStd = logVals.expanding().std()
            valStd = valStd.where(valStd >= 1, 0.0001).fillna(0.0001)
            df.loc[mask & posMask, 'Z Score'] = ((logVals - valMean) / valStd) * direction
        else:
            df.loc[mask, 'Z Score'] = 0 
            
        # Point-In-Time Expanding Mahalanobis
        sectorZCols = [f"{col} Sector Z" for col in valuationCols if f"{col} Sector Z" in df.columns]
        useCols = sectorZCols if len(sectorZCols) == len(valuationCols) else valuationCols
        
        # Ensure chronological order for expanding window
        if 'Date' in clusterData.index.names:
            clusterData = clusterData.sort_index(level='Date')
            dates = clusterData.index.get_level_values('Date')
        else:
            clusterData = clusterData.sort_values('Date')
            dates = clusterData['Date']
            
        metricsAll = clusterData[useCols].fillna(clusterData[useCols].median()).fillna(0)
        requiredSamples = len(useCols) * 4 # Buffer for covariance stability
        
        for date in np.unique(dates):
            histMask = dates <= date
            histData = metricsAll[histMask]
            
            if len(histData) < requiredSamples: continue
            
            currentIdx = clusterData[dates == date].index
            try:
                # Fallback to Empirical if MinCovDet is too strictly bounded by support fraction
                try: covModel = MinCovDet(support_fraction=0.9).fit(histData)
                except: covModel = EmpiricalCovariance().fit(histData)
                
                df.loc[currentIdx, 'MDist'] = covModel.mahalanobis(metricsAll.loc[currentIdx])
            except Exception: pass

    # Dynamic Programmatic Grid Search
    survAssets = df[df['MDist'] <= chiCutoff]
    maxZ = survAssets['Z Score'].abs().max() if not survAssets.empty else 1.75
    zGrid = np.linspace(1.0, max(1.75, maxZ), 10)
    
    bestYield = -1
    bestZ = 1.0

    for z in zGrid:
        validBuys = ((df['MDist'] <= chiCutoff) & (df['Predicted Buy'] == 1) & (df['Z Score'] <= -z)).sum()
        validSells = ((df['MDist'] <= chiCutoff) & (df['Predicted Sell'] == -1) & (df['Z Score'] >= z)).sum()
        validAssets = validBuys + validSells
        
        sensitivityResults.append({'Z-Score Min': z, 'Valid Assets': validAssets})
        if validAssets > bestYield:
            bestYield = validAssets
            bestZ = z

    # Vectorized Portfolio Assignment
    buyMask = (df['MDist'] <= chiCutoff) & (df['Predicted Buy'] == 1) & (df['Z Score'] <= -bestZ)
    sellMask = (df['MDist'] <= chiCutoff) & (df['Predicted Sell'] == -1) & (df['Z Score'] >= bestZ)
    outlierMask = (df['MDist'] > chiCutoff)
    
    df.loc[outlierMask, 'Validation'] = 'Rejected Outlier'
    df.loc[buyMask, 'Validation'] = 'Validated Buy'
    df.loc[sellMask, 'Validation'] = 'Validated Sell'
    
    finalPortfolioIdx = df[buyMask | sellMask].index

    # VISUALIZATION: Yield Curve
    sensDf = pd.DataFrame(sensitivityResults).set_index('Z-Score Min')
    if not sensDf.empty and sensDf['Valid Assets'].max() > 0:
        plt.figure(figsize=(8, 4))
        plt.plot(sensDf.index, sensDf['Valid Assets'], marker='o', color='teal', linewidth=2, markersize=8)
        plt.fill_between(sensDf.index, sensDf['Valid Assets'], color='teal', alpha=0.2)
        plt.title(f"Portfolio Yield Curve vs Strictness (Selected Z: {bestZ})")
        plt.xlabel("Absolute Z-Score Minimum Threshold")
        plt.ylabel("Validated Assets")
        plt.grid(True, linestyle=':', alpha=0.6)
        for z, val in zip(sensDf.index, sensDf['Valid Assets']):
            plt.annotate(f"{val}", (z, val), textcoords="offset points", xytext=(0,10), ha='center', fontweight='bold')
        plt.tight_layout()
        plt.show()

    # VISUALIZATION: Risk/Reward Yield Matrix
    plotDf = df[df['Validation'] != 'Neutral/Noise']
    if not plotDf.empty:
        rrym = sns.JointGrid(data=plotDf, x='Z Score', y='MDist', height=8, ratio=4)
        sns.kdeplot(data=plotDf, x='Z Score', fill=True, ax=rrym.ax_marg_x, color='gray', alpha=0.3)
        sns.kdeplot(data=plotDf, x='MDist', fill=True, ax=rrym.ax_marg_y, color='gray', alpha=0.3)
        
        
        outliers = plotDf[plotDf['Validation'] == 'Rejected Outlier']
        valid = plotDf[plotDf['Validation'].isin(['Validated Buy', 'Validated Sell'])]
        
        if not outliers.empty:
            rrym.ax_joint.scatter(outliers['Z Score'], outliers['MDist'], color='black', alpha=0.2, s=20, marker='x', label='Rejected')
        if not valid.empty:
            sns.scatterplot(data=valid, x='Z Score', y='MDist', hue='Validation', style='Validation', 
                            s=120, alpha=0.9, edgecolor='white', palette={'Validated Buy': 'green', 'Validated Sell': 'red'}, ax=rrym.ax_joint)
        rrym.ax_joint.axvline(-bestZ, color='green', linestyle='--', alpha=0.6)
        rrym.ax_joint.axvline(bestZ, color='red', linestyle='--', alpha=0.6)
        rrym.ax_joint.axhline(chiCutoff, color='black', linestyle='--', alpha=0.4)
        rrym.ax_joint.legend()
        plt.show()

    # VISUALIZATION: Final 2D Portfolio Projection
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    noiseDf = df[df['Validation'] == 'Neutral/Noise']
    ax.scatter(noiseDf['Alpha DimOne'], noiseDf['Alpha DimTwo'], noiseDf['Alpha DimThree'], c='gray', s=5, alpha=0.1, label='Neutral Density')
    if not outliers.empty:
        ax.scatter(outliers['Alpha DimOne'], outliers['Alpha DimTwo'], outliers['Alpha DimThree'], c='black', marker='x', s=15, alpha=0.4, label='Rejected Outlier')
    if not valid.empty:
        markers = ['o', '*', '^', 's', 'D', 'v', 'p', 'h', 'X', 'd', '<', '>']
        sectors = valid['Sector'].unique()
        for i, sector in enumerate(sectors):
            marker = markers[i % len(markers)]
            sectorMask = valid['Sector'] == sector
            
            buyMask = sectorMask & (valid['Validation'] == 'Validated Buy')
            sellMask = sectorMask & (valid['Validation'] == 'Validated Sell')
            ax.scatter(valid.loc[buyMask, 'Alpha DimOne'], valid.loc[buyMask, 'Alpha DimTwo'], valid.loc[buyMask, 'Alpha DimThree'], c='green', marker=marker, s=120, edgecolor='white', alpha=0.9, label=f'Buy {sector}')
            ax.scatter(valid.loc[sellMask, 'Alpha DimOne'], valid.loc[sellMask, 'Alpha DimTwo'], valid.loc[sellMask, 'Alpha DimThree'], c='red', marker=marker, s=120, edgecolor='white', alpha=0.9, label=f'Sell {sector}')

    ax.set_title("Final Evaluated Portfolio Mapped to Alpha Prediction Manifold", fontweight='bold', pad=20)
    ax.set_xlabel('Dim One')
    ax.set_ylabel('Dim Two')
    ax.set_zlabel('Dim Three')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.tight_layout()
    plt.show()
    
    # Return cleanly utilizing the exact MultiIndex
    return df.loc[finalPortfolioIdx].copy()

# ==========================================
# MASTER EXECUTION PIPELINE
# ==========================================
def execute_master_screener(rawDf, featureCols, valuationCols, priceCol='Adj Close', sectorCol='Sector'):
    dfOne, optimalFeatures = variable_selection(rawDf, featureCols, priceCol=priceCol, forwardWindow=20, sectorCol=sectorCol)
    dfTwo = manifold_clustering(dfOne, optimalFeatures, sectorCol)
    dfThree, localModels = multi_classification(dfTwo, optimalFeatures)
    finalPortfolio = manifold_validation(dfThree, valuationCols)
    
    return finalPortfolio, localModels

# ==========================================
# RUNTIME TRIGGER
# ==========================================
if __name__ == "__main__":
    # 1. Load Data (Force Date as string to preserve the exact YYYYMMDDHHMMSS format)
    df = pd.read_csv('market_data.csv', dtype={'Date': str})
    df.set_index(['Date', 'Ticker'], inplace=True)
    
    fundamentals = ['Operating Margin', 'Gross Margin', 'ROE', 'ROA', 'Debt-to-Equity', 'Current Ratio',
                    'Free Cash Flow', 'EV/EBITDA']
    for col in fundamentals:
        if col in df.columns:
            globalMean = df.groupby('Date')[col].transform('mean')
            globalStd = df.groupby('Date')[col].transform('std').replace(0, 1)
            sectorCount = df.groupby(['Date', 'Sector'])[col].transform('count')
            sectorMean = df.groupby(['Date', 'Sector'])[col].transform('mean')
            bayesMean = ((sectorCount * sectorMean) + (priorWeight * globalMean)) / (sectorCount + priorWeight)
            df[f"{col} Sector Z"] = (df[col] - bayesMean) / globalStd

    # 2. Define the Feature Space
    features = [
        'Real 10Y Yield', 'Yield Curve Spread', 'Month Momentum', 'Quarter Momentum', 
        'CPI YoY Inflation', 'Federal Funds Rate Delta', 'Unemployment Rate Delta', 'Relative Volume',
        'Financial Stress Index Delta', 'Volume', 'Operating Margin Sector Z', 'Gross Margin Sector Z', 
        'ROE Sector Z', 'ROA Sector Z', 'Debt-to-Equity Sector Z', 'Current Ratio Sector Z', 
        'Free Cash Flow Sector Z', 'EV/EBITDA Sector Z', 'High Yield Credit Spread', 'Month Volatility',
        "Credit Stress Exposure", "10Y Yield Beta", "FCF Risk Premium"
        ]
    
    # 3. Define the Validation Space
    valuationMetrics = ['EV/EBITDA', 'Free Cash Flow', 'Debt-to-Equity']
    
    # 4. Execute the Pipeline
    portfolioDf, trainedModels = execute_master_screener(
        rawDf=df,
        featureCols=features,
        valuationCols=valuationMetrics,
        priceCol='Adj Close',
        sectorCol='Sector'
    )

    # 5. Extract Current Market Payload for D3 Dashboard
    if not portfolioDf.empty:
        portfolioDf = portfolioDf.reset_index()
        # Isolate only the most recent cross-section to avoid plotting ghost history
        latestDate = portfolioDf['Date'].max()
        currentPortfolio = portfolioDf[portfolioDf['Date'] == latestDate].copy()

        print(f"Pipeline Complete. Validated {len(currentPortfolio)} active assets for {latestDate}.")

        # Define the exact columns the D3 JS requires
        keepCols = [
            'Date', 'Ticker', 'Sector', 'Alpha DimOne', 'Alpha DimTwo', 'Alpha DimThree', 'Alpha ClusterID',
            'Peer DimOne', 'Peer DimTwo', 'Peer DimThree', 'Peer ClusterID', 'MDist', 'Z Score', 'Validation',
            'Predicted Buy', 'Predicted Sell', 'Buy Prob', 'Sell Prob'  
        ] + valuationMetrics

        # Export flat JSON for the frontend
        exportPayload = currentPortfolio[keepCols]
        exportPayload.to_json('portfolio_dashboard_payload.json', orient='records', date_format='iso')
        print("Exported interactive payload to 'portfolio_dashboard_payload.json'")
    else:
        print("Pipeline finished, but 0 assets passed the strict Mahalanobis and Z-Score thresholds.")

