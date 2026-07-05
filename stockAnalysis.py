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

from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN, KMeans
from sklearn.covariance import MinCovDet
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import partial_dependence

randSeed = 37
corrThres = 0.85
varMIThres = 0.0001 #0.01
trustThres = 0.3 #0.75
valChiThres = 0.5 #0.95

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
    expandingMed = X.expanding().median().shift(1).bfill().fillna(X.median())
    imputed = X.fillna(expandingMed)

    if scalerType is None: return imputed

    # 2. Expanding Scaling
    if scalerType == 'standard':
        mean = imputed.expanding().mean().shift(1).bfill().fillna(imputed.mean())
        std = imputed.expanding().std().shift(1).bfill().fillna(imputed.std()).replace(0, 1)
        return (imputed - mean) / std
    else:
        minVal = imputed.expanding().min().shift(1).bfill().fillna(imputed.min())
        maxVal = imputed.expanding().max().shift(1).bfill().fillna(imputed.max())
        rangeVal = (maxVal - minVal).replace(0, 1)
        return (imputed - minVal) / rangeVal

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

    # Negative shift pulls future prices t+N backward to the current row t
    dfEval['Forward Alpha'] = np.log(dfEval.groupby('Ticker')[priceCol].shift(-forwardWindow) / dfEval[priceCol])
    # Drop rows at the end of the time-series where forward returns cannot yet be calculated
    dfEval = dfEval.dropna(subset=['Forward Alpha'])

    # Robust Targeting via QuantileTransformer on a Gaussian Distribution
    qt = QuantileTransformer(output_distribution='normal', random_state=randSeed)
    valTransformed = qt.fit_transform(dfEval['Forward Alpha'].values.reshape(-1, 1)).flatten()
    
    # Trinary Assignment of Quantile-based Targeting 
    conditions = [(valTransformed < -1), (valTransformed > 1)]
    dfEval['Target State'] = np.select(conditions, [-1, 1], default=0)
    X = expanding_preprocess(dfEval[featureCols], scalerType=None)
    yBinary = (dfEval['Target State'] != 0).astype(int)
    
    # Finding MI Classification values against the Target State 
    miScores = mutual_info_classif(X, yBinary, random_state=randSeed)
    miSeries = pd.Series(miScores, index=featureCols).sort_values(ascending=False)
    
    # Find Correlated variables through Spearman Correlation
    corrMatrix = X.corr(method='spearman').abs()
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
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Bar Plot fo How much MI each included variable contributes
    topFeatures = miSeries[selectedFeatures].sort_values(ascending=False).head(10)
    sns.barplot(x=topFeatures.values, y=topFeatures.index, palette='viridis', ax=axes[0])
    axes[0].set_title("Information Gain (Surviving Features)")
    axes[0].set_xlabel("Mutual Information Score")
    
    # Heat Map showing which variables were selected or dropped
    mask = np.triu(np.ones_like(corrMatrix.loc[selectedFeatures, selectedFeatures], dtype=bool))
    sns.heatmap(corrMatrix.loc[selectedFeatures, selectedFeatures], mask=mask, annot=True, cmap='RdBu_r', fmt=".2f", vmin=-1, vmax=1, ax=axes[1])
    axes[1].set_title("Post-Pruning Spearman Correlation")
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
    neighbors = [5, 10, 15, 20]
    validNeighbors = [n for n in neighbors if n < len(xEval)]
    if not validNeighbors: return 0.0
    return np.mean([trustworthiness(xEval, embEval, n_neighbors=n) for n in validNeighbors])

def manifold_clustering(df, features, sectorCol):
    XRaw = df[features].copy()
    ySector = LabelEncoder().fit_transform(df[sectorCol].astype(str)) if sectorCol in df.columns else None
    
    # Dynamic Manifold Hyperparamater Selection
    def objective(trial):
        # Select a Scaler and perform a fit to the median
        scalerChoice = trial.suggest_categorical('scaler', ['standard', 'minmax'])
        try:
            XScaled = expanding_preprocess(XRaw, scalerType=scalerChoice).values
            # Select a Dimensionality Reduction Algorithm
            dimAlgo = trial.suggest_categorical('dimAlgo', ['PCA', 'UMAP']) #'t-SNE', 
            # Set Supervision mode
            supervision = trial.suggest_categorical('supervision', ['Unsupervised', 'Targeted'])
            # Select the number of Components of the manifold
            nComp = 2 #trial.suggest_int('nComp', 2, 3)
            # Linear Primary Component Analysis
            if dimAlgo == 'PCA':
                emb = PCA(n_components=nComp, random_state=randSeed).fit_transform(XScaled)
            # t-Distributed Stochastic Neighbor with different perplexity neighbor parameters
            elif dimAlgo == 't-SNE':
                perp = trial.suggest_int('perplexity', 10, min(50, len(XScaled) - 1))
                emb = TSNE(n_components=nComp, perplexity=perp, init='pca', random_state=randSeed).fit_transform(XScaled)
            # Uniform Manifold Approximation and Projection, checking different comginations of neighbors
            else:
                numNeighbors = trial.suggest_int('numNeighbors', 5, min(50, len(XScaled) - 1))
                yTarget = ySector if supervision == 'Targeted' and ySector is not None else None
                emb = umap.UMAP(n_components=nComp, n_neighbors=numNeighbors, min_dist=0.1, random_state=randSeed).fit_transform(XScaled, y=yTarget)
                
            # Multi-Scale Trust Metric Evaluation
            trust = get_multi_trust(XScaled, emb)
            if trust < trustThres: return -1.0 
            
            # Evaluating clustering across both KMeans and DBSCAN
            clusterAlgo = trial.suggest_categorical('clusterAlgo', ['KMeans', 'DBSCAN'])
            # Evaluate different clustering sizes of k for KMeans
            if clusterAlgo == 'KMeans':
                labels = KMeans(n_clusters=trial.suggest_int('k', 3, 25), random_state=randSeed, n_init='auto').fit_predict(emb)
            # Evaluate different minimum sample sizes and epsilon search radius
            else:
                labels = DBSCAN(eps=trial.suggest_categorical('eps', [0.1, 0.25, 0.5, 1.0, 1.5]), min_samples=trial.suggest_int('minSamples', 3, 25)).fit_predict(StandardScaler().fit_transform(emb))
            # Filtering out combinations which are below the trustworthiness threshold
            validMask = labels != -1
            if len(set(labels[validMask])) < 2: return -1.0
            # Find the silhouette score of valid combinations
            score = silhouette_score(emb[validMask], labels[validMask], sample_size=15000, random_state=randSeed)
            return score - ((1.0 - (validMask.sum() / len(labels))) * 0.5)
        except ValueError: raise optuna.TrialPruned()   
        except Exception as e: 
            print(f"Trial failed with unexpected error: {e}")
            raise optuna.TrialPruned()
    
    numStudies = 3
    # Trigger Optuna Hyperparameter tuning for Manifold clustering
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=randSeed))
    study.optimize(objective, n_trials=numStudies, show_progress_bar=False)
    
    # VISUALIZATION: Manifold Optuna Convergence tracking
    trialsDf = study.trials_dataframe()
    plt.figure(figsize=(10, 4))
    plt.plot(trialsDf['number'] + 1, trialsDf['value'].cummax(), marker='o', linestyle='--', color='b', alpha=0.5, label='Best Score')
    plt.plot(trialsDf['number'] + 1, trialsDf['value'], marker='o', linestyle='', color='r', alpha=0.7, label='Trial Score')
    plt.locator_params(axis='x', integer=True)
    plt.title(f"Optuna Convergence: Best Silhouette Score ({study.best_value:.3f})")
    plt.xlabel("Trial Number")
    plt.ylabel("Penalized Silhouette Score")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.show()

    # Rebuild Optimal State
    best = study.best_params
    scaler = StandardScaler() if best['scaler'] == 'StandardScaler' else MinMaxScaler()
    XScaled = scaler.fit_transform(SimpleImputer(strategy='median').fit_transform(XRaw))
    
    if best['dimAlgo'] == 'PCA':
        emb = PCA(n_components=2, random_state=randSeed).fit_transform(XScaled)
        #emb = PCA(n_components=best['nComp'], random_state=randSeed).fit_transform(XScaled)
    elif best['dimAlgo'] == 't-SNE':
        emb = TSNE(n_components=2, perplexity=best['perplexity'], init='pca', random_state=randSeed).fit_transform(XScaled)
        #emb = TSNE(n_components=best['nComp'], perplexity=best['perplexity'], init='pca', random_state=randSeed).fit_transform(XScaled)
    else:
        y_target = ySector if best['supervision'] == 'Targeted' and ySector is not None else None
        emb = umap.UMAP(n_components=2, n_neighbors=best['numNeighbors'], min_dist=0.1, random_state=randSeed).fit_transform(XScaled, y=y_target)
        #emb = umap.UMAP(n_components=best['nComp'], n_neighbors=best['numNeighbors'], min_dist=0.1, random_state=randSeed).fit_transform(XScaled, y=y_target)

    if best['clusterAlgo'] == 'KMeans':
        labels = KMeans(n_clusters=best['k'], random_state=randSeed, n_init='auto').fit_predict(emb)
    else:
        labels = DBSCAN(eps=best['eps'], min_samples=best['minSamples']).fit_predict(StandardScaler().fit_transform(emb))
    
    labels = np.where(labels != -1, labels + 1, -1)
    df['DimOne'], df['DimTwo'], df['ClusterID'] = emb[:, 0], emb[:, 1], labels
    
    # VISUALIZATION: Embedding Feature Loadings (Mutual Information)
    miDimOne = mutual_info_regression(XScaled, emb[:, 0], random_state=randSeed)
    miDimTwo = mutual_info_regression(XScaled, emb[:, 1], random_state=randSeed)
    loadingsDf = pd.DataFrame({'DimOne': miDimOne, 'DimTwo': miDimTwo}, index=features).sort_values('DimOne', ascending=False)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(loadingsDf.head(10), annot=True, cmap='viridis', fmt=".3f")
    plt.title(f"Top 10 Feature Loadings to Optimal Manifold ({best['dimAlgo']} - {best['supervision']})")
    plt.tight_layout()
    plt.show()
    
    if sectorCol in df.columns:
        validClusters = df[df['ClusterID'] != -1]
        validClusters['Cluster Name'] = "Cluster" + validClusters['ClusterID'].astype(str)
        compTable = pd.crosstab(validClusters['Cluster Name'], validClusters[sectorCol], normalize='index') * 100
        compTable.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='Set3', edgecolor='black')
        plt.title("Sector Composition Map (Manifold Output)")
        plt.xlabel("")
        plt.ylabel("Percentage (%)")
        plt.legend(title="Sector", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.show()
    
    return df


# ==========================================
# PHASE 3: MASTER PROOF & LOCAL EXPERTS
# ==========================================

def calculate_decay_weights(dates, hlDays=180):
    """Calculates exponential decay weights combined with class balancing."""
    # Parse the 14-digit packed timestamp (YYYYMMDDHHMMSS)
    parsedDates = pd.to_datetime(dates.astype(str), format='%Y%m%d%H%M%S')
    latestDate = parsedDates.max()
    deltaDays = (latestDate - parsedDates).dt.total_seconds() / (24 * 3600)

    # Calculate exponential decay (lambda derived from half-life)
    decayRate = np.log(2) / hlDays
    return np.exp(-decayRate * deltaDays).values

def optimize_cluster_model(X, y, datesSeries, isLinear):
    """Sub-routine to tune Local Expert hyperparameters per cluster."""
    def objective(trial):
        hlDays = 180 #trial.suggest_int('hlDays', 30, 365)
        timeWeights = calculate_decay_weights(datesSeries, hlDays)
        cv = TimeSeriesSplit(n_splits=3)
        if isLinear:
            cValue = trial.suggest_float('cValue', 1e-3, 1e2, log=True)
            penaltyType = trial.suggest_categorical('penaltyType', ['l1', 'l2'])
            model = LogisticRegression(C=cValue, penalty=penaltyType, solver='saga', max_iter=1000, random_state=randSeed, class_weight='balanced')
            score = cross_val_score(model, X, y, cv=cv, scoring='matthews_corrcoef', error_score=0, params={'sample_weight': timeWeights}).mean()
        else:
            maxD = min(6, max(2, len(X) // 15))
            maxEst = min(100, max(30, len(X) * 2))
            maxDepth = trial.suggest_int('maxDepth', 2, maxD)
            learningRate = trial.suggest_float('learningRate', 1e-3, 0.3, log=True)
            nEstimators = trial.suggest_int('nEstimators', 30, maxEst)

            # Handle class imbalance explicitly for XGBoost
            posMask = (y == 1)
            numPos = max(1, posMask.sum())
            negWeight = (len(y) - posMask.sum()) / numPos

            combinedWeights = timeWeights.copy()
            combinedWeights[posMask] *= negWeight

            model = xgb.XGBClassifier(max_depth=maxDepth, learning_rate=learningRate, n_estimators=nEstimators, random_state=randSeed)
            score = cross_val_score(model, X, y, cv=cv, scoring='matthews_corrcoef', error_score=0, params={'sample_weight': combinedWeights}).mean()
        return score

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=randSeed))
    study.optimize(objective, n_trials=15, show_progress_bar=False)
    return study.best_params

def multi_classification(df, features): 
    clusterResults = []
    localExperts = {}
    shapVisuals = []
    
    # Adding Manifold "Loadings" as classification variables
    modelFeatures = features + ['DimOne', 'DimTwo']
    # Setup separate probability/prediction columns
    df['Buy Prob'] = 0.5
    df['Sell Prob'] = 0.5
    df['Predicted Buy'] = 0
    df['Predicted Sell'] = 0

    for clusterId in df['ClusterID'].unique():
        if clusterId == -1: continue

        mask = df['ClusterID'] == clusterId
        localDf = df[mask].copy()

        # Enforce strict chronology before TimeSeriesSplit
        if 'Date' in localDf.columns: localDf = localDf.sort_values('Date')
        else: localDf = localDf.sort_index(level='Date')

        XLocal = expanding_preprocess(localDf[modelFeatures], scalerType=None)
        XScaled = expanding_preprocess(XLocal, scalerType='standard').values
        localDates = localDf.index.get_level_values('Date') if 'Date' in localDf.index.names else localDf['Date']

        # Train two separate classifiers (1 for Upside, -1 for Downside)
        for targetVal, probCol, predCol in [(1, 'Buy Prob', 'Predicted Buy'), (-1, 'Sell Prob', 'Predicted Sell')]:
            yLocal = (localDf['Target State'] == targetVal).astype(int)

            if yLocal.nunique() < 2 or yLocal.sum() < 5: continue
            cv = TimeSeriesSplit(n_splits=3)

            # Tune Hyperparameters with nested Optuna
            linParams = optimize_cluster_model(XScaled, yLocal, localDates, isLinear=True)
            xgbParams = optimize_cluster_model(XLocal, yLocal, localDates, isLinear=False)

            # Pop the half-life out of the dictionary to rebuild optimal weights
            linWeights = calculate_decay_weights(localDates, hlDays=180)
            # linWeights = calculate_decay_weights(localDates, linParams.pop('hlDays'))
            xgbBaseWeights = calculate_decay_weights(localDates, hlDays=180)
            # xgbBaseWeights = calculate_decay_weights(localDates, xgbParams.pop('hlDays'))
    
            # Re-apply imbalance penalty to optimal XGBoost weights
            posMask = (yLocal == 1)
            xgbRatio = (len(yLocal) - posMask.sum()) / (max(1, posMask.sum()))
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
            
            linMcc = cross_val_score(linModel, XScaled, yLocal, cv=cv, scoring='matthews_corrcoef', error_score=0, params={'sample_weight': linWeights}).mean()
            xgbMcc = cross_val_score(xgbModel, XLocal, yLocal, cv=cv, scoring='matthews_corrcoef', error_score=0, params={'sample_weight': xgbCombinedWeights}).mean()
            
            dominance = "Linear" if linMcc > xgbMcc else "XGBoost"
            directionLabel = 'Buy' if targetVal == 1 else 'Sell'
            
            if dominance == "Linear":
                expert = linModel.fit(XScaled, yLocal, sample_weight=linWeights)
                df.loc[mask, probCol] = expert.predict_proba(XScaled)[:, 1]
                explainer = shap.LinearExplainer(expert, XScaled)
                shapValues = explainer.shap_values(XScaled)
            else:
                # Extract tree solely for SHAP, use Calibrated output for signal
                baseTree = xgbModel.fit(XLocal, yLocal, sample_weight=xgbCombinedWeights) 
                explainer = shap.TreeExplainer(baseTree)
                shapValues = explainer.shap_values(XLocal)
                expert = CalibratedClassifierCV(estimator=xgbModel, method='sigmoid', cv=cv)
                expert.fit(XLocal, yLocal, sample_weight=xgbCombinedWeights)
                df.loc[mask, probCol] = expert.predict_proba(XLocal)[:, 1]
    
            df.loc[mask, predCol] = np.where(df.loc[mask, probCol] > valChiThres, targetVal, 0)
            localExperts[f"Cluster {clusterId} {directionLabel}"] = expert
            
            # Extract Top 5 absolute SHAP values for the Matrix Grid
            meanShap = np.abs(shapValues[1] if isinstance(shapValues, list) else shapValues).mean(axis=0)
            topIdx = np.argsort(meanShap)[-5:][::-1]
            topFeaturesList = [modelFeatures[i] for i in topIdx]
            top5Str = ", ".join(topFeaturesList[:3]) + "\n" + ", ".join(topFeaturesList[3:])
            clusterLabel = f"Cluster {clusterId}\n({top5Str})"
            
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
                maxVal = min(subset['Lasso MCC'].max(), subset['XGB MCC'].max(), 0) + 0.05
                ax.set_xlim(minVal, maxVal)
                ax.set_ylim(minVal, maxVal)
                ax.plot([minVal, maxVal], [minVal, maxVal], 'k--', alpha=0.5, label='Linear Boundary')
                ax.fill_between([minVal, maxVal], [minVal, maxVal], maxVal, color='blue', alpha=0.05)
                ax.set_title(f"{direction} Models: Linear vs. Non-Linear Dominance")
                ax.set_xlabel("Linear Baseline (Lasso MCC)")
                ax.set_ylabel("Non-Linear Signal (XGBoost MCC)")
                ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, fontsize=9, title="Cluster Profile & Top 5 Features")
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
    largestCluster = df[df['ClusterID'] != -1]['ClusterID'].value_counts().index[0]
    buyKey = f"Cluster_{largestCluster}_Buy"
    
    if buyKey in localExperts:
        expert = localExperts[buyKey]
        expertCore = expert.estimator if hasattr(expert, 'estimator') else expert
        
        localMask = df['ClusterID'] == largestCluster
        XPdp = expanding_preprocess(df[localMask][modelFeatures], scaler_type=None)
        
        # Calculate 2-way PDP across the manifold components
        pdResults = partial_dependence(expertCore, XPdp, features=['DimOne', 'DimTwo'], grid_resolution=30)
        
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
    sensitivityResults = []
    
    # Chi-Square calculation for Mahalanobis statistical threshold 
    dof = len(valuationCols)
    chiCutoff = stats.chi2.ppf(valChiThres, dof) # 95% confidence interval limit

    for clusterId in df['ClusterID'].unique():
        if clusterId == -1: continue
        
        mask = df['ClusterID'] == clusterId
        clusterData = df[mask].copy()
        
        recentWindow = clusterData.tail(252)
        metricsRecent = recentWindow[valuationCols].fillna(recentWindow[valuationCols].median())
        metricsAll = clusterData[valuationCols].fillna(clusterData[valuationCols].median())
        
        try:
            covModel = MinCovDet().fit(metricsRecent)
            df.loc[mask, 'MDist'] = covModel.mahalanobis(metricsAll)
        except Exception: pass
        
        # Log-Transformation for Z-score on Primary Valuation Multiple
        valCol = valuationCols[0]
        posMask = clusterData[valCol] > 0
        if posMask.sum() > 1:
            logVals = np.log(clusterData.loc[posMask, valCol])
            zScores = (logVals - logVals.mean()) / logVals.std()
            df.loc[mask & posMask, 'Z Score'] = zScores
        else:
            df.loc[mask, 'Z Score'] = 0 

    # Dynamic programmatic grid search to select best Z-Score threshold
    zGrid = [1.0, 1.25, 1.5, 1.75]
    bestYield = -1
    bestZ = 1.0

    for z in zGrid:
        validAssets = ((df['MDist'] <= chiCutoff) & (df['Z Score'].abs() >= z)).sum()
        sensitivityResults.append({'Z-Score Min': z, 'Valid Assets': validAssets})
        if validAssets > bestYield:
            bestYield = validAssets
            bestZ = z

    # Execute Final Validation utilizing optimal programmatic threshold
    df['Validation'] = 'Neutral/Noise'
    finalPortfolio = []
    
    for idx, row in df.iterrows():
        if pd.isna(row['MDist']) or row['ClusterID'] == -1: continue
        
        if row['MDist'] > chiCutoff:
            df.at[idx, 'Validation'] = 'Rejected Outlier'
        elif row['Predicted Buy'] == 1 and row['Z Score'] <= -bestZ:
            df.at[idx, 'Validation'] = 'Validated Buy'
            finalPortfolio.append(row)
        elif row['Predicted Sell'] == -1 and row['Z Score'] >= bestZ:
            df.at[idx, 'Validation'] = 'Validated Sell'
            finalPortfolio.append(row)
         
    # VISUALIZATION: Yield Curve
    sensDf = pd.DataFrame(sensitivityResults).set_index('Z-Score Min')
    plt.figure(figsize=(8, 4))
    plt.plot(sensDf.index, sensDf['Valid Assets'], marker='o', color='teal', linewidth=2, markersize=8)
    plt.fill_between(sensDf.index, sensDf['Valid Assets'], color='teal', alpha=0.2)
    plt.title(f"Portfolio Yield Curve vs Strictness (Selected Z: {bestZ})")
    plt.xlabel("Absolute Z-Score Minimum Threshold")
    plt.ylabel("Validated Assets")
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # Annotate counts directly on the line chart
    for z, val in zip(sensDf.index, sensDf['Valid Assets']):
        plt.annotate(f"{val}", (z, val), textcoords="offset points", xytext=(0,10), ha='center', fontweight='bold')
    plt.tight_layout()
    plt.show()

    # VISUALIZATION: Risk/Reward Yield Matrix (Density Corrected)
    plotDf = df[df['Validation'] != 'Neutral/Noise']
    plt.figure(figsize=(8, 6))
    outliers = plotDf[plotDf['Validation'] == 'Rejected Outlier']
    valid = plotDf[plotDf['Validation'].isin(['Validated Buy', 'Validated Sell'])]
    # Layer 1: Outliers pushed to the background
    sns.scatterplot(data=outliers, x='Z Score', y='MDist', color='black', alpha=0.2, s=20, marker='x', label='Rejected Outlier')
    # Layer 2: Validated assets pushed to the front
    sns.scatterplot(data=valid, x='Z Score', y='MDist', hue='Validation', style='Validation', 
                    s=80, alpha=0.9, edgecolor='white', palette={'Validated Buy': 'green', 'Validated Sell': 'red'})
    plt.axvline(-bestZ, color='green', linestyle='--', alpha=0.4)
    plt.axvline(bestZ, color='red', linestyle='--', alpha=0.4)
    plt.axhline(chiCutoff, color='black', linestyle='--', alpha=0.4)
    plt.title("Risk/Reward Yield Matrix")
    plt.xlabel("Valuation Z-Score (Alpha Opportunity)")
    plt.ylabel("Mahalanobis Distance (Topological Risk)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # VISUALIZATION: Final 2D Portfolio Projection
    plt.figure(figsize=(10, 8))
    noiseDf = df[df['Validation'] == 'Neutral/Noise']
    # Layer 1: Hexbin grid prevents the 60,000 noise rows from rendering as a solid black blob
    plt.hexbin(noiseDf['DimOne'], noiseDf['DimTwo'], gridsize=40, cmap='Greys', mincnt=1, alpha=0.3, label='Neutral Density')
    # Layer 2 & 3: Outliers and Validated Assets overlay
    sns.scatterplot(data=outliers, x='DimOne', y='DimTwo', color='black', marker='x', s=15, alpha=0.3, label='Rejected')
    sns.scatterplot(data=valid, x='DimOne', y='DimTwo', hue='Validation', 
                    palette={'Validated Buy': 'green', 'Validated Sell': 'red'}, 
                    s=120, edgecolor='white', linewidth=1.5, alpha=1.0)
    plt.title("Final Evaluated Portfolio Mapped to Spatial Manifold")
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles, labels, loc='lower right')
    plt.tight_layout()
    plt.show()
    
    return pd.DataFrame(finalPortfolio, columns=df.columns)

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
    df = pd.read_csv('synthetic_market_data.csv', dtype={'Date': str})

    # 2. Define the Feature Space
    features = [
        '10-Year Treasury Yield', '2-Year Treasury Yield', 'Yield Curve Spread',
        'CPI', 'Federal Funds Rate', 'Unemployment Rate', 'Financial Stress Index',
        'Volume', 'Operating Margin', 'Gross Margin', 'ROE', 'ROA',
        'Debt-to-Equity', 'Current Ratio', 'Free Cash Flow', 'EV/EBITDA'
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
        # Isolate only the most recent cross-section to avoid plotting ghost history
        latestDate = portfolioDf['Date'].max()
        currentPortfolio = portfolioDf[portfolioDf['Date'] == latestDate].copy()

        print(f"Pipeline Complete. Validated {len(currentPortfolio)} active assets for {latestDate}.")

        # Define the exact columns the D3 JS requires
        keepCols = [
            'Date', 'Ticker', 'Sector', 'DimOne', 'DimTwo', 'ClusterID',
            'Predicted Buy', 'Predicted Sell', 'Buy Prob', 'Sell Prob', 
            'MDist', 'Z Score', 'Validation'
        ] + valuationMetrics

        # Export flat JSON for the frontend
        exportPayload = currentPortfolio[keepCols]
        exportPayload.to_json('portfolio_dashboard_payload.json', orient='records', date_format='iso')
        print("Exported interactive payload to 'portfolio_dashboard_payload.json'")
    else:
        print("Pipeline finished, but 0 assets passed the strict Mahalanobis and Z-Score thresholds.")

