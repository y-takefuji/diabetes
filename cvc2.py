import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.cluster import FeatureAgglomeration
from sklearn.metrics import f1_score
import xgboost as xgb
from lifelines import CoxPHFitter
from scipy.stats import spearmanr
import sklearn.base as skbase
import warnings
warnings.filterwarnings('ignore')

# ── 1. Load data ───────────────────────────────────────────────────────────────
df = pd.read_csv('diabetes_012_health_indicators_BRFSS2015.csv')
print("Dataset shape:", df.shape)

target_col = 'MentHlth'
y_raw = df[target_col]
X     = df.drop(columns=[target_col])

# ── Binary classification using mean as threshold ─────────────────────────────
mean_val = y_raw.mean()
y        = (y_raw >= mean_val).astype(int)

print(f"\nBinarisation threshold (mean): {mean_val:.4f}")
print(f"Class distribution:\n{y.value_counts().sort_index()}")
print(f"Class balance: {y.mean():.3f} positive rate")

feature_names = X.columns.tolist()
n_top = 6

# ══════════════════════════════════════════════════════════════════════════════
# Helper: cross-validate -> accuracy, F1(1), F1(0)
# ══════════════════════════════════════════════════════════════════════════════
def cv_metrics(model, X_sub, y_bin, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    acc_l, f1_1_l, f1_0_l = [], [], []

    for train_idx, test_idx in skf.split(X_sub, y_bin):
        X_tr = X_sub.iloc[train_idx]
        X_te = X_sub.iloc[test_idx]
        y_tr = y_bin.iloc[train_idx]
        y_te = y_bin.iloc[test_idx]

        m = skbase.clone(model)
        m.fit(X_tr, y_tr)
        y_pred = m.predict(X_te)

        acc_l.append(np.mean(y_pred == y_te.values))
        f1_1_l.append(f1_score(y_te, y_pred, pos_label=1, zero_division=0))
        f1_0_l.append(f1_score(y_te, y_pred, pos_label=0, zero_division=0))

    return {
        'accuracy': round(float(np.mean(acc_l)),   4),
        'f1_1':     round(float(np.mean(f1_1_l)),  4),
        'f1_0':     round(float(np.mean(f1_0_l)),  4)
    }

# ══════════════════════════════════════════════════════════════════════════════
# METHOD 1 - Random Forest Classifier
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("METHOD 1: Random Forest Feature Selection")
print("="*60)

rf = RandomForestClassifier(random_state=42)
rf.fit(X, y)
rf_importances = pd.Series(rf.feature_importances_, index=feature_names)
rf_top6        = rf_importances.nlargest(n_top).index.tolist()
print(f"RF Top 6: {rf_top6}")

rf_scores = cv_metrics(RandomForestClassifier(random_state=42), X[rf_top6], y)
print(f"RF CV Accuracy(6): {rf_scores['accuracy']} | F1(1): {rf_scores['f1_1']} | F1(0): {rf_scores['f1_0']}")

rf_highest   = rf_top6[0]
X_rf_reduced = X.drop(columns=[rf_highest])
rf2          = RandomForestClassifier(random_state=42)
rf2.fit(X_rf_reduced, y)
rf_imp2 = pd.Series(rf2.feature_importances_, index=X_rf_reduced.columns)
rf_top5 = rf_imp2.nlargest(5).index.tolist()
print(f"RF Top 5 (reduced, '{rf_highest}' removed): {rf_top5}")

# ══════════════════════════════════════════════════════════════════════════════
# METHOD 2 - XGBoost Classifier
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("METHOD 2: XGBoost Feature Selection")
print("="*60)

xgb_model = xgb.XGBClassifier(random_state=42, verbosity=0,
                                use_label_encoder=False,
                                eval_metric='logloss')
xgb_model.fit(X, y)
xgb_importances = pd.Series(xgb_model.feature_importances_, index=feature_names)
xgb_top6        = xgb_importances.nlargest(n_top).index.tolist()
print(f"XGB Top 6: {xgb_top6}")

xgb_scores = cv_metrics(
    xgb.XGBClassifier(random_state=42, verbosity=0,
                      use_label_encoder=False, eval_metric='logloss'),
    X[xgb_top6], y)
print(f"XGB CV Accuracy(6): {xgb_scores['accuracy']} | F1(1): {xgb_scores['f1_1']} | F1(0): {xgb_scores['f1_0']}")

xgb_highest   = xgb_top6[0]
X_xgb_reduced = X.drop(columns=[xgb_highest])
xgb2          = xgb.XGBClassifier(random_state=42, verbosity=0,
                                    use_label_encoder=False,
                                    eval_metric='logloss')
xgb2.fit(X_xgb_reduced, y)
xgb_imp2 = pd.Series(xgb2.feature_importances_, index=X_xgb_reduced.columns)
xgb_top5 = xgb_imp2.nlargest(5).index.tolist()
print(f"XGB Top 5 (reduced, '{xgb_highest}' removed): {xgb_top5}")

# ══════════════════════════════════════════════════════════════════════════════
# METHOD 3 - Stratified Cox Regression
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("METHOD 3: Stratified Cox Regression Feature Selection")
print("="*60)

CLINICAL_THRESHOLD = 14
strata_col         = 'Diabetes_012'

cox_df             = X.copy()
cox_df['duration'] = np.where(y_raw.values == 0, 1e-6, y_raw.values)
cox_df['event']    = (y_raw.values >= CLINICAL_THRESHOLD).astype(int)

print(f"Cox event rate (MentHlth >= {CLINICAL_THRESHOLD} days): "
      f"{cox_df['event'].mean():.3f} ({cox_df['event'].sum()} / {len(cox_df)})")

cph = CoxPHFitter()
cph.fit(cox_df, duration_col='duration', event_col='event',
        strata=[strata_col], show_progress=False)

cox_coefs = cph.params_.abs().copy()
for c in [strata_col, 'duration', 'event']:
    if c in cox_coefs.index:
        cox_coefs = cox_coefs.drop(c)

cox_top6 = cox_coefs.nlargest(n_top).index.tolist()
print(f"Cox Top 6: {cox_top6}")

def cox_cv_metrics(X_full, y_binary, y_raw_full, features, strata_col,
                   clinical_threshold=14, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    acc_l, f1_1_l, f1_0_l = [], [], []

    for train_idx, test_idx in skf.split(X_full, y_binary):
        train_X   = X_full.iloc[train_idx][features + [strata_col]].copy()
        test_X    = X_full.iloc[test_idx][features + [strata_col]].copy()
        train_raw = y_raw_full.iloc[train_idx]
        test_raw  = y_raw_full.iloc[test_idx]
        test_bin  = y_binary.iloc[test_idx].values

        train_X['duration'] = np.where(train_raw.values == 0, 1e-6, train_raw.values)
        train_X['event']    = (train_raw.values >= clinical_threshold).astype(int)
        test_X['duration']  = np.where(test_raw.values == 0, 1e-6, test_raw.values)
        test_X['event']     = (test_raw.values >= clinical_threshold).astype(int)

        try:
            _cph = CoxPHFitter()
            _cph.fit(train_X, duration_col='duration', event_col='event',
                     strata=[strata_col], show_progress=False)

            hazard   = _cph.predict_partial_hazard(test_X).values
            pred_bin = (hazard >= np.median(hazard)).astype(int)

            acc_l.append(np.mean(pred_bin == test_bin))
            f1_1_l.append(f1_score(test_bin, pred_bin, pos_label=1, zero_division=0))
            f1_0_l.append(f1_score(test_bin, pred_bin, pos_label=0, zero_division=0))
        except Exception as e:
            print(f"  Fold error: {e}")
            acc_l.append(np.nan)
            f1_1_l.append(np.nan)
            f1_0_l.append(np.nan)

    return {
        'accuracy': round(float(np.nanmean(acc_l)),   4),
        'f1_1':     round(float(np.nanmean(f1_1_l)),  4),
        'f1_0':     round(float(np.nanmean(f1_0_l)),  4)
    }

cox_scores = cox_cv_metrics(X, y, y_raw, cox_top6, strata_col,
                             clinical_threshold=CLINICAL_THRESHOLD)
print(f"Cox CV Accuracy(6): {cox_scores['accuracy']} | F1(1): {cox_scores['f1_1']} | F1(0): {cox_scores['f1_0']}")

cox_highest = cox_top6[0]
cox_df2     = cox_df.drop(columns=[cox_highest])
cph2        = CoxPHFitter()
cph2.fit(cox_df2, duration_col='duration', event_col='event',
         strata=[strata_col], show_progress=False)
cox_coefs2 = cph2.params_.abs().copy()
for c in [strata_col, 'duration', 'event']:
    if c in cox_coefs2.index:
        cox_coefs2 = cox_coefs2.drop(c)
cox_top5 = cox_coefs2.nlargest(5).index.tolist()
print(f"Cox Top 5 (reduced, '{cox_highest}' removed): {cox_top5}")

# ══════════════════════════════════════════════════════════════════════════════
# METHOD 4 - Pure Feature Agglomeration
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("METHOD 4: Pure Feature Agglomeration Feature Selection")
print("="*60)

def fa_top_features(X_data, n_select):
    X_arr    = X_data.values
    fa_model = FeatureAgglomeration(n_clusters=n_select)
    fa_model.fit(X_arr)

    labels      = fa_model.labels_
    X_centroids = fa_model.transform(X_arr)

    feat_scores = {}
    for i, feat in enumerate(X_data.columns):
        cluster_id        = labels[i]
        feat_vec          = X_arr[:, i]
        centroid          = X_centroids[:, cluster_id]
        eucl_dist         = np.sqrt(np.sum((feat_vec - centroid) ** 2))
        feat_scores[feat] = -eucl_dist

    return pd.Series(feat_scores, index=X_data.columns).nlargest(n_select).index.tolist()

fa_top6 = fa_top_features(X, n_top)
print(f"FA Top 6 (global ranking across all clusters): {fa_top6}")

fa_scores = cv_metrics(RandomForestClassifier(random_state=42), X[fa_top6], y)
print(f"FA CV Accuracy(6): {fa_scores['accuracy']} | F1(1): {fa_scores['f1_1']} | F1(0): {fa_scores['f1_0']}")

fa_highest   = fa_top6[0]
X_fa_reduced = X.drop(columns=[fa_highest])
fa_top5      = fa_top_features(X_fa_reduced, 5)
print(f"FA Top 5 (reduced, '{fa_highest}' removed): {fa_top5}")

# ══════════════════════════════════════════════════════════════════════════════
# METHOD 5 - HVGS (variance-based)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("METHOD 5: HVGS (Variance-based) Feature Selection")
print("="*60)

feat_var  = X.var()
hvgs_top6 = feat_var.nlargest(n_top).index.tolist()
print(f"HVGS Top 6: {hvgs_top6}")

hvgs_scores = cv_metrics(RandomForestClassifier(random_state=42), X[hvgs_top6], y)
print(f"HVGS CV Accuracy(6): {hvgs_scores['accuracy']} | F1(1): {hvgs_scores['f1_1']} | F1(0): {hvgs_scores['f1_0']}")

hvgs_highest   = hvgs_top6[0]
X_hvgs_reduced = X.drop(columns=[hvgs_highest])
feat_var2      = X_hvgs_reduced.var()
hvgs_top5      = feat_var2.nlargest(5).index.tolist()
print(f"HVGS Top 5 (reduced, '{hvgs_highest}' removed): {hvgs_top5}")

# ══════════════════════════════════════════════════════════════════════════════
# METHOD 6 - Spearman Correlation
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("METHOD 6: Spearman Correlation Feature Selection")
print("="*60)

spearman_corrs = pd.Series(
    {feat: abs(spearmanr(X[feat].values, y.values).statistic)
     for feat in feature_names},
    index=feature_names
)
spearman_top6 = spearman_corrs.nlargest(n_top).index.tolist()
print(f"Spearman Top 6: {spearman_top6}")

spearman_scores = cv_metrics(RandomForestClassifier(random_state=42), X[spearman_top6], y)
print(f"Spearman CV Accuracy(6): {spearman_scores['accuracy']} | F1(1): {spearman_scores['f1_1']} | F1(0): {spearman_scores['f1_0']}")

spearman_highest = spearman_top6[0]
X_sp_reduced     = X.drop(columns=[spearman_highest])
spearman_corrs2  = pd.Series(
    {feat: abs(spearmanr(X_sp_reduced[feat].values, y.values).statistic)
     for feat in X_sp_reduced.columns},
    index=X_sp_reduced.columns
)
spearman_top5 = spearman_corrs2.nlargest(5).index.tolist()
print(f"Spearman Top 5 (reduced, '{spearman_highest}' removed): {spearman_top5}")

# ── Summary Table ──────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SUMMARY TABLE")
print("="*60)

summary = pd.DataFrame({
    'Method': [
        'RF', 'XGB', 'Cox_Regression',
        'Feature_Agglomeration', 'HVGS', 'Spearman'
    ],
    'Accuracy': [
        rf_scores['accuracy'],   xgb_scores['accuracy'],  cox_scores['accuracy'],
        fa_scores['accuracy'],   hvgs_scores['accuracy'],  spearman_scores['accuracy']
    ],
    'F1_class1': [
        rf_scores['f1_1'],   xgb_scores['f1_1'],  cox_scores['f1_1'],
        fa_scores['f1_1'],   hvgs_scores['f1_1'],  spearman_scores['f1_1']
    ],
    'F1_class0': [
        rf_scores['f1_0'],   xgb_scores['f1_0'],  cox_scores['f1_0'],
        fa_scores['f1_0'],   hvgs_scores['f1_0'],  spearman_scores['f1_0']
    ],
    'Top_6_Features': [
        str(rf_top6), str(xgb_top6), str(cox_top6),
        str(fa_top6), str(hvgs_top6), str(spearman_top6)
    ],
    'Top_5_Features': [
        str(rf_top5), str(xgb_top5), str(cox_top5),
        str(fa_top5), str(hvgs_top5), str(spearman_top5)
    ]
})

print(summary.to_string(index=False))
summary.to_csv('result.csv', index=False)
print("\nSaved -> result.csv")
