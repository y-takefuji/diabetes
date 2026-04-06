import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.cluster import FeatureAgglomeration
from sklearn.model_selection import cross_val_score
from scipy.stats import spearmanr

# ============================================================
# 1. LOAD & ENCODE DATASET
# ============================================================
df = pd.read_csv('hypertension_dataset.csv')

print("=" * 60)
print("ORIGINAL DATASET")
print("=" * 60)
print(f"Shape: {df.shape}")
print(f"\nTarget Distribution (Original):")
print(df['Hypertension'].value_counts())
print(df['Hypertension'].value_counts(normalize=True).mul(100).round(2).astype(str) + '%')

# Identify string vs numeric columns
string_cols_idx = [0] + list(range(6, 11)) + list(range(len(df.columns)-4, len(df.columns)))
string_cols     = df.columns[string_cols_idx].tolist()

df_encoded   = df.copy()
le           = LabelEncoder()
encoding_map = {}

for col in string_cols:
    df_encoded[col] = le.fit_transform(df[col].astype(str))
    encoding_map[col] = dict(zip(le.classes_, le.transform(le.classes_)))

print("\nHypertension Encoding:", encoding_map['Hypertension'])

df_encoded.to_csv('data.csv', index=False)
print(f"\nSaved -> data.csv  |  Shape: {df_encoded.shape}")

# ============================================================
# 2. PREPARE FEATURES & TARGET
# ============================================================
X_full        = df_encoded.drop(columns=['Hypertension'])
y             = df_encoded['Hypertension']
feature_names = X_full.columns.tolist()

print("\n" + "=" * 60)
print("FEATURE SELECTION SETUP")
print("=" * 60)
print(f"Total Features : {len(feature_names)}")
print(f"Features       : {feature_names}")

# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def select_top_n(scores_dict, n):
    sorted_feats = sorted(scores_dict.items(), key=lambda x: x[1], reverse=True)
    return [f for f, _ in sorted_feats[:n]]

def cross_validate_model(model, X, y, cv=5):
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    return scores.mean()

def compute_variance(X_df):
    variances = {}
    for col in X_df.columns:
        values   = X_df[col].values
        mean_val = np.mean(values)
        var_val  = np.mean((values - mean_val) ** 2)
        variances[col] = var_val
    return variances

# ============================================================
# FA CORE FUNCTION
# ============================================================

def fa_select_top_features(X_df, n_top, n_clusters=None):
    if n_clusters is None:
        n_clusters = n_top

    X_arr = X_df.values
    cols  = X_df.columns.tolist()

    fa             = FeatureAgglomeration(n_clusters=n_clusters)
    fa.fit(X_arr)
    cluster_labels = fa.labels_

    scores = {}
    n      = X_arr.shape[0]
    for idx, col in enumerate(cols):
        x      = X_arr[:, idx].astype(float)
        total  = 0.0
        for val in x:
            total += val
        x_mean = total / n
        sq_sum = 0.0
        for val in x:
            sq_sum += (val - x_mean) ** 2
        scores[col] = sq_sum / n

    top_features = select_top_n(scores, n_top)
    return top_features, scores, cluster_labels

# ============================================================
# 4. METHOD 1 — RANDOM FOREST (RF)
# ============================================================
print("\n" + "=" * 60)
print("METHOD 1: RANDOM FOREST (RF)")
print("=" * 60)

rf_model  = RandomForestClassifier(random_state=42)
rf_model.fit(X_full, y)
rf_scores = dict(zip(feature_names, rf_model.feature_importances_))

rf_top6 = select_top_n(rf_scores, 6)
print(f"\nTop 6 Features (Full Set):")
for i, f in enumerate(rf_top6, 1):
    print(f"  {i}. {f:<35} Score: {rf_scores[f]:.6f}")

rf_highest   = rf_top6[0]
X_rf_reduced = X_full.drop(columns=[rf_highest])
print(f"\nRemoved Highest Feature: '{rf_highest}'")

rf_model2  = RandomForestClassifier(random_state=42)
rf_model2.fit(X_rf_reduced, y)
rf_scores2 = dict(zip(X_rf_reduced.columns, rf_model2.feature_importances_))
rf_top5    = select_top_n(rf_scores2, 5)
print(f"\nTop 5 Features (Reduced Set):")
for i, f in enumerate(rf_top5, 1):
    print(f"  {i}. {f:<35} Score: {rf_scores2[f]:.6f}")

rf_cv6_model = RandomForestClassifier(random_state=42)
rf_cv6       = cross_validate_model(rf_cv6_model, X_full[rf_top6], y, cv=5)
print(f"\nCV Accuracy (Top 6 Features): {rf_cv6:.4f}")

# ============================================================
# 5. METHOD 2 — XGBOOST (XGB)
# ============================================================
print("\n" + "=" * 60)
print("METHOD 2: XGBOOST (XGB)")
print("=" * 60)

xgb_model  = XGBClassifier(random_state=42, eval_metric='logloss', verbosity=0)
xgb_model.fit(X_full, y)
xgb_scores = dict(zip(feature_names, xgb_model.feature_importances_))

xgb_top6 = select_top_n(xgb_scores, 6)
print(f"\nTop 6 Features (Full Set):")
for i, f in enumerate(xgb_top6, 1):
    print(f"  {i}. {f:<35} Score: {xgb_scores[f]:.6f}")

xgb_highest   = xgb_top6[0]
X_xgb_reduced = X_full.drop(columns=[xgb_highest])
print(f"\nRemoved Highest Feature: '{xgb_highest}'")

xgb_model2  = XGBClassifier(random_state=42, eval_metric='logloss', verbosity=0)
xgb_model2.fit(X_xgb_reduced, y)
xgb_scores2 = dict(zip(X_xgb_reduced.columns, xgb_model2.feature_importances_))
xgb_top5    = select_top_n(xgb_scores2, 5)
print(f"\nTop 5 Features (Reduced Set):")
for i, f in enumerate(xgb_top5, 1):
    print(f"  {i}. {f:<35} Score: {xgb_scores2[f]:.6f}")

xgb_cv6_model = XGBClassifier(random_state=42, eval_metric='logloss', verbosity=0)
xgb_cv6       = cross_validate_model(xgb_cv6_model, X_full[xgb_top6], y, cv=5)
print(f"\nCV Accuracy (Top 6 Features): {xgb_cv6:.4f}")

# ============================================================
# 6. METHOD 3 — LOGISTIC REGRESSION (LR)
# ============================================================
print("\n" + "=" * 60)
print("METHOD 3: LOGISTIC REGRESSION (LR)")
print("=" * 60)

lr_model  = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_full, y)
lr_coefs  = np.abs(lr_model.coef_[0])
lr_scores = dict(zip(feature_names, lr_coefs))

lr_top6 = select_top_n(lr_scores, 6)
print(f"\nTop 6 Features (Full Set):")
for i, f in enumerate(lr_top6, 1):
    print(f"  {i}. {f:<35} |Coef|: {lr_scores[f]:.6f}")

lr_highest   = lr_top6[0]
X_lr_reduced = X_full.drop(columns=[lr_highest])
print(f"\nRemoved Highest Feature: '{lr_highest}'")

lr_model2  = LogisticRegression(random_state=42, max_iter=1000)
lr_model2.fit(X_lr_reduced, y)
lr_coefs2  = np.abs(lr_model2.coef_[0])
lr_scores2 = dict(zip(X_lr_reduced.columns, lr_coefs2))
lr_top5    = select_top_n(lr_scores2, 5)
print(f"\nTop 5 Features (Reduced Set):")
for i, f in enumerate(lr_top5, 1):
    print(f"  {i}. {f:<35} |Coef|: {lr_scores2[f]:.6f}")

lr_cv6_model = LogisticRegression(random_state=42, max_iter=1000)
lr_cv6       = cross_validate_model(lr_cv6_model, X_full[lr_top6], y, cv=5)
print(f"\nCV Accuracy (Top 6 Features): {lr_cv6:.4f}")

# ============================================================
# 7. METHOD 4 — FEATURE AGGLOMERATION (FA)
# ============================================================
print("\n" + "=" * 60)
print("METHOD 4: FEATURE AGGLOMERATION (FA)")
print("=" * 60)

fa_top6, fa_scores, fa_clusters = fa_select_top_features(
    X_full, n_top=6, n_clusters=6
)

print(f"\nCluster Assignments (FA Ward Linkage):")
for feat, clust in zip(feature_names, fa_clusters):
    mark = " <- selected" if feat in fa_top6 else ""
    print(f"  {feat:<35} -> Cluster {clust}{mark}")

print(f"\nAll Feature Variance Scores:")
for f, v in sorted(fa_scores.items(), key=lambda x: x[1], reverse=True):
    clust_id = fa_clusters[feature_names.index(f)]
    mark     = " <- TOP6" if f in fa_top6 else ""
    print(f"  {f:<35} Variance: {v:.6f}  Cluster: {clust_id}{mark}")

print(f"\nTop 6 Features Across ALL Clusters (no per-cluster restriction):")
for i, f in enumerate(fa_top6, 1):
    clust_id = fa_clusters[feature_names.index(f)]
    print(f"  {i}. {f:<35} Variance: {fa_scores[f]:.6f}  Cluster: {clust_id}")

fa_highest   = fa_top6[0]
X_fa_reduced = X_full.drop(columns=[fa_highest])
print(f"\nRemoved Highest Feature: '{fa_highest}'")

fa_top5, fa_scores2, fa_clusters2 = fa_select_top_features(
    X_fa_reduced, n_top=5, n_clusters=6
)
print(f"\nTop 5 Features (Reduced Set):")
for i, f in enumerate(fa_top5, 1):
    clust_id = fa_clusters2[X_fa_reduced.columns.tolist().index(f)]
    print(f"  {i}. {f:<35} Variance: {fa_scores2[f]:.6f}  Cluster: {clust_id}")

fa_cv6_model = RandomForestClassifier(random_state=42)
fa_cv6       = cross_validate_model(fa_cv6_model, X_full[fa_top6], y, cv=5)
print(f"\nCV Accuracy with RF (Top 6 Features): {fa_cv6:.4f}")

# ============================================================
# 8. METHOD 5 — HIGHLY VARIABLE GENE SELECTION (HVGS)
# ============================================================
print("\n" + "=" * 60)
print("METHOD 5: HIGHLY VARIABLE GENE SELECTION (HVGS)")
print("=" * 60)

hvgs_scores = compute_variance(X_full)

print("\nFeature Variances (manual computation):")
for f, v in sorted(hvgs_scores.items(), key=lambda x: x[1], reverse=True):
    print(f"  {f:<35} Variance: {v:.6f}")

hvgs_top6 = select_top_n(hvgs_scores, 6)
print(f"\nTop 6 Features (Full Set):")
for i, f in enumerate(hvgs_top6, 1):
    print(f"  {i}. {f:<35} Variance: {hvgs_scores[f]:.6f}")

hvgs_highest   = hvgs_top6[0]
X_hvgs_reduced = X_full.drop(columns=[hvgs_highest])
print(f"\nRemoved Highest Feature: '{hvgs_highest}'")

hvgs_scores2 = compute_variance(X_hvgs_reduced)
hvgs_top5    = select_top_n(hvgs_scores2, 5)
print(f"\nTop 5 Features (Reduced Set):")
for i, f in enumerate(hvgs_top5, 1):
    print(f"  {i}. {f:<35} Variance: {hvgs_scores2[f]:.6f}")

hvgs_cv6_model = RandomForestClassifier(random_state=42)
hvgs_cv6       = cross_validate_model(hvgs_cv6_model, X_full[hvgs_top6], y, cv=5)
print(f"\nCV Accuracy with RF (Top 6 Features): {hvgs_cv6:.4f}")

# ============================================================
# 9. METHOD 6 — SPEARMAN CORRELATION
# ============================================================
print("\n" + "=" * 60)
print("METHOD 6: SPEARMAN CORRELATION")
print("=" * 60)

spearman_scores = {}
for col in feature_names:
    corr, _ = spearmanr(X_full[col], y)
    spearman_scores[col] = abs(corr)

print("\nSpearman |Correlation| with Target:")
for f, v in sorted(spearman_scores.items(), key=lambda x: x[1], reverse=True):
    print(f"  {f:<35} |rho|: {v:.6f}")

sp_top6 = select_top_n(spearman_scores, 6)
print(f"\nTop 6 Features (Full Set):")
for i, f in enumerate(sp_top6, 1):
    print(f"  {i}. {f:<35} |rho|: {spearman_scores[f]:.6f}")

sp_highest   = sp_top6[0]
X_sp_reduced = X_full.drop(columns=[sp_highest])
print(f"\nRemoved Highest Feature: '{sp_highest}'")

spearman_scores2 = {}
for col in X_sp_reduced.columns:
    corr, _ = spearmanr(X_sp_reduced[col], y)
    spearman_scores2[col] = abs(corr)
sp_top5 = select_top_n(spearman_scores2, 5)
print(f"\nTop 5 Features (Reduced Set):")
for i, f in enumerate(sp_top5, 1):
    print(f"  {i}. {f:<35} |rho|: {spearman_scores2[f]:.6f}")

sp_cv6_model = RandomForestClassifier(random_state=42)
sp_cv6       = cross_validate_model(sp_cv6_model, X_full[sp_top6], y, cv=5)
print(f"\nCV Accuracy with RF (Top 6 Features): {sp_cv6:.4f}")

# ============================================================
# 10. SUMMARY TABLE + SAVE
# ============================================================
print("\n" + "=" * 70)
print("SUMMARY TABLE")
print("=" * 70)

sep = " ; "

results = {
    'RF'          : {'cv6': rf_cv6,   'top6': rf_top6,   'top5': rf_top5},
    'XGB'         : {'cv6': xgb_cv6,  'top6': xgb_top6,  'top5': xgb_top5},
    'LR'          : {'cv6': lr_cv6,   'top6': lr_top6,   'top5': lr_top5},
    'FA+RF'       : {'cv6': fa_cv6,   'top6': fa_top6,   'top5': fa_top5},
    'HVGS+RF'     : {'cv6': hvgs_cv6, 'top6': hvgs_top6, 'top5': hvgs_top5},
    'Spearman+RF' : {'cv6': sp_cv6,   'top6': sp_top6,   'top5': sp_top5},
}

rows = []
for method, res in results.items():
    row = {
        'Method'        : method,
        'CV_Accuracy'   : round(res['cv6'], 4),
        'Top6_Features' : sep.join(res['top6']),
        'Top5_Features' : sep.join(res['top5']),
    }
    rows.append(row)

summary_df = pd.DataFrame(rows)

pd.set_option('display.max_colwidth', 120)
pd.set_option('display.width', 300)
print(summary_df.to_string(index=False))

summary_df.to_csv('result.csv', index=False)
print(f"\nSaved -> result.csv")
print("\n" + "=" * 70)
print("ALL DONE!")
print("=" * 70)
