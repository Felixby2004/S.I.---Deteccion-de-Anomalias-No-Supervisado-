import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import confusion_matrix, recall_score, roc_auc_score, precision_recall_curve, auc, roc_curve
import matplotlib.pyplot as plt


st.set_page_config(page_title="Grupo 7 - Detección de Anomalías")
st.title("Detección de Anomalías (No Supervisado)")


# Subida de dataset
file = st.file_uploader("📂 Subir Financial Dataset.csv", type=["csv"])
if file is None:
    st.stop()

df = pd.read_csv(file)

# Slider para cantidad de registros a usar
max_rows = len(df)
rows = st.slider("Cantidad de registros a analizar:", min_value=5000, max_value=max_rows, step=500)
df = df.iloc[:rows].copy()

# Selección de columnas útiles
cat_cols = ["type"]
num_cols = ["step","amount","oldbalanceOrg","newbalanceOrig","oldbalanceDest","newbalanceDest"]

# Vista
st.subheader("👀 Vista previa")
st.dataframe(df.head())

# Resumen
st.subheader("📋 Resumen del Dataset subido")
st.markdown('<p>Distribución de "isFraud" (Normal = 0, Fraude = 1):</p>', unsafe_allow_html=True)
st.write(df['isFraud'].value_counts().to_frame("count"))
pct_fraud = 100 * df['isFraud'].mean()
st.markdown(f'<p>Porcentaje de fraude: {pct_fraud:.2f} %</p>', unsafe_allow_html=True)
st.write("---")

# Imputación
num_imputer = SimpleImputer(strategy='mean')
df[num_cols] = num_imputer.fit_transform(df[num_cols])
cat_imputer = SimpleImputer(strategy='most_frequent')
df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

# One-Hot Encoding + ColumnTransformer
ct = ColumnTransformer(transformers=[('one_hot', OneHotEncoder(drop='first'), cat_cols)], remainder='passthrough')
X = np.array(ct.fit_transform(df[cat_cols + num_cols]), dtype=np.float64)
y = df['isFraud'].values

# Estandarización o Normalización
scaler = StandardScaler()
X = scaler.fit_transform(X)

# División entrenamiento 70%, 30% prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# PCA 2D
pca = PCA(n_components=2, random_state=42)
pca.fit(X_train)
Xpca_test = pca.transform(X_test)


run = st.button("🚀 Ejecutar detección y evaluación")
if not run:
    st.stop()

# Función helper para métricas
def eval_and_metrics(y_true, y_pred, scores):
    cm = confusion_matrix(y_true, y_pred)
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0
    fpr = fp / (fp + tn + 1e-12)
    tpr = tp / (tp + fn + 1e-12)
    try:
        roc_auc = roc_auc_score(y_true, scores)
    except:
        roc_auc = np.nan
    try:
        prec, rec, _ = precision_recall_curve(y_true, scores)
        pr_auc = auc(rec, prec)
    except:
        pr_auc = np.nan
    return {"cm": cm, "tn": tn, "fp": fp, "fn": fn, "tp": tp,
            "FPR": fpr, "TPR": tpr, "ROC_AUC": roc_auc, "PR_AUC": pr_auc}



# Isolation Forest
st.subheader("1) Isolation Forest")
iso = IsolationForest(contamination=0.005, random_state=42, n_jobs=-1)
iso.fit(X_train)
iso_pred_raw = iso.predict(X_test)
iso_pred = (iso_pred_raw == -1).astype(int)
iso_scores = -iso.decision_function(X_test)
res_iso = eval_and_metrics(y_test, iso_pred, iso_scores)

# Métricas
col1, col2, col3, col4 = st.columns(4)
col1.metric("Tasa de falsos positivos", f"{res_iso['FPR']:.6f}")
col2.metric("Tasa de detección", f"{res_iso['TPR']:.6f}")
col3.metric("ROC-AUC", f"{res_iso['ROC_AUC']:.6f}")
col4.metric("PR-AUC", f"{res_iso['PR_AUC']:.6f}")

# PCA scatter
fig_iso, ax_iso = plt.subplots(figsize=(6,5))
mask_iso_norm = iso_pred == 0
ax_iso.scatter(Xpca_test[mask_iso_norm,0], Xpca_test[mask_iso_norm,1], s=8, alpha=0.6, label="Normal")
ax_iso.scatter(Xpca_test[~mask_iso_norm,0], Xpca_test[~mask_iso_norm,1], s=20, alpha=0.8, label="Anomalía (IF)", marker='x', color='red')
ax_iso.set_title("Isolation Forest - PCA 2D (Test set)")
ax_iso.set_xlabel("PC1"); ax_iso.set_ylabel("PC2"); ax_iso.legend()
st.pyplot(fig_iso)



# Learning Curve para Isolation Forest
st.markdown("### 📈 Learning Curve - Isolation Forest")
train_fracs = np.linspace(0.1, 1.0, 6)
iso_tprs, iso_fprs = [], []

for frac in train_fracs:
    n = max(50, int(len(X_train)*frac))
    X_sub = X_train[:n]
    model = IsolationForest(contamination=0.005, random_state=42)
    model.fit(X_sub)
    pred = (model.predict(X_test) == -1).astype(int)
    cm = confusion_matrix(y_test, pred)
    if cm.size==4:
        tn, fp, fn, tp = cm.ravel()
        f = fp/(fp+tn+1e-12)
    else:
        f = np.nan
    r = recall_score(y_test, pred, zero_division=0)
    iso_tprs.append(r)
    iso_fprs.append(f)

fig_l_iso, axl1 = plt.subplots(1,2, figsize=(12,4))
axl1[0].plot(train_fracs*len(X_train), iso_tprs, marker='o', label='TPR')
axl1[0].set_title("IF - TPR vs train size"); axl1[0].set_xlabel("Tamaño de entrenamiento"); axl1[0].set_ylabel("TPR"); axl1[0].grid(True)
axl1[1].plot(train_fracs*len(X_train), iso_fprs, marker='o', color='orange', label='FPR')
axl1[1].set_title("IF - FPR vs train size"); axl1[1].set_xlabel("Tamaño de entrenamiento"); axl1[1].set_ylabel("FPR"); axl1[1].grid(True)
st.pyplot(fig_l_iso)



# One-Class SVM
st.write("---")
st.subheader("2) One-Class SVM")
SVM_MAX_TRAIN = 5000
X_train_svm = X_train
if len(X_train) > SVM_MAX_TRAIN:
    rng = np.random.RandomState(42)
    idx = rng.choice(len(X_train), SVM_MAX_TRAIN, replace=False)
    X_train_svm = X_train[idx]

svm = OneClassSVM(kernel='rbf', nu=0.005, gamma='scale')
svm.fit(X_train_svm)
svm_pred = (svm.predict(X_test) == -1).astype(int)
svm_scores = -svm.decision_function(X_test)
res_svm = eval_and_metrics(y_test, svm_pred, svm_scores)

# Métricas
col1, col2, col3, col4 = st.columns(4)
col1.metric("Tasa de falsos positivos", f"{res_svm['FPR']:.6f}")
col2.metric("Tasa de detección", f"{res_svm['TPR']:.6f}")
col3.metric("ROC-AUC", f"{res_svm['ROC_AUC']:.6f}")
col4.metric("PR-AUC", f"{res_svm['PR_AUC']:.6f}")

# PCA scatter
fig_svm, ax_svm = plt.subplots(figsize=(6,5))
mask_svm_norm = svm_pred == 0
ax_svm.scatter(Xpca_test[mask_svm_norm,0], Xpca_test[mask_svm_norm,1], s=8, alpha=0.6, label="Normal")
ax_svm.scatter(Xpca_test[~mask_svm_norm,0], Xpca_test[~mask_svm_norm,1], s=20, alpha=0.8, label="Anomalía (OCSVM)", marker='x', color='red')
ax_svm.set_title("One-Class SVM - PCA 2D (Test set)")
ax_svm.set_xlabel("PC1"); ax_svm.set_ylabel("PC2"); ax_svm.legend()
st.pyplot(fig_svm)


# Learning Curve para One-Class SVM
st.markdown("### 📈 Learning Curve - One-Class SVM")
svm_tprs, svm_fprs = [], []

for frac in train_fracs:
    n = max(50, int(len(X_train)*frac))
    n_cap = min(n, SVM_MAX_TRAIN)
    if n_cap < n:
        rng = np.random.RandomState(42)
        idx_sub = rng.choice(len(X_train), n_cap, replace=False)
        X_sub = X_train[idx_sub]
    else:
        X_sub = X_train[:n]
    model = OneClassSVM(kernel='rbf', nu=0.005, gamma='scale')
    model.fit(X_sub)
    pred = (model.predict(X_test) == -1).astype(int)
    cm = confusion_matrix(y_test, pred)
    if cm.size==4:
        tn, fp, fn, tp = cm.ravel()
        f = fp/(fp+tn+1e-12)
    else:
        f = np.nan
    r = recall_score(y_test, pred, zero_division=0)
    svm_tprs.append(r)
    svm_fprs.append(f)

fig_l_svm, axl2 = plt.subplots(1,2, figsize=(12,4))
axl2[0].plot(train_fracs*len(X_train), svm_tprs, marker='o', label='TPR')
axl2[0].set_title("OCSVM - TPR vs train size"); axl2[0].set_xlabel("Tamaño de entrenamiento"); axl2[0].set_ylabel("TPR"); axl2[0].grid(True)
axl2[1].plot(train_fracs*len(X_train), svm_fprs, marker='o', color='orange', label='FPR')
axl2[1].set_title("OCSVM - FPR vs train size"); axl2[1].set_xlabel("Tamaño de entrenamiento"); axl2[1].set_ylabel("FPR"); axl2[1].grid(True)
st.pyplot(fig_l_svm)



# Mostrar anomalías
st.write("---")
anom_if = df.iloc[np.where(iso_pred==1)[0]]
st.subheader("🔴 Anomalías detectadas por Isolation Forest")
st.dataframe(anom_if)

anom_svm = df.iloc[np.where(svm_pred==1)[0]]
st.subheader("🔴 Anomalías detectadas por One-Class SVM")
st.dataframe(anom_svm)



# Tabla comparativa
st.write("---")
st.subheader("🏁 Tabla comparativa")
table = pd.DataFrame({
    "Modelo": ["Isolation Forest", "One-Class SVM"],
    "Anomalias Detectadas": [int(res_iso['tp']), int(res_svm['tp'])],
    "Falsos Positivos": [int(res_iso['fp']), int(res_svm['fp'])],
    "Ratio de Detection": [res_iso['TPR'], res_svm['TPR']],
    "FPR": [res_iso['FPR'], res_svm['FPR']],
    "ROC AUC": [res_iso['ROC_AUC'], res_svm['ROC_AUC']],
    "PR AUC": [res_iso['PR_AUC'], res_svm['PR_AUC']]
})
st.dataframe(table)



# Discusión de resultados
st.subheader("📝 Discusión de resultados")

# ¿Cuál detecta más anomalías?
if res_iso['tp'] > res_svm['tp']:
    detect_more = "Isolation Forest"
elif res_iso['tp'] < res_svm['tp']:
    detect_more = "One-Class SVM"
else:
    detect_more = "Ambos detectan igual"

# ¿Quién tiene más falsos positivos?
if res_iso['fp'] > res_svm['fp']:
    more_fp = "Isolation Forest"
elif res_iso['fp'] < res_svm['fp']:
    more_fp = "One-Class SVM"
else:
    more_fp = "Empate"

st.write(f"**¿Cuál detecta más anomalías?: {detect_more}**")
st.write(f"**¿Quién comete más falsos positivos?: {more_fp}**")

st.markdown("""
**Costo del error:**
- **Falso Negativo:** Alto costo como pérdidas financieras, pérdida de reputación.
- **Falso Positivo:** Transacción legítima marcada como fraude puede generar investigaciones fiscales, potencial pérdida de confianza.
""")



