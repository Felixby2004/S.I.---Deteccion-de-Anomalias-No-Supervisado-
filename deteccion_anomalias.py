import time
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve, auc, recall_score
from sklearn.model_selection import train_test_split



st.set_page_config(page_title="Grupo 7 - Detección de Anomalías")
st.title("Detección de Anomalías (No Supervisado)")
file = st.file_uploader("📂 Subir creditcard.csv", type=["csv"])

if file is None:
    st.stop()

# carga del dataset
df = pd.read_csv(file)

# Cantidad de registros a analizar
rows = st.slider("Cantidad de registros a analizar:", min_value=5000, max_value=20000, step=500)

# Limitar a los primeros 'rows' registros
df = df.head(rows)

# Resumen del dataset
st.subheader("📋 Resumen del Dataset subido")
st.markdown('<p>Distribución de "Class" (Normal = 0, Fraude = 1):</p>', unsafe_allow_html=True)
st.write(df['Class'].value_counts().to_frame("count"))
pct_fraud = 100 * df['Class'].mean()

st.markdown(f'<p>Porcentaje de fraude: {pct_fraud:.2f} %</p>', unsafe_allow_html=True)
st.write("---")


x = df.drop(columns=['Class'])          # variables independientes
y = df['Class'].values                  # variable dependiente


# Division de Dataset en entrenamiento (70%) y prueba (30%)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=0)


# PCA para visualización 2D
pca = PCA(n_components=2, random_state=42)
pca.fit(X_train)
Xpca_test = pca.transform(X_test)


run = st.button("🚀 Ejecutar detección y evaluación")
if not run:
    st.stop()

# Helper functions
def eval_and_metrics(y_true, y_pred, scores):
    # confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0
    # rates
    fpr = fp / (fp + tn + 1e-12)
    tpr = tp / (tp + fn + 1e-12)  # recall
    # ROC AUC (requires continuous scores)
    try:
        roc_auc = roc_auc_score(y_true, scores)
    except Exception:
        roc_auc = np.nan
    # PR AUC
    try:
        prec, rec, _ = precision_recall_curve(y_true, scores)
        pr_auc = auc(rec, prec)
    except Exception:
        pr_auc = np.nan
    return {
        "cm": cm, "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "FPR": fpr, "TPR": tpr, "ROC_AUC": roc_auc, "PR_AUC": pr_auc
    }




# Isolation Forest
st.subheader("1) Isolation Forest")
t0 = time.time()
iso = IsolationForest(contamination=0.005, random_state=42, n_jobs=-1)
iso.fit(X_train)

# predicción
iso_pred_raw = iso.predict(X_test)                  # 1 normal, -1 anormal
iso_pred = (iso_pred_raw == -1).astype(int)
iso_scores = -iso.decision_function(X_test)         # más alto = más anomalo
res_iso = eval_and_metrics(y_test, iso_pred, iso_scores)

# herramientas metricas
col1, col2, col3, col4 = st.columns(4)
col1.metric("Tasa de falsos positivos", f"{res_iso['FPR']:.6f}")
col2.metric("Tasa de detección", f"{res_iso['TPR']:.6f}")
col3.metric("ROC-AUC", f"{res_iso['ROC_AUC']:.6f}")
col4.metric("PR-AUC", f"{res_iso['PR_AUC']:.6f}")


# PCA scatter para el test (Isolation Forest)
fig_iso, ax_iso = plt.subplots(figsize=(6,5))
mask_iso_norm = iso_pred == 0
ax_iso.scatter(Xpca_test[mask_iso_norm,0], Xpca_test[mask_iso_norm,1], s=8, alpha=0.6, label="Normal")
ax_iso.scatter(Xpca_test[~mask_iso_norm,0], Xpca_test[~mask_iso_norm,1], s=20, alpha=0.8, label="Anomalía (IF)", marker='x', color='red')
ax_iso.set_title("Isolation Forest - PCA 2D (Test set)")
ax_iso.set_xlabel("PC1"); ax_iso.set_ylabel("PC2")
ax_iso.legend()
st.pyplot(fig_iso)


# Learning curve para Isolation Forest
st.markdown('<p><br><b>Learning Curve (Isolation Forest) — variando tamaño de entrenamiento (muestra) y evaluando en test</b></p>', unsafe_allow_html=True)
train_fracs = np.linspace(0.1, 1.0, 8)
iso_tprs = []
iso_fprs = []
for frac in train_fracs:
    n = max(50, int(len(X_train) * frac))
    X_sub = X_train[:n]
    model = IsolationForest(contamination=0.005, random_state=42)
    model.fit(X_sub)
    pred = (model.predict(X_test) == -1).astype(int)
    scores = -model.decision_function(X_test)
    r = recall_score(y_test, pred, zero_division=0)
    cm = confusion_matrix(y_test, pred)
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
        f = fp / (fp + tn + 1e-12)
    else:
        f = np.nan
    iso_tprs.append(r)
    iso_fprs.append(f)


fig_l_iso, axl1 = plt.subplots(1,2, figsize=(12,4))
axl1[0].plot(train_fracs * len(X_train), iso_tprs, marker='o', label='TPR (recall)')
axl1[0].set_xlabel('Tamaño de entrenamiento (n)')
axl1[0].set_ylabel('TPR')                   #Tasa de Verdaderos Positivos
axl1[0].set_title('IF - TPR vs train size')
axl1[0].grid(True)
axl1[0].legend()

axl1[1].plot(train_fracs * len(X_train), iso_fprs, marker='o', color='orange', label='FPR')
axl1[1].set_xlabel('Tamaño de entrenamiento (n)')
axl1[1].set_ylabel('FPR')                   #Tasa de Falsos Positivos
axl1[1].set_title('IF - FPR vs train size')
axl1[1].grid(True)
axl1[1].legend()

st.pyplot(fig_l_iso)

st.markdown("---")




# One-Class SVM
st.subheader("2) One-Class SVM")
SVM_MAX_TRAIN = 5000
X_train_svm = X_train
if len(X_train) > SVM_MAX_TRAIN:
    # escoje una cantidad aleatoria de datos porque éste demora más
    rng = np.random.RandomState(42)
    idx = rng.choice(len(X_train), SVM_MAX_TRAIN, replace=False)
    X_train_svm = X_train.iloc[idx]

svm = OneClassSVM(kernel='rbf', nu=0.005, gamma='scale')
svm.fit(X_train_svm)

svm_pred = (svm.predict(X_test) == -1).astype(int)
svm_scores = -svm.decision_function(X_test)

res_svm = eval_and_metrics(y_test, svm_pred, svm_scores)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Tasa de falsos positivos", f"{res_svm['FPR']:.6f}")
col2.metric("Tasa de detección", f"{res_svm['TPR']:.6f}")
col3.metric("ROC-AUC", f"{res_svm['ROC_AUC']:.6f}")
col4.metric("PR-AUC", f"{res_svm['PR_AUC']:.6f}")


# PCA scatter para test (One-Class SVM)
fig_svm, ax_svm = plt.subplots(figsize=(6,5))
mask_svm_norm = svm_pred == 0
ax_svm.scatter(Xpca_test[mask_svm_norm,0], Xpca_test[mask_svm_norm,1], s=8, alpha=0.6, label="Normal")
ax_svm.scatter(Xpca_test[~mask_svm_norm,0], Xpca_test[~mask_svm_norm,1], s=20, alpha=0.8, label="Anomalía (OCSVM)", marker='x', color='red')
ax_svm.set_title("One-Class SVM - PCA 2D (Test set)")
ax_svm.set_xlabel("PC1"); ax_svm.set_ylabel("PC2")
ax_svm.legend()
st.pyplot(fig_svm)

# Learning para One-Class SVM
st.markdown('<p><br><b>Learning Curve (One-Class SVM) — variando tamaño de entrenamiento (muestra) y evaluando en test</b></p>', unsafe_allow_html=True)
svm_tprs = []
svm_fprs = []
for frac in train_fracs:
    n = max(50, int(len(X_train) * frac))
    # for SVM respect cap
    n_cap = min(n, SVM_MAX_TRAIN)
    if n_cap < n:
        rng = np.random.RandomState(42)
        idx_sub = rng.choice(len(X_train), n_cap, replace=False)
        X_sub = X_train.iloc[idx_sub]
    else:
        X_sub = X_train[:n]
    model = OneClassSVM(kernel='rbf', nu=0.005, gamma='scale')
    model.fit(X_sub)
    # eval on test
    pred = (model.predict(X_test) == -1).astype(int)
    cm = confusion_matrix(y_test, pred)
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
        f = fp / (fp + tn + 1e-12)
    else:
        f = np.nan
        tp = 0
    r = recall_score(y_test, pred, zero_division=0)
    svm_tprs.append(r)
    svm_fprs.append(f)


# plot SVM learning curves
fig_l_svm, axl2 = plt.subplots(1,2, figsize=(12,4))
axl2[0].plot(train_fracs * len(X_train), svm_tprs, marker='o', label='TPR (recall)')
axl2[0].set_xlabel('Tamaño de entrenamiento (n)')
axl2[0].set_ylabel('TPR')
axl2[0].set_title('OCSVM - TPR vs train size')
axl2[0].grid(True)

axl2[1].plot(train_fracs * len(X_train), svm_fprs, marker='o', color='orange', label='FPR')
axl2[1].set_xlabel('Tamaño de entrenamiento (n)')
axl2[1].set_ylabel('FPR')
axl2[1].set_title('OCSVM - FPR vs train size')
axl2[1].grid(True)
st.pyplot(fig_l_svm)

st.markdown("---")




# ROC y PR curves (ambos modelos)
st.subheader("📉 Curvas ROC y Precision-Recall (post-hoc usando etiquetas reales)")

fpr_iso, tpr_iso, _ = roc_curve(y_test, res_iso['tp']*0 + iso_scores if False else iso_scores)
fpr_svm, tpr_svm, _ = roc_curve(y_test, res_svm['tp']*0 + svm_scores)

roc_auc_if = res_iso['ROC_AUC']
roc_auc_svm = res_svm['ROC_AUC']

figr, axr = plt.subplots(1,2, figsize=(14,5))
axr[0].plot(fpr_iso, tpr_iso, label=f'IF AUC={roc_auc_if:.4f}')
axr[0].plot([0,1],[0,1], linestyle='--', color='gray')
axr[0].set_title("ROC Curve - Isolation Forest")
axr[0].set_xlabel("FPR"); axr[0].set_ylabel("TPR"); axr[0].legend()

axr[1].plot(fpr_svm, tpr_svm, label=f'OCSVM AUC={roc_auc_svm:.4f}')
axr[1].plot([0,1],[0,1], linestyle='--', color='gray')
axr[1].set_title("ROC Curve - One-Class SVM")
axr[1].set_xlabel("FPR"); axr[1].set_ylabel("TPR"); axr[1].legend()
st.pyplot(figr)


# Precision-Recall
prec_i, rec_i, _ = precision_recall_curve(y_test, iso_scores)
prec_s, rec_s, _ = precision_recall_curve(y_test, svm_scores)
pr_auc_if = auc(rec_i, prec_i)
pr_auc_svm = auc(rec_s, prec_s)

figpr, axpr = plt.subplots(figsize=(7,6))
axpr.plot(rec_i, prec_i, label=f'IF PR-AUC={pr_auc_if:.4f}')
axpr.plot(rec_s, rec_s, label=f'OCSVM PR-AUC={pr_auc_svm:.4f}')
axpr.set_xlabel("Recall"); axpr.set_ylabel("Precision"); axpr.set_title("Precision-Recall Curve")
axpr.legend()
st.pyplot(figpr)



# Tabla comparativa
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
# quien detecta más anomalias
if res_iso['tp'] > res_svm['tp']:
    detect_more = "Isolation Forest"
elif res_iso['tp'] < res_svm['tp']:
    detect_more = "One-Class SVM"
else:
    detect_more = "Ambos detectan igual"

# quien tiene más falsos positivos
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

