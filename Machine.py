import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap, BoundaryNorm
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


# print(os.getcwd())
# 1. LOAD DATA
df = pd.read_csv("particle_trajectories.csv")
print("Shape:", df.shape)
print(df.head())
os.makedirs("imgs", exist_ok=True)

# вычисляем потенциальную энергию
V_x0 = -df['V0'] / (1.0 + df['x0'] / df['x0_scale'])

# полная начальная энергия
df['E0'] = 0.5 * df['v0']**2 + V_x0


# 2. EDA
print("\n=== РАСПРЕДЕЛЕНИЯ ПРИЗНАКОВ ===")
df.hist(figsize=(12, 6))
plt.tight_layout()
plt.savefig("imgs/hist.png")
plt.close()

print("\n=== SCATTER x0–v0 ===")
plt.figure(figsize=(6, 5))
sns.scatterplot(data=df, x="x0", y="v0", hue="trajectory_type", alpha=0.5)
plt.tight_layout()
plt.savefig("imgs/scatter_x0_v0.png")
plt.close()

print("\n=== TARGET DISTRIBUTION ===")
plt.figure(figsize=(5, 4))
sns.countplot(data=df, x="trajectory_type",
              order=df["trajectory_type"].value_counts().index)
plt.title("Распределение целевой переменной trajectory_type")
plt.xlabel("Класс траектории")
plt.ylabel("Количество")
plt.tight_layout()
plt.savefig("imgs/target_distribution.png")
plt.close()

print("\n=== Boxplot (без E0) ===")
plt.figure(figsize=(10, 5))
# Убираем E0 и целевую переменную, оставляем только исходные признаки
cols_to_plot = [col for col in df.columns if col not in ["trajectory_type", "E0"]]
sns.boxplot(data=df[cols_to_plot])
plt.tight_layout()
plt.savefig("imgs/boxplot_no_E0.png")
plt.close()

print("\nДоли классов:")
print(df["trajectory_type"].value_counts(normalize=True))

plt.figure(figsize=(6, 4))
sns.boxplot(
    data=df,
    x="trajectory_type",
    y="E0",
    showfliers=False
)
plt.title("Распределение начальной энергии E0 по классам")
plt.xlabel("Класс траектории")
plt.ylabel("E0")
plt.tight_layout()
plt.show()

# Считаем границы (отсекаем по 1% с каждой стороны)
lower_bound = df['E0'].quantile(0.01) # Нижняя граница (1-й перцентиль)
upper_bound = df['E0'].quantile(0.99) # Верхняя граница (99-й перцентиль)

# Фильтруем данные: оставляем только то, что больше нижней границы и меньше верхней
df_clean = df[(df['E0'] >= lower_bound) & (df['E0'] <= upper_bound)]

# Строим график по очищенным данным
plt.figure(figsize=(8, 5))
sns.histplot(
    data=df_clean,
    x='E0',
    hue="trajectory_type",
    kde=True,
    element="step",
    stat="density",
    common_norm=False,
    alpha=0.4
)
plt.title("Распределение E0 (отсечены по 1% с обеих сторон)")
plt.xlabel("E0")
plt.ylabel("Плотность")
plt.tight_layout()
plt.show()


features = df.drop(columns=["trajectory_type"]).columns.tolist()

for feature in features:

    plt.figure(figsize=(6, 4))
    sns.histplot(
        data=df,
        x=feature,
        hue="trajectory_type",
        kde=True,
        element="step",
        stat="density",
        common_norm=False,
        alpha=0.4
    )
    plt.title(f"Распределение признака {feature} по классам")
    plt.xlabel(feature)
    plt.ylabel("Плотность")
    plt.tight_layout()
    plt.show()


# 3. PREPROCESSING
feature_cols = df.drop(columns=["trajectory_type"]).columns.tolist()
X_df = df[feature_cols].copy()  # keep as DataFrame for clarity
y_sr = df["trajectory_type"].copy()

enc = LabelEncoder()
y = enc.fit_transform(y_sr)

scaler = StandardScaler()
X = scaler.fit_transform(X_df)  # scaler fitted on DataFrame -> knows feature names

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)


# 4. TRAIN MODELS + PARAM SEARCH
models = {}
cv = StratifiedKFold(3)


# 1) Logistic Regression search
logreg = LogisticRegression(max_iter=5000)
param_dist_logreg = {
    "C": np.logspace(-3, 2, 20),
    "penalty": ["l2"],
    "solver": ["lbfgs"]
}
logreg_search = RandomizedSearchCV(
    logreg,
    param_distributions=param_dist_logreg,
    n_iter=15,
    cv=cv,
    scoring="f1_macro",
    random_state=42,
    n_jobs=-1
)
logreg_search.fit(X_train, y_train)
models["logreg"] = logreg_search.best_estimator_
print("\nBest logreg params:", logreg_search.best_params_)


# 2) Random Forest search
rf = RandomForestClassifier(random_state=42)
param_dist_rf = {
    "n_estimators": [100, 150, 200, 300, 400],
    "max_depth": [3, 5, 8, 12, 15, None],
    "min_samples_split": [2, 5, 10, 20]
}
rf_search = RandomizedSearchCV(
    rf,
    param_distributions=param_dist_rf,
    n_iter=20,
    cv=cv,
    scoring="f1_macro",
    random_state=42,
    n_jobs=-1,
    verbose=1
)
rf_search.fit(X_train, y_train)
models["rf"] = rf_search.best_estimator_
print("\nBest RF params:", rf_search.best_params_)


# 3) XGBoost search
xgb = XGBClassifier(
    eval_metric="mlogloss",
    random_state=42
)
param_dist_xgb = {
    "n_estimators": [150, 300, 500],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "max_depth": [3, 5, 6, 8],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0]
}
xgb_search = RandomizedSearchCV(
    xgb,
    param_distributions=param_dist_xgb,
    n_iter=20,
    cv=cv,
    scoring="f1_macro",
    random_state=42,
    n_jobs=-1,
    verbose=1
)
xgb_search.fit(X_train, y_train)
models["xgb"] = xgb_search.best_estimator_
print("\nBest XGB params:", xgb_search.best_params_)


# 5. EVALUATION
print("\n========================")
print("=== MODEL RESULTS ===")
print("========================")

for name, model in models.items():
    print(f"\n\n======= {name.upper()} =======")
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Balanced accuracy:", balanced_accuracy_score(y_test, y_pred))
    print("F1-macro:", f1_score(y_test, y_pred, average="macro"))
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
    plt.title(f"Confusion matrix: {name}")
    plt.tight_layout()
    plt.savefig(f"imgs/cm_{name}.png")
    plt.close()

rf_model = models["rf"]

importances = rf_model.feature_importances_
importance_df = pd.DataFrame({
    "feature": feature_cols,
    "importance": importances
}).sort_values("importance", ascending=False)

print("\nFeature importances (Random Forest):")
print(importance_df)

plt.figure(figsize=(6, 4))
sns.barplot(
    data=importance_df,
    x="importance",
    y="feature"
)
plt.title("Feature importance (Random Forest)")
plt.tight_layout()
plt.show()

# 6. 2D-SLICE PREDICTIONS
print("\n=== 2D-slice predictions (x0–v0) ===")

V0_fixed = df["V0"].median()
x0_scale_fixed = df["x0_scale"].median()
gamma_fixed = df["gamma"].median()

x0_lin = np.linspace(df["x0"].min(), df["x0"].max(), 100)
v0_lin = np.linspace(df["v0"].min(), df["v0"].max(), 100)
xx, vv = np.meshgrid(x0_lin, v0_lin)

# вычисляем E0 для среза
V_x0_slice = -V0_fixed / (1.0 + xx.ravel() / x0_scale_fixed)
E0_slice = 0.5 * vv.ravel()**2 + V_x0_slice

slice_df = pd.DataFrame({
    "x0": xx.ravel(),
    "v0": vv.ravel(),
    "V0": V0_fixed,
    "x0_scale": x0_scale_fixed,
    "gamma": gamma_fixed,
    "E0": E0_slice
})

slice_df = slice_df[feature_cols]

slice_X = scaler.transform(slice_df)

class_names = enc.classes_
n_classes = len(class_names)
cmap = ListedColormap(plt.get_cmap("Set2").colors[:n_classes])
norm = BoundaryNorm(np.arange(n_classes + 1) - 0.5, cmap.N)

colors = {cls: cmap(i) for i, cls in enumerate(class_names)}

for name, model in models.items():
    y_slice_pred = model.predict(slice_X)

    plt.figure(figsize=(6, 5))

    plt.contourf(xx, vv, y_slice_pred.reshape(xx.shape), levels=np.arange(n_classes + 1) - 0.5,
                 cmap=cmap, norm=norm, alpha=0.3)

    for cls in class_names:
        mask = df["trajectory_type"] == cls
        plt.scatter(df.loc[mask, "x0"], df.loc[mask, "v0"],
                    color=colors[cls], label=cls, s=10, alpha=0.5)

    plt.xlabel("x0")
    plt.ylabel("v0")
    plt.title(
        f"{name.upper()} predictions on x0–v0 slice\n"
        f"Fixed: V0={V0_fixed:.2f}, x0_scale={x0_scale_fixed:.2f}, gamma={gamma_fixed:.2f}"
    )
    plt.legend(title="Class", loc="upper right")

    plt.tight_layout()
    plt.savefig(f"imgs/slice_x0_v0_{name}.png")
    plt.close()

print("\nГотово! Все графики — в папке imgs/")