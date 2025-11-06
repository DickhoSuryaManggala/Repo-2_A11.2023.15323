#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import joblib

def main(csv_path, target, test_size=0.33, random_state=42,
         max_depth=None, min_leaf=1, criterion="entropy",
         outname=None, label_map=None, ccp_alpha=0.0):
    # 1) Load
    df = pd.read_csv(csv_path)
    if target not in df.columns:
        raise ValueError(f"Kolom target '{target}' tidak ada. Kolom: {list(df.columns)}")

    # 2) Target & fitur
    y = df[target].copy()
    if label_map is not None:
        y = y.map(label_map)
        if y.isna().any():
            bad = df[target].unique()
            raise ValueError(f"Mapping label menghasilkan NaN. Cek nilai target: {bad}")
    X = df.drop(columns=[target])

    # 3) Deteksi tipe kolom
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    # 4) Preprocess
    pre = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), num_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_cols),
    ], remainder="drop")

    # 5) Model
    clf = DecisionTreeClassifier(
        criterion=criterion,
        class_weight="balanced",
        max_depth=max_depth,
        min_samples_leaf=min_leaf,
        ccp_alpha=ccp_alpha,
        random_state=random_state
    )
    model = Pipeline([("prep", pre), ("clf", clf)])

    # 6) Split & train
    strat = y if pd.Series(y).nunique() > 1 else None
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, stratify=strat, random_state=random_state)
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)

    # 7) Metrik
    acc = float(accuracy_score(yte, pred))
    report = classification_report(yte, pred, digits=4)
    cm = confusion_matrix(yte, pred)

    # 8) Output
    base = Path(csv_path).stem if outname is None else outname
    outdir = Path(f"outputs_{base}"); outdir.mkdir(exist_ok=True)

    with open(outdir / "metrics.json", "w") as f:
        json.dump({"accuracy": acc}, f, indent=2)
    with open(outdir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)
    pd.DataFrame(cm).to_csv(outdir / "confusion_matrix.csv", index=False)

    # 9) Plot pohon
    clf_est = model.named_steps["clf"]
    ohe = model.named_steps["prep"].named_transformers_.get("cat", None)
    cat_feature_names = ohe.named_steps["ohe"].get_feature_names_out(cat_cols) if ohe else []
    feature_names = list(num_cols) + list(cat_feature_names)

    plt.figure(figsize=(22, 12))
    plot_tree(clf_est, filled=True, rounded=True, fontsize=8, feature_names=feature_names)
    plt.tight_layout(); plt.savefig(outdir / "tree.png", dpi=200); plt.close()

    joblib.dump(model, outdir / "model.pkl")

    print("Train samples :", len(Xtr))
    print("Test samples  :", len(Xte))
    print("Akurasi       :", round(acc, 4))
    print("Keluaran ->", outdir.resolve())
    for f in ["metrics.json", "classification_report.txt", "confusion_matrix.csv", "tree.png", "model.pkl"]:
        print("-", f)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--test_size", type=float, default=0.33)
    p.add_argument("--random_state", type=int, default=42)
    p.add_argument("--max_depth", type=int, default=None)
    p.add_argument("--min_leaf", type=int, default=1)
    p.add_argument("--criterion", type=str, default="entropy", choices=["gini","entropy","log_loss"])
    p.add_argument("--outname", type=str, default=None)
    p.add_argument("--map", type=str, default=None, help="contoh: e:0,p:1  atau yes:1,no:0")
    p.add_argument("--ccp_alpha", type=float, default=0.0, help="cost-complexity pruning")
    args = p.parse_args()

    label_map = None
    if args.map:
        items = [kv.strip() for kv in args.map.split(",") if kv.strip()]
        label_map = {k.strip(): int(v.strip()) for k,v in (it.split(":") for it in items)}

    main(args.csv, args.target, args.test_size, args.random_state,
         args.max_depth, args.min_leaf, args.criterion,
         args.outname, label_map, args.ccp_alpha)
