
import re
import pickle
import warnings
from pathlib import Path
 
import numpy as np
import pandas as pd
 
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (StratifiedKFold, StratifiedGroupKFold,
                                     GridSearchCV)
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             roc_auc_score, brier_score_loss)
 
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, regularizers
 
 
FEAT_DIR = Path("data/features")      
CLIN_CSV = Path("data/clinical.csv")  
 
SF          = 250.0                       
WINDOW      = 25_000                      
SCALES      = [2**k for k in range(4, 12)]
QS          = np.linspace(-20, 20, 8)
N_HQ        = len(QS)
ELECTRODES  = ["FZ", "CZ", "PZ", "T3", "T4", "T5", "T6"]
 
TARGET_MINUTES  = [5, 10, 20, 30, 40, 60]
PRIMARY_MINUTES = 30
 
SEED     = 42
N_SPLITS = 5
N_BOOT   = 2000
N_PERM   = 500
ALPHA    = 0.05
MODELS   = ["Logistic Reg.", "Random Forest", "RBF-SVM", "DNN"]
 
CLIN_COLS = {"subject_id": "subject_id", "age": "age", "gcs": "gcs",
             "sex": "sex", "label": "late_pts", "early_pts": "early_pts"}
 
def seed_all(seed=SEED):
    np.random.seed(seed)
    tf.random.set_seed(seed)
 
 
PKL_RE = re.compile(r"^(\d+_\d+_\d+)_(\d+)_hq_(\d+)min$")
 
 
def load_clinical():
    """One row per patient: age, GCS, sex, early-PTS status, late-PTS outcome."""
    df = pd.read_csv(CLIN_CSV).rename(columns=str.strip)
    c = CLIN_COLS
    out = pd.DataFrame(index=df[c["subject_id"]].astype(str).str.strip())
    out["age"]   = pd.to_numeric(df[c["age"]], errors="coerce").values
    out["gcs"]   = pd.to_numeric(df[c["gcs"]], errors="coerce").values
    out["label"] = pd.to_numeric(df[c["label"]], errors="coerce").values
 
    sex = df[c["sex"]].astype(str).str.strip().str.lower()
    out["sex"] = sex.map(lambda v: 1.0 if v.startswith("m")
                         else (0.0 if v.startswith("f") else np.nan)).values
    out["early_pts"] = pd.to_numeric(df[c["early_pts"]], errors="coerce").values
 
    out = out[~out.index.duplicated(keep="first")]
    out = out[out["label"].notna()]
    out["label"] = out["label"].astype(int)
    return out
 
 
def load_features(minutes):
    """sid -> h(q) dict {electrode: 8-vector}, for the earliest recording day."""
    best = {}
    for f in FEAT_DIR.iterdir():
        m = PKL_RE.match(f.name[:-4]) if f.name.endswith(".pkl") else None
        if not m or int(m.group(3)) != minutes:
            continue
        sid, day = m.group(1), int(m.group(2))
        if sid not in best or day < best[sid][0]:
            best[sid] = (day, f)
 
    out = {}
    for sid, (_, fp) in best.items():
        with open(fp, "rb") as fh:
            obj = pickle.load(fh)
        out[sid] = obj["hq"] if isinstance(obj, dict) and "hq" in obj else obj
    return out
 
 
def cohort(minutes, clin, feats):
    """Patients with both a feature set at this duration and clinical data."""
    return sorted([s for s in feats
                   if s in clin.index and _valid_electrodes(feats[s])])
 
 
def _valid_electrodes(hq):
    return [np.asarray(v).ravel() for v in hq.values()
            if np.asarray(v).ravel().size == N_HQ
            and np.isfinite(np.asarray(v).ravel()).all()]
 
 
def _covars(clin, sid, which):
    v = []
    if "clinical" in which:
        v += [clin.at[sid, "age"], clin.at[sid, "gcs"], clin.at[sid, "sex"]]
    if "early" in which:
        v += [clin.at[sid, "early_pts"]]
    return v
 
 
def build_A(clin, sids, feats, which="none"):
    """Option A: h(q) averaged across a patient's electrodes. One row/patient."""
    X, y, used = [], [], []
    for sid in sids:
        vs = _valid_electrodes(feats[sid])
        if not vs:
            continue
        X.append(np.hstack([np.mean(vs, axis=0)] + _covars(clin, sid, which)))
        y.append(clin.at[sid, "label"]); used.append(sid)
    return np.vstack(X).astype(float), np.asarray(y, int), used
 
 
def build_B(clin, sids, feats, which="none"):
    """Option B: one row per electrode; groups carry patient identity."""
    X, y, groups = [], [], []
    for sid in sids:
        for v in feats[sid].values():
            v = np.asarray(v).ravel()
            if v.size == N_HQ and np.isfinite(v).all():
                X.append(np.hstack([v] + _covars(clin, sid, which)))
                y.append(clin.at[sid, "label"]); groups.append(sid)
    return np.vstack(X).astype(float), np.asarray(y, int), np.asarray(groups)
 
 
def build_C(clin, sids, feats, which="none"):
    """Option C: per-electrode h(q) blocks concatenated. One row/patient."""
    chs = sorted({c for s in sids for c in feats[s]},
                 key=lambda c: ELECTRODES.index(c) if c in ELECTRODES else 99)
    X, y, used = [], [], []
    for sid in sids:
        hq = feats[sid]
        blocks = []
        for ch in chs:
            v = np.asarray(hq[ch]).ravel() if ch in hq else None
            blocks.append(v if (v is not None and v.size == N_HQ
                                and np.isfinite(v).all())
                          else np.full(N_HQ, np.nan))
        if np.isnan(np.hstack(blocks)).all():
            continue
        X.append(np.hstack(blocks + [np.asarray(_covars(clin, sid, which), float)]))
        y.append(clin.at[sid, "label"]); used.append(sid)
    return np.vstack(X).astype(float), np.asarray(y, int), used
 
 
def build_covars_only(clin, sids, which):
    X = np.vstack([np.asarray(_covars(clin, s, which), float) for s in sids])
    return X, np.asarray([clin.at[s, "label"] for s in sids], int)
 
 
SVM_GRID = {"svm__C":     [2.0**k for k in range(-3, 10, 2)],
            "svm__gamma": [2.0**k for k in range(-15, 4, 3)]}
 
 
def make_model(name):
    if name == "Logistic Reg.":
        return make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(),
            LogisticRegression(penalty="l2", C=1.0, solver="liblinear",
                               class_weight="balanced", max_iter=5000,
                               random_state=SEED))
    if name == "Random Forest":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                   max_features="sqrt", class_weight="balanced",
                                   random_state=SEED, n_jobs=-1))
    if name == "RBF-SVM":
        return Pipeline([("impute", SimpleImputer(strategy="median")),
                         ("scale",  StandardScaler()),
                         ("svm",    SVC(kernel="rbf", class_weight="balanced",
                                        random_state=SEED))])
    raise ValueError(name)
 
 
def make_dnn(n_features):
    l2 = regularizers.l2(1e-4)
    m = models.Sequential([
        layers.Input((n_features,)),
        layers.Dense(128, activation="relu", kernel_regularizer=l2),
        layers.Dropout(0.30),
        layers.Dense(64,  activation="relu", kernel_regularizer=l2),
        layers.Dropout(0.30),
        layers.Dense(32,  activation="relu", kernel_regularizer=l2),
        layers.Dense(1,   activation="sigmoid")])
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss="binary_crossentropy", metrics=["accuracy"])
    return m
 
 
def to_proba(s):
    s = np.asarray(s, float)
    return s if (np.nanmin(s) >= 0 and np.nanmax(s) <= 1) else 1 / (1 + np.exp(-s))
 
 
def oof(model_name, X, y, groups=None, seed=SEED):
    """
    Pooled out-of-fold predictions.
 
    Options A and C use stratified 5-fold: each patient is one row, so folds are
    patient-disjoint by construction. Option B passes `groups`, which selects
    StratifiedGroupKFold so that no patient's electrodes span train and test.
    All preprocessing and SVM tuning happen inside the training fold.
    """
    if groups is None:
        splits = list(StratifiedKFold(N_SPLITS, shuffle=True,
                                      random_state=seed).split(X, y))
        inner = StratifiedKFold(5, shuffle=True, random_state=seed + 1)
    else:
        splits = list(StratifiedGroupKFold(N_SPLITS).split(X, y, groups=groups))
        inner = StratifiedGroupKFold(5)
 
    out = np.full(len(y), np.nan)
    for tr, te in splits:
        if groups is not None:
            assert not (set(groups[tr]) & set(groups[te])), "patient in both folds"
 
        if model_name == "DNN":
            seed_all()
            imp, sc = SimpleImputer(strategy="median"), StandardScaler()
            Xtr = sc.fit_transform(imp.fit_transform(X[tr]))
            Xte = sc.transform(imp.transform(X[te]))
            net = make_dnn(Xtr.shape[1])   # width after imputation drops NaN cols
            es = callbacks.EarlyStopping(monitor="val_loss", patience=15,
                                         restore_best_weights=True, verbose=0)
            net.fit(Xtr, y[tr], epochs=200, batch_size=16,
                    validation_split=0.15, callbacks=[es], verbose=0)
            out[te] = net.predict(Xte, verbose=0).ravel()
 
        elif model_name == "RBF-SVM":
            gs = GridSearchCV(make_model("RBF-SVM"), SVM_GRID, scoring="roc_auc",
                              cv=inner, n_jobs=-1, refit=True)
            gs.fit(X[tr], y[tr],
                   **({"groups": groups[tr]} if groups is not None else {}))
            out[te] = gs.best_estimator_.decision_function(X[te])
 
        else:
            est = make_model(model_name).fit(X[tr], y[tr])
            out[te] = (est.predict_proba(X[te])[:, 1]
                       if hasattr(est, "predict_proba")
                       else est.decision_function(X[te]))
    return out
 
 
def ci_metrics(y, scores, groups=None, n_boot=N_BOOT, seed=SEED):
    """
    Point estimates with bootstrap 95% CIs.
 
    Without `groups`, patients are resampled directly. With `groups` (Option B,
    where rows are electrodes), a cluster bootstrap resamples patients and takes
    all of each sampled patient's rows, since rows within a patient are not
    independent.
    """
    y = np.asarray(y); scores = np.asarray(scores, float)
    p = to_proba(scores); pred = (p >= 0.5).astype(int)
    point = dict(auc=roc_auc_score(y, scores), acc=accuracy_score(y, pred),
                 prec=precision_score(y, pred, zero_division=0),
                 sens=recall_score(y, pred), brier=brier_score_loss(y, p))
 
    rng = np.random.default_rng(seed)
    if groups is None:
        draw = lambda: rng.integers(0, len(y), len(y))
    else:
        rows_of = {s: np.where(groups == s)[0] for s in np.unique(groups)}
        pats = np.array(list(rows_of))
        draw = lambda: np.concatenate(
            [rows_of[s] for s in rng.choice(pats, len(pats), replace=True)])
 
    boot = {k: [] for k in point}
    for _ in range(n_boot):
        i = draw()
        if len(np.unique(y[i])) < 2:
            continue
        boot["auc"].append(roc_auc_score(y[i], scores[i]))
        boot["acc"].append(accuracy_score(y[i], pred[i]))
        boot["prec"].append(precision_score(y[i], pred[i], zero_division=0))
        boot["sens"].append(recall_score(y[i], pred[i]))
        boot["brier"].append(brier_score_loss(y[i], p[i]))
    return {k: (point[k], *np.percentile(v, [100*ALPHA/2, 100*(1-ALPHA/2)]))
            for k, v in boot.items()}
 
 
def delta_auc(y, s1, s2, n_boot=N_BOOT, seed=SEED):
    """Paired bootstrap: AUC(s1) - AUC(s2) on identically resampled patients."""
    y = np.asarray(y)
    d0 = roc_auc_score(y, s1) - roc_auc_score(y, s2)
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        d.append(roc_auc_score(y[i], np.asarray(s1)[i])
                 - roc_auc_score(y[i], np.asarray(s2)[i]))
    return (d0, *np.percentile(d, [100*ALPHA/2, 100*(1-ALPHA/2)]))
 
 
def permutation_test(model_name, X, y, n_perm=N_PERM, seed=SEED):
    """Shuffle labels, re-run the entire CV including preprocessing and tuning."""
    real = roc_auc_score(y, oof(model_name, X, y))
    rng = np.random.default_rng(seed)
    null = np.array([roc_auc_score(yp, oof(model_name, X, yp))
                     for yp in (rng.permutation(y) for _ in range(n_perm))])
    return real, null.mean(), null.std(), (np.sum(null >= real) + 1) / (n_perm + 1)
 
 
def fmt(m, k):
    v, lo, hi = m[k]
    return f"{v:.3f} [{lo:.3f}-{hi:.3f}]"
 
 
def table3(clin, minutes=PRIMARY_MINUTES):
    """Logistic regression under the three representations, h(q) alone."""
    feats = load_features(minutes)
    sids  = cohort(minutes, clin, feats)
 
    XA, yA, _  = build_A(clin, sids, feats)
    XB, yB, gB = build_B(clin, sids, feats)
    XC, yC, _  = build_C(clin, sids, feats)
 
    print(f"\n{'='*70}\nTABLE 3 — {minutes} min, n = {len(yA)} patients\n{'='*70}")
    rows = []
    for tag, X, y, g in [("A", XA, yA, None), ("B", XB, yB, gB),
                         ("C", XC, yC, None)]:
        sc = oof("Logistic Reg.", X, y, g)
        m  = ci_metrics(y, sc, g)
        rows.append(dict(option=tag, n=len(y), **{k: round(m[k][0], 3)
                                                  for k in m}))
        print(f"  Option {tag}  AUC {fmt(m,'auc')}  Acc {m['acc'][0]:.3f}  "
              f"Prec {m['prec'][0]:.3f}  Sens {m['sens'][0]:.3f}  "
              f"Brier {m['brier'][0]:.3f}")
    return pd.DataFrame(rows)
 
 
def table4(clin):
    """AUC across recording lengths, Option A, all four classifiers."""
    print(f"\n{'='*70}\nTABLE 4 — AUC by recording length\n{'='*70}")
    rows = []
    for minutes in TARGET_MINUTES:
        feats = load_features(minutes)
        sids  = cohort(minutes, clin, feats)
        if len(sids) < 20:
            continue
        X, y, _ = build_A(clin, sids, feats)
        for mdl in MODELS:
            auc = roc_auc_score(y, oof(mdl, X, y))
            rows.append(dict(window=minutes, n=len(y), model=mdl,
                             AUC=round(auc, 3)))
            print(f"  {minutes:>2} min  n={len(y):<3} {mdl:<15} AUC {auc:.3f}")
    return pd.DataFrame(rows)
 
 
def table5(clin, minutes=PRIMARY_MINUTES):
    """Baselines and nested models, all on identical patients and folds."""
    feats = load_features(minutes)
    sids  = cohort(minutes, clin, feats)
 
    Xh,  y, used = build_A(clin, sids, feats, "none")
    Xhc, _, _    = build_A(clin, used, feats, "clinical")
    Xhe, _, _    = build_A(clin, used, feats, "early")
    Xc,  _       = build_covars_only(clin, used, "clinical")
    Xe,  _       = build_covars_only(clin, used, "early")
 
    print(f"\n{'='*70}\nTABLE 5 — {minutes} min, n = {len(y)} patients\n{'='*70}")
    scores, rows = {}, []
    for name, X in [("h(q) only", Xh), ("h(q) + clinical", Xhc),
                    ("h(q) + early PTS", Xhe),
                    ("Clinical (age+GCS+sex)", Xc), ("Early PTS only", Xe)]:
        sc = oof("Logistic Reg.", X, y)
        scores[name] = sc
        m = ci_metrics(y, sc)
        rows.append(dict(model=name, n=len(y), AUC=round(m["auc"][0], 3),
                         AUC_lo=round(m["auc"][1], 3),
                         AUC_hi=round(m["auc"][2], 3),
                         Brier=round(m["brier"][0], 3)))
        print(f"  {name:<26} AUC {fmt(m,'auc')}  Brier {fmt(m,'brier')}")
 
    print(f"\n  Paired dAUC vs h(q) alone (* = CI excludes 0):")
    for name, sc in scores.items():
        if name == "h(q) only":
            continue
        d, lo, hi = delta_auc(y, sc, scores["h(q) only"])
        print(f"    {name:<26} {d:+.3f} [{lo:+.3f}, {hi:+.3f}]"
              + ("" if lo <= 0 <= hi else "  *"))
 
    d, lo, hi = delta_auc(y, scores["h(q) + early PTS"], scores["Early PTS only"])
    print(f"\n  h(q) + early PTS vs early PTS alone: "
          f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    return pd.DataFrame(rows), scores, y
 
 
def subgroup_no_early_pts(clin, minutes=PRIMARY_MINUTES):
    """Sensitivity analysis: patients who never had an early seizure."""
    feats = load_features(minutes)
    sids  = [s for s in cohort(minutes, clin, feats)
             if clin.at[s, "early_pts"] == 0]
    X, y, _ = build_A(clin, sids, feats)
    m = ci_metrics(y, oof("Logistic Reg.", X, y))
    print(f"\n{'='*70}\nSUBGROUP — no early PTS\n{'='*70}")
    print(f"  n = {len(y)} ({int(y.sum())} late PTS)   AUC {fmt(m,'auc')}")
    return m
 
 
def main():
    seed_all()
    clin = load_clinical()
    print(f"[clinical] n = {len(clin)}  late PTS = {int(clin['label'].sum())}")
 
    table3(clin)
    table4(clin)
    table5(clin)
    subgroup_no_early_pts(clin)
 
    feats = load_features(PRIMARY_MINUTES)
    sids  = cohort(PRIMARY_MINUTES, clin, feats)
    X, y, _ = build_A(clin, sids, feats)
    print(f"\n{'='*70}\nPERMUTATION TEST — {N_PERM} shuffles\n{'='*70}")
    real, mu, sd, p = permutation_test("Logistic Reg.", X, y)
    print(f"  observed {real:.3f} | null {mu:.3f} +/- {sd:.3f} | p = {p:.4f}")
 
 
if __name__ == "__main__":
    main()
