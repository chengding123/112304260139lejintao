from __future__ import annotations

import csv
import io
import itertools
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from scipy.stats import rankdata
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parent
COMPETITION_ZIP = ROOT / "word2vec-nlp-tutorial.zip"
SUBMISSION_PATH = ROOT / "submission_multiseed_oof.csv"
REPORT_PATH = ROOT / "submission_multiseed_oof_report.md"
SEEDS = (13, 42, 2024)


def strip_html(text: str) -> str:
    return BeautifulSoup(str(text), "html.parser").get_text(" ")


def normalize_word_text(text: str) -> str:
    text = strip_html(text).lower()
    text = re.sub(r"[^a-z'\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_char_text(text: str) -> str:
    text = strip_html(text).lower()
    return re.sub(r"\s+", " ", text).strip()


def read_competition_file(file_name: str, sep: str = "\t") -> pd.DataFrame:
    with zipfile.ZipFile(COMPETITION_ZIP) as outer_zip:
        if file_name.endswith(".zip"):
            with outer_zip.open(file_name) as zipped_bytes:
                with zipfile.ZipFile(io.BytesIO(zipped_bytes.read())) as inner_zip:
                    inner_name = inner_zip.namelist()[0]
                    with inner_zip.open(inner_name) as f:
                        if file_name == "unlabeledTrainData.tsv.zip":
                            return pd.read_csv(
                                f,
                                sep=sep,
                                engine="python",
                                quoting=csv.QUOTE_MINIMAL,
                                on_bad_lines="skip",
                            )
                        return pd.read_csv(f, sep=sep, quoting=csv.QUOTE_MINIMAL)
        with outer_zip.open(file_name) as f:
            return pd.read_csv(f, sep=sep, quoting=csv.QUOTE_MINIMAL)


def build_word_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="word",
        strip_accents="unicode",
        lowercase=False,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.9,
        sublinear_tf=True,
        max_features=220000,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z']+\b",
    )


def build_char_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="char_wb",
        lowercase=False,
        ngram_range=(3, 5),
        min_df=3,
        sublinear_tf=True,
        max_features=260000,
    )


def build_nb_vectorizer() -> CountVectorizer:
    return CountVectorizer(
        analyzer="word",
        lowercase=False,
        ngram_range=(1, 2),
        min_df=3,
        max_features=220000,
        binary=True,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z']+\b",
    )


def scaled_ranks(values: np.ndarray) -> np.ndarray:
    return rankdata(values, method="average") / len(values)


def fit_nbsvm(x_train, y_train):
    y = np.asarray(y_train)
    pos = x_train[y == 1].sum(axis=0) + 1
    neg = x_train[y == 0].sum(axis=0) + 1
    ratio = np.log(np.asarray(pos / neg)).ravel()
    model = LogisticRegression(solver="liblinear", C=4.0, max_iter=1000)
    model.fit(x_train.multiply(ratio), y)
    return ratio, model


def predict_nbsvm(x, ratio, model) -> np.ndarray:
    return model.predict_proba(x.multiply(ratio))[:, 1]


def search_best_blend(predictions: dict[str, np.ndarray], y_true: np.ndarray):
    names = list(predictions)
    ranked = {name: scaled_ranks(pred) for name, pred in predictions.items()}
    steps = [i / 20 for i in range(21)]
    best_auc = -1.0
    best_weights: dict[str, float] = {}

    for weights in itertools.product(steps, repeat=len(names)):
        if abs(sum(weights) - 1.0) > 1e-9:
            continue
        blend = np.zeros(len(y_true), dtype=float)
        for name, weight in zip(names, weights):
            if weight:
                blend += weight * ranked[name]
        auc = roc_auc_score(y_true, blend)
        if auc > best_auc:
            best_auc = auc
            best_weights = dict(zip(names, weights))

    return best_auc, best_weights


def blend_predictions(predictions: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    blended = np.zeros(len(next(iter(predictions.values()))), dtype=float)
    for name, pred in predictions.items():
        blended += weights.get(name, 0.0) * scaled_ranks(pred)
    return blended


def run_seed_models(x_word, x_char, x_nb, x_test_word, x_test_char, x_test_nb, y, seed: int):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    oof_word_svc = np.zeros(len(y), dtype=float)
    oof_nbsvm = np.zeros(len(y), dtype=float)
    oof_char_lr = np.zeros(len(y), dtype=float)

    test_word_svc_folds = []
    test_nbsvm_folds = []
    test_char_lr_folds = []

    for train_idx, valid_idx in skf.split(x_word, y):
        y_train = y[train_idx]

        x_train_word = x_word[train_idx]
        x_valid_word = x_word[valid_idx]
        x_train_char = x_char[train_idx]
        x_valid_char = x_char[valid_idx]
        x_train_nb = x_nb[train_idx]
        x_valid_nb = x_nb[valid_idx]

        word_svc = LinearSVC(C=0.5)
        word_svc.fit(x_train_word, y_train)
        oof_word_svc[valid_idx] = word_svc.decision_function(x_valid_word)
        test_word_svc_folds.append(word_svc.decision_function(x_test_word))

        char_lr = LogisticRegression(solver="liblinear", C=3.0, max_iter=1000)
        char_lr.fit(x_train_char, y_train)
        oof_char_lr[valid_idx] = char_lr.predict_proba(x_valid_char)[:, 1]
        test_char_lr_folds.append(char_lr.predict_proba(x_test_char)[:, 1])

        nb_ratio, nb_model = fit_nbsvm(x_train_nb, y_train)
        oof_nbsvm[valid_idx] = predict_nbsvm(x_valid_nb, nb_ratio, nb_model)
        test_nbsvm_folds.append(predict_nbsvm(x_test_nb, nb_ratio, nb_model))

    seed_predictions = {
        f"char_lr_s{seed}": oof_char_lr,
        f"word_svc_s{seed}": oof_word_svc,
        f"nbsvm_s{seed}": oof_nbsvm,
    }
    seed_test_predictions = {
        f"char_lr_s{seed}": np.mean(test_char_lr_folds, axis=0),
        f"word_svc_s{seed}": np.mean(test_word_svc_folds, axis=0),
        f"nbsvm_s{seed}": np.mean(test_nbsvm_folds, axis=0),
    }
    return seed_predictions, seed_test_predictions


def main() -> None:
    labeled_df = read_competition_file("labeledTrainData.tsv.zip", sep="\t")
    test_df = read_competition_file("testData.tsv.zip", sep="\t")
    unlabeled_df = read_competition_file("unlabeledTrainData.tsv.zip", sep="\t")
    sample_submission = read_competition_file("sampleSubmission.csv", sep=",")

    labeled_df = labeled_df.copy()
    test_df = test_df.copy()
    unlabeled_df = unlabeled_df.copy()

    labeled_df["word_text"] = labeled_df["review"].map(normalize_word_text)
    labeled_df["char_text"] = labeled_df["review"].map(normalize_char_text)
    test_df["word_text"] = test_df["review"].map(normalize_word_text)
    test_df["char_text"] = test_df["review"].map(normalize_char_text)
    unlabeled_df["word_text"] = unlabeled_df["review"].map(normalize_word_text)
    unlabeled_df["char_text"] = unlabeled_df["review"].map(normalize_char_text)

    all_word_text = pd.concat(
        [labeled_df["word_text"], test_df["word_text"], unlabeled_df["word_text"]],
        ignore_index=True,
    )
    all_char_text = pd.concat(
        [labeled_df["char_text"], test_df["char_text"], unlabeled_df["char_text"]],
        ignore_index=True,
    )

    word_vectorizer = build_word_vectorizer()
    word_vectorizer.fit(all_word_text)
    x_word = word_vectorizer.transform(labeled_df["word_text"])
    x_test_word = word_vectorizer.transform(test_df["word_text"])

    char_vectorizer = build_char_vectorizer()
    char_vectorizer.fit(all_char_text)
    x_char = char_vectorizer.transform(labeled_df["char_text"])
    x_test_char = char_vectorizer.transform(test_df["char_text"])

    nb_vectorizer = build_nb_vectorizer()
    nb_vectorizer.fit(all_word_text)
    x_nb = nb_vectorizer.transform(labeled_df["word_text"])
    x_test_nb = nb_vectorizer.transform(test_df["word_text"])

    y = labeled_df["sentiment"].to_numpy()

    all_oof_predictions: dict[str, np.ndarray] = {}
    all_test_predictions: dict[str, np.ndarray] = {}
    seed_blend_aucs: list[float] = []

    for seed in SEEDS:
        seed_oof, seed_test = run_seed_models(
            x_word, x_char, x_nb, x_test_word, x_test_char, x_test_nb, y, seed
        )
        all_oof_predictions.update(seed_oof)
        all_test_predictions.update(seed_test)
        seed_auc, _ = search_best_blend(seed_oof, y)
        seed_blend_aucs.append(seed_auc)
        print(f"seed_{seed}_best_auc={seed_auc:.6f}")

    final_auc, final_weights = search_best_blend(all_oof_predictions, y)
    final_test = blend_predictions(all_test_predictions, final_weights)

    submission = pd.DataFrame({"id": test_df["id"], "sentiment": final_test})
    submission = submission[sample_submission.columns.tolist()]
    submission.to_csv(SUBMISSION_PATH, index=False, encoding="utf-8-sig")

    weight_text = ", ".join(
        f"{name}={weight:.2f}" for name, weight in final_weights.items() if weight > 0
    )
    REPORT_PATH.write_text(
        "\n".join(
            [
                "# Multi-seed OOF AUC 提交说明",
                "",
                f"- 使用随机种子: {', '.join(map(str, SEEDS))}",
                f"- 各种子最优 OOF AUC: {', '.join(f'{x:.6f}' for x in seed_blend_aucs)}",
                f"- 多种子总融合 OOF AUC: {final_auc:.6f}",
                f"- 最终非零权重: {weight_text}",
                "- 基模型: char LR, word LinearSVC, NB-SVM",
                "- 无标签数据用途: 参与向量器词表拟合，但不参与监督训练。",
                "- 提交文件: submission_multiseed_oof.csv",
                "- 注意: 为适配 AUC，提交值使用融合后的排序分数，而不是硬标签。",
            ]
        ),
        encoding="utf-8",
    )

    print(f"multiseed_oof_auc={final_auc:.6f}")
    print(f"weights={final_weights}")
    print(f"saved: {SUBMISSION_PATH}")
    print(f"saved: {REPORT_PATH}")


if __name__ == "__main__":
    main()
