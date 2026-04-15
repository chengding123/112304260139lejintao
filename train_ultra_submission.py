from __future__ import annotations

import io
import itertools
import re
import zipfile
import csv
from pathlib import Path

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from scipy.stats import rankdata
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parent
COMPETITION_ZIP = ROOT / "word2vec-nlp-tutorial.zip"
SUBMISSION_PATH = ROOT / "submission_ultra.csv"
REPORT_PATH = ROOT / "submission_ultra_report.md"


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
                        return pd.read_csv(
                            f,
                            sep=sep,
                            quoting=csv.QUOTE_MINIMAL,
                        )

        with outer_zip.open(file_name) as f:
            return pd.read_csv(
                f,
                sep=sep,
                quoting=csv.QUOTE_MINIMAL,
            )


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
    ranks = rankdata(values, method="average")
    return ranks / len(ranks)


def fit_nbsvm(x_train, y_train):
    y = np.asarray(y_train)
    pos = x_train[y == 1].sum(axis=0) + 1
    neg = x_train[y == 0].sum(axis=0) + 1
    ratio = np.log(np.asarray(pos / neg)).ravel()

    model = LogisticRegression(
        solver="liblinear",
        C=4.0,
        max_iter=1000,
    )
    model.fit(x_train.multiply(ratio), y)
    return ratio, model


def predict_nbsvm(x, ratio, model) -> np.ndarray:
    return model.predict_proba(x.multiply(ratio))[:, 1]


def search_best_blend(predictions: dict[str, np.ndarray], y_true: pd.Series):
    names = list(predictions)
    ranked = {name: scaled_ranks(pred) for name, pred in predictions.items()}

    best_auc = -1.0
    best_weights: dict[str, float] = {}
    steps = [i / 20 for i in range(21)]

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

    train_df, valid_df = train_test_split(
        labeled_df,
        test_size=0.2,
        random_state=42,
        stratify=labeled_df["sentiment"],
    )

    y_train = train_df["sentiment"]
    y_valid = valid_df["sentiment"]

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
    x_train_word = word_vectorizer.transform(train_df["word_text"])
    x_valid_word = word_vectorizer.transform(valid_df["word_text"])

    char_vectorizer = build_char_vectorizer()
    char_vectorizer.fit(all_char_text)
    x_train_char = char_vectorizer.transform(train_df["char_text"])
    x_valid_char = char_vectorizer.transform(valid_df["char_text"])

    nb_vectorizer = build_nb_vectorizer()
    nb_vectorizer.fit(all_word_text)
    x_train_nb = nb_vectorizer.transform(train_df["word_text"])
    x_valid_nb = nb_vectorizer.transform(valid_df["word_text"])

    word_lr = LogisticRegression(solver="liblinear", C=4.0, max_iter=1000)
    word_lr.fit(x_train_word, y_train)
    valid_word_lr = word_lr.predict_proba(x_valid_word)[:, 1]
    auc_word_lr = roc_auc_score(y_valid, valid_word_lr)

    char_lr = LogisticRegression(solver="liblinear", C=3.0, max_iter=1000)
    char_lr.fit(x_train_char, y_train)
    valid_char_lr = char_lr.predict_proba(x_valid_char)[:, 1]
    auc_char_lr = roc_auc_score(y_valid, valid_char_lr)

    word_svc = LinearSVC(C=0.5)
    word_svc.fit(x_train_word, y_train)
    valid_word_svc = word_svc.decision_function(x_valid_word)
    auc_word_svc = roc_auc_score(y_valid, valid_word_svc)

    nb_ratio, nbsvm_model = fit_nbsvm(x_train_nb, y_train)
    valid_nbsvm = predict_nbsvm(x_valid_nb, nb_ratio, nbsvm_model)
    auc_nbsvm = roc_auc_score(y_valid, valid_nbsvm)

    valid_predictions = {
        "word_lr": valid_word_lr,
        "char_lr": valid_char_lr,
        "word_svc": valid_word_svc,
        "nbsvm": valid_nbsvm,
    }
    auc_blend, best_weights = search_best_blend(valid_predictions, y_valid)

    x_full_word = word_vectorizer.transform(labeled_df["word_text"])
    x_test_word = word_vectorizer.transform(test_df["word_text"])

    x_full_char = char_vectorizer.transform(labeled_df["char_text"])
    x_test_char = char_vectorizer.transform(test_df["char_text"])

    x_full_nb = nb_vectorizer.transform(labeled_df["word_text"])
    x_test_nb = nb_vectorizer.transform(test_df["word_text"])

    full_word_lr = LogisticRegression(solver="liblinear", C=4.0, max_iter=1000)
    full_word_lr.fit(x_full_word, labeled_df["sentiment"])
    test_word_lr = full_word_lr.predict_proba(x_test_word)[:, 1]

    full_char_lr = LogisticRegression(solver="liblinear", C=3.0, max_iter=1000)
    full_char_lr.fit(x_full_char, labeled_df["sentiment"])
    test_char_lr = full_char_lr.predict_proba(x_test_char)[:, 1]

    full_word_svc = LinearSVC(C=0.5)
    full_word_svc.fit(x_full_word, labeled_df["sentiment"])
    test_word_svc = full_word_svc.decision_function(x_test_word)

    full_nb_ratio, full_nbsvm_model = fit_nbsvm(x_full_nb, labeled_df["sentiment"])
    test_nbsvm = predict_nbsvm(x_test_nb, full_nb_ratio, full_nbsvm_model)

    test_predictions = {
        "word_lr": test_word_lr,
        "char_lr": test_char_lr,
        "word_svc": test_word_svc,
        "nbsvm": test_nbsvm,
    }
    blend_test = blend_predictions(test_predictions, best_weights)

    submission = pd.DataFrame({"id": test_df["id"], "sentiment": blend_test})
    submission = submission[sample_submission.columns.tolist()]
    submission.to_csv(SUBMISSION_PATH, index=False, encoding="utf-8-sig")

    weight_text = ", ".join(
        f"{name}={weight:.2f}" for name, weight in best_weights.items() if weight > 0
    )
    REPORT_PATH.write_text(
        "\n".join(
            [
                "# Ultra AUC 提交说明",
                "",
                f"- 词级 LR 验证 AUC: {auc_word_lr:.6f}",
                f"- 字符级 LR 验证 AUC: {auc_char_lr:.6f}",
                f"- 词级 LinearSVC 验证 AUC: {auc_word_svc:.6f}",
                f"- NB-SVM 验证 AUC: {auc_nbsvm:.6f}",
                f"- 最优融合验证 AUC: {auc_blend:.6f}",
                f"- 最优融合权重: {weight_text}",
                "- 无标签数据用途: 参与向量器词表拟合，但不参与监督训练。",
                "- 提交文件: submission_ultra.csv",
                "- 注意: 为适配 AUC，提交值使用融合后的排序分数，而不是硬标签。",
            ]
        ),
        encoding="utf-8",
    )

    print(f"auc_word_lr={auc_word_lr:.6f}")
    print(f"auc_char_lr={auc_char_lr:.6f}")
    print(f"auc_word_svc={auc_word_svc:.6f}")
    print(f"auc_nbsvm={auc_nbsvm:.6f}")
    print(f"auc_blend={auc_blend:.6f}")
    print(f"weights={best_weights}")
    print(f"saved: {SUBMISSION_PATH}")
    print(f"saved: {REPORT_PATH}")


if __name__ == "__main__":
    main()
