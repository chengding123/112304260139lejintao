from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup
from scipy.stats import rankdata
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent.parent
COMPETITION_ZIP = ROOT / "data" / "raw" / "word2vec-nlp-tutorial.zip"
SUBMISSION_PATH = ROOT / "submissions" / "submission_highscore.csv"
REPORT_PATH = ROOT / "docs" / "reports" / "submission_highscore_report.md"


def strip_html(text: str) -> str:
    return BeautifulSoup(str(text), "html.parser").get_text(" ")


def normalize_word_text(text: str) -> str:
    text = strip_html(text)
    text = text.lower()
    text = re.sub(r"[^a-z'\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_char_text(text: str) -> str:
    text = strip_html(text)
    text = text.lower()
    return re.sub(r"\s+", " ", text).strip()


def read_competition_file(file_name: str, sep: str = "\t") -> pd.DataFrame:
    with zipfile.ZipFile(COMPETITION_ZIP) as outer_zip:
        if file_name.endswith(".zip"):
            with outer_zip.open(file_name) as zipped_bytes:
                with zipfile.ZipFile(io.BytesIO(zipped_bytes.read())) as inner_zip:
                    inner_name = inner_zip.namelist()[0]
                    with inner_zip.open(inner_name) as f:
                        return pd.read_csv(f, sep=sep)

        with outer_zip.open(file_name) as f:
            return pd.read_csv(f, sep=sep)


def build_word_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="word",
        strip_accents="unicode",
        lowercase=False,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.9,
        sublinear_tf=True,
        max_features=180000,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z']+\b",
    )


def build_char_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="char_wb",
        lowercase=False,
        ngram_range=(3, 5),
        min_df=3,
        sublinear_tf=True,
        max_features=220000,
    )


def scaled_ranks(values) -> pd.Series:
    ranks = rankdata(values, method="average")
    return pd.Series(ranks / len(ranks))


def main() -> None:
    labeled_df = read_competition_file("labeledTrainData.tsv.zip", sep="\t")
    test_df = read_competition_file("testData.tsv.zip", sep="\t")
    sample_submission = read_competition_file("sampleSubmission.csv", sep=",")

    labeled_df = labeled_df.copy()
    test_df = test_df.copy()

    labeled_df["word_text"] = labeled_df["review"].map(normalize_word_text)
    labeled_df["char_text"] = labeled_df["review"].map(normalize_char_text)
    test_df["word_text"] = test_df["review"].map(normalize_word_text)
    test_df["char_text"] = test_df["review"].map(normalize_char_text)

    train_df, valid_df = train_test_split(
        labeled_df,
        test_size=0.2,
        random_state=42,
        stratify=labeled_df["sentiment"],
    )

    y_train = train_df["sentiment"]
    y_valid = valid_df["sentiment"]

    word_vectorizer = build_word_vectorizer()
    x_train_word = word_vectorizer.fit_transform(train_df["word_text"])
    x_valid_word = word_vectorizer.transform(valid_df["word_text"])

    char_vectorizer = build_char_vectorizer()
    x_train_char = char_vectorizer.fit_transform(train_df["char_text"])
    x_valid_char = char_vectorizer.transform(valid_df["char_text"])

    word_lr = LogisticRegression(
        solver="liblinear",
        C=4.0,
        max_iter=1000,
    )
    word_lr.fit(x_train_word, y_train)
    valid_word_lr = word_lr.predict_proba(x_valid_word)[:, 1]
    auc_word_lr = roc_auc_score(y_valid, valid_word_lr)

    char_lr = LogisticRegression(
        solver="liblinear",
        C=3.0,
        max_iter=1000,
    )
    char_lr.fit(x_train_char, y_train)
    valid_char_lr = char_lr.predict_proba(x_valid_char)[:, 1]
    auc_char_lr = roc_auc_score(y_valid, valid_char_lr)

    word_sgd = SGDClassifier(
        loss="modified_huber",
        alpha=1e-5,
        max_iter=2000,
        tol=1e-3,
        random_state=42,
    )
    word_sgd.fit(x_train_word, y_train)
    valid_word_sgd = word_sgd.decision_function(x_valid_word)
    auc_word_sgd = roc_auc_score(y_valid, valid_word_sgd)

    blend_valid = (
        0.45 * scaled_ranks(valid_word_lr).to_numpy()
        + 0.35 * scaled_ranks(valid_char_lr).to_numpy()
        + 0.20 * scaled_ranks(valid_word_sgd).to_numpy()
    )
    auc_blend = roc_auc_score(y_valid, blend_valid)

    full_word_vectorizer = build_word_vectorizer()
    x_full_word = full_word_vectorizer.fit_transform(labeled_df["word_text"])
    x_test_word = full_word_vectorizer.transform(test_df["word_text"])

    full_char_vectorizer = build_char_vectorizer()
    x_full_char = full_char_vectorizer.fit_transform(labeled_df["char_text"])
    x_test_char = full_char_vectorizer.transform(test_df["char_text"])

    full_word_lr = LogisticRegression(
        solver="liblinear",
        C=4.0,
        max_iter=1000,
    )
    full_word_lr.fit(x_full_word, labeled_df["sentiment"])
    test_word_lr = full_word_lr.predict_proba(x_test_word)[:, 1]

    full_char_lr = LogisticRegression(
        solver="liblinear",
        C=3.0,
        max_iter=1000,
    )
    full_char_lr.fit(x_full_char, labeled_df["sentiment"])
    test_char_lr = full_char_lr.predict_proba(x_test_char)[:, 1]

    full_word_sgd = SGDClassifier(
        loss="modified_huber",
        alpha=1e-5,
        max_iter=2000,
        tol=1e-3,
        random_state=42,
    )
    full_word_sgd.fit(x_full_word, labeled_df["sentiment"])
    test_word_sgd = full_word_sgd.decision_function(x_test_word)

    blend_test = (
        0.45 * scaled_ranks(test_word_lr).to_numpy()
        + 0.35 * scaled_ranks(test_char_lr).to_numpy()
        + 0.20 * scaled_ranks(test_word_sgd).to_numpy()
    )

    submission = pd.DataFrame(
        {
            "id": test_df["id"],
            "sentiment": blend_test,
        }
    )
    submission = submission[sample_submission.columns.tolist()]
    submission.to_csv(SUBMISSION_PATH, index=False, encoding="utf-8-sig")

    REPORT_PATH.write_text(
        "\n".join(
            [
                "# High-score AUC 提交说明",
                "",
                f"- 词级 LR 验证 AUC: {auc_word_lr:.6f}",
                f"- 字符级 LR 验证 AUC: {auc_char_lr:.6f}",
                f"- 词级 SGD 验证 AUC: {auc_word_sgd:.6f}",
                f"- 融合后验证 AUC: {auc_blend:.6f}",
                "- 融合权重: word_lr=0.45, char_lr=0.35, word_sgd=0.20",
                "- 提交文件: submission_highscore.csv",
                "- 注意: 为适配 AUC，提交值使用融合后的排序分数，而不是硬标签。",
            ]
        ),
        encoding="utf-8",
    )

    print(f"auc_word_lr={auc_word_lr:.6f}")
    print(f"auc_char_lr={auc_char_lr:.6f}")
    print(f"auc_word_sgd={auc_word_sgd:.6f}")
    print(f"auc_blend={auc_blend:.6f}")
    print(f"saved: {SUBMISSION_PATH}")
    print(f"saved: {REPORT_PATH}")


if __name__ == "__main__":
    main()
