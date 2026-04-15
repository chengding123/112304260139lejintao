from __future__ import annotations

import csv
import io
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent
COMPETITION_ZIP = ROOT / "word2vec-nlp-tutorial.zip"
SUBMISSION_PATH = ROOT / "submission_word2vec_avg.csv"
REPORT_PATH = ROOT / "submission_word2vec_avg_report.md"


def strip_html(text: str) -> str:
    return BeautifulSoup(str(text), "html.parser").get_text(" ")


def clean_and_tokenize(text: str) -> list[str]:
    text = strip_html(text).lower()
    text = re.sub(r"[^a-z'\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return [token for token in text.split(" ") if token]


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


def average_vector(tokens: list[str], model: Word2Vec) -> np.ndarray:
    vectors = [model.wv[token] for token in tokens if token in model.wv]
    if not vectors:
        return np.zeros(model.vector_size, dtype=np.float32)
    return np.mean(vectors, axis=0)


def build_matrix(token_lists: list[list[str]], model: Word2Vec) -> np.ndarray:
    return np.vstack([average_vector(tokens, model) for tokens in token_lists])


def main() -> None:
    labeled_df = read_competition_file("labeledTrainData.tsv.zip", sep="\t")
    test_df = read_competition_file("testData.tsv.zip", sep="\t")
    unlabeled_df = read_competition_file("unlabeledTrainData.tsv.zip", sep="\t")
    sample_submission = read_competition_file("sampleSubmission.csv", sep=",")

    labeled_df = labeled_df.copy()
    test_df = test_df.copy()
    unlabeled_df = unlabeled_df.copy()

    labeled_df["tokens"] = labeled_df["review"].map(clean_and_tokenize)
    test_df["tokens"] = test_df["review"].map(clean_and_tokenize)
    unlabeled_df["tokens"] = unlabeled_df["review"].map(clean_and_tokenize)

    train_df, valid_df = train_test_split(
        labeled_df,
        test_size=0.2,
        random_state=42,
        stratify=labeled_df["sentiment"],
    )

    training_sentences = (
        train_df["tokens"].tolist()
        + valid_df["tokens"].tolist()
        + test_df["tokens"].tolist()
        + unlabeled_df["tokens"].tolist()
    )

    word2vec = Word2Vec(
        sentences=training_sentences,
        vector_size=300,
        window=10,
        min_count=5,
        workers=1,
        sg=1,
        negative=10,
        sample=1e-3,
        epochs=15,
        seed=42,
    )

    x_train = build_matrix(train_df["tokens"].tolist(), word2vec)
    x_valid = build_matrix(valid_df["tokens"].tolist(), word2vec)

    model = LogisticRegression(
        solver="liblinear",
        C=4.0,
        max_iter=1000,
    )
    model.fit(x_train, train_df["sentiment"])
    valid_pred = model.predict_proba(x_valid)[:, 1]
    valid_auc = roc_auc_score(valid_df["sentiment"], valid_pred)

    x_full = build_matrix(labeled_df["tokens"].tolist(), word2vec)
    x_test = build_matrix(test_df["tokens"].tolist(), word2vec)

    final_model = LogisticRegression(
        solver="liblinear",
        C=4.0,
        max_iter=1000,
    )
    final_model.fit(x_full, labeled_df["sentiment"])
    test_pred = final_model.predict_proba(x_test)[:, 1]

    submission = pd.DataFrame({"id": test_df["id"], "sentiment": test_pred})
    submission = submission[sample_submission.columns.tolist()]
    submission.to_csv(SUBMISSION_PATH, index=False, encoding="utf-8-sig")

    REPORT_PATH.write_text(
        "\n".join(
            [
                "# Word2Vec 均值向量提交说明",
                "",
                f"- 验证集 AUC: {valid_auc:.6f}",
                "- 流程: 清洗 + Word2Vec + 均值 embedding + Logistic Regression",
                "- Word2Vec 参数: vector_size=300, window=10, min_count=5, sg=1, epochs=15",
                "- 无标签数据用途: 参与 Word2Vec 训练。",
                "- 提交文件: submission_word2vec_avg.csv",
                "- 注意: 为适配 AUC，提交值使用正类概率，而不是硬标签。",
            ]
        ),
        encoding="utf-8",
    )

    print(f"validation_auc={valid_auc:.6f}")
    print(f"saved: {SUBMISSION_PATH}")
    print(f"saved: {REPORT_PATH}")


if __name__ == "__main__":
    main()
