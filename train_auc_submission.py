from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent
COMPETITION_ZIP = ROOT / "word2vec-nlp-tutorial.zip"
SUBMISSION_PATH = ROOT / "submission.csv"
REPORT_PATH = ROOT / "submission_report.md"


def clean_review(text: str) -> str:
    text = BeautifulSoup(str(text), "html.parser").get_text(" ")
    text = re.sub(r"[^a-zA-Z]", " ", text)
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


def prepare_features(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df["review_clean"] = train_df["review"].map(clean_review)
    test_df["review_clean"] = test_df["review"].map(clean_review)
    return train_df, test_df


def build_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        strip_accents="unicode",
        lowercase=False,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.9,
        sublinear_tf=True,
        max_features=120000,
    )


def build_model() -> LogisticRegression:
    return LogisticRegression(
        solver="liblinear",
        C=4.0,
        max_iter=1000,
    )


def main() -> None:
    labeled_df = read_competition_file("labeledTrainData.tsv.zip", sep="\t")
    test_df = read_competition_file("testData.tsv.zip", sep="\t")
    sample_submission = read_competition_file("sampleSubmission.csv", sep=",")

    labeled_df, test_df = prepare_features(labeled_df, test_df)

    train_df, valid_df = train_test_split(
        labeled_df,
        test_size=0.2,
        random_state=42,
        stratify=labeled_df["sentiment"],
    )

    vectorizer = build_vectorizer()
    x_train = vectorizer.fit_transform(train_df["review_clean"])
    x_valid = vectorizer.transform(valid_df["review_clean"])

    model = build_model()
    model.fit(x_train, train_df["sentiment"])

    valid_pred = model.predict_proba(x_valid)[:, 1]
    valid_auc = roc_auc_score(valid_df["sentiment"], valid_pred)

    # Refit on the full labeled set before generating the final submission.
    full_vectorizer = build_vectorizer()
    x_full = full_vectorizer.fit_transform(labeled_df["review_clean"])
    x_test = full_vectorizer.transform(test_df["review_clean"])

    final_model = build_model()
    final_model.fit(x_full, labeled_df["sentiment"])
    test_pred = final_model.predict_proba(x_test)[:, 1]

    submission = pd.DataFrame(
        {
            "id": test_df["id"],
            "sentiment": test_pred,
        }
    )
    submission = submission[sample_submission.columns.tolist()]
    submission.to_csv(SUBMISSION_PATH, index=False, encoding="utf-8-sig")

    REPORT_PATH.write_text(
        "\n".join(
            [
                "# AUC 验证与提交文件说明",
                "",
                f"- 训练集样本数: {len(labeled_df)}",
                f"- 测试集样本数: {len(test_df)}",
                f"- 验证集 AUC: {valid_auc:.6f}",
                "- 模型: TF-IDF (1-2 gram) + Logistic Regression",
                "- 提交文件: submission.csv",
                "- 提交列说明: `id`, `sentiment`",
                "- 注意: 为匹配 AUC 指标，`sentiment` 输出的是正类概率，而不是硬分类 0/1。",
            ]
        ),
        encoding="utf-8",
    )

    print(f"validation_auc={valid_auc:.6f}")
    print(f"saved: {SUBMISSION_PATH}")
    print(f"saved: {REPORT_PATH}")


if __name__ == "__main__":
    main()
