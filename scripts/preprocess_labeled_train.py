from __future__ import annotations

import re
import zipfile
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent.parent
ZIP_PATH = ROOT / "data" / "raw" / "labeledTrainData.tsv.zip"
RAW_TSV_PATH = ROOT / "data" / "raw" / "labeledTrainData.tsv"
OUTPUT_DIR = ROOT / "data" / "processed"
FULL_OUTPUT_PATH = OUTPUT_DIR / "labeledTrainData_clean.csv"
TRAIN_OUTPUT_PATH = OUTPUT_DIR / "train_split.csv"
VALID_OUTPUT_PATH = OUTPUT_DIR / "valid_split.csv"
SUMMARY_PATH = OUTPUT_DIR / "processing_summary.md"


def clean_review(text: str) -> str:
    """Convert raw IMDB review text into plain lowercase tokens."""
    text = BeautifulSoup(text, "html.parser").get_text(" ")
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_raw_tsv() -> None:
    with zipfile.ZipFile(ZIP_PATH) as archive:
        archive.extract("labeledTrainData.tsv", path=ROOT)


def build_summary(df: pd.DataFrame, train_df: pd.DataFrame, valid_df: pd.DataFrame) -> str:
    label_counts = df["sentiment"].value_counts().sort_index()
    return "\n".join(
        [
            "# labeledTrainData.tsv 预处理结果",
            "",
            f"- 原始样本数: {len(df)}",
            f"- 字段: {', '.join(df.columns)}",
            f"- 标签分布: 0={label_counts.get(0, 0)}, 1={label_counts.get(1, 0)}",
            f"- 训练集大小: {len(train_df)}",
            f"- 验证集大小: {len(valid_df)}",
            "- 清洗规则: 去除 HTML 标签、仅保留英文字母、转小写、压缩多余空白。",
            "",
            "## 输出文件",
            "",
            f"- {FULL_OUTPUT_PATH.name}: 完整清洗数据，含 `review_clean` 等辅助列。",
            f"- {TRAIN_OUTPUT_PATH.name}: 分层抽样后的训练子集。",
            f"- {VALID_OUTPUT_PATH.name}: 分层抽样后的验证子集。",
        ]
    )


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    extract_raw_tsv()

    df = pd.read_csv(RAW_TSV_PATH, sep="\t")
    df["review_clean"] = df["review"].astype(str).map(clean_review)
    df["review_word_count"] = df["review_clean"].str.split().str.len()
    df["review_char_count"] = df["review_clean"].str.len()

    train_df, valid_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["sentiment"],
    )

    df.to_csv(FULL_OUTPUT_PATH, index=False, encoding="utf-8-sig")
    train_df.to_csv(TRAIN_OUTPUT_PATH, index=False, encoding="utf-8-sig")
    valid_df.to_csv(VALID_OUTPUT_PATH, index=False, encoding="utf-8-sig")
    SUMMARY_PATH.write_text(
        build_summary(df, train_df, valid_df),
        encoding="utf-8",
    )

    print(f"saved: {FULL_OUTPUT_PATH}")
    print(f"saved: {TRAIN_OUTPUT_PATH}")
    print(f"saved: {VALID_OUTPUT_PATH}")
    print(f"saved: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
