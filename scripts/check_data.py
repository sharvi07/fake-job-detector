"""Step 1 data check for Person 2.

Usage:
    python scripts/check_data.py --train path/to/train.csv --test path/to/test.csv

It prints what you need to know before building features, and writes a small
stratified sample (data_sample.csv) that you can attach to a chat or share.
Works with .csv or .parquet files.
"""
import argparse
import re
import sys

import pandas as pd

LABEL_CANDIDATES = ["fraudulent", "label", "is_fraud", "target"]
RAW_FIELDS = {
    "title": ["title"],
    "description": ["description"],
    "requirements": ["requirements"],
    "company_profile": ["company_profile", "profile"],
    "salary_range": ["salary_range", "salary"],
    "contact email / url in text": [],  # checked separately below
}
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
URL_RE = re.compile(r"https?://\S+|www\.\S+")


def load(path):
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def find_label(df):
    for c in LABEL_CANDIDATES:
        if c in df.columns:
            return c
    return None


def text_cols(df):
    return [c for c in df.columns if pd.api.types.is_string_dtype(df[c])
            or df[c].dtype == object]


def report(name, df, label):
    print(f"\n===== {name} =====")
    print("shape:", df.shape)
    print("\ncolumns and dtypes:")
    print(df.dtypes.to_string())
    print("\nmissing values (fraction):")
    print(df.isna().mean().round(3).to_string())
    if label:
        print(f"\nlabel column: {label}")
        print(df[label].value_counts().to_string())
        print("fraud rate:", round(df[label].mean(), 4))
    else:
        print("\n!! No label column found. Tell me what it is called.")

    print("\nraw fields present?")
    for field, names in RAW_FIELDS.items():
        if names:
            hit = [n for n in names if n in df.columns]
            print(f"  {field}: {'YES (' + hit[0] + ')' if hit else 'NO'}")

    # Are emails/URLs still in the text, or were they stripped by cleaning?
    n_email = n_url = 0
    for c in text_cols(df):
        s = df[c].dropna().astype(str)
        n_email += s.str.contains(EMAIL_RE).sum()
        n_url += s.str.contains(URL_RE).sum()
    print(f"  emails still in text: {n_email} cells")
    print(f"  URLs still in text:   {n_url} cells")
    if n_email == 0 and n_url == 0:
        print("  -> cleaning likely removed emails/URLs. Ask Person 1 for the "
              "raw text columns; email/URL features need them.")

    print("\nexample text lengths (chars) for text columns:")
    for c in text_cols(df)[:6]:
        print(f"  {c}: median {int(df[c].fillna('').astype(str).str.len().median())}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", default=None)
    ap.add_argument("--sample-size", type=int, default=750, help="rows per class")
    args = ap.parse_args()

    train = load(args.train)
    label = find_label(train)
    report("TRAIN", train, label)

    if args.test:
        test = load(args.test)
        report("TEST", test, find_label(test))
        if label and label in test.columns:
            print("\ncolumns identical in train and test:",
                  list(train.columns) == list(test.columns))

    if label:
        sample = (train.groupby(label, group_keys=False)
                  .apply(lambda g: g.sample(min(len(g), args.sample_size),
                                            random_state=42)))
    else:
        sample = train.sample(min(len(train), 1500), random_state=42)
    sample.to_csv("data_sample.csv", index=False)
    print(f"\nWrote data_sample.csv ({len(sample)} rows). "
          "Attach it to the chat if it is under the upload limit.")


if __name__ == "__main__":
    sys.exit(main())
