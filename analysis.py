"""Course renewal analysis.

Loads tables from the published_data directory of the
fulldecent/mts_course_renewal_marketing HuggingFace dataset and reports
basic statistics.
"""

import pandas as pd
from huggingface_hub import list_repo_files, hf_hub_download

DATASET_REPO = "fulldecent/mts_course_renewal_marketing"
DATA_DIR = "published_data"


def load_tables() -> dict[str, pd.DataFrame]:
    """Download and load all data files from the published_data directory."""
    all_files = list(
        list_repo_files(DATASET_REPO, repo_type="dataset")
    )
    data_files = [
        f for f in all_files
        if f.startswith(DATA_DIR + "/") and not f.endswith("/")
    ]

    tables: dict[str, pd.DataFrame] = {}
    for file_path in sorted(data_files):
        local_path = hf_hub_download(
            repo_id=DATASET_REPO,
            filename=file_path,
            repo_type="dataset",
        )
        name = file_path[len(DATA_DIR) + 1:]  # strip "published_data/"
        if file_path.endswith(".parquet"):
            tables[name] = pd.read_parquet(local_path)
        elif file_path.endswith(".csv"):
            tables[name] = pd.read_csv(local_path)
        elif file_path.endswith(".jsonl") or file_path.endswith(".json"):
            tables[name] = pd.read_json(local_path, lines=file_path.endswith(".jsonl"))
        else:
            print(f"Skipping unsupported file type: {file_path}")

    return tables


def main() -> None:
    tables = load_tables()

    if not tables:
        print("No tables found in published_data.")
        return

    print("Row counts per table:")
    print("-" * 40)
    for name, df in tables.items():
        print(f"  {name}: {len(df):,} rows")
    print("-" * 40)
    print(f"  Total tables: {len(tables)}")


if __name__ == "__main__":
    main()
