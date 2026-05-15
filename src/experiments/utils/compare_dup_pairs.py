#!/usr/bin/env python3
"""compare_dup_pairs.py

Compare and analyze duplicate pair exports from Steps 02 (exact) and 03 (near-duplicate).
Provides filtering, lookup, statistics, and export utilities for manual review and analysis.
"""

from pathlib import Path
import pandas as pd
import argparse
from typing import Optional, Dict, Any


# ----------------------------
# CONFIG
# ----------------------------

REPORT_DIR = Path("output")
STEP02_PAIRS = Path(
    "data/_02_exact_deduplicated_datasets/_02_dup_pairs/_02_exact_duplicate_pairs.parquet"
)
STEP02_PAIRS_CSV = Path(
    "data/_02_exact_deduplicated_datasets/_02_dup_pairs/_02_exact_duplicate_pairs.csv"
)
STEP03_PAIRS = Path(
    "data/_03_near_deduplicated_datasets/_03_dup_pairs/_03_near_duplicate_pairs.parquet"
)
STEP03_PAIRS_CSV = Path(
    "data/_03_near_deduplicated_datasets/_03_dup_pairs/_03_near_duplicate_pairs.csv"
)

STEP02_SOURCE_DIR = Path("data/_01_normalized_datasets")
STEP03_SOURCE_DIR = Path("data/_02_exact_deduplicated_datasets")

COMPARISON_OUTPUT_DIR = REPORT_DIR / "pair_comparison"
COMPARISON_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CODE_COLUMN_CANDIDATES = ["code", "source_code", "source", "text"]


def get_step_source_dir(step_name: str) -> Path:
    """Return source dataset directory for each pair step."""
    if step_name == "02":
        return STEP02_SOURCE_DIR
    if step_name == "03":
        return STEP03_SOURCE_DIR
    raise ValueError(f"Unknown step: {step_name}")


def parse_sample_id(sample_id: str) -> tuple[str, int]:
    """Parse sample id in format dataset::idx."""
    if "::" not in sample_id:
        raise ValueError(f"Invalid sample_id format: {sample_id}")
    ds_name, idx_str = sample_id.split("::", 1)
    return ds_name, int(idx_str)


def resolve_code_column(df: pd.DataFrame) -> str:
    """Choose best code-like column from known candidates."""
    for col in CODE_COLUMN_CANDIDATES:
        if col in df.columns:
            return col
    raise KeyError(
        f"No code column found. Expected one of {CODE_COLUMN_CANDIDATES}, got {list(df.columns)}"
    )


def load_dataset_df(source_dir: Path, ds_name: str, use_parquet: bool = True) -> pd.DataFrame:
    """Load dataset file by name from a source directory."""
    if use_parquet:
        p = source_dir / f"{ds_name}.parquet"
        if p.exists():
            return pd.read_parquet(p)
    c = source_dir / f"{ds_name}.csv"
    if c.exists():
        return pd.read_csv(c)
    raise FileNotFoundError(f"Dataset file not found for '{ds_name}' in {source_dir}")


def fetch_code_from_sample_id(
    sample_id: str,
    source_dir: Path,
    dataset_cache: dict[str, pd.DataFrame],
    use_parquet: bool = True,
) -> str:
    """Resolve a sample_id to its code text from source datasets."""
    ds_name, idx = parse_sample_id(sample_id)
    if ds_name not in dataset_cache:
        dataset_cache[ds_name] = load_dataset_df(source_dir, ds_name, use_parquet=use_parquet)
    ds_df = dataset_cache[ds_name]
    code_col = resolve_code_column(ds_df)
    if idx < 0 or idx >= len(ds_df):
        raise IndexError(
            f"Sample index out of range for {sample_id}: dataset '{ds_name}' has {len(ds_df)} rows"
        )
    val = ds_df.iloc[idx][code_col]
    return "" if pd.isna(val) else str(val)


def print_pair_codes(
    row: pd.Series,
    step_name: str,
    dataset_cache: dict[str, pd.DataFrame],
    use_parquet: bool = True,
    max_chars: int = 1200,
) -> None:
    """Print removed/survivor code for one pair row."""
    source_dir = get_step_source_dir(step_name)
    removed_sid = row.get("removed_sample_id", "")
    survivor_sid = row.get("survivor_sample_id", "")

    print("\n" + "-" * 70)
    print(
        f"Pair {row.get('pair_id', '?')} | type={row.get('pair_type', '?')} "
        f"| similarity={row.get('similarity', 0):.6f}"
    )
    print(f"Removed:  {removed_sid}")
    print(f"Survivor: {survivor_sid}")

    try:
        removed_code = fetch_code_from_sample_id(
            removed_sid, source_dir, dataset_cache, use_parquet=use_parquet
        )
    except Exception as exc:
        removed_code = f"<error loading removed code: {exc}>"

    try:
        survivor_code = fetch_code_from_sample_id(
            survivor_sid, source_dir, dataset_cache, use_parquet=use_parquet
        )
    except Exception as exc:
        survivor_code = f"<error loading survivor code: {exc}>"

    if max_chars > 0:
        removed_code = removed_code[:max_chars]
        survivor_code = survivor_code[:max_chars]

    print("\n[REMOVED CODE]")
    print(removed_code)
    print("\n[SURVIVOR CODE]")
    print(survivor_code)


# ----------------------------
# DATA LOADING
# ----------------------------


def load_pairs(use_parquet: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load Step 02 and Step 03 pair exports.

    Returns:
        (step02_df, step03_df) or raises FileNotFoundError if files missing
    """
    step02_file = STEP02_PAIRS if use_parquet else STEP02_PAIRS_CSV
    step03_file = STEP03_PAIRS if use_parquet else STEP03_PAIRS_CSV

    if not step02_file.exists():
        raise FileNotFoundError(f"Step 02 pairs not found: {step02_file}")

    step02_df = pd.read_parquet(step02_file) if use_parquet else pd.read_csv(step02_file)

    # Step 03 may not exist yet (first run)
    step03_df = None
    if step03_file.exists():
        step03_df = pd.read_parquet(step03_file) if use_parquet else pd.read_csv(step03_file)
    else:
        print(f"Warning: Step 03 pairs not found: {step03_file}")
        print("  Creating empty DataFrame. Run Step 03 to populate.")
        step03_df = pd.DataFrame()

    return step02_df, step03_df


# ----------------------------
# FILTERING
# ----------------------------


class PairFilter:
    """Flexible pair filtering with multiple criteria."""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.filtered = df.copy()

    def by_dataset(self, dataset: str, column: str = "removed_dataset") -> "PairFilter":
        """Filter by removed or survivor dataset."""
        if column not in self.filtered.columns:
            raise ValueError(
                f"Column '{column}' not found. Available: {list(self.filtered.columns)}"
            )
        self.filtered = self.filtered[self.filtered[column] == dataset]
        return self

    def by_removed_dataset(self, dataset: str) -> "PairFilter":
        """Filter by removed dataset."""
        return self.by_dataset(dataset, "removed_dataset")

    def by_survivor_dataset(self, dataset: str) -> "PairFilter":
        """Filter by survivor dataset."""
        return self.by_dataset(dataset, "survivor_dataset")

    def by_pair_type(self, pair_type: str) -> "PairFilter":
        """Filter by pair type ('exact' or 'near')."""
        if pair_type not in ("exact", "near"):
            raise ValueError(f"pair_type must be 'exact' or 'near', got '{pair_type}'")
        self.filtered = self.filtered[self.filtered["pair_type"] == pair_type]
        return self

    def by_similarity_range(self, min_sim: float = 0.0, max_sim: float = 1.0) -> "PairFilter":
        """Filter by similarity score range."""
        if "similarity" not in self.filtered.columns:
            raise ValueError("similarity column not found in dataframe")
        self.filtered = self.filtered[
            (self.filtered["similarity"] >= min_sim) & (self.filtered["similarity"] <= max_sim)
        ]
        return self

    def by_group_size(self, min_size: int = 1, max_size: Optional[int] = None) -> "PairFilter":
        """Filter by group size (collision count)."""
        if "group_size" not in self.filtered.columns:
            raise ValueError("group_size column not found in dataframe")
        self.filtered = self.filtered[self.filtered["group_size"] >= min_size]
        if max_size is not None:
            self.filtered = self.filtered[self.filtered["group_size"] <= max_size]
        return self

    def by_sample_id(self, sample_id: str, column: str = "removed_sample_id") -> "PairFilter":
        """Filter by removed or survivor sample ID."""
        if column not in self.filtered.columns:
            raise ValueError(f"Column '{column}' not found")
        self.filtered = self.filtered[self.filtered[column] == sample_id]
        return self

    def by_removed_sample_id(self, sample_id: str) -> "PairFilter":
        """Filter by removed sample ID."""
        return self.by_sample_id(sample_id, "removed_sample_id")

    def by_survivor_sample_id(self, sample_id: str) -> "PairFilter":
        """Filter by survivor sample ID."""
        return self.by_sample_id(sample_id, "survivor_sample_id")

    def head(self, n: int = 10) -> pd.DataFrame:
        """Return first n rows."""
        return self.filtered.head(n)

    def tail(self, n: int = 10) -> pd.DataFrame:
        """Return last n rows."""
        return self.filtered.tail(n)

    def get(self) -> pd.DataFrame:
        """Return filtered dataframe."""
        return self.filtered

    def count(self) -> int:
        """Return row count."""
        return len(self.filtered)


# ----------------------------
# STATISTICS
# ----------------------------


def compute_statistics(step02_df: pd.DataFrame, step03_df: pd.DataFrame) -> Dict[str, Any]:
    """Compute summary statistics for both pair exports."""

    stats = {
        "step02": {},
        "step03": {},
        "combined": {},
    }

    # Step 02 stats
    if not step02_df.empty:
        stats["step02"] = {
            "total_pairs": len(step02_df),
            "unique_groups": step02_df["group_id"].nunique(),
            "datasets": step02_df["removed_dataset"].unique().tolist(),
            "avg_group_size": step02_df["group_size"].mean(),
            "max_group_size": step02_df["group_size"].max(),
            "similarity": {
                "min": step02_df["similarity"].min(),
                "max": step02_df["similarity"].max(),
                "mean": step02_df["similarity"].mean(),
            },
        }

    # Step 03 stats
    if not step03_df.empty:
        stats["step03"] = {
            "total_pairs": len(step03_df),
            "unique_groups": step03_df["group_id"].nunique(),
            "datasets": step03_df["removed_dataset"].unique().tolist(),
            "avg_group_size": step03_df["group_size"].mean(),
            "max_group_size": step03_df["group_size"].max(),
            "similarity": {
                "min": step03_df["similarity"].min(),
                "max": step03_df["similarity"].max(),
                "mean": step03_df["similarity"].mean(),
            },
        }

    # Combined stats
    if not step02_df.empty or not step03_df.empty:
        combined = pd.concat([step02_df, step03_df], ignore_index=True)
        stats["combined"] = {
            "total_pairs": len(combined),
            "step02_pairs": len(step02_df),
            "step03_pairs": len(step03_df),
            "unique_groups": combined["group_id"].nunique() if not combined.empty else 0,
        }

    return stats


def print_statistics(stats: Dict[str, Any]) -> None:
    """Pretty-print statistics."""
    print("\n" + "=" * 70)
    print("PAIR COMPARISON STATISTICS")
    print("=" * 70)

    if stats.get("step02"):
        print("\n[STEP 02 - EXACT DUPLICATES]")
        s2 = stats["step02"]
        print(f"  Total pairs:        {s2['total_pairs']}")
        print(f"  Unique groups:      {s2['unique_groups']}")
        print(f"  Datasets affected:  {', '.join(s2['datasets'])}")
        print(f"  Group size:         avg={s2['avg_group_size']:.2f}, max={s2['max_group_size']}")
        print(
            f"  Similarity:         min={s2['similarity']['min']:.6f}, "
            f"max={s2['similarity']['max']:.6f}, mean={s2['similarity']['mean']:.6f}"
        )

    if stats.get("step03"):
        print("\n[STEP 03 - NEAR DUPLICATES]")
        s3 = stats["step03"]
        print(f"  Total pairs:        {s3['total_pairs']}")
        print(f"  Unique groups:      {s3['unique_groups']}")
        print(f"  Datasets affected:  {', '.join(s3['datasets'])}")
        print(f"  Group size:         avg={s3['avg_group_size']:.2f}, max={s3['max_group_size']}")
        print(
            f"  Similarity:         min={s3['similarity']['min']:.6f}, "
            f"max={s3['similarity']['max']:.6f}, mean={s3['similarity']['mean']:.6f}"
        )

    if stats.get("combined"):
        print("\n[COMBINED SUMMARY]")
        c = stats["combined"]
        print(f"  Total pairs:        {c['total_pairs']}")
        print(f"    - From Step 02:   {c['step02_pairs']}")
        print(f"    - From Step 03:   {c['step03_pairs']}")
        print(f"  Unique groups:      {c['unique_groups']}")

    print("\n" + "=" * 70 + "\n")


# ----------------------------
# LOOKUP & RETRIEVAL
# ----------------------------


def find_sample_pairs(
    df: pd.DataFrame, sample_id: str, as_removed: bool = True, as_survivor: bool = True
) -> pd.DataFrame:
    """Find all pairs involving a sample ID (as removed or survivor)."""
    results = []

    if as_removed:
        results.append(df[df["removed_sample_id"] == sample_id])

    if as_survivor:
        results.append(df[df["survivor_sample_id"] == sample_id])

    if results:
        return pd.concat(results, ignore_index=True).drop_duplicates()
    return pd.DataFrame()


def find_group_pairs(df: pd.DataFrame, group_id: str) -> pd.DataFrame:
    """Find all pairs in a group."""
    return df[df["group_id"] == group_id]


# ----------------------------
# EXPORT & REPORTING
# ----------------------------


def export_filtered_pairs(df: pd.DataFrame, name: str, include_csv: bool = True) -> None:
    """Export filtered pairs to parquet and optionally CSV."""
    parquet_file = COMPARISON_OUTPUT_DIR / f"{name}.parquet"
    csv_file = COMPARISON_OUTPUT_DIR / f"{name}.csv"

    df.to_parquet(parquet_file, index=False)
    print(f"Exported {len(df)} pairs to {parquet_file}")

    if include_csv:
        df.to_csv(csv_file, index=False)
        print(f"Exported {len(df)} pairs to {csv_file}")


def generate_markdown_report(df: pd.DataFrame, title: str = "Pair Report") -> str:
    """Generate markdown report of pair data."""
    report = []
    report.append(f"# {title}\n")
    report.append(f"**Total Pairs:** {len(df)}\n")

    if not df.empty:
        # Summary stats
        report.append("\n## Summary Statistics\n")
        if "similarity" in df.columns:
            report.append(
                f"- Similarity (min/max/mean): "
                f"{df['similarity'].min():.6f} / "
                f"{df['similarity'].max():.6f} / "
                f"{df['similarity'].mean():.6f}\n"
            )

        if "group_size" in df.columns:
            report.append(
                f"- Group sizes (min/max/avg): "
                f"{df['group_size'].min()} / "
                f"{df['group_size'].max()} / "
                f"{df['group_size'].mean():.2f}\n"
            )

        if "pair_type" in df.columns:
            report.append(f"- Pair types: {df['pair_type'].value_counts().to_dict()}\n")

        # Dataset distribution
        if "removed_dataset" in df.columns:
            report.append("\n## Removed by Dataset\n")
            for ds, count in df["removed_dataset"].value_counts().items():
                report.append(f"- {ds}: {count}\n")

        # Sample table
        report.append("\n## Pair Details\n")
        report.append("\n| Pair ID | Type | Group | Removed | Survivor | Similarity |\n")
        report.append("|---------|------|-------|---------|----------|------------|\n")

        for _, row in df.head(100).iterrows():
            pair_id = row.get("pair_id", "?")
            ptype = row.get("pair_type", "?")
            gid = str(row.get("group_id", "?"))[:8]
            removed = row.get("removed_sample_id", "?")
            survivor = row.get("survivor_sample_id", "?")
            sim = row.get("similarity", 0)

            report.append(f"| {pair_id} | {ptype} | {gid} | {removed} | {survivor} | {sim:.6f} |\n")

        if len(df) > 100:
            report.append(f"\n*... and {len(df) - 100} more pairs*\n")

    return "".join(report)


def save_markdown_report(df: pd.DataFrame, name: str, title: str | None = None) -> None:
    """Save markdown report to file."""
    if title is None:
        title = f"{name} Report"

    report_content = generate_markdown_report(df, title)
    report_file = COMPARISON_OUTPUT_DIR / f"{name}.md"

    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"Saved markdown report to {report_file}")


# ----------------------------
# CLI
# ----------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Compare and analyze duplicate pair exports from Steps 02 and 03",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show statistics
  python compare_dup_pairs.py --stats
  
  # Filter Step 02 exact duplicates
  python compare_dup_pairs.py --step 02 --show 20
  
  # Find pairs involving a specific sample
  python compare_dup_pairs.py --lookup ai_detector::12345
  
  # Export low-similarity pairs from Step 03
  python compare_dup_pairs.py --step 03 --sim-max 0.95 --export low_similarity_pairs
  
  # Find all pairs in a group
  python compare_dup_pairs.py --group-lookup <group_id>
        """,
    )

    parser.add_argument("--stats", action="store_true", help="Print summary statistics")

    parser.add_argument(
        "--step",
        choices=["02", "03", "both"],
        default="both",
        help="Filter by step (default: both)",
    )

    parser.add_argument("--show", type=int, default=10, help="Show N rows (default: 10)")

    parser.add_argument(
        "--show-code", action="store_true", help="Show removed/survivor code for displayed rows"
    )

    parser.add_argument(
        "--code-max-chars",
        type=int,
        default=1200,
        help="Max chars per code block when --show-code is used (default: 1200)",
    )

    parser.add_argument("--dataset", help="Filter by removed dataset")

    parser.add_argument("--survivor-dataset", help="Filter by survivor dataset")

    parser.add_argument("--pair-type", choices=["exact", "near"], help="Filter by pair type")

    parser.add_argument(
        "--sim-min", type=float, default=0.0, help="Minimum similarity (default: 0.0)"
    )

    parser.add_argument(
        "--sim-max", type=float, default=1.0, help="Maximum similarity (default: 1.0)"
    )

    parser.add_argument("--group-size-min", type=int, help="Minimum group size")

    parser.add_argument("--group-size-max", type=int, help="Maximum group size")

    parser.add_argument("--lookup", help="Find pairs involving sample ID (format: dataset::idx)")

    parser.add_argument("--group-lookup", help="Find all pairs in a group (group_id)")

    parser.add_argument(
        "--export", help="Export filtered results to COMPARISON_OUTPUT_DIR/{export}.parquet/.csv"
    )

    parser.add_argument("--report", help="Generate markdown report")

    parser.add_argument("--csv", action="store_true", help="Use CSV instead of parquet for loading")

    args = parser.parse_args()

    # Load data
    try:
        step02_df, step03_df = load_pairs(use_parquet=not args.csv)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1

    # Stats command
    if args.stats:
        stats = compute_statistics(step02_df, step03_df)
        print_statistics(stats)
        return 0

    # Select dataframe(s)
    if args.step == "02":
        dfs = {"02": step02_df}
    elif args.step == "03":
        dfs = {"03": step03_df if not step03_df.empty else pd.DataFrame()}
    else:  # both
        dfs = {"02": step02_df, "03": step03_df}

    # Process each step
    for step_name, df in dfs.items():
        dataset_cache: dict[str, pd.DataFrame] = {}
        if df.empty:
            if step_name == "03":
                print("Step 03 pairs not available (may not have been run yet)\n")
            continue

        print(f"\n{'='*70}")
        print(f"STEP {step_name}")
        print(f"{'='*70}\n")

        # Lookup by sample ID
        if args.lookup:
            result = find_sample_pairs(df, args.lookup)
            if result.empty:
                print(f"No pairs found for sample: {args.lookup}")
            else:
                print(f"Found {len(result)} pair(s) for sample: {args.lookup}\n")
                print(result.to_string())
                if args.show_code:
                    for _, row in result.head(args.show).iterrows():
                        print_pair_codes(
                            row,
                            step_name,
                            dataset_cache,
                            use_parquet=not args.csv,
                            max_chars=args.code_max_chars,
                        )
            continue

        # Lookup by group
        if args.group_lookup:
            result = find_group_pairs(df, args.group_lookup)
            if result.empty:
                print(f"No pairs found for group: {args.group_lookup}")
            else:
                print(f"Found {len(result)} pair(s) in group: {args.group_lookup}\n")
                print(result.to_string())
                if args.show_code:
                    for _, row in result.head(args.show).iterrows():
                        print_pair_codes(
                            row,
                            step_name,
                            dataset_cache,
                            use_parquet=not args.csv,
                            max_chars=args.code_max_chars,
                        )
            continue

        # Apply filters
        filt = PairFilter(df)

        if args.dataset:
            filt.by_removed_dataset(args.dataset)

        if args.survivor_dataset:
            filt.by_survivor_dataset(args.survivor_dataset)

        if args.pair_type:
            filt.by_pair_type(args.pair_type)

        filt.by_similarity_range(args.sim_min, args.sim_max)

        if args.group_size_min or args.group_size_max:
            filt.by_group_size(args.group_size_min or 1, args.group_size_max)

        filtered_df = filt.get()

        # Display results
        print(f"Filtered: {len(filtered_df)} / {len(df)} pairs")
        if len(filtered_df) > 0:
            print(f"\nFirst {min(args.show, len(filtered_df))} rows:\n")
            print(filtered_df.head(args.show).to_string())
            if args.show_code:
                for _, row in filtered_df.head(args.show).iterrows():
                    print_pair_codes(
                        row,
                        step_name,
                        dataset_cache,
                        use_parquet=not args.csv,
                        max_chars=args.code_max_chars,
                    )

        # Export
        if args.export:
            export_name = f"{args.export}_step{step_name}"
            export_filtered_pairs(filtered_df, export_name)

        # Report
        if args.report:
            report_name = f"{args.report}_step{step_name}"
            save_markdown_report(filtered_df, report_name)


if __name__ == "__main__":
    import sys

    sys.exit(main() or 0)
