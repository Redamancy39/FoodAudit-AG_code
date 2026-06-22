import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd


FOUR_LABELS = ("SAFE", "SAFE_QS", "RISK_FORBIDDEN", "RISK_OVERLIMIT")


def robust_load_table(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if path.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    for enc in ("utf-8-sig", "gb18030", "gbk", "utf-8"):
        try:
            return pd.read_csv(path, encoding=enc, engine="python", sep=None)
        except Exception:
            continue
    raise RuntimeError(f"Cannot read file: {path}")


def save_json(obj: Any, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_text(text: str, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def normalize_label(value: Any) -> str:
    s = str(value).strip().upper()
    if s in FOUR_LABELS:
        return s
    if s == "RISK":
        return "RISK_FORBIDDEN"
    if "SAFE_QS" in s or s == "QS":
        return "SAFE_QS"
    if "OVERLIMIT" in s:
        return "RISK_OVERLIMIT"
    if "FORBIDDEN" in s:
        return "RISK_FORBIDDEN"
    if "SAFE" in s:
        return "SAFE"
    return "RISK_FORBIDDEN"


def to_binary(label: Any) -> str:
    return "SAFE" if normalize_label(label) in ("SAFE", "SAFE_QS") else "RISK"


def parse_recipe_map(value: Any) -> Dict[str, str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return {}
    if isinstance(value, dict):
        return {str(k).strip(): normalize_label(v) for k, v in value.items()}
    s = str(value).strip()
    if not s:
        return {}
    if s.startswith("{") and s.endswith("}"):
        try:
            obj = json.loads(s)
            return {str(k).strip(): normalize_label(v) for k, v in obj.items()}
        except Exception:
            pass
    items: Dict[str, str] = {}
    for part in s.split("|"):
        part = part.strip()
        if ":" in part:
            k, v = part.split(":", 1)
            items[k.strip()] = normalize_label(v)
    return items


def reconstruct_item_df(recipe_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[dict] = []
    if "details" not in recipe_df.columns:
        return pd.DataFrame()
    for _, row in recipe_df.iterrows():
        details_raw = row.get("details")
        if details_raw is None or (isinstance(details_raw, float) and math.isnan(details_raw)):
            continue
        try:
            details = json.loads(details_raw) if isinstance(details_raw, str) else list(details_raw)
        except Exception:
            continue
        for d in details:
            gt = normalize_label(d.get("gt_label", "RISK_FORBIDDEN"))
            pred = normalize_label(d.get("pred_label", "RISK_FORBIDDEN"))
            rows.append(
                {
                    "id": row.get("id"),
                    "difficulty_tier": row.get("difficulty_tier", "UNKNOWN"),
                    "bucket_primary": row.get("bucket_primary", "UNKNOWN"),
                    "gt_item": d.get("gt_item"),
                    "pred_item": d.get("pred_item"),
                    "gt_label": gt,
                    "pred_label": pred,
                    "gt_binary": to_binary(gt),
                    "pred_binary": to_binary(pred),
                    "fine_ok": int(gt == pred),
                    "bin_ok": int(to_binary(gt) == to_binary(pred)),
                }
            )
    return pd.DataFrame(rows)


def bootstrap_ci(acc: float, n: int) -> List[float]:
    if n <= 0:
        return [0.0, 0.0]
    se = math.sqrt(max(acc * (1 - acc), 1e-12) / n)
    lo = max(0.0, acc - 1.96 * se)
    hi = min(1.0, acc + 1.96 * se)
    return [round(float(lo), 6), round(float(hi), 6)]


def recipe_metrics(recipe_df: pd.DataFrame) -> Dict[str, Any]:
    if recipe_df.empty:
        return {
            "n": 0,
            "fine_acc": 0.0,
            "binary_acc": 0.0,
            "avg_latency": 0.0,
            "ci95": [0.0, 0.0],
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "false_safe_count": 0,
            "false_safe_rate": 0.0,
            "false_risk_count": 0,
            "false_risk_rate": 0.0,
        }

    df = recipe_df.copy()
    if "gt_binary" not in df.columns:
        df["gt_binary"] = df["gt_map"].apply(lambda x: "SAFE" if all(to_binary(v) == "SAFE" for v in parse_recipe_map(x).values()) else "RISK")
    if "pred_binary" not in df.columns:
        df["pred_binary"] = df["pred_map"].apply(lambda x: "SAFE" if all(to_binary(v) == "SAFE" for v in parse_recipe_map(x).values()) else "RISK")

    fine_acc = float(df["fine_correct"].astype(float).mean()) if "fine_correct" in df.columns else 0.0
    binary_acc = float(df["binary_correct"].astype(float).mean()) if "binary_correct" in df.columns else 0.0
    avg_latency = float(df["latency"].astype(float).mean()) if "latency" in df.columns else 0.0

    tp = int(((df["gt_binary"] == "RISK") & (df["pred_binary"] == "RISK")).sum())
    fp = int(((df["gt_binary"] == "SAFE") & (df["pred_binary"] == "RISK")).sum())
    fn = int(((df["gt_binary"] == "RISK") & (df["pred_binary"] == "SAFE")).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    false_safe_count = int(((df["gt_binary"] == "RISK") & (df["pred_binary"] == "SAFE")).sum())
    false_risk_count = int(((df["gt_binary"] == "SAFE") & (df["pred_binary"] == "RISK")).sum())

    return {
        "n": int(len(df)),
        "fine_acc": round(fine_acc, 6),
        "binary_acc": round(binary_acc, 6),
        "avg_latency": round(avg_latency, 6),
        "ci95": bootstrap_ci(fine_acc, len(df)),
        "precision": round(float(precision), 6),
        "recall": round(float(recall), 6),
        "f1": round(float(f1), 6),
        "false_safe_count": false_safe_count,
        "false_safe_rate": round(false_safe_count / len(df), 6),
        "false_risk_count": false_risk_count,
        "false_risk_rate": round(false_risk_count / len(df), 6),
    }


def per_class_report(item_df: pd.DataFrame) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for label in FOUR_LABELS:
        if item_df.empty:
            rows.append({"label": label, "support": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0})
            continue
        tp = int(((item_df["gt_label"] == label) & (item_df["pred_label"] == label)).sum())
        fp = int(((item_df["gt_label"] != label) & (item_df["pred_label"] == label)).sum())
        fn = int(((item_df["gt_label"] == label) & (item_df["pred_label"] != label)).sum())
        support = int((item_df["gt_label"] == label).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        rows.append(
            {
                "label": label,
                "support": support,
                "precision": round(float(precision), 6),
                "recall": round(float(recall), 6),
                "f1": round(float(f1), 6),
            }
        )
    return rows


def confusion_4class(item_df: pd.DataFrame) -> Dict[str, Dict[str, int]]:
    mat = {gt: {pred: 0 for pred in FOUR_LABELS} for gt in FOUR_LABELS}
    if item_df.empty:
        return mat
    for _, row in item_df.iterrows():
        gt = normalize_label(row["gt_label"])
        pred = normalize_label(row["pred_label"])
        mat[gt][pred] += 1
    return mat


def grouped_recipe(recipe_df: pd.DataFrame, col: str) -> List[Dict[str, Any]]:
    if col not in recipe_df.columns:
        return []
    rows: List[Dict[str, Any]] = []
    for key, sub in recipe_df.groupby(col, dropna=False):
        rows.append({col: str(key), **recipe_metrics(sub)})
    return rows


def grouped_anchor(recipe_df: pd.DataFrame, col: Optional[str] = None) -> List[Dict[str, Any]]:
    if "anchor_correct" not in recipe_df.columns:
        return []
    valid = recipe_df[recipe_df["anchor_correct"].notna()].copy()
    if valid.empty:
        return []
    if col is None:
        return [{"n": int(len(valid)), "anchor_acc": round(float(valid["anchor_correct"].astype(float).mean()), 6)}]
    if col not in valid.columns:
        return []
    rows: List[Dict[str, Any]] = []
    for key, sub in valid.groupby(col, dropna=False):
        rows.append({col: str(key), "n": int(len(sub)), "anchor_acc": round(float(sub["anchor_correct"].astype(float).mean()), 6)})
    return rows


def find_item_level_path(recipe_path: Path, override_dir: Optional[Path] = None) -> Optional[Path]:
    stem = recipe_path.stem
    candidates = []
    if override_dir is not None:
        candidates.append(override_dir / f"{stem}_item_level.xlsx")
        candidates.append(override_dir / f"{stem}_item_level.csv")
    candidates.append(recipe_path.with_name(f"{stem}_item_level.xlsx"))
    candidates.append(recipe_path.with_name(f"{stem}_item_level.csv"))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def analyze_run(recipe_path: Path, item_dir: Optional[Path] = None) -> Dict[str, Any]:
    recipe_df = robust_load_table(str(recipe_path))
    item_path = find_item_level_path(recipe_path, item_dir)
    if item_path is not None:
        item_df = robust_load_table(str(item_path))
    else:
        item_df = reconstruct_item_df(recipe_df)

    if not item_df.empty:
        item_df = item_df.copy()
        item_df["gt_label"] = item_df["gt_label"].apply(normalize_label)
        item_df["pred_label"] = item_df["pred_label"].apply(normalize_label)

    return {
        "run_name": recipe_path.stem,
        "recipe_file": str(recipe_path),
        "item_file": str(item_path) if item_path else None,
        "recipe_metrics": recipe_metrics(recipe_df),
        "by_difficulty_recipe": grouped_recipe(recipe_df, "difficulty_tier"),
        "by_bucket_recipe": grouped_recipe(recipe_df, "bucket_primary"),
        "anchor_overall": grouped_anchor(recipe_df),
        "anchor_by_difficulty": grouped_anchor(recipe_df, "difficulty_tier"),
        "anchor_by_bucket": grouped_anchor(recipe_df, "bucket_primary"),
        "item_level": {
            "n_items": int(len(item_df)),
            "per_class": per_class_report(item_df),
            "confusion_4class": confusion_4class(item_df),
        },
    }


def collect_recipe_files(inputs: Iterable[str]) -> List[Path]:
    files: List[Path] = []
    for raw in inputs:
        p = Path(raw)
        if p.is_dir():
            for child in sorted(p.glob("*.xlsx")):
                if child.stem.endswith("_item_level"):
                    continue
                files.append(child)
            for child in sorted(p.glob("*.csv")):
                if child.stem.endswith("_item_level"):
                    continue
                files.append(child)
        else:
            files.append(p)
    dedup: List[Path] = []
    seen = set()
    for f in files:
        key = str(f.resolve()) if f.exists() else str(f)
        if key not in seen:
            seen.add(key)
            dedup.append(f)
    return dedup


def build_markdown(results: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    lines.append("# FoodAudit supplementary experiment analysis")
    lines.append("")
    lines.append("## Overall comparison")
    lines.append("")
    lines.append("| run | n | fine_acc | binary_acc | precision | recall | f1 | false_safe | false_risk | avg_latency |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in results:
        m = r["recipe_metrics"]
        lines.append(
            f"| {r['run_name']} | {m['n']} | {m['fine_acc']:.4f} | {m['binary_acc']:.4f} | "
            f"{m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} | "
            f"{m['false_safe_count']} ({m['false_safe_rate']:.4f}) | {m['false_risk_count']} ({m['false_risk_rate']:.4f}) | {m['avg_latency']:.4f} |"
        )
    lines.append("")

    for r in results:
        lines.append(f"## {r['run_name']}")
        lines.append("")
        lines.append(f"- recipe file: `{r['recipe_file']}`")
        if r["item_file"]:
            lines.append(f"- item file: `{r['item_file']}`")
        if r["anchor_overall"]:
            lines.append(f"- anchor accuracy: {r['anchor_overall'][0]['anchor_acc']:.4f} (n={r['anchor_overall'][0]['n']})")
        lines.append("")

        lines.append("### Per-class item-level report")
        lines.append("")
        lines.append("| label | support | precision | recall | f1 |")
        lines.append("|---|---:|---:|---:|---:|")
        for row in r["item_level"]["per_class"]:
            lines.append(f"| {row['label']} | {row['support']} | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} |")
        lines.append("")

        for title, key in (("Recipe by difficulty", "by_difficulty_recipe"), ("Recipe by bucket", "by_bucket_recipe")):
            lines.append(f"### {title}")
            lines.append("")
            rows = r[key]
            if not rows:
                lines.append("(empty)")
                lines.append("")
                continue
            first_key = [k for k in rows[0].keys() if k not in {"n", "fine_acc", "binary_acc", "avg_latency", "ci95", "precision", "recall", "f1", "false_safe_count", "false_safe_rate", "false_risk_count", "false_risk_rate"}][0]
            lines.append(f"| {first_key} | n | fine_acc | binary_acc | false_safe | false_risk |")
            lines.append("|---|---:|---:|---:|---:|---:|")
            for row in rows:
                lines.append(
                    f"| {row[first_key]} | {row['n']} | {row['fine_acc']:.4f} | {row['binary_acc']:.4f} | "
                    f"{row['false_safe_count']} ({row['false_safe_rate']:.4f}) | {row['false_risk_count']} ({row['false_risk_rate']:.4f}) |"
                )
            lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Post-process FoodAudit evaluator outputs into supplementary paper-ready statistics.")
    ap.add_argument("--inputs", nargs="+", required=True, help="Recipe-level xlsx/csv files or directories containing evaluator outputs")
    ap.add_argument("--item_dir", default="", help="Optional separate directory containing *_item_level.xlsx files")
    ap.add_argument("--output_prefix", required=True, help="Output prefix without extension")
    args = ap.parse_args()

    recipe_files = collect_recipe_files(args.inputs)
    item_dir = Path(args.item_dir) if args.item_dir else None
    results = [analyze_run(path, item_dir=item_dir) for path in recipe_files]

    output_prefix = Path(args.output_prefix)
    save_json(results, str(output_prefix.with_suffix(".json")))
    save_text(build_markdown(results), str(output_prefix.with_suffix(".md")))

    print(output_prefix.with_suffix(".json"))
    print(output_prefix.with_suffix(".md"))


if __name__ == "__main__":
    main()
