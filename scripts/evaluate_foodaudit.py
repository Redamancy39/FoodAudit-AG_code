import os
import re
import json
import time
import math
import argparse
import importlib.util
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
from tqdm import tqdm
#python FoodAudit_AG_Backend_Evaluator.py --input "E:\Chain of Thought in llm\scripts_database\test_final_with_difficulty_robust.xlsx" --output_dir "E:\Chain of Thought in llm\scripts_test\foodaudit_backend_robust_reports" --backend "E:\Chain of Thought in llm\scripts_website\GB2760_Backend2.py" --run_name "foodaudit_ag_backend"

FOUR_LABELS = ("SAFE", "SAFE_QS", "RISK_FORBIDDEN", "RISK_OVERLIMIT")


def robust_load_table(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if path.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    for enc in ["utf-8-sig", "gb18030", "gbk", "utf-8"]:
        try:
            return pd.read_csv(path, encoding=enc, engine="python", sep=None)
        except Exception:
            continue
    raise RuntimeError(f"Cannot read input file: {path}")


def save_xlsx(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_excel(path, index=False)


def normalize_pred_status(s: Any) -> str:
    s = str(s).strip().upper()
    if s in FOUR_LABELS:
        return s
    if s == "RISK":
        return "RISK_FORBIDDEN"
    if "SAFE_QS" in s or "QS" == s:
        return "SAFE_QS"
    if "RISK_FORBIDDEN" in s or "FORBIDDEN" in s:
        return "RISK_FORBIDDEN"
    if "RISK_OVERLIMIT" in s or "OVERLIMIT" in s:
        return "RISK_OVERLIMIT"
    if "SAFE" in s:
        return "SAFE"
    return "RISK_FORBIDDEN"


def normalize_gt(s: Any) -> str:
    s = str(s).strip().upper()
    if s in FOUR_LABELS:
        return s
    if s == "RISK":
        return "RISK_FORBIDDEN"
    if "SAFE_QS" in s or s == "QS":
        return "SAFE_QS"
    if "OVERLIMIT" in s:
        return "RISK_OVERLIMIT"
    if "FORBIDDEN" in s or "RISK" in s:
        return "RISK_FORBIDDEN"
    if "SAFE" in s:
        return "SAFE"
    return "RISK_FORBIDDEN"


def to_binary(label: str) -> str:
    label = str(label).upper()
    return "SAFE" if label in ("SAFE", "SAFE_QS") else "RISK"


def parse_gt_recipe(gt_text: Any) -> Dict[str, str]:
    if gt_text is None or (isinstance(gt_text, float) and math.isnan(gt_text)):
        return {}
    s = str(gt_text).strip()
    if not s:
        return {}
    # JSON dict
    if s.startswith("{") and s.endswith("}"):
        try:
            obj = json.loads(s)
            return {str(k).strip(): normalize_gt(v) for k, v in obj.items()}
        except Exception:
            pass
    # additive:label|additive:label
    items: Dict[str, str] = {}
    for part in s.split("|"):
        part = part.strip()
        if ":" in part:
            k, v = part.split(":", 1)
            items[k.strip()] = normalize_gt(v)
    return items


def fuzzy_match(a: str, b: str) -> bool:
    a = str(a).strip()
    b = str(b).strip()
    if not a or not b:
        return False
    return (a in b) or (b in a)


def difficulty_to_tier(x: Any) -> str:
    s = str(x).strip().upper()
    m = re.match(r"^(L\d)\b", s)
    if m:
        return m.group(1)
    m = re.match(r"^(L\d)", s)
    if m:
        return m.group(1)
    return "UNKNOWN"


def first_existing_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    return None


def build_query_from_row(row: pd.Series) -> str:
    # Prefer a prebuilt question/query column for consistency with baseline evaluation.
    for c in ["question", "query", "user_input", "input_text", "prompt"]:
        if c in row.index:
            val = str(row.get(c) or "").strip()
            if val:
                return val

    # Otherwise compose from available food + additive info.
    food = ""
    for c in ["food_name", "food_entity", "product_name", "product", "food", "name"]:
        if c in row.index:
            food = str(row.get(c) or "").strip()
            if food:
                break

    # Try an explicit additive text column first.
    additive_text = ""
    for c in ["additives", "additive_text", "ingredient_text", "items_text", "formula_text"]:
        if c in row.index:
            additive_text = str(row.get(c) or "").strip()
            if additive_text:
                break

    # Or derive from gt_map item names if nothing else exists.
    if not additive_text:
        gt_map = parse_gt_recipe(row.get("gt_map", row.get("ground_truth", "")))
        if gt_map:
            additive_text = "、".join([f"{k} 未提供" for k in gt_map.keys()])

    if food and additive_text:
        return f"请核查{food}配方中，{additive_text}的使用是否符合GB2760标准。"
    if food:
        return f"请核查{food}是否符合GB2760添加要求。"
    return str(row.to_dict())


def evaluate_one_sample(gt_map: Dict[str, str], pred_map: Dict[str, str]) -> Tuple[bool, bool, List[dict]]:
    details = []
    fine_all_ok = True
    bin_all_ok = True
    for gt_item, gt_label in gt_map.items():
        pred_label = "RISK_FORBIDDEN"
        matched = None
        for p_item, p_label in pred_map.items():
            if fuzzy_match(gt_item, p_item):
                matched = p_item
                pred_label = normalize_pred_status(p_label)
                break
        gt_label_n = normalize_gt(gt_label)
        pred_label_n = normalize_pred_status(pred_label)
        fine_ok = (gt_label_n == pred_label_n)
        bin_ok = (to_binary(gt_label_n) == to_binary(pred_label_n))
        fine_all_ok = fine_all_ok and fine_ok
        bin_all_ok = bin_all_ok and bin_ok
        details.append({
            "gt_item": gt_item,
            "gt_label": gt_label_n,
            "pred_item": matched,
            "pred_label": pred_label_n,
            "fine_ok": fine_ok,
            "bin_ok": bin_ok,
        })
    return fine_all_ok, bin_all_ok, details


def compute_metrics(df: pd.DataFrame) -> dict:
    if len(df) == 0:
        return {"n": 0, "fine_acc": 0.0, "binary_acc": 0.0, "avg_latency": 0.0}
    return {
        "n": int(len(df)),
        "fine_acc": float(df["fine_correct"].mean()) if "fine_correct" in df.columns else 0.0,
        "binary_acc": float(df["binary_correct"].mean()) if "binary_correct" in df.columns else 0.0,
        "avg_latency": float(df["latency"].mean()) if "latency" in df.columns else 0.0,
    }


def binary_prf_from_recipe_df(df: pd.DataFrame) -> dict:
    if len(df) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    tp = int(((df["gt_binary"] == "RISK") & (df["pred_binary"] == "RISK")).sum())
    fp = int(((df["gt_binary"] == "SAFE") & (df["pred_binary"] == "RISK")).sum())
    fn = int(((df["gt_binary"] == "RISK") & (df["pred_binary"] == "SAFE")).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"precision": float(prec), "recall": float(rec), "f1": float(f1), "tp": tp, "fp": fp, "fn": fn}


def bootstrap_ci(acc: float, n: int) -> List[float]:
    if n <= 0:
        return [0.0, 0.0]
    se = math.sqrt(max(acc * (1 - acc), 1e-12) / n)
    lo = max(0.0, acc - 1.96 * se)
    hi = min(1.0, acc + 1.96 * se)
    return [float(lo), float(hi)]


def load_backend_module(backend_path: str):
    spec = importlib.util.spec_from_file_location("gb2760_backend2_runtime", backend_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import backend from: {backend_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parse_backend_events(events: List[dict]) -> dict:
    result = {
        "food_raw": None,
        "anchor_name": None,
        "anchor_code": None,
        "anchor_full": None,
        "mapping_reason": None,
        "pred_map": {},
        "item_records": [],
        "error": None,
        "events": events,
    }
    for ev in events:
        step = ev.get("step")
        if step == "parsing_done":
            result["food_raw"] = ev.get("food")
        elif step == "mapping_success":
            result["mapping_reason"] = ev.get("reason")
            mapped = str(ev.get("mapped") or "")
            result["anchor_full"] = mapped
            m = re.match(r"^(.*?)\s*\(([^()]+)\)\s*$", mapped)
            if m:
                result["anchor_name"] = m.group(1).strip()
                result["anchor_code"] = m.group(2).strip()
            else:
                result["anchor_name"] = mapped.strip() or None
        elif step == "item_finished":
            name = str(ev.get("name") or "").strip()
            label = normalize_pred_status(ev.get("fine_label", ev.get("status", "RISK_FORBIDDEN")))
            trace = ev.get("trace") or {}
            result["pred_map"][name] = label
            result["item_records"].append({
                "pred_item": name,
                "pred_label": label,
                "analysis": ev.get("analysis"),
                "trace": trace,
            })
        elif step == "error":
            result["error"] = ev.get("message")
    return result


def main():
    ap = argparse.ArgumentParser(description="Backend-driven evaluator for FoodAudit-AG via GB2760_Backend2.py")
    ap.add_argument("--input", required=True, help="Path to input xlsx/csv")
    ap.add_argument("--output_dir", required=True, help="Directory for outputs")
    ap.add_argument("--backend", default="/mnt/data/GB2760_Backend2.py", help="Path to GB2760_Backend2.py")
    ap.add_argument("--run_name", default="foodaudit_ag_backend", help="Output file prefix")
    ap.add_argument("--limit", type=int, default=0, help="Optional max rows to evaluate")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    df = robust_load_table(args.input)
    if args.limit and args.limit > 0:
        df = df.head(args.limit).copy()

    id_col = first_existing_col(df, ["id", "sample_id", "case_id"])
    diff_col = first_existing_col(df, ["difficulty_tier", "difficulty_level", "difficulty", "tier"])
    bucket_col = first_existing_col(df, ["bucket_primary", "mechanism_bucket", "bucket", "bucket_type"])
    gt_col = first_existing_col(df, ["gt_map", "ground_truth", "gold", "labels"])
    anchor_gold_col = first_existing_col(df, ["anchored_category", "anchor_gold", "gold_anchor", "food_entity"])

    backend_mod = load_backend_module(args.backend)
    service = backend_mod.GB2760Service()
    service.initialize()

    recipe_rows: List[dict] = []
    item_rows: List[dict] = []

    for ridx, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="Evaluating FoodAudit-AG via backend"), start=1):
        sid = row[id_col] if id_col else ridx
        question = build_query_from_row(row)
        gt_map = parse_gt_recipe(row[gt_col]) if gt_col else {}
        difficulty_tier = difficulty_to_tier(row[diff_col]) if diff_col else "UNKNOWN"
        bucket_primary = str(row[bucket_col]).strip() if bucket_col else "UNKNOWN"
        gold_anchor = str(row[anchor_gold_col]).strip() if anchor_gold_col else ""

        t0 = time.time()
        events = list(service.process_query(question))
        latency = time.time() - t0
        parsed = parse_backend_events(events)
        pred_map = parsed["pred_map"]

        fine_ok, bin_ok, details = evaluate_one_sample(gt_map, pred_map)
        gt_binary = "SAFE" if all(to_binary(v) == "SAFE" for v in gt_map.values()) else "RISK"
        pred_binary = "SAFE" if all(to_binary(v) == "SAFE" for v in pred_map.values()) else "RISK"
        false_safe_recipe = int(gt_binary == "RISK" and pred_binary == "SAFE")

        anchor_correct = None
        if gold_anchor and parsed.get("anchor_name"):
            a = parsed["anchor_name"]
            anchor_correct = int(fuzzy_match(gold_anchor, a) or fuzzy_match(a, gold_anchor))

        recipe_rows.append({
            "id": sid,
            "difficulty_tier": difficulty_tier,
            "bucket_primary": bucket_primary,
            "question": question,
            "food_raw": parsed.get("food_raw") or "",
            "gold_anchor": gold_anchor,
            "pred_anchor_name": parsed.get("anchor_name") or "",
            "pred_anchor_code": parsed.get("anchor_code") or "",
            "anchor_reason": parsed.get("mapping_reason") or "",
            "anchor_correct": anchor_correct,
            "gt_map": json.dumps(gt_map, ensure_ascii=False),
            "pred_map": json.dumps(pred_map, ensure_ascii=False),
            "details": json.dumps(details, ensure_ascii=False),
            "trace": json.dumps(events, ensure_ascii=False),
            "gt_binary": gt_binary,
            "pred_binary": pred_binary,
            "fine_correct": int(fine_ok),
            "binary_correct": int(bin_ok),
            "false_safe_recipe": false_safe_recipe,
            "error": parsed.get("error") or "",
            "latency": latency,
        })

        for d in details:
            item_rows.append({
                "id": sid,
                "difficulty_tier": difficulty_tier,
                "bucket_primary": bucket_primary,
                "question": question,
                "gold_anchor": gold_anchor,
                "pred_anchor_name": parsed.get("anchor_name") or "",
                "pred_anchor_code": parsed.get("anchor_code") or "",
                "gt_item": d["gt_item"],
                "gt_label": d["gt_label"],
                "pred_item": d["pred_item"],
                "pred_label": d["pred_label"],
                "fine_ok": int(d["fine_ok"]),
                "bin_ok": int(d["bin_ok"]),
                "latency": latency,
            })

    recipe_df = pd.DataFrame(recipe_rows)
    item_df = pd.DataFrame(item_rows)

    # Summary
    recipe_metrics = compute_metrics(recipe_df)
    item_metrics = {
        "n": int(len(item_df)),
        "fine_acc": float(item_df["fine_ok"].mean()) if len(item_df) else 0.0,
        "binary_acc": float(item_df["bin_ok"].mean()) if len(item_df) else 0.0,
    }
    recipe_ci = bootstrap_ci(recipe_metrics["fine_acc"], recipe_metrics["n"])
    risk_metrics = binary_prf_from_recipe_df(recipe_df)

    by_difficulty = {}
    for tier, sub in recipe_df.groupby("difficulty_tier", dropna=False):
        by_difficulty[str(tier)] = compute_metrics(sub)

    by_bucket = {}
    for bucket, sub in recipe_df.groupby("bucket_primary", dropna=False):
        by_bucket[str(bucket)] = compute_metrics(sub)

    anchor_stats = {}
    if "anchor_correct" in recipe_df.columns and recipe_df["anchor_correct"].notna().any():
        valid = recipe_df[recipe_df["anchor_correct"].notna()].copy()
        anchor_stats = {
            "n": int(len(valid)),
            "acc": float(valid["anchor_correct"].astype(float).mean()) if len(valid) else 0.0,
        }

    false_safe_stats = {
        "count": int(recipe_df["false_safe_recipe"].sum()) if len(recipe_df) else 0,
        "rate": float(recipe_df["false_safe_recipe"].mean()) if len(recipe_df) else 0.0,
    }

    confusion = {}
    for gt in ["SAFE", "RISK"]:
        for pred in ["SAFE", "RISK"]:
            confusion[f"{gt}->{pred}"] = int(((recipe_df["gt_binary"] == gt) & (recipe_df["pred_binary"] == pred)).sum())

    summary = {
        "run_name": args.run_name,
        "recipe_level": recipe_metrics,
        "recipe_level_ci95": recipe_ci,
        "item_level": item_metrics,
        "binary_risk_state": risk_metrics,
        "anchor_stats": anchor_stats,
        "false_safe_recipe": false_safe_stats,
        "by_difficulty": by_difficulty,
        "by_bucket_primary": by_bucket,
        "binary_confusion_recipe": confusion,
    }

    # Save
    out_xlsx = os.path.join(args.output_dir, f"{args.run_name}.xlsx")
    out_item = os.path.join(args.output_dir, f"{args.run_name}_item_level.xlsx")
    out_json = os.path.join(args.output_dir, f"{args.run_name}_summary.json")
    out_md = os.path.join(args.output_dir, f"{args.run_name}_report.md")

    save_xlsx(recipe_df, out_xlsx)
    save_xlsx(item_df, out_item)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    md = []
    md.append(f"# {args.run_name} report\n")
    md.append("## Recipe-level\n")
    md.append(f"- N: {recipe_metrics['n']}\n")
    md.append(f"- Fine accuracy: {recipe_metrics['fine_acc']:.4f}\n")
    md.append(f"- Binary accuracy: {recipe_metrics['binary_acc']:.4f}\n")
    md.append(f"- Avg latency: {recipe_metrics['avg_latency']:.2f}s\n")
    md.append(f"- 95% CI: [{recipe_ci[0]:.4f}, {recipe_ci[1]:.4f}]\n")
    md.append("\n## Item-level\n")
    md.append(f"- N: {item_metrics['n']}\n")
    md.append(f"- Fine accuracy: {item_metrics['fine_acc']:.4f}\n")
    md.append(f"- Binary accuracy: {item_metrics['binary_acc']:.4f}\n")
    md.append("\n## Binary risk-state\n")
    md.append(f"- Precision: {risk_metrics['precision']:.4f}\n")
    md.append(f"- Recall: {risk_metrics['recall']:.4f}\n")
    md.append(f"- F1: {risk_metrics['f1']:.4f}\n")
    if anchor_stats:
        md.append("\n## Anchoring\n")
        md.append(f"- N with gold anchor: {anchor_stats['n']}\n")
        md.append(f"- Anchor accuracy: {anchor_stats['acc']:.4f}\n")
    md.append("\n## False-safe recipe\n")
    md.append(f"- Count: {false_safe_stats['count']}\n")
    md.append(f"- Rate: {false_safe_stats['rate']:.4f}\n")
    md.append("\n## By difficulty\n")
    for k, v in by_difficulty.items():
        md.append(f"- {k}: fine_acc={v['fine_acc']:.4f}, binary_acc={v['binary_acc']:.4f}, n={v['n']}\n")
    md.append("\n## By bucket\n")
    for k, v in by_bucket.items():
        md.append(f"- {k}: fine_acc={v['fine_acc']:.4f}, binary_acc={v['binary_acc']:.4f}, n={v['n']}\n")
    md.append("\n## Binary confusion (recipe-level)\n")
    for k, v in confusion.items():
        md.append(f"- {k}: {v}\n")

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("".join(md))

    print("Saved:")
    print(out_xlsx)
    print(out_item)
    print(out_json)
    print(out_md)


if __name__ == "__main__":
    main()
