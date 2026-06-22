import os
import re
import json
import time
import math
import argparse
import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
from tqdm import tqdm

from neo4j import GraphDatabase
import torch
from sentence_transformers import SentenceTransformer, util
from openai import OpenAI

# python GB2760_Baseline_Evaluator_CRAG_LogicAware.py --input "E:\Chain of Thought in llm\scripts_database\test_final_with_difficulty.xlsx" --output_dir "E:\Chain of Thought in llm\scripts_test\baseline_reports_new" --vector_topk 8 --corrective_keep_n 3 --max_evidence 8

# =========================
# Config
# =========================
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")

EMBEDDING_MODEL_PATH = os.getenv("GB2760_EMBED_PATH", r"D:\foodllm_website\bge-small-zh-v1.5")

API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
BASE_URL = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
MODEL_NAME = os.getenv("DASHSCOPE_MODEL", "qwen3-max")
TEMPERATURE = 0.1

CACHE_DIR = os.getenv("GB2760_CACHE_DIR", "./cache_gb2760")
os.makedirs(CACHE_DIR, exist_ok=True)

FOUR_LABELS = ("SAFE", "SAFE_QS", "RISK_FORBIDDEN", "RISK_OVERLIMIT")


# =========================
# Helpers: IO
# =========================
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

    raise RuntimeError("Cannot read input file.")


def save_xlsx(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_excel(path, index=False)


def save_json(obj: Any, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_text(text: str, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


# =========================
# Label normalization
# =========================
def normalize_pred_status(s: str) -> str:
    s = str(s).strip().upper()
    if s in FOUR_LABELS:
        return s
    if s.lower() == "safe":
        return "SAFE"
    if s.lower() == "risk":
        return "RISK_FORBIDDEN"
    if s.lower() == "unknown":
        return "RISK_FORBIDDEN"
    if "FORBIDDEN" in s:
        return "RISK_FORBIDDEN"
    if "OVERLIMIT" in s:
        return "RISK_OVERLIMIT"
    if "SAFE_QS" in s or "QS" in s or "GMP" in s:
        return "SAFE_QS"
    if "SAFE" in s:
        return "SAFE"
    return "RISK_FORBIDDEN"


def normalize_gt(s: str) -> str:
    s = str(s).strip().upper()
    if "SAFE_QS" in s:
        return "SAFE_QS"
    if "RISK_FORBIDDEN" in s or "FORBIDDEN" in s:
        return "RISK_FORBIDDEN"
    if "RISK_OVERLIMIT" in s or "OVERLIMIT" in s:
        return "RISK_OVERLIMIT"
    if "SAFE" in s:
        return "SAFE"
    if s == "RISK":
        return "RISK_FORBIDDEN"
    return "SAFE"


def to_binary(label: str) -> str:
    label = str(label).upper()
    if label in ("SAFE", "SAFE_QS"):
        return "SAFE"
    if label.startswith("RISK"):
        return "RISK"
    return "RISK"


def parse_gt_recipe(gt_text: str) -> Dict[str, str]:
    items = {}
    for part in str(gt_text).split("|"):
        part = part.strip()
        if ":" in part:
            k, v = part.split(":", 1)
            items[k.strip()] = normalize_gt(v)
    return items


def fuzzy_match(a: str, b: str) -> bool:
    a = str(a).strip()
    b = str(b).strip()
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


# =========================
# Parser / Cleaner
# =========================
INVALID_PATTERNS = ["是否符合", "是否合规", "用于", "标准", "GB2760", "使用情况"]


def clean_item_name(item_name: str) -> Optional[str]:
    if not item_name:
        return None
    if any(p in item_name for p in INVALID_PATTERNS):
        return None
    return str(item_name).strip()


def robust_json(s: str) -> Optional[dict]:
    if not s:
        return None
    s = s.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)
    try:
        return json.loads(s)
    except Exception:
        pass
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


@dataclass
class ParsedItem:
    name: str
    amount: str


def extract_structured_data_llm(client: OpenAI, text: str) -> Optional[dict]:
    prompt = f"""
你是一个严谨的食品数据解析员。请从用户文本中提取：
1) 核心食品名称 (food_name)
2) 添加剂列表 (items)，包含名称(name)和用量(amount)。

【示例】
输入："核查芬达配方：爱德万甜 0.005g/kg、液体二氧化碳(煤气化法) 适量。"
输出：
{{
  "food_name": "芬达",
  "items": [
    {{"name":"爱德万甜","amount":"0.005g/kg"}},
    {{"name":"液体二氧化碳(煤气化法)","amount":"适量"}}
  ]
}}

【待处理文本】
{text}

注意：
- 若用户只是问“面包能加什么”，food_name提取“面包”，items 为空数组。
- 仅返回 JSON。
""".strip()

    try:
        res = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=TEMPERATURE,
        )
        return robust_json(res.choices[0].message.content)
    except Exception:
        return None


def cache_key(prefix: str, s: str) -> str:
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    return os.path.join(CACHE_DIR, f"{prefix}_{h}.json")


def cached_extract_structured(client: OpenAI, text: str) -> Optional[dict]:
    key = cache_key("parse_v2", f"{MODEL_NAME}|{TEMPERATURE}|{text}")
    if os.path.exists(key):
        try:
            with open(key, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    data = extract_structured_data_llm(client, text)
    if data is not None:
        try:
            with open(key, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
    return data


def normalize_parsed(parsed: dict) -> Tuple[Optional[str], List[ParsedItem]]:
    if not parsed:
        return None, []
    food = str(parsed.get("food_name") or "").strip()
    if not food:
        return None, []
    items_raw = parsed.get("items", []) or []
    items: List[ParsedItem] = []
    for it in items_raw:
        name = clean_item_name(str(it.get("name") or "").strip())
        if not name:
            continue
        amount = str(it.get("amount") or "").strip()
        items.append(ParsedItem(name=name, amount=amount))
    return food, items


# =========================
# Neo4j Access
# =========================
class GBNeo4j:
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def get_all_food_categories(self) -> List[dict]:
        cypher = "MATCH (c:FoodCategory) RETURN c.code AS code, c.name AS name"
        with self.driver.session() as session:
            return [r.data() for r in session.run(cypher)]

    def retrieve_hierarchy_with_meta(self, food_code: str, additive_name: str) -> List[dict]:
        cypher = """
MATCH (target:FoodCategory {code: $food_code})
MATCH path = (target)<-[:PARENT_OF*0..5]-(ancestor)
MATCH (ancestor)<-[r:ALLOWED_IN]-(a:Additive)
WHERE a.name CONTAINS $add_name OR $add_name IN a.synonyms
RETURN
  target.code as 目标分类编码,
  target.name as 目标分类,
  ancestor.code as 依据分类编码,
  ancestor.name as 依据分类,
  a.name as 添加剂,
  r.max_amount as 限量,
  r.unit as 单位,
  r.remark as 备注,
  length(path) as depth
ORDER BY depth ASC, size(ancestor.code) DESC
"""
        with self.driver.session() as session:
            return [r.data() for r in session.run(cypher, food_code=food_code, add_name=additive_name)]


# =========================
# Food Mapper (vector)
# =========================
class FoodMapper:
    def __init__(self, neo: GBNeo4j, embedder: SentenceTransformer):
        self.neo = neo
        self.embedder = embedder
        cats = neo.get_all_food_categories()
        self.food_kb = [{"code": c["code"], "name": c["name"]} for c in cats if c.get("code") and c.get("name")]
        if not self.food_kb:
            raise RuntimeError("FoodCategory empty in Neo4j.")
        self.names = [x["name"] for x in self.food_kb]
        self.emb = embedder.encode(self.names, convert_to_tensor=True)

    def map(self, raw: str, topk: int = 8) -> Tuple[Optional[dict], str, List[dict]]:
        raw = str(raw).strip()
        if not raw:
            return None, "empty raw food", []
        q = self.embedder.encode(raw, convert_to_tensor=True)
        scores = util.cos_sim(q, self.emb)[0]
        vals, idxs = torch.topk(scores, k=min(topk, len(self.food_kb)))

        cands = []
        for v, idx in zip(vals.tolist(), idxs.tolist()):
            item = self.food_kb[int(idx)]
            cands.append({"code": item["code"], "name": item["name"], "score": float(v)})

        best = cands[0]
        reason = f"vector_top1={best['name']} score={best['score']:.4f}"
        return {"code": best["code"], "name": best["name"]}, reason, cands


# =========================
# Evidence helpers
# =========================
def normalize_text(s: Any) -> str:
    return str(s or "").strip().lower()


def is_qs_evidence(e: dict) -> bool:
    try:
        ma = e.get("限量", None)
        if ma is not None and str(ma).strip() != "":
            if float(ma) == -1.0:
                return True
    except Exception:
        pass

    unit = str(e.get("单位") or "").strip().upper()
    if unit in ("GMP", "QS"):
        return True

    remark = str(e.get("备注") or "")
    if ("适量" in remark) or ("按生产需要" in remark) or ("根据生产需要" in remark):
        return True

    return False


def parse_amount_float(amount_str: str) -> Optional[float]:
    if not amount_str:
        return None
    s = str(amount_str)
    if "适量" in s or "QS" in s.upper() or "GMP" in s.upper():
        return None
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)", s)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def parse_limit_float(raw: Any) -> Optional[float]:
    if raw is None:
        return None
    try:
        v = float(str(raw))
        if v >= 0:
            return v
        return None
    except Exception:
        return None


def evidence_signature(e: dict) -> Tuple:
    return (
        e.get("目标分类编码"),
        e.get("依据分类编码"),
        e.get("依据分类"),
        e.get("添加剂"),
        str(e.get("限量")),
        e.get("单位"),
        e.get("备注"),
        e.get("depth"),
    )


def verdict_from_evidence(food_name: str, item_name: str, amount_str: str, evidence: List[dict]) -> Tuple[str, str]:
    if not evidence:
        text = (
            f"❌ FORBIDDEN (whitelist negation)\n"
            f"Food={food_name} Additive={item_name}: No allowed entry found in GB2760 for target category or ancestors."
        )
        return text, "RISK_FORBIDDEN"

    for e in evidence[:5]:
        if is_qs_evidence(e):
            text = (
                f"✅ SAFE_QS\n"
                f"Food={food_name} Additive={item_name} Amount={amount_str if amount_str else 'N/A'}\n"
                f"Reason: QS/GMP detected in evidence."
            )
            return text, "SAFE_QS"

    actual = parse_amount_float(amount_str)
    limits = []
    for e in evidence:
        v = parse_limit_float(e.get("限量"))
        if v is not None:
            limits.append(v)

    if actual is None or not limits:
        text = (
            f"✅ SAFE\n"
            f"Food={food_name} Additive={item_name} Amount={amount_str if amount_str else 'N/A'}\n"
            f"Reason: Non-QS evidence exists; numeric compare unavailable (ignored)."
        )
        return text, "SAFE"

    limit = min(limits)
    if actual > limit:
        text = (
            f"⚠️ RISK_OVERLIMIT\n"
            f"Food={food_name} Additive={item_name} Actual={actual} Limit={limit}\n"
            f"Reason: actual > limit (unit ignored by design)."
        )
        return text, "RISK_OVERLIMIT"

    text = (
        f"✅ SAFE\n"
        f"Food={food_name} Additive={item_name} Actual={actual} Limit={limit}\n"
        f"Reason: actual <= limit (unit ignored by design)."
    )
    return text, "SAFE"


def score_evidence_record(e: dict, item_name: str) -> float:
    depth = e.get("depth", 999)
    try:
        depth = int(depth)
    except Exception:
        depth = 999

    score = 0.0
    if depth == 0:
        score += 3.0
    elif depth == 1:
        score += 2.0
    elif depth == 2:
        score += 1.5
    elif depth >= 3:
        score += 1.0

    add_name = normalize_text(e.get("添加剂"))
    item_name_n = normalize_text(item_name)
    if add_name == item_name_n:
        score += 2.0
    elif item_name_n in add_name or add_name in item_name_n:
        score += 1.0

    if is_qs_evidence(e):
        score += 0.6

    unit = str(e.get("单位") or "").strip()
    remark = str(e.get("备注") or "").strip()
    if unit:
        score += 0.2
    if remark:
        score += 0.2
    if parse_limit_float(e.get("限量")) is not None:
        score += 0.4

    basis_code = str(e.get("依据分类编码") or "")
    target_code = str(e.get("目标分类编码") or "")
    if basis_code and target_code and basis_code == target_code:
        score += 1.0

    return score


def correct_retrieved_evidence(evidence: List[dict], item_name: str, keep_n: int = 3) -> Tuple[List[dict], dict]:
    if not evidence:
        return [], {
            "raw_n": 0,
            "kept_n": 0,
            "raw_depths": [],
            "kept_depths": [],
            "mode": "empty",
        }

    unique = []
    seen = set()
    for e in evidence:
        sig = evidence_signature(e)
        if sig not in seen:
            unique.append(dict(e))
            seen.add(sig)

    scored = []
    for e in unique:
        e2 = dict(e)
        e2["retrieval_score"] = round(score_evidence_record(e2, item_name), 4)
        scored.append(e2)

    scored.sort(key=lambda x: (-float(x.get("retrieval_score", 0.0)), int(x.get("depth", 999)), str(x.get("依据分类编码") or "")))

    depth0 = [e for e in scored if int(e.get("depth", 999)) == 0]
    depth1 = [e for e in scored if int(e.get("depth", 999)) == 1]

    if depth0:
        kept = depth0[:keep_n]
        if len(kept) < keep_n:
            kept.extend(depth1[: max(0, keep_n - len(kept))])
        mode = "prefer_depth0"
    else:
        min_depth = min(int(e.get("depth", 999)) for e in scored)
        same_min = [e for e in scored if int(e.get("depth", 999)) == min_depth]
        if len(same_min) >= keep_n:
            kept = same_min[:keep_n]
            mode = f"prefer_min_depth_{min_depth}"
        else:
            kept = same_min + [e for e in scored if int(e.get("depth", 999)) != min_depth][: max(0, keep_n - len(same_min))]
            mode = f"min_depth_plus_support_{min_depth}"

    final_kept = []
    best_limit_by_depth = {}
    for e in kept:
        depth = int(e.get("depth", 999))
        limit = parse_limit_float(e.get("限量"))
        if limit is None:
            final_kept.append(e)
            continue
        prev = best_limit_by_depth.get(depth)
        if prev is None or limit < prev["limit"]:
            best_limit_by_depth[depth] = {"limit": limit, "evidence": e}

    taken_sigs = set()
    for e in final_kept:
        taken_sigs.add(evidence_signature(e))
    for _, pack in sorted(best_limit_by_depth.items(), key=lambda kv: kv[0]):
        sig = evidence_signature(pack["evidence"])
        if sig not in taken_sigs:
            final_kept.append(pack["evidence"])
            taken_sigs.add(sig)

    final_kept.sort(key=lambda x: (int(x.get("depth", 999)), -float(x.get("retrieval_score", 0.0))))
    final_kept = final_kept[:keep_n]

    diag = {
        "raw_n": len(evidence),
        "dedup_n": len(unique),
        "kept_n": len(final_kept),
        "raw_depths": [int(e.get("depth", 999)) for e in evidence],
        "kept_depths": [int(e.get("depth", 999)) for e in final_kept],
        "mode": mode,
    }
    return final_kept, diag


def build_logic_aware_context(food_node: dict, item: ParsedItem, evidence: List[dict], max_evidence: int = 8) -> str:
    if not evidence:
        return "Anchored category: unknown\nEvidence path: (no evidence)"

    lines = []
    target_code = str(food_node.get("code") or "")
    target_name = str(food_node.get("name") or "")
    lines.append(f"Anchored category: {target_code} {target_name}".strip())
    lines.append(f"Additive under audit: {item.name}")
    lines.append("Evidence path (nearest valid ancestor first):")

    ordered = sorted(evidence, key=lambda x: (int(x.get("depth", 999)), str(x.get("依据分类编码") or "")))
    for e in ordered[:max_evidence]:
        depth = int(e.get("depth", 999))
        basis_code = str(e.get("依据分类编码") or "")
        basis_name = str(e.get("依据分类") or "")
        add_name = str(e.get("添加剂") or "")
        limit = str(e.get("限量") or "")
        unit = str(e.get("单位") or "")
        remark = str(e.get("备注") or "")
        lines.append(
            f"[depth={depth}] {basis_code} {basis_name} | additive={add_name} | limit={limit} | unit={unit} | remark={remark}".strip()
        )
    return "\n".join(lines)


# =========================
# LLM classifiers
# =========================
def llm_classify_with_evidence(client: OpenAI, question: str, food: str, item: str, amount: str, evidence: List[dict], max_evidence: int = 8) -> Tuple[str, str]:
    ev_lines = []
    for e in evidence[:max_evidence]:
        ev_lines.append(
            f"- 分类依据: {e.get('依据分类')} | 添加剂: {e.get('添加剂')} | 限量: {e.get('限量')} | 单位: {e.get('单位')} | 备注: {e.get('备注')} | depth: {e.get('depth')}"
        )
    ev_text = "\n".join(ev_lines) if ev_lines else "(no evidence)"

    prompt = f"""
You are a food additive compliance auditor for GB 2760.
Return ONLY a JSON object with fields:
- label: one of ["SAFE","SAFE_QS","RISK_FORBIDDEN","RISK_OVERLIMIT"]
- rationale: short, evidence-grounded

Rules:
1) If no allowed entry is found in evidence -> RISK_FORBIDDEN.
2) If evidence indicates QS/GMP (e.g., max_amount=-1 OR unit=GMP/QS OR remark mentions "适量"/"按生产需要") -> SAFE_QS.
3) Otherwise, compare actual amount to the strictest numerical limit:
   - if actual > limit -> RISK_OVERLIMIT
   - else -> SAFE

Question: {question}
Parsed food: {food}
Additive: {item}
Actual amount: {amount}

Evidence:
{ev_text}
""".strip()

    try:
        res = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=TEMPERATURE,
        )
        obj = robust_json(res.choices[0].message.content) or {}
        label = normalize_pred_status(obj.get("label", "RISK_FORBIDDEN"))
        rationale = str(obj.get("rationale") or "").strip()
        if not rationale:
            rationale = "LLM rationale unavailable."
        return rationale, label
    except Exception:
        return verdict_from_evidence(food, item, amount, evidence)


def llm_classify_logic_aware(client: OpenAI, question: str, food_node: dict, item: ParsedItem, evidence: List[dict], max_evidence: int = 8) -> Tuple[str, str]:
    ctx = build_logic_aware_context(food_node, item, evidence, max_evidence=max_evidence)
    prompt = f"""
You are a hierarchy-aware food additive compliance auditor for GB 2760.
Return ONLY a JSON object with fields:
- label: one of ["SAFE","SAFE_QS","RISK_FORBIDDEN","RISK_OVERLIMIT"]
- rationale: short, evidence-grounded

Decision protocol:
1) Treat the nearest path-valid rule along the anchored category path as primary evidence.
2) Prefer depth=0 over depth=1, and shallower evidence over deeper ancestors.
3) Do not combine unrelated ancestor entries into a broader permission.
4) If no path-valid authorization exists -> RISK_FORBIDDEN.
5) If the nearest valid rule is QS/GMP-like -> SAFE_QS.
6) Otherwise compare the actual amount against the strictest valid numeric limit among the nearest logically consistent evidence.
7) If actual > limit -> RISK_OVERLIMIT, else -> SAFE.

Question: {question}
Actual amount: {item.amount}

Path-aware evidence:
{ctx}
""".strip()

    try:
        res = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=TEMPERATURE,
        )
        obj = robust_json(res.choices[0].message.content) or {}
        label = normalize_pred_status(obj.get("label", "RISK_FORBIDDEN"))
        rationale = str(obj.get("rationale") or "").strip() or "LLM rationale unavailable."
        return rationale, label
    except Exception:
        return verdict_from_evidence(food_node.get("name", "UNKNOWN"), item.name, item.amount, evidence)


# =========================
# Evaluation
# =========================
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


def expand_item_level_records(row_df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for _, row in row_df.iterrows():
        try:
            details = json.loads(row["details"]) if isinstance(row["details"], str) else (row["details"] or [])
        except Exception:
            details = []
        for d in details:
            gt_label = normalize_gt(d.get("gt_label", "RISK_FORBIDDEN"))
            pred_label = normalize_pred_status(d.get("pred_label", "RISK_FORBIDDEN"))
            records.append({
                "id": row.get("id"),
                "difficulty_tier": row.get("difficulty_tier"),
                "bucket_primary": row.get("bucket_primary"),
                "gt_item": d.get("gt_item"),
                "pred_item": d.get("pred_item"),
                "gt_label": gt_label,
                "pred_label": pred_label,
                "fine_ok": bool(d.get("fine_ok", False)),
                "bin_ok": bool(d.get("bin_ok", False)),
                "gt_binary": to_binary(gt_label),
                "pred_binary": to_binary(pred_label),
            })
    return pd.DataFrame(records)


def compute_recipe_metrics(df: pd.DataFrame) -> dict:
    if len(df) == 0:
        return {"n": 0, "fine_acc": 0.0, "binary_acc": 0.0, "avg_latency": 0.0}
    return {
        "n": int(len(df)),
        "fine_acc": float(df["fine_correct"].mean()),
        "binary_acc": float(df["binary_correct"].mean()),
        "avg_latency": float(df["latency"].mean()),
    }


def compute_item_metrics(item_df: pd.DataFrame) -> dict:
    if len(item_df) == 0:
        return {"n_items": 0, "fine_acc": 0.0, "binary_acc": 0.0}
    return {
        "n_items": int(len(item_df)),
        "fine_acc": float(item_df["fine_ok"].mean()),
        "binary_acc": float(item_df["bin_ok"].mean()),
    }


def confusion_matrix_dict(item_df: pd.DataFrame) -> dict:
    mat = {gt: {pred: 0 for pred in FOUR_LABELS} for gt in FOUR_LABELS}
    if len(item_df) == 0:
        return mat
    for _, r in item_df.iterrows():
        gt = normalize_gt(r["gt_label"])
        pred = normalize_pred_status(r["pred_label"])
        mat[gt][pred] += 1
    return mat


def per_class_report(item_df: pd.DataFrame) -> List[dict]:
    rows = []
    if len(item_df) == 0:
        for label in FOUR_LABELS:
            rows.append({
                "label": label,
                "support": 0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
            })
        return rows

    for label in FOUR_LABELS:
        tp = int(((item_df["gt_label"] == label) & (item_df["pred_label"] == label)).sum())
        fp = int(((item_df["gt_label"] != label) & (item_df["pred_label"] == label)).sum())
        fn = int(((item_df["gt_label"] == label) & (item_df["pred_label"] != label)).sum())
        support = int((item_df["gt_label"] == label).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        rows.append({
            "label": label,
            "support": support,
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
        })
    return rows


def grouped_metrics_recipe(df: pd.DataFrame, col: str) -> List[dict]:
    rows = []
    if col not in df.columns:
        return rows
    for k, sub in df.groupby(col, dropna=False):
        m = compute_recipe_metrics(sub)
        rows.append({col: k, **m})
    return rows


def grouped_metrics_item(item_df: pd.DataFrame, col: str) -> List[dict]:
    rows = []
    if col not in item_df.columns:
        return rows
    for k, sub in item_df.groupby(col, dropna=False):
        m = compute_item_metrics(sub)
        rows.append({col: k, **m})
    return rows


def top_error_cases(df: pd.DataFrame, topn: int = 30) -> List[dict]:
    bad = df[df["fine_correct"] == False].copy()
    if len(bad) == 0:
        return []
    bad = bad.sort_values(["difficulty_tier", "bucket_primary", "latency"], ascending=[True, True, False])
    cols = [
        "id", "difficulty_tier", "bucket_primary", "question", "food_entity",
        "gt_map", "pred_map", "details", "trace", "latency"
    ]
    out = []
    for _, row in bad.head(topn).iterrows():
        out.append({c: row.get(c) for c in cols})
    return out


def build_markdown_report(name: str, recipe_metrics: dict, item_metrics: dict, by_diff_recipe: List[dict], by_bucket_recipe: List[dict], by_diff_item: List[dict], by_bucket_item: List[dict], per_class: List[dict]) -> str:
    lines = []
    lines.append(f"# {name} report")
    lines.append("")
    lines.append("## Overall")
    lines.append("")
    lines.append(f"- Recipe-level N: {recipe_metrics['n']}")
    lines.append(f"- Recipe-level fine accuracy: {recipe_metrics['fine_acc']:.4f}")
    lines.append(f"- Recipe-level binary accuracy: {recipe_metrics['binary_acc']:.4f}")
    lines.append(f"- Average latency: {recipe_metrics['avg_latency']:.4f}s")
    lines.append(f"- Item-level N: {item_metrics['n_items']}")
    lines.append(f"- Item-level fine accuracy: {item_metrics['fine_acc']:.4f}")
    lines.append(f"- Item-level binary accuracy: {item_metrics['binary_acc']:.4f}")
    lines.append("")

    lines.append("## Per-class item-level report")
    lines.append("")
    lines.append("| label | support | precision | recall | f1 |")
    lines.append("|---|---:|---:|---:|---:|")
    for r in per_class:
        lines.append(f"| {r['label']} | {r['support']} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1']:.4f} |")
    lines.append("")

    def add_table(title: str, rows: List[dict], key: str):
        lines.append(f"## {title}")
        lines.append("")
        if not rows:
            lines.append("(empty)")
            lines.append("")
            return
        header = f"| {key} | n | fine_acc | binary_acc | avg_latency |"
        sep = "|---|---:|---:|---:|---:|"
        lines.append(header)
        lines.append(sep)
        for r in rows:
            lines.append(f"| {r.get(key)} | {r.get('n', 0)} | {r.get('fine_acc', 0.0):.4f} | {r.get('binary_acc', 0.0):.4f} | {r.get('avg_latency', 0.0):.4f} |")
        lines.append("")

    def add_item_table(title: str, rows: List[dict], key: str):
        lines.append(f"## {title}")
        lines.append("")
        if not rows:
            lines.append("(empty)")
            lines.append("")
            return
        header = f"| {key} | n_items | fine_acc | binary_acc |"
        sep = "|---|---:|---:|---:|"
        lines.append(header)
        lines.append(sep)
        for r in rows:
            lines.append(f"| {r.get(key)} | {r.get('n_items', 0)} | {r.get('fine_acc', 0.0):.4f} | {r.get('binary_acc', 0.0):.4f} |")
        lines.append("")

    add_table("Recipe-level by difficulty", by_diff_recipe, "difficulty_tier")
    add_table("Recipe-level by bucket", by_bucket_recipe, "bucket_primary")
    add_item_table("Item-level by difficulty", by_diff_item, "difficulty_tier")
    add_item_table("Item-level by bucket", by_bucket_item, "bucket_primary")
    return "\n".join(lines)


# =========================
# Runner
# =========================
class BaselineRunner:
    def __init__(self, vector_topk: int = 8, corrective_keep_n: int = 3, max_evidence: int = 8):
        self.client = OpenAI(api_key=API_KEY, base_url=BASE_URL) if API_KEY else None
        self.neo = GBNeo4j(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
        self.embedder = SentenceTransformer(EMBEDDING_MODEL_PATH)
        self.mapper = FoodMapper(self.neo, self.embedder)
        self.vector_topk = vector_topk
        self.corrective_keep_n = corrective_keep_n
        self.max_evidence = max_evidence

    def close(self):
        self.neo.close()

    def predict_corrective_vector_rag(self, food_raw: str, items: List[ParsedItem]) -> Tuple[Dict[str, str], List[dict]]:
        trace = []
        cand, reason, cands = self.mapper.map(food_raw, topk=self.vector_topk)
        if not cand:
            pred = {it.name: "RISK_FORBIDDEN" for it in items}
            trace.append({"step": "vector_map_fail", "food_raw": food_raw})
            return pred, trace

        trace.append({
            "step": "vector_map",
            "food_raw": food_raw,
            "food_code": cand["code"],
            "food_name": cand["name"],
            "reason": reason,
            "top_candidates": cands,
        })

        pred = {}
        for it in items:
            raw_ev = self.neo.retrieve_hierarchy_with_meta(cand["code"], it.name)
            corr_ev, diag = correct_retrieved_evidence(raw_ev, it.name, keep_n=self.corrective_keep_n)
            rationale, label = llm_classify_with_evidence(
                self.client,
                f"{food_raw} {it.name} {it.amount}".strip(),
                cand["name"],
                it.name,
                it.amount,
                corr_ev,
                max_evidence=self.max_evidence,
            )
            pred[it.name] = label
            trace.append({
                "step": "corrective_retrieval",
                "item": it.name,
                "amount": it.amount,
                "raw_evidence_n": len(raw_ev),
                "corrected_evidence_n": len(corr_ev),
                "correction_diag": diag,
                "corrected_evidence": corr_ev,
                "rationale": rationale,
                "pred_label": label,
            })
        return pred, trace

    def predict_logic_aware_hierarchy_rag(self, food_raw: str, items: List[ParsedItem]) -> Tuple[Dict[str, str], List[dict]]:
        trace = []
        cand, reason, cands = self.mapper.map(food_raw, topk=self.vector_topk)
        if not cand:
            pred = {it.name: "RISK_FORBIDDEN" for it in items}
            trace.append({"step": "vector_map_fail", "food_raw": food_raw})
            return pred, trace

        trace.append({
            "step": "vector_map",
            "food_raw": food_raw,
            "food_code": cand["code"],
            "food_name": cand["name"],
            "reason": reason,
            "top_candidates": cands,
        })

        pred = {}
        for it in items:
            ev = self.neo.retrieve_hierarchy_with_meta(cand["code"], it.name)
            rationale, label = llm_classify_logic_aware(
                self.client,
                question=f"{food_raw} {it.name} {it.amount}".strip(),
                food_node=cand,
                item=it,
                evidence=ev,
                max_evidence=self.max_evidence,
            )
            pred[it.name] = label
            trace.append({
                "step": "logic_aware_reasoning",
                "item": it.name,
                "amount": it.amount,
                "evidence_n": len(ev),
                "path_context": build_logic_aware_context(cand, it, ev, max_evidence=self.max_evidence),
                "rationale": rationale,
                "pred_label": label,
            })
        return pred, trace


# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, required=True, help="test_final_with_difficulty.xlsx")
    ap.add_argument("--output_dir", type=str, required=True, help="output directory")
    ap.add_argument("--vector_topk", type=int, default=8)
    ap.add_argument("--corrective_keep_n", type=int, default=3)
    ap.add_argument("--max_evidence", type=int, default=8)
    args = ap.parse_args()

    df = robust_load_table(args.input)
    df = df.dropna(subset=["question", "ground_truth"]).reset_index(drop=True)

    runner = BaselineRunner(
        vector_topk=args.vector_topk,
        corrective_keep_n=args.corrective_keep_n,
        max_evidence=args.max_evidence,
    )

    baselines = [
        ("corrective_vector_rag", runner.predict_corrective_vector_rag),
        ("logic_aware_hierarchy_rag", runner.predict_logic_aware_hierarchy_rag),
    ]

    try:
        for name, fn in baselines:
            rows = []
            for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Running {name}"):
                q = str(row["question"])
                gt_map = parse_gt_recipe(row["ground_truth"])

                t0 = time.time()
                parsed = cached_extract_structured(runner.client, q)
                food, items = normalize_parsed(parsed)
                if food is None:
                    food = str(row.get("food_entity") or "").strip() or "UNKNOWN"
                    items = []

                pred_map, trace = fn(food, items)
                latency = time.time() - t0

                fine_ok, bin_ok, details = evaluate_one_sample(gt_map, pred_map)

                rows.append({
                    "id": row.get("id"),
                    "type": row.get("type"),
                    "question": q,
                    "food_entity": row.get("food_entity"),
                    "bucket_primary": row.get("bucket_primary"),
                    "bucket_multi": row.get("bucket_multi"),
                    "difficulty_level": row.get("difficulty_level"),
                    "difficulty_tier": difficulty_to_tier(row.get("difficulty_level")),
                    "gt_map": json.dumps(gt_map, ensure_ascii=False),
                    "pred_map": json.dumps(pred_map, ensure_ascii=False),
                    "details": json.dumps(details, ensure_ascii=False),
                    "trace": json.dumps(trace, ensure_ascii=False),
                    "parsed_food": food,
                    "parsed_items": json.dumps([it.__dict__ for it in items], ensure_ascii=False),
                    "fine_correct": bool(fine_ok),
                    "binary_correct": bool(bin_ok),
                    "latency": float(latency),
                })

            out_df = pd.DataFrame(rows)
            item_df = expand_item_level_records(out_df)

            recipe_metrics = compute_recipe_metrics(out_df)
            item_metrics = compute_item_metrics(item_df)
            by_diff_recipe = grouped_metrics_recipe(out_df, "difficulty_tier")
            by_bucket_recipe = grouped_metrics_recipe(out_df, "bucket_primary")
            by_diff_item = grouped_metrics_item(item_df, "difficulty_tier")
            by_bucket_item = grouped_metrics_item(item_df, "bucket_primary")
            per_class = per_class_report(item_df)
            conf4 = confusion_matrix_dict(item_df)
            error_cases = top_error_cases(out_df, topn=30)

            # detailed outputs
            out_path = os.path.join(args.output_dir, f"{name}.xlsx")
            save_xlsx(out_df, out_path)

            item_path = os.path.join(args.output_dir, f"{name}_item_level.xlsx")
            save_xlsx(item_df, item_path)

            summary_obj = {
                "baseline": name,
                "model": MODEL_NAME,
                "temperature": TEMPERATURE,
                "vector_topk": args.vector_topk,
                "corrective_keep_n": args.corrective_keep_n,
                "max_evidence": args.max_evidence,
                "recipe_metrics": recipe_metrics,
                "item_metrics": item_metrics,
                "by_difficulty_recipe": by_diff_recipe,
                "by_bucket_recipe": by_bucket_recipe,
                "by_difficulty_item": by_diff_item,
                "by_bucket_item": by_bucket_item,
                "per_class_item": per_class,
                "confusion_4class_item": conf4,
                "top_error_cases": error_cases,
            }
            meta_path = os.path.join(args.output_dir, f"{name}_summary.json")
            save_json(summary_obj, meta_path)

            md_report = build_markdown_report(
                name=name,
                recipe_metrics=recipe_metrics,
                item_metrics=item_metrics,
                by_diff_recipe=by_diff_recipe,
                by_bucket_recipe=by_bucket_recipe,
                by_diff_item=by_diff_item,
                by_bucket_item=by_bucket_item,
                per_class=per_class,
            )
            md_path = os.path.join(args.output_dir, f"{name}_report.md")
            save_text(md_report, md_path)

            print(f"\n==== {name} Summary ====")
            print(f"Recipe-level: N={recipe_metrics['n']} Fine={recipe_metrics['fine_acc']:.4f} Binary={recipe_metrics['binary_acc']:.4f} Latency={recipe_metrics['avg_latency']:.3f}s")
            print(f"Item-level:   N={item_metrics['n_items']} Fine={item_metrics['fine_acc']:.4f} Binary={item_metrics['binary_acc']:.4f}")
            print(f"Saved sample-level: {out_path}")
            print(f"Saved item-level:   {item_path}")
            print(f"Saved summary:      {meta_path}")
            print(f"Saved markdown:     {md_path}")

    finally:
        runner.close()


if __name__ == "__main__":
    main()
