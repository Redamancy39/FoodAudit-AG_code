import os
import re
import json
import time
import math
import argparse
import hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
from tqdm import tqdm

from neo4j import GraphDatabase
import torch
from sentence_transformers import SentenceTransformer, util
from openai import OpenAI

# python GB2760_Baseline_Evaluator.py ^
#   --input "E:\Chain of Thought in llm\scripts_database\test_final_with_difficulty.xlsx" ^
#   --output_dir "E:\Chain of Thought in llm\scripts_test\baseline_reports" ^
#   --vector_topk 8

# =========================
# Config (keep API_KEY placeholder)
# =========================
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")

EMBEDDING_MODEL_PATH = os.getenv("GB2760_EMBED_PATH", r"D:\foodllm_website\bge-small-zh-v1.5")

# user requested: keep placeholder, do NOT rely on env var
API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
BASE_URL = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
MODEL_NAME = os.getenv("DASHSCOPE_MODEL", "qwen3-max")
TEMPERATURE = 0.1

CACHE_DIR = os.getenv("GB2760_CACHE_DIR", "./cache_gb2760")
os.makedirs(CACHE_DIR, exist_ok=True)


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


# =========================
# Label normalization (aligned with Evaluator2.py)
# =========================
FOUR_LABELS = ("SAFE", "SAFE_QS", "RISK_FORBIDDEN", "RISK_OVERLIMIT")


def normalize_pred_status(s: str) -> str:
    s = str(s).strip().upper()
    if s in FOUR_LABELS:
        return s
    # legacy compat
    if s.lower() == "safe":
        return "SAFE"
    if s.lower() == "risk":
        return "RISK_FORBIDDEN"  # conservative mapping
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
    # dataset might contain "RISK" legacy
    if s == "RISK":
        return "RISK_FORBIDDEN"
    return "SAFE"  # fallback (should be rare)


def to_binary(label: str) -> str:
    label = str(label).upper()
    if label in ("SAFE", "SAFE_QS"):
        return "SAFE"
    if label.startswith("RISK"):
        return "RISK"
    return "RISK"


def parse_gt_recipe(gt_text: str) -> Dict[str, str]:
    """
    ground_truth format:
      "additiveA:SAFE|additiveB:RISK_FORBIDDEN|..."
    """
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
# Parser/Cleaner (aligned with GB2760_Backend2.py)
# =========================
INVALID_PATTERNS = ["是否符合", "是否合规", "用于", "标准", "GB2760", "使用情况"]


def clean_item_name(item_name: str) -> Optional[str]:
    if not item_name:
        return None
    if any(p in item_name for p in INVALID_PATTERNS):
        return None
    return str(item_name).strip()


def robust_json(s: str) -> Optional[dict]:
    """
    LLM may return json fenced / extra text.
    """
    if not s:
        return None
    s = s.strip()
    # remove code fences
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)
    # try direct
    try:
        return json.loads(s)
    except Exception:
        pass
    # try find first {...}
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
    """
    exact prompt copied from GB2760_Backend2.py style (same semantics).
    """
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
            extra_body={"enable_thinking": False},
        )
        return robust_json(res.choices[0].message.content)
    except Exception as e:
        return None


def cache_key(prefix: str, s: str) -> str:
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    return os.path.join(CACHE_DIR, f"{prefix}_{h}.json")


def cached_extract_structured(client: OpenAI, text: str) -> Optional[dict]:
    key = cache_key("parse_v2", f"{MODEL_NAME}|{TEMPERATURE}|{text}")
    if os.path.exists(key):
        try:
            return json.load(open(key, "r", encoding="utf-8"))
        except Exception:
            pass
    data = extract_structured_data_llm(client, text)
    if data is not None:
        try:
            json.dump(data, open(key, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
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
# Neo4j Access (baseline retrieval)
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

    def find_food_by_name_keyword(self, raw_food: str) -> Optional[dict]:
        """
        keyword baseline mapping (very simple): find best contains match
        """
        raw_food = str(raw_food).strip()
        if not raw_food:
            return None
        cypher = """
MATCH (c:FoodCategory)
WHERE c.name CONTAINS $q OR $q CONTAINS c.name
RETURN c.code AS code, c.name AS name
ORDER BY size(c.code) DESC
LIMIT 1
"""
        with self.driver.session() as session:
            rec = session.run(cypher, q=raw_food).single()
            return rec.data() if rec else None

    def retrieve_nohier(self, food_code: str, additive_name: str) -> List[dict]:
        cypher = """
MATCH (target:FoodCategory {code: $food_code})
MATCH (target)<-[r:ALLOWED_IN]-(a:Additive)
WHERE a.name CONTAINS $add_name OR $add_name IN a.synonyms
RETURN
  target.name as 依据分类,
  a.name as 添加剂,
  r.max_amount as 限量,
  r.unit as 单位,
  r.remark as 备注
"""
        with self.driver.session() as session:
            return [r.data() for r in session.run(cypher, food_code=food_code, add_name=additive_name)]

    def retrieve_hierarchy(self, food_code: str, additive_name: str) -> List[dict]:
        """
        identical to backend _retrieve_evidence cypher
        """
        cypher = """
MATCH (target:FoodCategory {code: $food_code})
MATCH (target)<-[:PARENT_OF*0..5]-(ancestor)
MATCH (ancestor)<-[r:ALLOWED_IN]-(a:Additive)
WHERE a.name CONTAINS $add_name OR $add_name IN a.synonyms
RETURN
  ancestor.name as 依据分类,
  a.name as 添加剂,
  r.max_amount as 限量,
  r.unit as 单位,
  r.remark as 备注
ORDER BY size(ancestor.code) DESC
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

    def map(self, raw: str, topk: int = 8) -> Tuple[Optional[dict], str]:
        raw = str(raw).strip()
        if not raw:
            return None, "empty raw food"
        q = self.embedder.encode(raw, convert_to_tensor=True)
        scores = util.cos_sim(q, self.emb)[0]
        vals, idxs = torch.topk(scores, k=min(topk, len(self.food_kb)))
        best = int(idxs[0])
        cand = self.food_kb[best]
        reason = f"vector_top1={cand['name']} score={float(vals[0]):.4f}"
        return cand, reason


# =========================
# Evidence -> Verdict (4-class, aligned with backend semantics)
# =========================
def is_qs_evidence(e: dict) -> bool:
    """
    aligned with backend: max_amount=-1 OR unit in {GMP,QS} OR remark contains '适量'
    """
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
    """
    unit ignored per user request; just extract the first number
    """
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


def verdict_from_evidence(food_name: str, item_name: str, amount_str: str, evidence: List[dict]) -> Tuple[str, str]:
    """
    Deterministic baseline verdict:
    - no evidence => RISK_FORBIDDEN
    - any QS evidence => SAFE_QS
    - else compare numeric: if actual > min(limit) => RISK_OVERLIMIT else SAFE
    """
    if not evidence:
        text = (
            f"❌ FORBIDDEN (whitelist negation)\n"
            f"Food={food_name} Additive={item_name}: No allowed entry found in GB2760 for target category or ancestors."
        )
        return text, "RISK_FORBIDDEN"

    # QS short-circuit
    for e in evidence[:5]:
        if is_qs_evidence(e):
            text = (
                f"✅ SAFE_QS\n"
                f"Food={food_name} Additive={item_name} Amount={amount_str if amount_str else 'N/A'}\n"
                f"Reason: QS/GMP detected in evidence."
            )
            return text, "SAFE_QS"

    # numeric compare (unit ignored)
    actual = parse_amount_float(amount_str)
    limits = []
    for e in evidence:
        try:
            ma = e.get("限量", None)
            if ma is None:
                continue
            v = float(str(ma))
            if v >= 0:
                limits.append(v)
        except Exception:
            continue

    if actual is None or not limits:
        # conservative: if cannot compare, treat as SAFE (you can switch to SAFE_QS/SAFE depending your dataset policy)
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


# =========================
# LLM baselines (vector_rag + llm_only)
# =========================
def llm_classify_with_evidence(client: OpenAI, question: str, food: str, item: str, amount: str, evidence: List[dict]) -> Tuple[str, str]:
    """
    vector_rag baseline: LLM decides among 4 labels using evidence.
    IMPORTANT: QS/GMP must map to SAFE_QS (not UNKNOWN).
    """
    ev_lines = []
    for e in evidence[:8]:
        ev_lines.append(
            f"- 分类依据: {e.get('依据分类')} | 添加剂: {e.get('添加剂')} | 限量: {e.get('限量')} | 单位: {e.get('单位')} | 备注: {e.get('备注')}"
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
            extra_body={"enable_thinking": False},
        )
        obj = robust_json(res.choices[0].message.content) or {}
        label = normalize_pred_status(obj.get("label", "RISK_FORBIDDEN"))
        rationale = str(obj.get("rationale") or "").strip()
        if not rationale:
            rationale = "LLM rationale unavailable."
        return rationale, label
    except Exception as e:
        # fallback to deterministic verdict
        return verdict_from_evidence(food, item, amount, evidence)


def llm_only_classify(client: OpenAI, question: str) -> Tuple[str, str]:
    """
    llm_only baseline: no Neo4j evidence; forced 4-class decision.
    """
    prompt = f"""
You are a food additive compliance auditor for GB 2760.
Given only the user question (no database access), output ONLY a JSON:
- label: one of ["SAFE","SAFE_QS","RISK_FORBIDDEN","RISK_OVERLIMIT"]
- rationale: short

Question: {question}
""".strip()

    try:
        res = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=TEMPERATURE,
            extra_body={"enable_thinking": False},
        )
        obj = robust_json(res.choices[0].message.content) or {}
        label = normalize_pred_status(obj.get("label", "RISK_FORBIDDEN"))
        rationale = str(obj.get("rationale") or "").strip() or "LLM rationale unavailable."
        return rationale, label
    except Exception:
        return "LLM call failed; default RISK_FORBIDDEN.", "RISK_FORBIDDEN"


# =========================
# Evaluation
# =========================
def compute_metrics(df: pd.DataFrame) -> dict:
    if len(df) == 0:
        return {"n": 0, "fine_acc": 0.0, "binary_acc": 0.0, "avg_latency": 0.0}
    return {
        "n": int(len(df)),
        "fine_acc": float(df["fine_correct"].mean()),
        "binary_acc": float(df["binary_correct"].mean()),
        "avg_latency": float(df["latency"].mean()),
    }


def evaluate_one_sample(gt_map: Dict[str, str], pred_map: Dict[str, str]) -> Tuple[bool, bool, List[dict]]:
    """
    aligned with Evaluator2 concept: recipe-level map compare with fuzzy match.
    """
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


# =========================
# Runner
# =========================
class BaselineRunner:
    def __init__(self, vector_topk: int = 8):
        # OpenAI compatible client for DashScope
        self.client = OpenAI(api_key=API_KEY, base_url=BASE_URL) if API_KEY else None
        self.neo = GBNeo4j(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
        self.embedder = SentenceTransformer(EMBEDDING_MODEL_PATH)
        self.mapper = FoodMapper(self.neo, self.embedder)
        self.vector_topk = vector_topk

    def close(self):
        self.neo.close()

    def predict_keyword_nohier(self, food_raw: str, items: List[ParsedItem]) -> Tuple[Dict[str, str], List[dict]]:
        """
        keyword mapping + no hierarchy retrieval
        """
        trace = []
        food_node = self.neo.find_food_by_name_keyword(food_raw)
        if not food_node:
            # no mapping -> treat as forbidden for all items
            pred = {it.name: "RISK_FORBIDDEN" for it in items}
            trace.append({"step": "keyword_map_fail", "food_raw": food_raw})
            return pred, trace

        trace.append({"step": "keyword_map", "food_raw": food_raw, "food_code": food_node["code"], "food_name": food_node["name"]})

        pred = {}
        for it in items:
            ev = self.neo.retrieve_nohier(food_node["code"], it.name)
            _, label = verdict_from_evidence(food_node["name"], it.name, it.amount, ev)
            pred[it.name] = label
        return pred, trace

    def predict_keyword_hierarchy(self, food_raw: str, items: List[ParsedItem]) -> Tuple[Dict[str, str], List[dict]]:
        food_node = self.neo.find_food_by_name_keyword(food_raw)
        trace = []
        if not food_node:
            pred = {it.name: "RISK_FORBIDDEN" for it in items}
            trace.append({"step": "keyword_map_fail", "food_raw": food_raw})
            return pred, trace

        trace.append({"step": "keyword_map", "food_raw": food_raw, "food_code": food_node["code"], "food_name": food_node["name"]})

        pred = {}
        for it in items:
            ev = self.neo.retrieve_hierarchy(food_node["code"], it.name)
            _, label = verdict_from_evidence(food_node["name"], it.name, it.amount, ev)
            pred[it.name] = label
        return pred, trace

    def predict_vector_rag(self, food_raw: str, items: List[ParsedItem]) -> Tuple[Dict[str, str], List[dict]]:
        """
        vector mapping + hierarchy retrieval + LLM decision (4-class)
        """
        trace = []
        cand, reason = self.mapper.map(food_raw, topk=self.vector_topk)
        if not cand:
            pred = {it.name: "RISK_FORBIDDEN" for it in items}
            trace.append({"step": "vector_map_fail", "food_raw": food_raw})
            return pred, trace

        trace.append({"step": "vector_map", "food_raw": food_raw, "food_code": cand["code"], "food_name": cand["name"], "reason": reason})

        pred = {}
        for it in items:
            ev = self.neo.retrieve_hierarchy(cand["code"], it.name)
            rationale, label = llm_classify_with_evidence(self.client, f"{food_raw} {it.name} {it.amount}".strip(), cand["name"], it.name, it.amount, ev)
            pred[it.name] = label
        return pred, trace

    def predict_llm_only(self, question: str, items: List[ParsedItem]) -> Tuple[Dict[str, str], List[dict]]:
        """
        No DB access; classify each item (or the whole question if items empty).
        """
        trace = [{"step": "llm_only"}]
        pred = {}
        if items:
            for it in items:
                rationale, label = llm_only_classify(self.client, f"{question}\nAdditive={it.name} Amount={it.amount}".strip())
                pred[it.name] = label
        else:
            # if no items parsed, still output empty pred_map (evaluation recipe may be empty)
            pass
        return pred, trace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, required=True, help="test_final_with_difficulty.xlsx")
    ap.add_argument("--output_dir", type=str, required=True, help="output directory")
    ap.add_argument("--vector_topk", type=int, default=8)
    args = ap.parse_args()

    df = robust_load_table(args.input)
    df = df.dropna(subset=["question", "ground_truth"]).reset_index(drop=True)

    runner = BaselineRunner(vector_topk=args.vector_topk)

    baselines = [
        ("keyword_nohier", runner.predict_keyword_nohier),
        ("keyword_hierarchy", runner.predict_keyword_hierarchy),
        ("vector_rag", runner.predict_vector_rag),
        ("llm_only", None),  # special case
    ]

    try:
        for name, fn in baselines:
            rows = []
            for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Running {name}"):
                q = str(row["question"])
                gt_map = parse_gt_recipe(row["ground_truth"])

                # Parse (aligned) + Clean
                t0 = time.time()
                parsed = cached_extract_structured(runner.client, q)
                food, items = normalize_parsed(parsed)
                if food is None:
                    food = str(row.get("food_entity") or "").strip() or "UNKNOWN"
                    items = []

                # Predict
                if name == "llm_only":
                    pred_map, trace = runner.predict_llm_only(q, items)
                else:
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
                    "fine_correct": bool(fine_ok),
                    "binary_correct": bool(bin_ok),
                    "latency": float(latency),
                })

            out_df = pd.DataFrame(rows)

            # metrics
            overall = compute_metrics(out_df)

            # stratified: L1-L4
            strat_L = out_df.groupby("difficulty_tier", dropna=False).apply(compute_metrics).to_dict()
            # stratified: bucket_primary
            strat_B = out_df.groupby("bucket_primary", dropna=False).apply(compute_metrics).to_dict()

            # save
            out_path = os.path.join(args.output_dir, f"{name}.xlsx")
            save_xlsx(out_df, out_path)

            # print summary
            print(f"\n==== {name} Summary ====")
            print(f"N={overall['n']}  Fine={overall['fine_acc']:.4f}  Binary={overall['binary_acc']:.4f}  Latency={overall['avg_latency']:.3f}s")
            print(f"Saved: {out_path}")

            # also save a compact overview json for paper writing
            meta = {
                "baseline": name,
                "model": MODEL_NAME,
                "temperature": TEMPERATURE,
                "overall": overall,
                "by_difficulty": strat_L,
                "by_bucket_primary": strat_B,
            }
            meta_path = os.path.join(args.output_dir, f"{name}_meta.json")
            json.dump(meta, open(meta_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    finally:
        runner.close()


if __name__ == "__main__":
    main()
