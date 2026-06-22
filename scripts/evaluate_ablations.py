# -*- coding: utf-8 -*-
"""
GB2760_Ablation_Evaluator_v2.py  (FINAL INTEGRATED)
==================================================
Purpose:
- Ablation evaluator aligned to backend logic (FoodAudit-AG, GB2760).
- Fixes:
  (1) ground_truth parsing drops items due to Chinese colon "："  -> fixed
  (2) limit selection across hierarchical evidence should be specificity-first,
      NOT global min across ancestors -> fixed
  (3) vector mapping should be backend-style: vector topK candidates + LLM choose -> fixed
- Adds:
  - --id_list / --max_n / --shuffle for cheap debug runs
  - pred_cache + --resume to avoid re-paying for re-runs
  - --only to run selected experiments

Notes:
- Unit conversion is intentionally ignored (per user requirement). Assumes dataset units are consistent.
- Keep API_KEY="" placeholder (user will fill manually).
"""

import os
import re
import json
import time
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


# =========================
# Config (keep API_KEY placeholder)
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


# =========================
# Labels (4-class)
# =========================
FOUR_LABELS = ("SAFE", "SAFE_QS", "RISK_FORBIDDEN", "RISK_OVERLIMIT")


def normalize_pred_status(s: str) -> str:
    s = str(s).strip().upper()
    if s in FOUR_LABELS:
        return s
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


# =========================
# Ground truth parser (FIXED ":" and "：")
# =========================
def parse_gt_recipe(gt_text: str) -> Dict[str, str]:
    """
    Parse ground_truth like:
      "A: SAFE | B: SAFE_QS | C：RISK_OVERLIMIT | ..."
    Supports both ':' and '：'.
    """
    items: Dict[str, str] = {}
    if gt_text is None:
        return items

    text = str(gt_text).replace("：", ":").replace("\u3000", " ").strip()
    parts = [p.strip() for p in text.split("|") if p.strip()]
    for part in parts:
        part = part.replace("\u3000", " ").strip()
        if ":" not in part:
            continue
        k, v = part.split(":", 1)
        k = k.strip()
        v = v.strip()
        if not k:
            continue
        items[k] = normalize_gt(v)
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
# JSON robustness + caching helpers
# =========================
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


def cache_key(prefix: str, s: str) -> str:
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    return os.path.join(CACHE_DIR, f"{prefix}_{h}.json")


# =========================
# Parser/Cleaner (aligned with backend)
# =========================
INVALID_PATTERNS = ["是否符合", "是否合规", "用于", "标准", "GB2760", "使用情况"]


def clean_item_name(item_name: str) -> Optional[str]:
    if not item_name:
        return None
    if any(p in item_name for p in INVALID_PATTERNS):
        return None
    return str(item_name).strip()


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
# Neo4j Access (FIX: return 依据编码)
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
  target.code as 依据编码,
  target.name as 依据分类,
  a.name as 添加剂,
  r.max_amount as 限量,
  r.unit as 单位,
  r.remark as 备注
"""
        with self.driver.session() as session:
            return [r.data() for r in session.run(cypher, food_code=food_code, add_name=additive_name)]

    def retrieve_hierarchy(self, food_code: str, additive_name: str) -> List[dict]:
        cypher = """
MATCH (target:FoodCategory {code: $food_code})
MATCH (target)<-[:PARENT_OF*0..5]-(ancestor)
MATCH (ancestor)<-[r:ALLOWED_IN]-(a:Additive)
WHERE a.name CONTAINS $add_name OR $add_name IN a.synonyms
RETURN
  ancestor.code as 依据编码,
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
# Food Mapper (vector topK candidates + LLM choose) (FIX: backend-style)
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

    def map_topk(self, raw: str, topk: int = 10) -> List[dict]:
        raw = str(raw).strip()
        if not raw:
            return []
        q = self.embedder.encode(raw, convert_to_tensor=True)
        scores = util.cos_sim(q, self.emb)[0]
        vals, idxs = torch.topk(scores, k=min(topk, len(self.food_kb)))
        out = []
        for v, idx in zip(vals, idxs):
            item = self.food_kb[int(idx)]
            out.append({"code": item["code"], "name": item["name"], "score": float(v)})
        return out


def cached_map_food_anchor(client: OpenAI, mapper: FoodMapper, raw_name: str, topk: int = 10) -> Tuple[Optional[dict], str]:
    """
    Backend-style mapping:
    - vector retrieve topk candidates
    - LLM chooses the best category among candidates
    - cached by raw_name + model+temp
    """
    raw_name = str(raw_name).strip()
    if not raw_name:
        return None, "empty raw_name"

    key = cache_key("food_anchor_v2", f"{MODEL_NAME}|{TEMPERATURE}|topk={topk}|{raw_name}")
    if os.path.exists(key):
        try:
            obj = json.load(open(key, "r", encoding="utf-8"))
            return obj.get("item"), obj.get("reason", "cached")
        except Exception:
            pass

    cands = mapper.map_topk(raw_name, topk=topk)
    if not cands:
        return None, "vector search empty"

    # concise candidate strings
    candidates = [f"{x['name']} (Code: {x['code']})" for x in cands]

    prompt = f"""
你是一名 GB2760 食品分类专家。
用户输入食品名称："{raw_name}"

候选分类（来自向量检索 Top{len(candidates)}）：
{json.dumps(candidates, ensure_ascii=False)}

任务：从候选中选择一个最能涵盖 "{raw_name}" 的分类。

规则：
- 如果输入是品牌名/俗称，选择对应产品类型；
- 若无完美匹配，选择最接近且更合理的上位概念；
- 优先选择更具体（更细粒度）的分类，但不要误选不相关子类。

只返回 JSON：
{{"target_name":"...","target_code":"...","reason":"..."}}
""".strip()

    try:
        res = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=TEMPERATURE,
        )
        parsed = robust_json(res.choices[0].message.content) or {}
        target_code = str(parsed.get("target_code") or "").strip()
        target_name = str(parsed.get("target_name") or "").strip()
        reason = str(parsed.get("reason") or "LLM choose among candidates").strip()

        chosen = None
        if target_code:
            for x in mapper.food_kb:
                if x["code"] == target_code:
                    chosen = x
                    break
        if chosen is None and target_name:
            for x in mapper.food_kb:
                if x["name"] == target_name:
                    chosen = x
                    break
        # fallback: use top1 vector if LLM output invalid
        if chosen is None:
            chosen = {"code": cands[0]["code"], "name": cands[0]["name"]}
            reason = f"fallback to vector_top1; llm_invalid target_code/name. top1={chosen['name']}"

        try:
            json.dump({"item": chosen, "reason": reason, "candidates": cands}, open(key, "w", encoding="utf-8"),
                      ensure_ascii=False, indent=2)
        except Exception:
            pass
        return chosen, reason
    except Exception as e:
        # fallback: vector top1
        chosen = {"code": cands[0]["code"], "name": cands[0]["name"]}
        return chosen, f"mapping LLM error: {e}; fallback vector_top1={chosen['name']}"


# =========================
# Evidence interpretation (FIX: specificity-first)
# =========================
def is_qs_evidence(e: dict) -> bool:
    # QS via max_amount == -1
    try:
        ma = e.get("限量", None)
        if ma is not None and str(ma).strip() != "":
            if float(ma) == -1.0:
                return True
    except Exception:
        pass
    # QS via unit
    unit = str(e.get("单位") or "").strip().upper()
    if unit in ("GMP", "QS"):
        return True
    # QS via remark
    remark = str(e.get("备注") or "")
    if ("适量" in remark) or ("按生产需要" in remark) or ("根据生产需要" in remark):
        return True
    return False


def parse_amount_float(amount_str: str) -> Optional[float]:
    """
    Unit conversion ignored. Extract numeric; return None for QS/适量.
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


def pick_most_specific_evidence(evidence: List[dict]) -> List[dict]:
    """
    Select evidence rows from the most specific category level (longest 依据编码).
    If 依据编码 missing, fallback to all evidence.
    """
    codes = [str(e.get("依据编码") or "").strip() for e in evidence]
    codes = [c for c in codes if c]
    if not codes:
        return evidence
    max_len = max(len(c) for c in codes)
    return [e for e in evidence if len(str(e.get("依据编码") or "")) == max_len]


def verdict_from_evidence(food_name: str, item_name: str, amount_str: str, evidence: List[dict]) -> Tuple[str, str]:
    """
    Deterministic fallback:
    - no evidence -> RISK_FORBIDDEN (negative whitelist)
    - QS in most-specific level -> SAFE_QS
    - else compare within most-specific level; if numeric compare unavailable -> SAFE
    """
    if not evidence:
        return "No allowed entry found (whitelist negation).", "RISK_FORBIDDEN"

    ev = pick_most_specific_evidence(evidence)

    # QS within most-specific
    for e in ev:
        if is_qs_evidence(e):
            return "QS/GMP detected (most-specific evidence level).", "SAFE_QS"

    actual = parse_amount_float(amount_str)
    limits: List[float] = []
    for e in ev:
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
        return "Numeric compare unavailable (most-specific evidence level).", "SAFE"

    limit = min(limits)  # only within the same specificity level
    if actual > limit:
        return f"actual({actual}) > limit({limit}) at most-specific level.", "RISK_OVERLIMIT"
    return f"actual({actual}) <= limit({limit}) at most-specific level.", "SAFE"


# =========================
# LLM judge (evidence-grounded 4-class)
# =========================
def llm_classify_with_evidence(client: OpenAI, question: str, food: str, item: str, amount: str, evidence: List[dict]) -> Tuple[str, str]:
    """
    Keep LLM judge for parity with backend agent behavior, but provide deterministic fallback.
    IMPORTANT: evidence may include multiple ancestors; we pass full evidence and explicit rule:
               choose most specific.
    """
    # include evidence lines with code to help LLM follow specificity rule
    ev_lines = []
    for e in evidence[:10]:
        ev_lines.append(
            f"- code={e.get('依据编码')} | 分类={e.get('依据分类')} | 添加剂={e.get('添加剂')} | 限量={e.get('限量')} | 单位={e.get('单位')} | 备注={e.get('备注')}"
        )
    ev_text = "\n".join(ev_lines) if ev_lines else "(no evidence)"

    prompt = f"""
You are a food additive compliance auditor for GB 2760.
Return ONLY a JSON object:
- label: one of ["SAFE","SAFE_QS","RISK_FORBIDDEN","RISK_OVERLIMIT"]
- rationale: short, evidence-grounded

Core rules (must follow strictly):
1) If no allowed entry is found in evidence -> RISK_FORBIDDEN. (negative whitelist)
2) When evidence contains multiple category levels (ancestors), DO NOT take the global minimum limit.
   Instead, use the MOST SPECIFIC category level (longest category code) as the governing basis.
3) If the most specific level indicates QS/GMP (e.g., max_amount=-1 OR unit=GMP/QS OR remark contains "适量"/"按生产需要") -> SAFE_QS.
4) Otherwise compare actual amount to the strictest numerical limit within the most specific level:
   - if actual > limit -> RISK_OVERLIMIT
   - else -> SAFE

Question: {question}
Mapped food category: {food}
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
        rationale = str(obj.get("rationale") or "").strip() or "LLM rationale unavailable."
        return rationale, label
    except Exception:
        # fallback deterministic (specificity-first)
        return verdict_from_evidence(food, item, amount, evidence)


def llm_only_classify(client: OpenAI, question: str) -> Tuple[str, str]:
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
    Recipe-level: all GT items must be correct.
    NOTE: Extra predicted items not in GT are currently not penalized (your existing design).
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


def save_xlsx(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_excel(path, index=False)


# =========================
# Ablation Runner
# =========================
class AblationRunner:
    def __init__(self, vector_topk: int = 10):
        self.client = OpenAI(api_key=API_KEY or "EMPTY", base_url=BASE_URL)
        self.neo = GBNeo4j(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
        self.embedder = SentenceTransformer(EMBEDDING_MODEL_PATH)
        self.mapper = FoodMapper(self.neo, self.embedder)
        self.vector_topk = vector_topk

    def close(self):
        self.neo.close()

    def map_food_vector_backend_style(self, food_raw: str) -> Tuple[Optional[dict], dict]:
        item, reason = cached_map_food_anchor(self.client, self.mapper, food_raw, topk=self.vector_topk)
        trace = {"step": "vector_topk_llm_choose", "food_raw": food_raw, "mapped": item, "reason": reason, "topk": self.vector_topk}
        return item, trace

    def map_food_keyword(self, food_raw: str) -> Tuple[Optional[dict], dict]:
        cand = self.neo.find_food_by_name_keyword(food_raw)
        trace = {"step": "keyword_map", "food_raw": food_raw, "mapped": cand}
        return cand, trace

    def predict(
        self,
        question: str,
        food_raw: str,
        items: List[ParsedItem],
        *,
        use_hierarchy: bool,
        use_whitelist_negation: bool,
        use_vector_mapping: bool,
        use_llm_judge: bool,
    ) -> Tuple[Dict[str, str], List[dict]]:
        trace = []

        # mapping
        if use_vector_mapping:
            food_node, t = self.map_food_vector_backend_style(food_raw)
        else:
            food_node, t = self.map_food_keyword(food_raw)
        trace.append(t)

        if not food_node:
            # mapping fail policy
            pred: Dict[str, str] = {}
            for it in items:
                if use_whitelist_negation:
                    pred[it.name] = "RISK_FORBIDDEN"
                else:
                    _, label = llm_only_classify(self.client, f"{question}\nAdditive={it.name} Amount={it.amount}".strip())
                    pred[it.name] = label
            trace.append({"step": "map_fail_policy", "use_whitelist_negation": use_whitelist_negation})
            return pred, trace

        food_code = food_node["code"]
        food_name = food_node["name"]

        pred: Dict[str, str] = {}
        for it in items:
            # retrieval
            if use_hierarchy:
                evidence = self.neo.retrieve_hierarchy(food_code, it.name)
                trace.append({"step": "retrieve_hierarchy", "item": it.name, "n": len(evidence)})
            else:
                evidence = self.neo.retrieve_nohier(food_code, it.name)
                trace.append({"step": "retrieve_nohier", "item": it.name, "n": len(evidence)})

            # whitelist negation OFF: if no evidence, don't force forbidden
            if (not evidence) and (not use_whitelist_negation):
                _, label = llm_only_classify(self.client, f"{question}\nAdditive={it.name} Amount={it.amount}".strip())
                pred[it.name] = label
                trace.append({"step": "whitelist_off_fallback_llm_only", "item": it.name, "label": label})
                continue

            # judge
            if use_llm_judge:
                _, label = llm_classify_with_evidence(self.client, question, food_name, it.name, it.amount, evidence)
                pred[it.name] = label
            else:
                _, label = verdict_from_evidence(food_name, it.name, it.amount, evidence)
                pred[it.name] = label

        return pred, trace


# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, required=True, help="test_final_with_difficulty.xlsx")
    ap.add_argument("--output_dir", type=str, required=True, help="output directory")
    ap.add_argument("--vector_topk", type=int, default=10, help="vector candidates for LLM choose (default 10)")
    ap.add_argument("--only", type=str, default=None,
                    help="comma-separated experiments to run, e.g., full_system,hierarchy_off")
    ap.add_argument("--id_list", type=str, default=None,
                    help="comma-separated ids to run, e.g., 3,7,9")
    ap.add_argument("--max_n", type=int, default=None,
                    help="debug: run only first N samples (after optional shuffle)")
    ap.add_argument("--shuffle", action="store_true",
                    help="shuffle dataset before slicing (for max_n)")
    ap.add_argument("--resume", action="store_true",
                    help="use per-sample pred cache to skip already computed items")
    args = ap.parse_args()

    df = pd.read_excel(args.input) if args.input.lower().endswith((".xlsx", ".xls")) else pd.read_csv(args.input)
    df = df.dropna(subset=["question", "ground_truth"]).reset_index(drop=True)

    # filter by ids
    if args.id_list:
        keep = set([x.strip() for x in args.id_list.split(",") if x.strip()])
        df = df[df["id"].astype(str).isin(keep)].reset_index(drop=True)

    if args.shuffle:
        df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    if args.max_n is not None:
        df = df.head(int(args.max_n)).reset_index(drop=True)

    runner = AblationRunner(vector_topk=args.vector_topk)

    # Experiments
    settings_all = [
        ("full_system", dict(use_hierarchy=True,  use_whitelist_negation=True,  use_vector_mapping=True,  use_llm_judge=True)),
        ("hierarchy_off", dict(use_hierarchy=False, use_whitelist_negation=True,  use_vector_mapping=True,  use_llm_judge=True)),
        ("whitelist_negation_off", dict(use_hierarchy=True,  use_whitelist_negation=False, use_vector_mapping=True,  use_llm_judge=True)),
        ("bi_retrieval_off", dict(use_hierarchy=True,  use_whitelist_negation=True,  use_vector_mapping=False, use_llm_judge=True)),
    ]

    if args.only:
        wanted = set([x.strip() for x in args.only.split(",") if x.strip()])
        settings = [(n, c) for (n, c) in settings_all if n in wanted]
        if not settings:
            raise ValueError(f"--only provided but no valid experiments matched: {args.only}")
    else:
        settings = settings_all

    try:
        for exp_name, cfg in settings:
            rows = []
            out_path = os.path.join(args.output_dir, f"{exp_name}.xlsx")
            meta_path = os.path.join(args.output_dir, f"{exp_name}_meta.json")
            os.makedirs(args.output_dir, exist_ok=True)

            for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Running {exp_name}"):
                q = str(row["question"])
                gt_map = parse_gt_recipe(row["ground_truth"])

                # parse cached
                t0 = time.time()
                parsed = cached_extract_structured(runner.client, q)
                food, items = normalize_parsed(parsed)

                # fallback for missing food/items (keep conservative)
                if food is None:
                    food = str(row.get("food_entity") or "").strip() or "UNKNOWN"
                    items = []

                # per-sample pred cache
                sid = str(row.get("id"))
                pred_cache_path = cache_key(f"pred_{exp_name}", f"{MODEL_NAME}|{TEMPERATURE}|{sid}")
                if args.resume and os.path.exists(pred_cache_path):
                    try:
                        obj = json.load(open(pred_cache_path, "r", encoding="utf-8"))
                        pred_map = obj.get("pred_map", {})
                        trace = obj.get("trace", [])
                        latency = float(obj.get("latency", 0.0))
                    except Exception:
                        pred_map, trace = runner.predict(question=q, food_raw=food, items=items, **cfg)
                        latency = time.time() - t0
                        json.dump({"pred_map": pred_map, "trace": trace, "latency": latency},
                                  open(pred_cache_path, "w", encoding="utf-8"),
                                  ensure_ascii=False, indent=2)
                else:
                    pred_map, trace = runner.predict(question=q, food_raw=food, items=items, **cfg)
                    latency = time.time() - t0
                    if args.resume:
                        try:
                            json.dump({"pred_map": pred_map, "trace": trace, "latency": latency},
                                      open(pred_cache_path, "w", encoding="utf-8"),
                                      ensure_ascii=False, indent=2)
                        except Exception:
                            pass

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
            overall = compute_metrics(out_df)

            cols = ["fine_correct", "binary_correct", "latency"]
            strat_L = out_df.groupby("difficulty_tier", dropna=False)[cols].apply(lambda x: {
                "n": int(len(x)),
                "fine_acc": float(x["fine_correct"].mean()) if len(x) else 0.0,
                "binary_acc": float(x["binary_correct"].mean()) if len(x) else 0.0,
                "avg_latency": float(x["latency"].mean()) if len(x) else 0.0,
            }).to_dict()

            strat_B = out_df.groupby("bucket_primary", dropna=False)[cols].apply(lambda x: {
                "n": int(len(x)),
                "fine_acc": float(x["fine_correct"].mean()) if len(x) else 0.0,
                "binary_acc": float(x["binary_correct"].mean()) if len(x) else 0.0,
                "avg_latency": float(x["latency"].mean()) if len(x) else 0.0,
            }).to_dict()

            save_xlsx(out_df, out_path)

            meta = {
                "experiment": exp_name,
                "model": MODEL_NAME,
                "temperature": TEMPERATURE,
                "vector_topk": args.vector_topk,
                "settings": cfg,
                "overall": overall,
                "by_difficulty": strat_L,
                "by_bucket_primary": strat_B,
            }
            json.dump(meta, open(meta_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

            print(f"\n==== {exp_name} Summary ====")
            print(f"N={overall['n']}  Fine={overall['fine_acc']:.4f}  Binary={overall['binary_acc']:.4f}  Latency={overall['avg_latency']:.3f}s")
            print(f"Saved: {out_path}")

    finally:
        runner.close()


if __name__ == "__main__":
    main()
