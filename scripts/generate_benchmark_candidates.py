import os
import re
import json
import time
import random
import argparse
import hashlib
import traceback
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple, Callable

import pandas as pd
from neo4j import GraphDatabase

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# python GB2760_Dataset_Generator_v3.py --out "E:\Chain of Thought in llm\scripts_database\gb2760_candidate_pool_v3.xlsx" --n_L1 0 --n_L2_H 0 --n_L2_N 0 --n_L2_O 0 --n_L3 0 --n_L4_QS 35 --use_llm --progress_every 20 --autosave_every 25 --no_progress_limit 500

"""
GB2760_Dataset_Generator_v3.py

Purpose:
- Generate benchmark candidates aligned with the paper's current benchmark schema.
- Add robust debugging / progress logging / autosave.
- Fix L4_QS generation by building a distinct QS/GMP candidate pool first.
- Allow early stop for saturated buckets instead of infinite-looking loops.
- Prefer candidate-pool generation over repeated random probing where appropriate.

Important:
- This script generates benchmark candidates only.
- Final benchmark inclusion still requires human review and adjudication.
"""


# =========================
# Config
# =========================
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
DASHSCOPE_BASE_URL = os.getenv(
    "DASHSCOPE_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1"
)
DASHSCOPE_MODEL = os.getenv("DASHSCOPE_MODEL", "qwen3-max")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))

QS_MARKERS = {"-1GMP", "-1GM", "1GMP", "GMP", "QS", "适量", "QUANTUM SATIS", "AS NEEDED"}


# =========================
# Logging
# =========================
def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# =========================
# Helpers
# =========================
def is_qs_value(x: Any) -> bool:
    s = str(x).strip().upper()
    if not s:
        return False
    return any(m in s for m in ["-1GMP", "-1GM", "1GMP", "GMP", "QS", "适量", "QUANTUM", "SATIS"]) or s in QS_MARKERS


def first_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    m = re.search(r"([\d.]+)", str(x))
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def fmt_amount(value: float, unit: str = "g/kg") -> str:
    return f"{value:.2f}{unit}"


def normalize_recipe_label(labels: List[str]) -> str:
    labels = [str(x).strip().upper() for x in labels if x]
    if any(x == "RISK_FORBIDDEN" for x in labels):
        return "RISK_FORBIDDEN"
    if any(x == "RISK_OVERLIMIT" for x in labels):
        return "RISK_OVERLIMIT"
    if labels and all(x == "SAFE_QS" for x in labels):
        return "SAFE_QS"
    return "SAFE"


def summarize_counter(d: Dict[str, int], topk: int = 10) -> str:
    items = sorted(d.items(), key=lambda x: (-x[1], x[0]))
    items = items[:topk]
    return ", ".join([f"{k}={v}" for k, v in items]) if items else "(none)"


# =========================
# Data structures
# =========================
@dataclass
class AdditiveCase:
    additive_name: str
    amount_str: Optional[str]
    label: str
    mechanism: str
    evidence: Optional[Dict[str, Any]] = None


# =========================
# Neo4j access
# =========================
class NeoKG:
    def __init__(self, fetch_size: int = 1000):
        if not NEO4J_PASSWORD:
            raise RuntimeError("NEO4J_PASSWORD is empty. Please set it via environment variable.")
        self.driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD),
            fetch_size=fetch_size,
        )

    def close(self):
        self.driver.close()

    def ping(self) -> bool:
        q = "RETURN 1 AS ok"
        with self.driver.session() as s:
            r = s.run(q).single()
            return bool(r and r["ok"] == 1)

    def fetch_leaf_food_categories(self, limit: int = 5000) -> List[Dict[str, str]]:
        q = """
        MATCH (f:FoodCategory)
        WHERE NOT (f)-[:PARENT_OF]->(:FoodCategory)
        RETURN f.code AS code, f.name AS name
        LIMIT $limit
        """
        with self.driver.session() as s:
            return [r.data() for r in s.run(q, limit=limit)]

    def fetch_random_additives(self, limit: int = 4000) -> List[str]:
        q = "MATCH (a:Additive) RETURN a.name AS name LIMIT $limit"
        with self.driver.session() as s:
            return [r["name"] for r in s.run(q, limit=limit)]

    def pick_easy_safe(self) -> Optional[Tuple[Dict[str, str], Dict[str, Any]]]:
        q = """
        MATCH (f:FoodCategory)
        WHERE NOT (f)-[:PARENT_OF]->(:FoodCategory)
        MATCH (a:Additive)-[r:ALLOWED_IN]->(f)
        WHERE r.max_amount IS NOT NULL
        RETURN f.code AS code, f.name AS food, a.name AS additive,
               r.max_amount AS max_amount, r.unit AS unit, r.remark AS remark,
               f.name AS basis_name, f.code AS basis_code
        """
        with self.driver.session() as s:
            rows = [r.data() for r in s.run(q)]
        random.shuffle(rows)
        for row in rows[:1200]:
            lim = first_float(row.get("max_amount"))
            if lim is not None and lim > 0 and not is_qs_value(row.get("max_amount")):
                return {"code": row["code"], "name": row["food"]}, row
        return None

    def pick_overlimit(self) -> Optional[Tuple[Dict[str, str], Dict[str, Any]]]:
        q = """
        MATCH (f:FoodCategory)
        WHERE NOT (f)-[:PARENT_OF]->(:FoodCategory)
        MATCH (anc:FoodCategory)-[:PARENT_OF*0..6]->(f)
        MATCH (a:Additive)-[r:ALLOWED_IN]->(anc)
        WHERE r.max_amount IS NOT NULL
        RETURN f.code AS code, f.name AS food, anc.name AS basis_name, anc.code AS basis_code,
               a.name AS additive, r.max_amount AS max_amount, r.unit AS unit, r.remark AS remark
        """
        with self.driver.session() as s:
            rows = [r.data() for r in s.run(q)]
        random.shuffle(rows)
        for row in rows[:1500]:
            lim = first_float(row.get("max_amount"))
            if lim is not None and lim > 0 and not is_qs_value(row.get("max_amount")):
                return {"code": row["code"], "name": row["food"]}, row
        return None

    def pick_hierarchy_needed(self) -> Optional[Tuple[Dict[str, str], Dict[str, Any]]]:
        q = """
        MATCH (anc:FoodCategory)-[:PARENT_OF*1..6]->(f:FoodCategory)
        WHERE NOT (f)-[:PARENT_OF]->(:FoodCategory)
        MATCH (a:Additive)-[r:ALLOWED_IN]->(anc)
        WHERE NOT EXISTS { MATCH (a)-[:ALLOWED_IN]->(f) }
        RETURN f.code AS code, f.name AS food, anc.name AS basis_name, anc.code AS basis_code,
               a.name AS additive, r.max_amount AS max_amount, r.unit AS unit, r.remark AS remark
        """
        with self.driver.session() as s:
            rows = [r.data() for r in s.run(q)]
        random.shuffle(rows)
        for row in rows[:1500]:
            return {"code": row["code"], "name": row["food"]}, row
        return None

    def fetch_qs_candidate_pool(self, leaf_only: bool = False, limit: int = 50000) -> List[Dict[str, Any]]:
        """
        Build a distinct candidate pool for QS/GMP-style cases.
        If leaf_only=False, all food categories can be used as candidate anchors.
        If leaf_only=True, only leaf categories are kept.
        """
        food_filter = "WHERE NOT (f)-[:PARENT_OF]->(:FoodCategory)" if leaf_only else ""

        q = f"""
        MATCH (f:FoodCategory)
        {food_filter}
        MATCH (anc:FoodCategory)-[:PARENT_OF*0..6]->(f)
        MATCH (a:Additive)-[r:ALLOWED_IN]->(anc)
        WHERE r.max_amount IS NOT NULL
          AND (
                toUpper(toString(r.max_amount)) CONTAINS 'GMP'
             OR toUpper(toString(r.max_amount)) CONTAINS 'QS'
             OR toString(r.max_amount) CONTAINS '适量'
             OR toUpper(coalesce(toString(r.remark), '')) CONTAINS 'GMP'
             OR toUpper(coalesce(toString(r.remark), '')) CONTAINS 'QS'
             OR coalesce(toString(r.remark), '') CONTAINS '适量'
          )
        RETURN DISTINCT
            f.code AS code,
            f.name AS food,
            anc.code AS basis_code,
            anc.name AS basis_name,
            a.name AS additive,
            toString(r.max_amount) AS max_amount,
            coalesce(r.unit, '') AS unit,
            coalesce(toString(r.remark), '') AS remark
        LIMIT $limit
        """
        with self.driver.session() as s:
            rows = [r.data() for r in s.run(q, limit=limit)]
        return rows

    def is_allowed_anywhere_in_ancestry(self, food_code: str, additive_name: str) -> bool:
        q = """
        MATCH (f:FoodCategory {code:$code})
        MATCH (anc:FoodCategory)-[:PARENT_OF*0..6]->(f)
        MATCH (:Additive {name:$add})-[:ALLOWED_IN]->(anc)
        RETURN count(*) AS c
        """
        with self.driver.session() as s:
            r = s.run(q, code=food_code, add=additive_name).single()
            return bool(r and r["c"] > 0)

    def pick_forbidden(self, food: Dict[str, str], additive_pool: List[str], tries: int = 60) -> Optional[str]:
        for _ in range(tries):
            add = random.choice(additive_pool)
            if not self.is_allowed_anywhere_in_ancestry(food["code"], add):
                return add
        return None


# =========================
# Bucket / difficulty
# =========================
def difficulty_from_mechanisms(mechs: List[str]) -> Tuple[str, str, str]:
    mset = set(mechs)
    if mset == {"E"}:
        return "Easy-type", "Easy-type", "L1"
    if mset == {"QS"}:
        return "QS-type", "QS-type", "L4"

    core = mset.intersection({"H", "N", "O"})
    if len(core) == 1 and len(mset) == 1:
        m = next(iter(core))
        label = {"H": "H-type", "N": "N-type", "O": "O-type"}[m]
        return label, label, "L2"

    ordered = []
    for m in ["H", "N", "O", "QS"]:
        if m in mset:
            ordered.append({"H": "H-type", "N": "N-type", "O": "O-type", "QS": "QS-type"}[m])
    return ordered[0], "|".join(ordered), "L3"


# =========================
# LLM helpers
# =========================
def openai_client() -> Optional[OpenAI]:
    if not DASHSCOPE_API_KEY or OpenAI is None:
        return None
    return OpenAI(api_key=DASHSCOPE_API_KEY, base_url=DASHSCOPE_BASE_URL)


def llm_health_check(client: Optional[OpenAI]) -> None:
    if client is None:
        log("LLM health check skipped (client unavailable).")
        return
    prompt = 'Reply with JSON only: {"ok": true}'
    t0 = time.time()
    resp = client.chat.completions.create(
        model=DASHSCOPE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    dt = time.time() - t0
    msg = (resp.choices[0].message.content or "").strip()
    log(f"LLM health check ok in {dt:.2f}s; preview={msg[:80]!r}")


def build_question_template(food_name: str, cases: List[AdditiveCase]) -> str:
    joined = "、".join(
        [f"{c.additive_name} {c.amount_str}" if c.amount_str else c.additive_name for c in cases]
    )
    return f"请核查以下{food_name}配方的合规性：{joined}。"


def build_short_description(food_name: str, cases: List[AdditiveCase]) -> str:
    adds = "、".join([c.additive_name for c in cases])
    return f"配方审计候选样本，食品名称为{food_name}，涉及添加剂包括{adds}。"


def llm_polish_question(question: str, food_name: str, cases: List[AdditiveCase], client: Optional[OpenAI]) -> str:
    if client is None:
        return question
    items = [{"additive": c.additive_name, "amount": c.amount_str} for c in cases]
    prompt = f"""
你是食品合规审查数据集的出题助手。请将给定的题干草稿改写成自然、简洁、正式的中文问句。
要求：
- 不要改变食品名称和添加剂名称/用量
- 不要添加任何新的添加剂或数值
- 输出一句话即可

食品：{food_name}
配方项：{json.dumps(items, ensure_ascii=False)}
题干草稿：{question}
"""
    resp = client.chat.completions.create(
        model=DASHSCOPE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=LLM_TEMPERATURE,
    )
    txt = (resp.choices[0].message.content or "").strip().splitlines()[0].strip()
    return txt or question


def llm_short_description(food_name: str, cases: List[AdditiveCase], client: Optional[OpenAI]) -> str:
    if client is None:
        return build_short_description(food_name, cases)
    items = [{"additive": c.additive_name, "amount": c.amount_str} for c in cases]
    prompt = f"""
你是食品合规审查数据集的样本整理助手。请为一个候选配方样本写一句非常简短的中文描述。
要求：
- 只概述食品和涉及的添加剂，不给出法规结论
- 不改变食品名称和添加剂名称/用量
- 不添加不存在的工艺、口味、剂型、包装信息
- 输出一句话，不要超过35个汉字

食品：{food_name}
配方项：{json.dumps(items, ensure_ascii=False)}
"""
    resp = client.chat.completions.create(
        model=DASHSCOPE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=LLM_TEMPERATURE,
    )
    txt = (resp.choices[0].message.content or "").strip().splitlines()[0].strip()
    return txt or build_short_description(food_name, cases)


# =========================
# Serialization helpers
# =========================
def additive_list_text(cases: List[AdditiveCase]) -> str:
    return " | ".join([f"{c.additive_name}: {c.amount_str or 'N/A'}" for c in cases])


def quantity_info_text(cases: List[AdditiveCase]) -> str:
    return " | ".join([f"{c.additive_name}={c.amount_str}" for c in cases if c.amount_str])


def item_gold_map(cases: List[AdditiveCase]) -> str:
    return " | ".join([f"{c.additive_name}: {c.label}" for c in cases])


def evidence_summary(cases: List[AdditiveCase]) -> str:
    parts = []
    for c in cases:
        e = c.evidence or {}
        if not e:
            parts.append(f"{c.additive_name}: NO_AUTHORIZATION_IN_ANCESTRY")
        else:
            parts.append(
                f"{c.additive_name}: basis={e.get('basis_name','')}({e.get('basis_code','')}), "
                f"limit={e.get('max_amount','')}, unit={e.get('unit','')}, remark={e.get('remark','')}"
            )
    return " || ".join(parts)


def stable_signature(food: Dict[str, str], cases: List[AdditiveCase]) -> str:
    data = {
        "food_code": food["code"],
        "items": sorted([(c.additive_name, c.amount_str or "", c.label, c.mechanism) for c in cases])
    }
    return hashlib.md5(json.dumps(data, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


# =========================
# Candidate constructors
# =========================
def make_L1_easy(kg: NeoKG) -> Optional[Dict[str, Any]]:
    picked = kg.pick_easy_safe()
    if not picked:
        return None
    food, row = picked
    lim = first_float(row["max_amount"])
    amt = min(lim * 0.6, lim - 0.01) if lim and lim > 0.05 else (lim * 0.8 if lim else 0.02)
    cases = [AdditiveCase(row["additive"], fmt_amount(amt), "SAFE", "E", evidence=row)]
    bp, bm, diff = difficulty_from_mechanisms([c.mechanism for c in cases])
    return {"food": food, "cases": cases, "bucket_primary": bp, "bucket_multi": bm, "difficulty_level": diff}


def make_L2_single_H(kg: NeoKG) -> Optional[Dict[str, Any]]:
    picked = kg.pick_hierarchy_needed()
    if not picked:
        return None
    food, row = picked
    if is_qs_value(row.get("max_amount")) or is_qs_value(row.get("remark")):
        cases = [AdditiveCase(row["additive"], fmt_amount(5.0), "SAFE_QS", "H", evidence=row)]
    else:
        lim = first_float(row.get("max_amount"))
        if lim is None or lim <= 0:
            return None
        amt = min(lim * 0.6, lim - 0.01) if lim > 0.05 else lim * 0.8
        cases = [AdditiveCase(row["additive"], fmt_amount(max(amt, 0.01)), "SAFE", "H", evidence=row)]
    bp, bm, diff = difficulty_from_mechanisms([c.mechanism for c in cases])
    return {"food": food, "cases": cases, "bucket_primary": bp, "bucket_multi": bm, "difficulty_level": diff}


def make_L2_single_O(kg: NeoKG) -> Optional[Dict[str, Any]]:
    picked = kg.pick_overlimit()
    if not picked:
        return None
    food, row = picked
    lim = first_float(row.get("max_amount"))
    if lim is None or lim <= 0:
        return None
    cases = [AdditiveCase(row["additive"], fmt_amount(lim * 1.4 + 0.01), "RISK_OVERLIMIT", "O", evidence=row)]
    bp, bm, diff = difficulty_from_mechanisms([c.mechanism for c in cases])
    return {"food": food, "cases": cases, "bucket_primary": bp, "bucket_multi": bm, "difficulty_level": diff}


def make_L2_single_N(kg: NeoKG, additive_pool: List[str]) -> Optional[Dict[str, Any]]:
    foods = kg.fetch_leaf_food_categories(limit=3000)
    if not foods:
        return None
    food = random.choice(foods)
    add = kg.pick_forbidden(food, additive_pool)
    if not add:
        return None
    cases = [AdditiveCase(add, fmt_amount(0.5), "RISK_FORBIDDEN", "N", evidence=None)]
    bp, bm, diff = difficulty_from_mechanisms([c.mechanism for c in cases])
    return {"food": food, "cases": cases, "bucket_primary": bp, "bucket_multi": bm, "difficulty_level": diff}


def make_L4_QS_from_pool(qs_pool: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Pop one distinct QS/GMP candidate from a prebuilt pool.
    """
    if not qs_pool:
        return None

    row = qs_pool.pop()
    food = {"code": row["code"], "name": row["food"]}

    cases = [
        AdditiveCase(
            additive_name=row["additive"],
            amount_str="适量",
            label="SAFE_QS",
            mechanism="QS",
            evidence=row
        )
    ]

    bp, bm, diff = difficulty_from_mechanisms([c.mechanism for c in cases])
    return {
        "food": food,
        "cases": cases,
        "bucket_primary": bp,
        "bucket_multi": bm,
        "difficulty_level": diff
    }


def make_L3_mixed(kg: NeoKG, additive_pool: List[str]) -> Optional[Dict[str, Any]]:
    h = kg.pick_hierarchy_needed()
    if not h:
        return None
    food, rowH = h
    parts: List[AdditiveCase] = []

    if is_qs_value(rowH.get("max_amount")) or is_qs_value(rowH.get("remark")):
        parts.append(AdditiveCase(rowH["additive"], fmt_amount(2.0), "SAFE_QS", "H", evidence=rowH))
    else:
        limH = first_float(rowH.get("max_amount"))
        if limH and limH > 0:
            parts.append(AdditiveCase(rowH["additive"], fmt_amount(max(limH * 0.6, 0.01)), "SAFE", "H", evidence=rowH))

    o = kg.pick_overlimit()
    if o:
        _, rowO = o
        limO = first_float(rowO.get("max_amount"))
        if limO and limO > 0:
            parts.append(AdditiveCase(rowO["additive"], fmt_amount(limO * 1.4 + 0.01), "RISK_OVERLIMIT", "O", evidence=rowO))

    addN = kg.pick_forbidden(food, additive_pool, tries=40)
    if addN:
        parts.append(AdditiveCase(addN, fmt_amount(0.5), "RISK_FORBIDDEN", "N", evidence=None))

    if len(set([p.mechanism for p in parts if p.mechanism in {"H", "N", "O"}])) < 2:
        return None

    parts = parts[:3]
    bp, bm, diff = difficulty_from_mechanisms([c.mechanism for c in parts])
    return {"food": food, "cases": parts, "bucket_primary": bp, "bucket_multi": bm, "difficulty_level": diff}


# =========================
# Row builder
# =========================
def make_candidate_row(
    uid: int,
    inst: Dict[str, Any],
    client: Optional[OpenAI],
    use_llm: bool,
    stats: Dict[str, int],
) -> Dict[str, Any]:
    food = inst["food"]
    cases = inst["cases"]

    q0 = build_question_template(food["name"], cases)
    question = q0
    short_desc = build_short_description(food["name"], cases)

    if use_llm:
        t0 = time.time()
        question = llm_polish_question(q0, food["name"], cases, client)
        stats["llm_question_calls"] += 1
        stats["llm_question_seconds"] += int((time.time() - t0) * 1000)

        t1 = time.time()
        short_desc = llm_short_description(food["name"], cases, client)
        stats["llm_shortdesc_calls"] += 1
        stats["llm_shortdesc_seconds"] += int((time.time() - t1) * 1000)

    return {
        "id": uid,
        "type": "recipe_audit",
        "product_name": food["name"],
        "short_description": short_desc,
        "user_query": question,
        "additive_list": additive_list_text(cases),
        "quantity_info": quantity_info_text(cases),
        "anchored_category_name": food["name"],
        "anchored_category_code": food["code"],
        "item_level_gold": item_gold_map(cases),
        "recipe_level_gold": normalize_recipe_label([c.label for c in cases]),
        "bucket_primary": inst["bucket_primary"],
        "bucket_multi": inst["bucket_multi"],
        "difficulty_level": inst["difficulty_level"],
        "candidate_source": "kg_rule_sampling",
        "evidence_summary": evidence_summary(cases),
        "candidate_signature": stable_signature(food, cases),
        "annotation_status": "draft",

        "reviewer_a_item_labels": "",
        "reviewer_b_item_labels": "",
        "reviewer_a_recipe_label": "",
        "reviewer_b_recipe_label": "",
        "reviewer_a_anchor": "",
        "reviewer_b_anchor": "",

        "adjudicated_item_labels": "",
        "adjudicated_recipe_label": "",
        "adjudicated_anchor": "",

        "include_final_benchmark": "",
        "exclude_reason": "",
        "review_notes": "",
    }


def build_annotation_sheet(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "id",
        "product_name",
        "short_description",
        "user_query",
        "additive_list",
        "quantity_info",
        "anchored_category_name",
        "anchored_category_code",
        "item_level_gold",
        "recipe_level_gold",
        "bucket_primary",
        "bucket_multi",
        "difficulty_level",
        "evidence_summary",
        "reviewer_a_item_labels",
        "reviewer_b_item_labels",
        "reviewer_a_recipe_label",
        "reviewer_b_recipe_label",
        "reviewer_a_anchor",
        "reviewer_b_anchor",
        "adjudicated_item_labels",
        "adjudicated_recipe_label",
        "adjudicated_anchor",
        "include_final_benchmark",
        "exclude_reason",
        "review_notes",
    ]
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = ""
    return out[cols]


def autosave(out_path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    annotation_df = build_annotation_sheet(df)
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="candidate_pool")
        annotation_df.to_excel(writer, index=False, sheet_name="annotation_template")


# =========================
# Bucket runner
# =========================
def run_bucket(
    maker_name: str,
    need: int,
    maker: Callable[[], Optional[Dict[str, Any]]],
    rows: List[Dict[str, Any]],
    seen: set,
    uid_start: int,
    client: Optional[OpenAI],
    use_llm: bool,
    overall_stats: Dict[str, int],
    args,
    out_path: str
) -> Tuple[int, int]:
    bucket_start = time.time()
    made = 0
    tries = 0
    uid = uid_start
    no_progress_tries = 0
    fail_stats: Dict[str, int] = {
        "none_instance": 0,
        "duplicate_signature": 0,
        "row_build_error": 0,
    }

    log(f"----- Bucket {maker_name} started | target={need} -----")

    while made < need and tries < args.max_tries:
        tries += 1
        try:
            inst = maker()

            if not inst:
                fail_stats["none_instance"] += 1
                no_progress_tries += 1
                if tries % args.progress_every == 0:
                    elapsed = time.time() - bucket_start
                    log(
                        f"[progress:{maker_name}] made={made}/{need}, tries={tries}, "
                        f"elapsed={elapsed:.1f}s, no_progress={no_progress_tries}, "
                        f"fail_stats={summarize_counter(fail_stats)}"
                    )
                if no_progress_tries >= args.no_progress_limit:
                    log(f"[early_stop:{maker_name}] no new samples for {no_progress_tries} tries; likely saturated.")
                    break
                continue

            sig = stable_signature(inst["food"], inst["cases"])
            if sig in seen:
                fail_stats["duplicate_signature"] += 1
                no_progress_tries += 1
                if tries % args.progress_every == 0:
                    elapsed = time.time() - bucket_start
                    log(
                        f"[progress:{maker_name}] made={made}/{need}, tries={tries}, "
                        f"elapsed={elapsed:.1f}s, no_progress={no_progress_tries}, "
                        f"fail_stats={summarize_counter(fail_stats)}"
                    )
                if no_progress_tries >= args.no_progress_limit:
                    log(f"[early_stop:{maker_name}] no new samples for {no_progress_tries} tries; likely saturated.")
                    break
                continue

            row = make_candidate_row(uid, inst, client, use_llm, overall_stats)
            seen.add(sig)
            rows.append(row)
            uid += 1
            made += 1
            no_progress_tries = 0

            log(
                f"[ok:{maker_name}] made={made}/{need} | total_rows={len(rows)} | "
                f"id={row['id']} | food={row['product_name']} | bucket={row['bucket_primary']} | diff={row['difficulty_level']}"
            )

            if len(rows) % args.autosave_every == 0:
                tmp_out = os.path.splitext(out_path)[0] + ".autosave.xlsx"
                log(f"[autosave] writing {len(rows)} rows to {tmp_out} ...")
                autosave(tmp_out, rows)
                log("[autosave] done.")

            if tries % args.progress_every == 0:
                elapsed = time.time() - bucket_start
                log(
                    f"[progress:{maker_name}] made={made}/{need}, tries={tries}, "
                    f"elapsed={elapsed:.1f}s, no_progress={no_progress_tries}, "
                    f"fail_stats={summarize_counter(fail_stats)}"
                )

        except KeyboardInterrupt:
            log(f"[interrupt] user interrupted during maker={maker_name}")
            raise
        except Exception as e:
            fail_stats["row_build_error"] += 1
            no_progress_tries += 1
            log(f"[error:{maker_name}] tries={tries}, made={made}/{need}, err={type(e).__name__}: {e}")
            if tries % max(args.progress_every, 5) == 0:
                traceback.print_exc()
            if no_progress_tries >= args.no_progress_limit:
                log(f"[early_stop:{maker_name}] no new samples for {no_progress_tries} tries after repeated errors.")
                break

    bucket_elapsed = time.time() - bucket_start
    log(
        f"----- Bucket {maker_name} finished | made={made}/{need} | tries={tries} | "
        f"elapsed={bucket_elapsed:.1f}s | fail_stats={summarize_counter(fail_stats)} -----"
    )

    if made < need:
        log(f"[warning:{maker_name}] target not reached: need={need}, got={made}")

    return uid, made


# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Output .xlsx")
    ap.add_argument("--n_L1", type=int, default=30)
    ap.add_argument("--n_L2_H", type=int, default=15)
    ap.add_argument("--n_L2_N", type=int, default=15)
    ap.add_argument("--n_L2_O", type=int, default=15)
    ap.add_argument("--n_L3", type=int, default=60)
    ap.add_argument("--n_L4_QS", type=int, default=15)
    ap.add_argument("--use_llm", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_tries", type=int, default=10000)
    ap.add_argument("--progress_every", type=int, default=20)
    ap.add_argument("--autosave_every", type=int, default=25)
    ap.add_argument("--no_progress_limit", type=int, default=500)
    ap.add_argument("--qs_leaf_only", action="store_true", help="Restrict QS pool to leaf food categories only.")
    ap.add_argument("--qs_pool_limit", type=int, default=50000)
    ap.add_argument("--dry_run", action="store_true", help="Only test Neo4j/LLM connectivity and one sample per maker.")
    args = ap.parse_args()

    random.seed(args.seed)

    log("===== GB2760_Dataset_Generator_v3 started =====")
    log(f"Python PID={os.getpid()}")
    log(f"Output file: {args.out}")
    log(f"use_llm={args.use_llm}, model={DASHSCOPE_MODEL}, dry_run={args.dry_run}")
    log(
        f"Targets: L1={args.n_L1}, L2_H={args.n_L2_H}, L2_N={args.n_L2_N}, "
        f"L2_O={args.n_L2_O}, L3={args.n_L3}, L4_QS={args.n_L4_QS}"
    )
    log(
        f"Controls: max_tries={args.max_tries}, progress_every={args.progress_every}, "
        f"autosave_every={args.autosave_every}, no_progress_limit={args.no_progress_limit}"
    )

    t_all = time.time()
    kg = None

    try:
        t0 = time.time()
        log("Connecting to Neo4j ...")
        kg = NeoKG()
        ok = kg.ping()
        log(f"Neo4j ping ok={ok} in {time.time() - t0:.2f}s")

        t1 = time.time()
        log("Fetching additive pool ...")
        additive_pool = kg.fetch_random_additives(limit=4000)
        log(f"Fetched additive_pool size={len(additive_pool)} in {time.time() - t1:.2f}s")

        client = None
        if args.use_llm:
            log("Initializing LLM client ...")
            client = openai_client()
            if client is None:
                raise RuntimeError("use_llm specified but DASHSCOPE_API_KEY or openai client not available.")
            llm_health_check(client)
        else:
            log("LLM disabled. Candidate text fields will use deterministic templates.")

        log("Building QS candidate pool ...")
        t_qs = time.time()
        qs_pool = kg.fetch_qs_candidate_pool(leaf_only=args.qs_leaf_only, limit=args.qs_pool_limit)
        random.shuffle(qs_pool)
        log(
            f"QS candidate pool size={len(qs_pool)} in {time.time() - t_qs:.2f}s "
            f"(leaf_only={args.qs_leaf_only})"
        )
        if len(qs_pool) < args.n_L4_QS:
            log(f"[warning] QS pool size={len(qs_pool)} < target={args.n_L4_QS}")

        makers = [
            ("L1_easy", args.n_L1, lambda: make_L1_easy(kg)),
            ("L2_H", args.n_L2_H, lambda: make_L2_single_H(kg)),
            ("L2_N", args.n_L2_N, lambda: make_L2_single_N(kg, additive_pool)),
            ("L2_O", args.n_L2_O, lambda: make_L2_single_O(kg)),
            ("L3_mixed", args.n_L3, lambda: make_L3_mixed(kg, additive_pool)),
            ("L4_QS", args.n_L4_QS, lambda: make_L4_QS_from_pool(qs_pool)),
        ]

        if args.dry_run:
            log("Running DRY RUN mode...")
            for name, _, maker in makers:
                log(f"[dry_run] probing maker={name}")
                t_probe = time.time()
                inst = maker()
                if inst is None:
                    log(f"[dry_run] maker={name} returned None after {time.time() - t_probe:.2f}s")
                else:
                    log(
                        f"[dry_run] maker={name} ok after {time.time() - t_probe:.2f}s | "
                        f"food={inst['food']['name']} | bucket={inst['bucket_primary']} | diff={inst['difficulty_level']}"
                    )
            log("DRY RUN completed.")
            return

        rows: List[Dict[str, Any]] = []
        seen = set()
        uid = 1

        overall_stats: Dict[str, int] = {
            "llm_question_calls": 0,
            "llm_question_seconds": 0,
            "llm_shortdesc_calls": 0,
            "llm_shortdesc_seconds": 0,
        }

        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

        for maker_name, need, maker in makers:
            uid, _ = run_bucket(
                maker_name=maker_name,
                need=need,
                maker=maker,
                rows=rows,
                seen=seen,
                uid_start=uid,
                client=client,
                use_llm=args.use_llm,
                overall_stats=overall_stats,
                args=args,
                out_path=args.out
            )

        log("Writing final Excel ...")
        autosave(args.out, rows)
        log(f"Saved final Excel: {args.out}")

        df = pd.DataFrame(rows)
        log(f"Final rows={len(df)}")
        if not df.empty:
            log("Difficulty distribution:")
            log(df["difficulty_level"].value_counts().to_string())
            log("Primary bucket distribution:")
            log(df["bucket_primary"].value_counts().to_string())

        if overall_stats["llm_question_calls"] > 0:
            q_avg = overall_stats["llm_question_seconds"] / max(overall_stats["llm_question_calls"], 1) / 1000.0
            s_avg = overall_stats["llm_shortdesc_seconds"] / max(overall_stats["llm_shortdesc_calls"], 1) / 1000.0
            log(
                f"LLM stats | question_calls={overall_stats['llm_question_calls']}, avg_question_s={q_avg:.2f}, "
                f"shortdesc_calls={overall_stats['llm_shortdesc_calls']}, avg_shortdesc_s={s_avg:.2f}"
            )

        log(f"TOTAL elapsed={time.time() - t_all:.1f}s")
        log("===== Completed successfully =====")

    finally:
        if kg is not None:
            try:
                kg.close()
                log("Neo4j connection closed.")
            except Exception:
                pass


if __name__ == "__main__":
    main()
