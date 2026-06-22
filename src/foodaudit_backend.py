import os
import json
import re
import logging
import torch
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer, util
from openai import OpenAI

# ================= 配置区域 =================
# 建议用环境变量，不要把 Key 写进代码
# setx DASHSCOPE_API_KEY "xxx"
API_CONFIG = {
    "api_key": os.getenv("DASHSCOPE_API_KEY", ""),
    "base_url": os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    "model": os.getenv("DASHSCOPE_MODEL", "qwen3-max"),  # ✅ 改为 qwen3-max
}

NEO4J_CONFIG = {
    "uri": os.getenv("NEO4J_URI", "bolt://localhost:7687"),
    "user": os.getenv("NEO4J_USER", "neo4j"),
    "password": os.getenv("NEO4J_PASSWORD", ""),
}

# 本地路径配置（保持你原有逻辑）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TREE_FILE = os.path.join(r"E:\Chain of Thought in llm\output", "GB2760_Category_Tree.jsonl")

# 本地向量模型路径（保持你原有逻辑）
EMBEDDING_MODEL_PATH = r"D:\foodllm_website\bge-small-zh-v1.5"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class GB2760Service:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GB2760Service, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance

    # -------------------- 初始化 --------------------
    def initialize(self):
        if self.initialized:
            return

        logging.info("🚀 [Backend] 正在初始化服务...")

        # 1) Neo4j
        try:
            self.driver = GraphDatabase.driver(
                NEO4J_CONFIG["uri"],
                auth=(NEO4J_CONFIG["user"], NEO4J_CONFIG["password"]),
            )
            self.driver.verify_connectivity()
            logging.info("✅ Neo4j 连接成功")
        except Exception as e:
            logging.error(f"❌ Neo4j 连接失败: {e}")
            raise

        # 2) LLM Client (DashScope OpenAI-compatible)
        if not API_CONFIG["api_key"]:
            logging.warning("⚠️ DASHSCOPE_API_KEY 未设置，LLM 相关功能将失败。")
        self.client = OpenAI(api_key=API_CONFIG["api_key"], base_url=API_CONFIG["base_url"])

        # 3) Embedding
        logging.info(f"🧠 正在加载本地向量模型: {EMBEDDING_MODEL_PATH} ...")
        try:
            if os.path.exists(EMBEDDING_MODEL_PATH):
                self.embedder = SentenceTransformer(EMBEDDING_MODEL_PATH)
            else:
                logging.warning("⚠️ 本地 embedding 路径不存在，改用在线模型 BAAI/bge-small-zh-v1.5")
                self.embedder = SentenceTransformer("BAAI/bge-small-zh-v1.5")
            logging.info("✅ Embedding 模型加载成功")
        except Exception as e:
            logging.error(f"❌ Embedding 模型加载失败: {e}")
            raise

        # 4) Load category tree
        self.food_kb = self._load_category_tree()
        self._vectorize_categories()

        self.initialized = True
        logging.info("✅ 服务初始化完成")

    def _load_category_tree(self):
        kb = []
        if os.path.exists(TREE_FILE):
            with open(TREE_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        inner = data.get("data", data)
                        name = inner.get("name", "")
                        code = inner.get("cns_code") or inner.get("code", "")
                        if name and code:
                            kb.append({"code": code, "name": name, "full": f"{code} {name}"})
                    except:
                        continue
        logging.info(f"📚 加载了 {len(kb)} 个 GB2760 标准分类节点")
        return kb

    def _vectorize_categories(self):
        if not self.food_kb:
            return
        logging.info("📊 正在为分类树计算语义向量...")
        names = [x["name"] for x in self.food_kb]
        self.category_embeddings = self.embedder.encode(names, convert_to_tensor=True)
        logging.info("✅ 向量化完成")

    # -------------------- 工具函数 --------------------
    def _robust_json(self, text: str):
        try:
            return json.loads(text)
        except:
            m = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group(1))
                except:
                    pass
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group())
                except:
                    pass
        return None

    def _parse_amount_float(self, amount_str: str):
        if not amount_str:
            return None
        s = str(amount_str).strip()
        if not s:
            return None
        # 如果用户写“适量/QS/GMP”，视为无数值
        if any(k in s.upper() for k in ["适量", "QS", "GMP", "N/A"]):
            return None
        m = re.search(r"(-?[\d.]+)", s)
        if not m:
            return None
        try:
            return float(m.group(1))
        except:
            return None

    def _is_qs_evidence(self, e: dict) -> bool:
        """
        ✅ 核心：结构化 QS/GMP 判别
        图谱中 QS 常存为：max_amount=-1 且 unit="GMP"
        """
        if not e:
            return False

        lim = e.get("限量")  # r.max_amount
        unit = str(e.get("单位") or "").strip().upper()
        remark = str(e.get("备注") or "").strip()

        # 1) 数值哨兵：-1
        try:
            if lim is not None and float(lim) == -1.0:
                return True
        except:
            pass

        # 2) unit 直接是 GMP/QS
        if unit in {"GMP", "QS"}:
            return True

        # 3) remark 语义
        if any(k in remark for k in ["适量", "按生产需要", "按需要", "量按生产需要"]):
            return True

        # 4) 字符串兜底（兼容少量历史数据）
        s = str(lim).strip().upper()
        if any(k in s for k in ["-1GMP", "1GMP", "GMP", "QS", "适量"]):
            return True

        return False

    def _clean_item_name(self, item_name: str):
        INVALID_PATTERNS = ["是否符合", "是否合规", "用于", "标准", "GB2760", "使用情况"]
        if not item_name:
            return None
        if any(p in item_name for p in INVALID_PATTERNS):
            return None
        return item_name.strip()

    # -------------------- 主流水线 --------------------
    def process_query(self, user_input: str):
        """
        Pipeline: Extraction -> Mapping -> Retrieval -> Verdict
        """
        yield {"step": "parsing", "message": "🧠 正在解析配方结构 (Extraction)..."}

        parsed_data = self._extract_structured_data(user_input)
        if not parsed_data or not parsed_data.get("food_name"):
            yield {"step": "error", "message": "❌ 无法识别有效的食品名称。请尝试更清晰的表述。"}
            return

        target_food_raw = parsed_data["food_name"]
        items = parsed_data.get("items", []) or []
        yield {"step": "parsing_done", "food": target_food_raw, "count": len(items)}

        yield {"step": "anchoring", "message": f"⚓ 正在进行语义映射: '{target_food_raw}' -> GB2760 标准分类..."}

        food_anchor, mapping_reason = self._map_food_anchor(target_food_raw)
        if not food_anchor:
            yield {"step": "error", "message": f"⚠️ 系统无法将“{target_food_raw}”映射到任何已知的 GB2760 分类。"}
            return

        yield {
            "step": "mapping_success",
            "original": target_food_raw,
            "mapped": f"{food_anchor['name']} ({food_anchor['code']})",
            "reason": mapping_reason,
        }

        yield {"step": "retrieving", "message": "🔍 正在 GB2760 图谱中检索..."}

        if not items:
            yield {"step": "result", "content": f"已定位标准分类：**{food_anchor['name']}**。请提供要查询的添加剂与用量。"}
            return

        for item in items:
            add_name = self._clean_item_name(item.get("name"))
            if not add_name:
                continue

            amount = item.get("amount")  # 可能 None

            evidence_records = self._retrieve_evidence(food_anchor["code"], add_name)
            analysis_text, fine_status = self._generate_verdict(
                food_anchor["name"], add_name, amount, evidence_records
            )

            trace_data = {
                "food_anchor": f"{food_anchor['name']} ({food_anchor['code']})",
                "top_evidence": [e for e in evidence_records[:3]],
            }

            yield {
                "step": "item_finished",
                "name": add_name,
                "analysis": analysis_text,
                "trace": trace_data,
                "status": fine_status,   # ✅ 直接输出四分类
                "fine_label": fine_status,
            }

    # -------------------- Extraction --------------------
    def _extract_structured_data(self, text: str):
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
"""
        try:
            res = self.client.chat.completions.create(
                model=API_CONFIG["model"],
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            return self._robust_json(res.choices[0].message.content)
        except Exception as e:
            logging.error(f"LLM Extraction Error: {e}")
            return None

    # -------------------- Mapping --------------------
    def _map_food_anchor(self, raw_name: str):
        # 1) Vector retrieval
        try:
            query_vec = self.embedder.encode(raw_name, convert_to_tensor=True)
            cos_scores = util.cos_sim(query_vec, self.category_embeddings)[0]
            top_results = torch.topk(cos_scores, k=10)

            candidates = []
            for score, idx in zip(top_results[0], top_results[1]):
                idx = int(idx)
                item = self.food_kb[idx]
                candidates.append(f"{item['name']} (Code: {item['code']})")
        except Exception as e:
            logging.error(f"Vector search failed: {e}")
            return None, "向量检索异常"

        # 2) LLM choose one
        prompt = f"""
你是一名食品分类专家。
用户输入名称："{raw_name}"

候选 GB2760 分类如下：
{json.dumps(candidates, ensure_ascii=False)}

任务：选择一个最能涵盖 "{raw_name}" 的分类。

规则：
- 如果输入是品牌名，归到对应产品类型。
- 若无完美匹配，选择最接近的上位概念。

只返回 JSON：
{{"target_name":"...","target_code":"...","reason":"..."}}
"""
        try:
            res = self.client.chat.completions.create(
                model=API_CONFIG["model"],
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            parsed = self._robust_json(res.choices[0].message.content) or {}
            target_code = parsed.get("target_code")
            target_name = parsed.get("target_name")
            reason = parsed.get("reason", "模型推理匹配")

            for item in self.food_kb:
                if item["code"] == target_code:
                    return item, reason
            for item in self.food_kb:
                if item["name"] == target_name:
                    return item, reason

            return None, "模型选出的分类不在标准库中"
        except Exception as e:
            logging.error(f"Mapping LLM Error: {e}")
            return None, str(e)

    # -------------------- Retrieval --------------------
    def _retrieve_evidence(self, food_code: str, additive_name: str):
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
            result = session.run(cypher, food_code=food_code, add_name=additive_name)
            return [record.data() for record in result]

    # -------------------- Verdict --------------------
    def _generate_verdict(self, food_name: str, item_name: str, amount_str: str, evidence: list):
        """
        口径：
        - 无 evidence：RISK_FORBIDDEN（固定文本）
        - evidence 含 QS/GMP（max_amount=-1 or unit=GMP...）：SAFE_QS（固定文本，不让 LLM 推断）
        - 其余：交给 LLM 判 SAFE 或 RISK_OVERLIMIT
        """

        # 1) 白名单否定：禁用
        if not evidence:
            text = (
                f"**❌ 禁止添加（白名单否定）**\n\n"
                f"依据 GB2760，在 **{food_name}** 及其父类中未检索到 **{item_name}** 的允许使用条目。"
            )
            return text, "RISK_FORBIDDEN"

        # 2) QS/GMP 短路：max_amount=-1 / unit=GMP / remark 含适量
        for e in evidence[:5]:
            if self._is_qs_evidence(e):
                fixed = (
                    f"**✅ 允许使用（SAFE_QS）**\n\n"
                    f"- 食品分类：{food_name}\n"
                    f"- 添加剂：{item_name}\n"
                    f"- 实际使用量：{amount_str if amount_str else '未提供'}\n\n"
                    f"**法规依据（GB2760）**：该条目属于 **适量制度** 。\n\n"
                    f"**说明**：适量表示按生产需要适量使用，法规未给出可用于数值比较的最大限量。"
                    f"因此本系统输出 **SAFE_QS**（允许使用但不适用“数值超限比较”）。\n\n"
                    f"**提示**：请结合企业工艺、配方设计与风险评估合理控制添加量。"
                )
                return fixed, "SAFE_QS"

        # 3) 非 QS：交给 LLM 判 SAFE / RISK_OVERLIMIT
        evidence_txt = "\n".join(
            [f"- [{e['依据分类']}] 限量 {e['限量']}{e.get('单位','')} ({e.get('备注','')})" for e in evidence[:3]]
        )

        prompt = f"""
你是一名严格的食品合规审计AI。你必须基于给定法规依据进行判断。

【输入数据】
- 食品分类：{food_name}
- 添加剂：{item_name}
- 实际使用量：{amount_str if amount_str else "N/A"}

【法规依据 (GB2760, 来自知识图谱检索)】
{evidence_txt}

【判定规则】
1) 若未提供实际使用量（N/A），仅作科普提示，输出 SAFE。
2) 若法规限量为 GMP/QS/适量，则不允许输出超限结论（本题已由系统在上游短路排除，故此处无需考虑）。
3) 若存在明确数值限量，并且 实际使用量 > 法规限量，则输出 RISK_OVERLIMIT，否则输出 SAFE。

【输出格式要求】
- 第一行必须且只能输出下列之一：
  [[SAFE]]
  [[RISK_OVERLIMIT]]
- 后续输出详细分析理由（Markdown），说明比较依据与结论。
"""

        try:
            res = self.client.chat.completions.create(
                model=API_CONFIG["model"],
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            content = (res.choices[0].message.content or "").strip()

            if content.startswith("[[RISK_OVERLIMIT]]"):
                return content.replace("[[RISK_OVERLIMIT]]", "").strip(), "RISK_OVERLIMIT"
            if content.startswith("[[SAFE]]"):
                return content.replace("[[SAFE]]", "").strip(), "SAFE"

            # 兜底：若模型不按格式输出，做关键词 fallback（尽量保守）
            # 有“超标/超限/超过限量”才判 overlimit
            if any(k in content for k in ["超标", "超限", "超过限量", "超过法规限量"]):
                return content, "RISK_OVERLIMIT"
            return content, "SAFE"

        except Exception as e:
            logging.error(f"Verdict LLM Error: {e}")
            # 保守：LLM 挂了，且有 evidence -> 不贸然判禁用，给 SAFE（可改为 error）
            return "生成判定失败（LLM 调用异常）。建议稍后重试。", "SAFE"
