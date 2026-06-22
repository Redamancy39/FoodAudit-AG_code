import os
import json
from neo4j import GraphDatabase
from tqdm import tqdm

# ================= 配置区域 =================
# Neo4j Desktop 默认连接地址
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")

# 文件路径 (请修改为你实际存放的路径)
BASE_DIR = r"E:\Chain of Thought in llm\output"
TREE_FILE = os.path.join(BASE_DIR, "GB2760_Category_Tree.jsonl")
# 这里指向你刚刚手动导出并下载的那个文件
CLEANED_DATA_FILE = os.path.join(BASE_DIR, "GB2760_Manual_Calibrated_V3.jsonl")
SYNONYM_FILE = os.path.join(BASE_DIR, "GB2760_Synonyms.jsonl")


class GB2760DesktopImporter:
    def __init__(self):
        print(f"🔌 正在连接 Neo4j Desktop ({NEO4J_URI})...")
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    def close(self):
        self.driver.close()

    def clear_database(self):
        print("\n💥 [Step 0] 正在清空数据库 (为了保证数据纯净)...")
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            # 删除旧索引 (可选，防止报错)
            try:
                indexes = session.run("SHOW INDEXES").data()
                for idx in indexes:
                    if idx['type'] != 'LOOKUP':  # 保留系统索引
                        session.run(f"DROP INDEX {idx['name']}")
                constraints = session.run("SHOW CONSTRAINTS").data()
                for c in constraints:
                    session.run(f"DROP CONSTRAINT {c['name']}")
            except:
                pass
        print("✅ 数据库已清空")

    def create_schema(self):
        print("\n🛠️ [Step 1] 创建索引和约束...")
        with self.driver.session() as session:
            # 唯一性约束：食品分类号必须唯一
            session.run(
                "CREATE CONSTRAINT category_code_uniq IF NOT EXISTS FOR (c:FoodCategory) REQUIRE c.code IS UNIQUE")
            # 索引：添加剂名称加速查询
            session.run("CREATE INDEX additive_name_idx IF NOT EXISTS FOR (a:Additive) ON (a.name)")
            # 索引：功能加速查询 (支持数组包含查询)
            session.run("CREATE INDEX additive_func_idx IF NOT EXISTS FOR (a:Additive) ON (a.function)")
            # 索引：俗名
            session.run("CREATE INDEX synonym_name_idx IF NOT EXISTS FOR (s:Synonym) ON (s.name)")
        print("✅ 索引创建完成")

    def import_category_tree(self):
        print("\n🌳 [Step 2] 导入食品分类树...")
        if not os.path.exists(TREE_FILE):
            print(f"❌ 错误：找不到文件 {TREE_FILE}")
            return

        count = 0
        with self.driver.session() as session:
            with open(TREE_FILE, 'r', encoding='utf-8') as f:
                # 预读所有行
                lines = f.readlines()

                # 1. 先创建所有节点 (Node)
                for line in tqdm(lines, desc="   创建节点"):
                    try:
                        d = json.loads(line)
                        inner = d.get("data", d)
                        code = inner.get("cns_code") or inner.get("code")
                        name = inner.get("name")
                        if code:
                            session.run("MERGE (c:FoodCategory {code: $code}) SET c.name = $name", code=code, name=name)
                            count += 1
                    except:
                        continue

                # 2. 再建立关系 (Relationship)
                for line in tqdm(lines, desc="   建立层级"):
                    try:
                        d = json.loads(line)
                        inner = d.get("data", d)
                        code = inner.get("cns_code") or inner.get("code")
                        parent = inner.get("parent_code")

                        if code and parent and parent not in ["root", "null", ""]:
                            session.run("""
                            MATCH (c:FoodCategory {code: $code})
                            MATCH (p:FoodCategory {code: $parent})
                            MERGE (p)-[:PARENT_OF]->(c)
                            """, code=code, parent=parent)
                    except:
                        continue
        print(f"✅ 分类树导入完成 ({count} 节点)")

    def import_regulations(self):
        print("\n💊 [Step 3] 导入法规数据 (含多功能属性)...")
        if not os.path.exists(CLEANED_DATA_FILE):
            print(f"❌ 错误：找不到文件 {CLEANED_DATA_FILE}")
            return

        batch_data = []
        BATCH_SIZE = 1000  # 批量提交以提高性能

        # Cypher 语句：处理列表类型的 function
        query = """
        UNWIND $batch AS row

        // 1. 创建添加剂节点
        MERGE (a:Additive {name: row.head_entity.name})
        SET a.cns = row.head_entity.cns,
            a.ins = row.head_entity.ins,
            a.function = row.head_entity.function
            // 注意：如果 function 是列表，Neo4j 会自动存为 String Array

        // 2. 匹配食品节点 (如果分类树里漏了，这里自动补全)
        MERGE (f:FoodCategory {code: row.tail_entity.code})
        ON CREATE SET f.name = row.tail_entity.name

        // 3. 建立 ALLOWED_IN 关系
        MERGE (a)-[r:ALLOWED_IN]->(f)
        SET r.max_amount = row.properties.max_amount_value,
            r.unit = row.properties.unit,
            r.remark = row.properties.remark,
            r.calc_basis = row.properties.calculation_basis
        """

        with self.driver.session() as session:
            with open(CLEANED_DATA_FILE, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in tqdm(lines, desc="   导入规则"):
                    try:
                        item = json.loads(line)
                        # 简单校验
                        if not item.get('tail_entity', {}).get('code'): continue

                        batch_data.append(item)

                        if len(batch_data) >= BATCH_SIZE:
                            session.run(query, batch=batch_data)
                            batch_data = []
                    except:
                        continue

                # 处理剩余尾巴
                if batch_data:
                    session.run(query, batch=batch_data)

        print(f"✅ 法规数据导入完成 (共 {len(lines)} 条)")

    def import_synonyms(self):
        print("\n🔗 [Step 4] 导入同义词节点...")
        if not os.path.exists(SYNONYM_FILE):
            print(f"⚠️ 跳过：找不到同义词文件 {SYNONYM_FILE}")
            return

        with self.driver.session() as session:
            with open(SYNONYM_FILE, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in tqdm(lines, desc="   导入同义词"):
                    try:
                        d = json.loads(line)
                        session.run("""
                        MATCH (f:FoodCategory {code: $code})
                        MERGE (s:Synonym {name: $name})
                        MERGE (s)-[:ALIASED_AS]->(f)
                        """, code=d['target_code'], name=d['name'])
                    except:
                        continue
        print("✅ 同义词导入完成")

    def verify(self):
        print("\n📊 [Final] 最终数据校验...")
        with self.driver.session() as session:
            n_additives = session.run("MATCH (n:Additive) RETURN count(n)").single()[0]
            n_rels = session.run("MATCH ()-[r:ALLOWED_IN]->() RETURN count(r)").single()[0]

            # 检查多功能属性是否成功存为数组
            try:
                sample = session.run(
                    "MATCH (a:Additive) WHERE size(a.function) > 1 LIMIT 1 RETURN a.name, a.function").single()
                print(f"   🔹 添加剂总数: {n_additives}")
                print(f"   🔹 法规关系数: {n_rels}")
                if sample:
                    print(f"   ✨ 成功验证数组属性: {sample['a.name']} -> {sample['a.function']} (类型: List)")
                else:
                    print("   ⚠️ 提示: 未发现多功能的添加剂，请确认数据源。")
            except Exception as e:
                print(f"   校验时出错: {e}")


if __name__ == "__main__":
    importer = GB2760DesktopImporter()
    try:
        importer.clear_database()  # 1
        importer.create_schema()  # 2
        importer.import_category_tree()  # 3
        importer.import_regulations()  # 4
        importer.import_synonyms()  # 5
        importer.verify()  # 6
    except Exception as e:
        print(f"\n❌ 发生严重错误: {e}")
        print("💡 提示: 请检查 Neo4j Desktop 是否已启动，以及密码是否正确。")
    finally:
        importer.close()
