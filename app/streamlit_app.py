import os
import streamlit as st
import re
from GB2760_Backend2 import GB2760Service

# ================= 1. 页面配置 =================
st.set_page_config(
    page_title="FoodAudit-AG",
    page_icon="🛡️",
    layout="wide"
)

# ================= 2. 样式定义 =================
st.markdown("""
<style>
    .stApp { font-family: 'Inter', sans-serif; }

    /* 标题栏 */
    .header-box {
        background: #f8fafc; border-bottom: 2px solid #e2e8f0; padding: 20px;
        margin-bottom: 20px; text-align: center;
    }
    .header-title { color: #0f172a; font-size: 1.8rem; font-weight: 800; margin: 0; }
    .header-subtitle { color: #64748b; font-size: 0.9rem; margin-top: 5px; }

    /* 映射提示框 */
    .mapping-box {
        background-color: #eff6ff; border: 1px solid #bfdbfe; color: #1e3a8a;
        padding: 12px 16px; border-radius: 8px; margin-bottom: 20px;
        display: flex; align-items: center; gap: 10px; font-size: 0.95rem;
    }
    .arrow { color: #3b82f6; font-weight: bold; font-size: 1.2rem; }

    /* 结果卡片 */
    .audit-card {
        background: white; border: 1px solid #e2e8f0; border-radius: 10px;
        margin-bottom: 16px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .card-header {
        background: #f8fafc; padding: 10px 15px; border-bottom: 1px solid #e2e8f0;
        display: flex; justify-content: space-between; align-items: center;
    }
    .item-name { font-weight: 700; color: #334155; font-size: 1.05rem; }
    .status-badge { font-size: 0.75rem; padding: 3px 10px; border-radius: 20px; font-weight: 700; }
    .safe { background: #dcfce7; color: #166534; border: 1px solid #86efac; }
    .risk { background: #fee2e2; color: #991b1b; border: 1px solid #fca5a5; }

    .card-body { padding: 15px; font-size: 0.9rem; color: #334155; line-height: 1.6; }
    .card-body p { margin-bottom: 8px; }
    .card-body strong { color: #0f172a; }

    /* 溯源详情 */
    details.trace { border-top: 1px solid #e2e8f0; }
    details.trace summary {
        padding: 8px 15px; cursor: pointer; color: #64748b; font-size: 0.8rem; font-weight: 600;
        background: #fff;
    }
    details.trace summary:hover { background: #f1f5f9; }
    .trace-content { padding: 10px 15px; background: #f8fafc; font-family: monospace; font-size: 0.8rem; color: #475569; }
    .trace-list { list-style: none; padding: 0; margin: 5px 0 0 0; }
    .trace-list li { margin-bottom: 4px; padding-bottom: 4px; border-bottom: 1px dashed #e2e8f0; }
    .cat-label { background: #dbeafe; color: #1e40af; padding: 1px 5px; border-radius: 3px; font-weight: bold; }

</style>
""", unsafe_allow_html=True)


# ================= Markdown 渲染引擎 =================
def render_md(text):
    if not text: return ""
    # 移除代码块
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    text = text.replace('```', '')

    # 转换
    text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'###\s*(.*)', r'<h4>\1</h4>', text)

    lines = text.split('\n')
    html = []
    in_list = False

    for line in lines:
        line = line.strip()
        if not line: continue

        if line.startswith('- ') or line.startswith('* '):
            if not in_list:
                html.append('<ul style="margin:5px 0 10px 20px; padding:0;">')
                in_list = True
            content = line[2:]
            content = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', content)
            html.append(f'<li>{content}</li>')
        else:
            if in_list:
                html.append('</ul>')
                in_list = False
            if not line.startswith('<h'):
                html.append(f'<p>{line}</p>')
            else:
                html.append(line)

    if in_list: html.append('</ul>')
    return "".join(html)


# ================= 后端连接 =================
@st.cache_resource
def get_backend():
    s = GB2760Service()
    s.initialize()
    return s


try:
    backend = get_backend()
except Exception as e:
    st.error(f"Backend Error: {e}")
    st.stop()

# ================= UI 逻辑 =================

st.markdown("""
<div class="header-box">
    <div class="header-title">🛡️ GB2760 智能合规审查</div>
    <div class="header-subtitle">基于向量语义映射与图谱推理 (Vector-RAG + Graph Reasoning)</div>
</div>
""", unsafe_allow_html=True)

# 侧边栏
with st.sidebar:
    st.info("💡 **测试指南**\n\n试试输入：\n\n> *核查芬达配方：爱德万甜 0.005g/kg*\n\n> *蔬菜罐头能加什么防腐剂？*")
    if st.button("🗑️ 清空历史"):
        st.session_state.messages = []
        st.rerun()

# 聊天历史
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "请发送配方或问题，系统将自动进行语义映射与合规审查。"}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)

# 输入处理
if prompt := st.chat_input("输入内容..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        html_buffer = ""
        status_box = st.status("🧠 分析中...", expanded=True)

        try:
            for update in backend.process_query(prompt):

                # 状态流
                if update.get("step") in ["parsing", "anchoring", "retrieving"]:
                    status_box.write(update["message"])

                elif update.get("step") == "mapping_success":
                    html = f"""
                    <div class="mapping-box">
                        <span style="font-size:1.2rem">🏷️</span>
                        <div>
                            <b>语义映射成功：</b>
                            <span>"{update['original']}"</span>
                            <span class="arrow">➔</span>
                            <b>"{update['mapped']}"</b>
                            <div style="font-size:0.8em; opacity:0.8; margin-top:2px;">(依据: {update['reason']})</div>
                        </div>
                    </div>
                    """
                    html_buffer += html
                    placeholder.markdown(html_buffer, unsafe_allow_html=True)
                    status_box.write(f"✅ 映射完成：{update['mapped']}")

                elif update.get("step") == "error":
                    status_box.update(label="❌ 错误", state="error")
                    err = f"<div style='color:red; background:#fff0f0; padding:10px; border-radius:5px;'>❌ {update['message']}</div>"
                    html_buffer += err
                    placeholder.markdown(html_buffer, unsafe_allow_html=True)

                elif update.get("step") == "item_finished":
                    name = update["name"]
                    body_html = render_md(update["analysis"])
                    trace = update["trace"]

                    fine_label = (update.get("fine_label") or "").upper().strip()

                    if fine_label:
                        is_risk = fine_label.startswith("RISK")
                        badge_txt = fine_label
                    else:
                        status_txt = (update.get("status") or "").upper().strip()
                        is_risk = status_txt.startswith("RISK")
                        badge_txt = status_txt if status_txt else ("RISK" if is_risk else "SAFE")

                    badge_cls = "risk" if is_risk else "safe"
                    icon = "⚠️" if is_risk else "🛡️"

                    # 让 SAFE_QS 更醒目（可选）
                    if (fine_label == "SAFE_QS") or (badge_txt == "SAFE_QS"):
                        badge_txt = "SAFE_QS"
                        icon = "🛡️"

                    ev_list = ""
                    if trace.get("top_evidence"):
                        for e in trace["top_evidence"]:
                            ev_list += (
                                f"<li><span class='cat-label'>{e['依据分类']}</span> "
                                f"限量:{e['限量']}{e['单位']} "
                                f"<span style='color:#94a3b8'>({e.get('备注', '')})</span></li>"
                            )
                    else:
                        ev_list = "<li>❌ 未检索到允许依据 (白名单拦截)</li>"

                    card = f"""
                    <div class="audit-card">
                        <div class="card-header">
                            <div class="item-name">{icon} {name}</div>
                            <span class="status-badge {badge_cls}">{badge_txt}</span>
                        </div>
                        <div class="card-body">{body_html}</div>
                        <details class="trace">
                            <summary>🔍 查看法规依据链</summary>
                            <div class="trace-content">
                                <div>📍 锚点分类: {trace.get('food_anchor', '')}</div>
                                <ul class="trace-list">{ev_list}</ul>
                            </div>
                        </details>
                    </div>
                    """
                    html_buffer += card
                    placeholder.markdown(html_buffer, unsafe_allow_html=True)

                elif update.get("step") == "result":
                    body_html = render_md(update["content"])
                    card = f"<div class='audit-card'><div class='card-body'>{body_html}</div></div>"
                    html_buffer += card
                    placeholder.markdown(html_buffer, unsafe_allow_html=True)

            status_box.update(label="✅ 完成", state="complete", expanded=False)
            st.session_state.messages.append({"role": "assistant", "content": html_buffer})

        except Exception as e:
            st.error(str(e))
