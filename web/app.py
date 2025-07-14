# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import torch
from adaptive_classifier import AdaptiveClassifier   # 确保安装或路径正确
import os
import json
import tempfile
import uuid

# -------------------- 页面配置 --------------------
st.set_page_config(page_title="AI4X agents", layout="wide")
st.title("🧠 AI4X - Adaptive Classifier Playground")
st.markdown("交互式体验动态文本分类、增量学习与策略预测")

# -------------------- 侧边栏：全局配置 --------------------
with st.sidebar:
    st.header("⚙️ 全局配置")
    model_name = st.text_input("Embedding 模型名", value="bge-large-zh-v1.5", help="支持 Hugging Face Hub 上的模型")
    device = st.selectbox("设备", ["auto", "cpu", "cuda"], index=0)
    strategic = st.checkbox("启用策略模式", value=False)
    seed = st.number_input("随机种子", value=42, step=1)

    if st.button("🚀 初始化 / 重置模型"):
        st.session_state["clf"] = AdaptiveClassifier(
            model_name=model_name,
            device=None if device == "auto" else device,
            config={"enable_strategic_mode": strategic},
            seed=seed
        )
        st.success("模型已初始化！")

# -------------------- 初始化 session_state --------------------
if "clf" not in st.session_state:
    st.session_state["clf"] = AdaptiveClassifier(
        model_name=model_name,
        config={"enable_strategic_mode": strategic},
        seed=seed
    )

clf: AdaptiveClassifier = st.session_state["clf"]

# -------------------- 主区域：分栏 --------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📝 添加示例", "📊 统计信息", "💾 保存/加载", "🔍 单条预测", "📦 批量预测"]
)

# ---------- 1. 添加示例 ----------
with tab1:
    st.subheader("📝 添加示例")
    texts = st.text_area("文本（一行一条）").splitlines()
    labels = st.text_area("对应标签（一行一条）").splitlines()

    if st.button("增量学习"):
        if texts and labels and len(texts) == len(labels):
            with st.spinner("正在学习..."):
                clf.add_examples(texts, labels)
            st.success("完成！")
        else:
            st.warning("文本和标签不能为空，且数量一致")

# ---------- 2. 统计信息 ----------
with tab2:
    st.subheader("📊 统计信息")
    stats = clf.get_memory_stats()
    st.json(stats)

    if stats["num_classes"] > 0:
        df = pd.DataFrame(
            list(stats["examples_per_class"].items()),
            columns=["label", "count"]
        )
        fig = px.bar(df, x="label", y="count", title="类别示例分布")
        st.plotly_chart(fig, use_container_width=True)

# ---------- 3. 保存 / 加载 ----------
with tab3:
    st.subheader("💾 保存 / 加载")
    col1, col2 = st.columns(2)

    with col1:
        st.write("**保存模型**")
        save_name = st.text_input("保存目录名", value=f"adaptive_model_{uuid.uuid4().hex[:6]}")
        if st.button("保存"):
            os.makedirs(save_name, exist_ok=True)
            clf._save_pretrained(save_name)
            st.success(f"已保存到 `{save_name}`")

    with col2:
        st.write("**加载模型**")
        load_path = st.text_input("模型目录 / HF Hub ID", value=save_name)
        if st.button("加载"):
            try:
                st.session_state["clf"] = AdaptiveClassifier._from_pretrained(
                    load_path,
                    device=None if device == "auto" else device,
                )
                st.success("加载完成！")
                st.rerun()
            except Exception as e:
                st.error(str(e))

# ---------- 4. 单条预测 ----------
with tab4:
    st.subheader("🔍 单条预测")
    text = st.text_area("输入文本", height=100)
    k = st.slider("返回 Top-k", 1, 10, 5)

    if st.button("预测"):
        if text.strip():
            preds = clf.predict(text, k=k)
            st.write("**预测结果：**")
            for label, score in preds:
                st.write(f"{label}: {score:.3f}")
        else:
            st.warning("请输入文本")

# ---------- 5. 批量预测 ----------
with tab5:
    st.subheader("📦 批量预测")
    src = st.radio("数据来源", ["上传 csv", "粘贴文本"])
    if src == "上传 csv":
        uploaded = st.file_uploader("csv 文件（一列 text）", type="csv")
        if uploaded:
            df = pd.read_csv(uploaded)
            texts = df.iloc[:, 0].astype(str).tolist()
    else:
        texts = st.text_area("每行一条文本").splitlines()

    k = st.slider("Top-k（批量）", 1, 10, 3, key="batchk")
    if st.button("批量预测"):
        if texts:
            with st.spinner("预测中..."):
                preds = clf.predict_batch(texts, k=k)
            out = [{"text": t, "top1": p[0][0], "top1_score": p[0][1]} for t, p in zip(texts, preds)]
            st.write(pd.DataFrame(out))
            csv = pd.DataFrame(out).to_csv(index=False)
            st.download_button("下载结果 csv", csv, file_name="batch_preds.csv")
        else:
            st.warning("无文本")