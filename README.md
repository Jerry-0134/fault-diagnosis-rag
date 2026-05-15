\# 工业故障智能问答系统



基于 RAG（检索增强生成）的工业设备故障诊断助手。一线操作员通过自然语言提问，即可获得专业、可操作的故障排查建议。



\## 📸 功能演示



\[点击观看演示视频]([https://你的视频链接](https://www.bilibili.com/video/BV1ZB5v6PELd/?spm_id_from=333.1387.homepage.video_card.click&vd_source=e81b0e7ddf1c19806bcb2a4163c4cd12))   <!-- 后续录好视频后替换这个链接 -->



\## 🚀 快速开始



\### 环境要求

\- Python 3.8+

\- DeepSeek API Key（\[获取地址](https://platform.deepseek.com/)）



\### 安装步骤



```bash

\# 1. 克隆仓库

git clone https://github.com/Jerry-0134/fault-diagnosis-rag.git

cd fault-diagnosis-rag



\# 2. 安装依赖

pip install -r requirements.txt



\# 3. 配置 API Key

echo "DEEPSEEK\_API\_KEY=你的Key" > .env



\# 4. 启动应用

streamlit run app.py



🏗️ 系统架构

层级	技术	说明

前端	Streamlit	Web 交互界面

检索	关键词匹配	零延迟，适配小规模数据

生成	DeepSeek API	专业故障诊断回答

数据	data.txt	10+ 工业故障案例

✨ 核心亮点

落地思维：针对工厂网络受限场景，采用轻量级关键词匹配，无需下载大模型



工程化：模块化设计，配置与逻辑分离



AI 辅助开发：深度使用 Claude Code 辅助调试



📈 改进方向

数据量扩充至 1000+ 条后升级为向量检索（ChromaDB + text2vec）



增加多轮对话能力



部署到云端，支持多用户并发



📝 项目结构

text

rag-fault-system/

├── app.py              # Streamlit 主界面

├── config.py           # 配置文件

├── data\_loader.py      # 数据解析模块

├── rag\_engine.py       # RAG 检索+生成

├── vector\_store.py     # 向量/关键词检索

├── data.txt            # 故障知识库

├── requirements.txt    # 依赖列表

└── .env                # API Key（不提交）

👤 作者

\[郝健跃] - 控制工程/人工智能学院 研究生



GitHub: \[https://github.com/Jerry-0134]



邮箱: \[1344151482@qq.com]

## 📊 系统评估

| 测试问题 | 知识库状态 | 系统行为 | 结果 |
|---------|-----------|---------|------|
| 电机振动超限 | ✅ 完全匹配 | 基于案例生成专业回答 | ✅ 通过 |
| 电机温度高但不报警 | ⚠️ 部分匹配 | 标注“仅供参考”，给出诊断思路 | ✅ 通过 |
| 打印机卡纸 | ❌ 完全不匹配 | 引导用户补充信息 + 通用指南 | ✅ 通过 |
