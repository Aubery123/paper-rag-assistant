"""评测判分提示词（RAGAS 风格，LLM-as-judge）。"""

# 忠实度：把答案拆成原子陈述，逐条判断是否被【上下文】支持
FAITHFULNESS_SYSTEM = """你是严格的事实核查员。给定【上下文】和一段【答案】，请：
1. 把答案拆成若干条原子事实陈述（atomic statements）；
2. 对每条判断它是否**能由上下文直接推出/支持**（不依赖上下文之外的常识）。
只输出 JSON，格式：{"statements": [{"text": "陈述", "supported": true/false}, ...]}"""

FAITHFULNESS_USER = """【上下文】
{context}

【答案】
{answer}

请输出 JSON。"""

# 答案相关性：由答案反推它能回答的问题，再与原问题比相似度
RELEVANCY_SYSTEM = """给定一段【答案】，请生成 3 个该答案能够直接、完整回答的中文问题。
只输出 JSON，格式：{"questions": ["问题1", "问题2", "问题3"]}"""

RELEVANCY_USER = """【答案】
{answer}

请输出 JSON。"""
