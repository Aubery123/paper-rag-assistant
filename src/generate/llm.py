"""LLM 客户端（流式）。

职责：封装通义千问 qwen-plus（默认，OpenAI 兼容接口），支持流式输出；可换 DeepSeek。
API key 走环境变量 DASHSCOPE_API_KEY。

TODO(P1): 封装 chat / chat_stream，模型与 base_url 取自 config。
"""
