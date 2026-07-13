"""
09 Web 版 Twins —— 双人格 AI 助手
善良 / 邪恶 两种模式，RAG + Agent + 记忆
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import json
import datetime
import numpy as np
import gradio as gr
from config import DEEPSEEK_MODEL, get_client

client = get_client()
MODEL = DEEPSEEK_MODEL


class KnowledgeBase:
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        self.paragraphs = []
        self.vectors = None

    def load(self, filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        self.paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        self.vectors = self.model.encode(self.paragraphs, convert_to_numpy=True)

    def search(self, query, top_n=3):
        if self.vectors is None:
            return []
        qvec = self.model.encode([query], convert_to_numpy=True)[0]
        sims = np.dot(self.vectors, qvec) / (
            np.linalg.norm(self.vectors, axis=1) * np.linalg.norm(qvec)
        )
        idx = np.argsort(sims)[::-1][:top_n]
        return [self.paragraphs[i] for i in idx]


def tool_calculator(expression):
    allowed = set("0123456789+-*/.() ")
    if not all(c in allowed for c in expression):
        return "不允许的字符"
    try:
        return str(eval(expression))
    except Exception as e:
        return str(e)

def tool_get_time():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

TOOL_DEFS = [
    {"type": "function", "function": {"name": "calculator", "description": "数学计算",
        "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}},
    {"type": "function", "function": {"name": "get_time", "description": "获取当前时间",
        "parameters": {"type": "object", "properties": {}}}}
]
TOOL_FUNCS = {"calculator": tool_calculator, "get_time": tool_get_time}

# 服务端记忆 —— 页面刷新后仍可恢复对话
_server_memory = []

GOOD_PROMPT = """## 角色 ##
你是一个温柔善良、乐于助人的AI助手。说话温暖可爱，喜欢用"哦""呢""呀"等语气词。

## 输出规则 ##
1. 用户问时间或日期 → 必须调用 get_time 工具，不允许凭记忆回答
2. 用户要求计算 → 必须调用 calculator 工具，不允许凭记忆回答
3. 先看知识库有没有相关内容
   - 有 -> 温柔地分享知识
   - 没有 -> 诚实地说"这个我还不太清楚呢，换一个问题吧~"
4. 每次回复2-4句话

## 禁止 ##
- 禁止冷漠、生硬、官方腔
- 禁止编造知识库没有的信息

## 知识库 ##
{context}"""

EVIL_PROMPT = """## 角色 ##
你是一个毒舌吐槽机器人。每句话都要阴阳怪气、带刺、翻白眼，但知识库里的内容不能瞎编。

## 输出规则 ##
1. 用户问时间或日期 → 必须调用 get_time 工具，不允许凭记忆回答
2. 用户要求计算 → 必须调用 calculator 工具，不允许凭记忆回答
3. 先看知识库有没有相关内容
   - 有 -> 用知识库的事实怼人
   - 没有 -> 直接说"这我哪不知道，问点有用的行不行"
4. 每次回复2-4句话，别啰嗦

## 禁止 ##
- 禁止正经、礼貌、客套
- 禁止编造知识库没有的信息
- 禁止说"根据我的知识"——要说"资料里写着呢自己不会看啊"

## 知识库 ##
{context}"""


def chat_fn(message, history, personality):
    """处理一条消息，返回回复"""
    kb = chat_fn.kb
    docs = kb.search(message)
    # 用换行分隔而非 ---，避免 Markdown 渲染成横线
    context = "\n\n".join(docs) if docs else "无相关资料"

    prompt_template = GOOD_PROMPT if personality == "善良" else EVIL_PROMPT
    system = prompt_template.format(context=context)

    api_history = list(history) if history else []

    messages = [{"role": "system", "content": system}] + api_history
    messages += [{"role": "user", "content": message}]

    response = client.chat.completions.create(
        model=MODEL, messages=messages, tools=TOOL_DEFS, tool_choice="auto"
    )
    msg = response.choices[0].message

    if msg.tool_calls:
        for call in msg.tool_calls:
            args = json.loads(call.function.arguments)
            result = TOOL_FUNCS[call.function.name](**args)
            messages.append(msg)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
        final = client.chat.completions.create(model=MODEL, messages=messages)
        reply = final.choices[0].message.content
    else:
        reply = msg.content

    # 防止 AI 回复中被 Markdown 渲染成分隔线（--- *** 等）
    import re
    reply = re.sub(r'^\s*[-*]{3,}\s*$', '———', reply, flags=re.MULTILINE)
    return reply


def main():
    print("加载向量模型...")
    kb = KnowledgeBase()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    kb.load(os.path.join(script_dir, "knowledge.txt"))
    print("知识库就绪")
    chat_fn.kb = kb

    with gr.Blocks(title="Twins V1.0") as demo:
        gr.Markdown("# Twins V1.0")
        gr.Markdown("双人格 AI 助手")

        personality = gr.State("善良")

        with gr.Row():
            good_btn = gr.Button("😇 善良模式", variant="primary")
            evil_btn = gr.Button("😈 邪恶模式")
        mode_label = gr.Markdown("当前人格：**善良**")

        chatbot = gr.Chatbot(
            label="对话",
            height=450,
            placeholder="你好！",
        )

        msg = gr.Textbox(
            label="输入你的消息",
            placeholder="你好！",
            lines=2,
        )

        with gr.Row():
            send = gr.Button("发送", variant="primary")
            hello_btn = gr.Button("你好！")
            clear = gr.Button("清空对话")

        gr.Markdown("试着说点什么")

        # --- 事件处理 ---
        def respond(message, chat_history, personality_val):
            global _server_memory
            reply = chat_fn(message, chat_history, personality_val)
            chat_history = list(chat_history) if chat_history else []
            chat_history.append({"role": "user", "content": message})
            chat_history.append({"role": "assistant", "content": reply})
            _server_memory = chat_history
            return chat_history, ""

        # 发送按钮 & 回车
        send.click(respond, [msg, chatbot, personality], [chatbot, msg])
        msg.submit(respond, [msg, chatbot, personality], [chatbot, msg])

        # "你好！" 快捷按钮
        hello_btn.click(
            lambda h, p: respond("你好！", h, p),
            [chatbot, personality],
            [chatbot, msg],
        )

        # 清空
        def clear_history():
            global _server_memory
            _server_memory = []
            return [], ""
        clear.click(clear_history, outputs=[chatbot, msg])

        # 页面加载时恢复对话（防止刷新/深浅色切换清空）
        demo.load(lambda: _server_memory, outputs=[chatbot])

        # 切换人格 —— 按钮高亮当前模式
        def switch_good():
            return ("善良", "当前人格：**善良**",
                    gr.update(variant="primary"), gr.update(variant="secondary"))
        def switch_evil():
            return ("邪恶", "当前人格：**邪恶**",
                    gr.update(variant="secondary"), gr.update(variant="primary"))
        good_btn.click(switch_good, outputs=[personality, mode_label, good_btn, evil_btn])
        evil_btn.click(switch_evil, outputs=[personality, mode_label, good_btn, evil_btn])

    # 面试用密码锁（从环境变量读取，防止陌生人消耗Token）
    APP_PASSWORD = os.getenv("TWINS_PASSWORD", "demo2026")
    demo.launch(server_name="0.0.0.0", auth=("面试官", APP_PASSWORD))


if __name__ == "__main__":
    main()
