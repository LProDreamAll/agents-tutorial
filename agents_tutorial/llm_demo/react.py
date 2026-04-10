# 智能体范式ReAct (Reason + Act)
"""
Thought (思考)： 这是智能体的“内心独白”。它会分析当前情况、分解任务、制定下一步计划，或者反思上一步的结果。
Action (行动)： 这是智能体决定采取的具体动作，通常是调用一个外部工具，例如 Search['华为最新款手机']。
Observation (观察)： 这是执行Action后从外部工具返回的结果，例如搜索结果的摘要或API的返回值。

"""
import os
from typing import Dict, Any, Callable

try:
    from serpapi import SerpApiClient
except Exception:
    SerpApiClient = None

from llm_client import HelloAgentsLLM


class ToolExecutor:
    """
    一个工具执行器，负责管理和执行工具。
    """

    def __init__(self):
        self.tools: Dict[str, Dict[str, Any]] = {}

    def registerTool(self, name: str, description: str, func: Callable):
        """
        向工具箱中注册一个新工具。
        """
        if name in self.tools:
            print(f"警告:工具 '{name}' 已存在，将被覆盖。")
        self.tools[name] = {"description": description, "func": func}
        print(f"工具 '{name}' 已注册。")

    def getTool(self, name: str) -> Callable | None:
        """
        根据名称获取一个工具的执行函数。
        """
        return self.tools.get(name, {}).get("func")

    def getAvailableTools(self) -> str:
        """
        获取所有可用工具的格式化描述字符串。
        """
        return "\n".join([
            f"- {name}: {info['description']}"
            for name, info in self.tools.items()
        ])


def _mock_search(query: str) -> str:
    """
    离线Mock搜索结果，便于在没有SerpApi或没有API Key时演示流程。
    """
    if "华为" in query and "手机" in query:
        return "\n\n".join([
            "[1] 华为手机- 华为官网（Mock）\nMate 系列 / Pura 系列 / Pocket 系列 / nova 系列。",
            "[2] 2025华为手机推荐（Mock）\nMate 70 系列常见卖点：影像、做工、续航、鸿蒙生态。",
            "[3] Pura 系列信息（Mock）\nPura 80 Pro+ 常见卖点：影像能力与设计。"
        ])
    if "英伟达" in query and ("GPU" in query or "显卡" in query):
        return "\n\n".join([
            "[1] NVIDIA GeForce（Mock）\n消费级显卡产品线信息页。",
            "[2] RTX 50 系列（Mock）\n用于演示ReAct流程的静态结果，不代表实时信息。",
            "[3] 数据中心GPU（Mock）\nH 系列/Blackwell 相关新闻摘要（演示数据）。"
        ])
    return f"[Mock] 未命中预置数据，查询词: {query}"


def search(query: str) -> str:
    """
    一个基于SerpApi的实战网页搜索引擎工具。
    它会智能地解析搜索结果，优先返回直接答案或知识图谱信息。
    """
    use_mock = os.getenv("USE_MOCK_SEARCH", "1").lower() in {"1", "true", "yes"}
    if use_mock:
        print(f"🔍 正在执行 [MockSearch] 网页搜索: {query}")
        return _mock_search(query)

    print(f"🔍 正在执行 [SerpApi] 网页搜索: {query}")
    try:
        if SerpApiClient is None:
            print("⚠️ 未安装 google-search-results，自动降级为 MockSearch。")
            return _mock_search(query)

        api_key = os.getenv("SERPAPI_API_KEY")
        if not api_key:
            print("⚠️ SERPAPI_API_KEY 未配置，自动降级为 MockSearch。")
            return _mock_search(query)

        params = {
            "engine": "google",
            "q": query,
            "api_key": api_key,
            "gl": "cn",  # 国家代码
            "hl": "zh-cn",  # 语言代码
        }

        client = SerpApiClient(params)
        results = client.get_dict()

        # 智能解析:优先寻找最直接的答案
        if "answer_box_list" in results:
            return "\n".join(results["answer_box_list"])
        if "answer_box" in results and "answer" in results["answer_box"]:
            return results["answer_box"]["answer"]
        if "knowledge_graph" in results and "description" in results["knowledge_graph"]:
            return results["knowledge_graph"]["description"]
        if "organic_results" in results and results["organic_results"]:
            # 如果没有直接答案，则返回前三个有机结果的摘要
            snippets = [
                f"[{i + 1}] {res.get('title', '')}\n{res.get('snippet', '')}"
                for i, res in enumerate(results["organic_results"][:3])
            ]
            return "\n\n".join(snippets)

        return f"对不起，没有找到关于 '{query}' 的信息。"

    except Exception as e:
        print(f"⚠️ SerpApi 请求异常，自动降级为 MockSearch: {e}")
        return _mock_search(query)


def build_default_tool_executor() -> ToolExecutor:
    tool_executor = ToolExecutor()
    search_description = "一个网页搜索引擎。当你需要回答关于时事、事实以及在你的知识库中找不到的信息时，应使用此工具。"
    tool_executor.registerTool("Search", search_description, search)
    return tool_executor

REACT_PROMPT_TEMPLATE = """
请注意，你是一个有能力调用外部工具的智能助手。

可用工具如下:
{tools}

请严格按照以下格式进行回应:

Thought: 你的思考过程，用于分析问题、拆解任务和规划下一步行动。
Action: 你决定采取的行动，必须是以下格式之一:
- `{{tool_name}}[{{tool_input}}]`:调用一个可用工具。
- `Finish[最终答案]`:当你认为已经获得最终答案时。
- 当你收集到足够的信息，能够回答用户的最终问题时，你必须在Action:字段后使用 Finish[最终答案] 来输出最终答案。

现在，请开始解决以下问题:
Question: {question}
History: {history}
"""

import re


class ReActAgent:
    def __init__(self, llm_client: HelloAgentsLLM, tool_executor: ToolExecutor, max_steps: int = 5):
        self.llm_client = llm_client
        self.tool_executor = tool_executor
        self.max_steps = max_steps
        self.history = []

    def run(self, question: str):
        """
        运行ReAct智能体来回答一个问题。
        """
        self.history = []  # 每次运行时重置历史记录
        current_step = 0

        while current_step < self.max_steps:
            current_step += 1
            print(f"--- 第 {current_step} 步 ---")

            # 1. 格式化提示词
            tools_desc = self.tool_executor.getAvailableTools()
            history_str = "\n".join(self.history)
            prompt = REACT_PROMPT_TEMPLATE.format(
                tools=tools_desc,
                question=question,
                history=history_str
            )

            # 2. 调用LLM进行思考
            messages = [{"role": "user", "content": prompt}]
            response_text = self.llm_client.think(messages=messages)

            if not response_text:
                print("错误:LLM未能返回有效响应。")
                break

            # 3. 解析LLM的输出
            thought, action = self._parse_output(response_text)

            if thought:
                print(f"思考: {thought}")

            if not action:
                print("警告:未能解析出有效的Action，流程终止。")
                break

            # 4. 执行Action
            if action.startswith("Finish"):
                # 如果是Finish指令，提取最终答案并结束
                final_answer = self._extract_finish_answer(action)
                if final_answer is None:
                    print("警告: Finish 指令格式无效，流程终止。")
                    break
                print(f"🎉 最终答案: {final_answer}")
                return final_answer

            tool_name, tool_input = self._parse_action(action)
            if not tool_name or not tool_input:
                # ... 处理无效Action格式 ...
                continue

            print(f"🎬 行动: {tool_name}[{tool_input}]")

            tool_function = self.tool_executor.getTool(tool_name)
            if not tool_function:
                observation = f"错误:未找到名为 '{tool_name}' 的工具。"
            else:
                observation = tool_function(tool_input)  # 调用真实工具
            print(f"👀 观察: {observation}")

            # 将本轮的Action和Observation添加到历史记录中
            self.history.append(f"Action: {action}")
            self.history.append(f"Observation: {observation}")

            # 循环结束
        print("已达到最大步数，流程终止。")
        return None

    def _parse_output(self, text: str):
        """解析LLM的输出，提取Thought和Action。
        """
        # Thought: 匹配到 Action: 或文本末尾
        thought_match = re.search(r"Thought:\s*(.*?)(?=\nAction:|$)", text, re.DOTALL)
        # Action: 匹配到文本末尾
        action_match = re.search(r"Action:\s*(.*?)$", text, re.DOTALL)
        thought = thought_match.group(1).strip() if thought_match else None
        action = action_match.group(1).strip() if action_match else None
        return thought, action

    def _parse_action(self, action_text: str):
        """解析Action字符串，提取工具名称和输入。
        """
        match = re.match(r"(\w+)\[(.*)\]", action_text, re.DOTALL)
        if match:
            return match.group(1), match.group(2)
        return None, None

    def _extract_finish_answer(self, action_text: str):
        """
        解析 Finish[最终答案]，兼容多行内容。
        """
        match = re.search(r"Finish\[(.*)\]\s*$", action_text, re.DOTALL)
        if not match:
            return None
        return match.group(1).strip()

def main():
    react_agent = ReActAgent(
        llm_client=HelloAgentsLLM(),
        tool_executor=build_default_tool_executor()
    )
    react_agent.run("英伟达最新的GPU型号是什么")


if __name__ == '__main__':
    main()
