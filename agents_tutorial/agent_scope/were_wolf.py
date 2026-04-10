# -*- coding: utf-8 -*-
"""
三国狼人杀 - 基于AgentScope的中文版狼人杀游戏
融合三国演义角色和传统狼人杀玩法
AgentScope 的优势与局限性分析
通过这个"三国狼人杀"案例，我们深度体验了 AgentScope 框架的核心优势。该框架以其消息驱动的架构为核心，将复杂的游戏流程优雅地映射为一系列并发、异步的消息传递事件，从而避免了传统状态机的僵硬与复杂。结合其强大的结构化输出能力，
我们将游戏规则直接转化为代码层面的约束，极大地提升了系统的稳定性和可预测性。这种设计范式不仅在性能上展现了其原生并发的优势，更在容错处理上保证了即使单个智能体出现异常，整体流程也能稳健运行。

然而，AgentScope 的工程化优势也带来了一定的复杂性成本。其消息驱动架构虽然强大，但对开发者的技术要求较高，需要理解异步编程、分布式通信等概念。对于简单的多智能体对话场景，这种架构可能显得过于复杂，
存在"过度工程化"的风险。此外，作为相对较新的框架，其生态系统和社区资源还有待进一步完善。因此，AgentScope 更适合需要构建大规模、高可靠性的生产级多智能体系统，而对于快速原型开发或简单应用场景，选择更轻量级的框架可能更为合适。
"""
import asyncio
import logging
import os
import random
from pathlib import Path
from typing import List, Dict, Optional, Any

from agentscope.agent import ReActAgent
from agentscope.model import OpenAIChatModel
from agentscope.pipeline import MsgHub, sequential_pipeline, fanout_pipeline
from agentscope.formatter import OpenAIMultiAgentFormatter
from dotenv import load_dotenv

from prompt_cn import ChinesePrompts
from game_roles import GameRoles
from structured_output_cn import (
    DiscussionModelCN,
    get_vote_model_cn,
    WitchActionModelCN,
    get_seer_model_cn,
    get_hunter_model_cn,
    WerewolfKillModelCN,
)
from utils_cn import (
    check_winning_cn,
    majority_vote_cn,
    get_chinese_name,
    format_player_list,
    GameModerator,
    MAX_GAME_ROUND,
    MAX_DISCUSSION_ROUND,
)

load_dotenv()
load_dotenv(Path(__file__).resolve().parents[2] / ".env")


class ThreeKingdomsWerewolfGame:
    """三国狼人杀游戏主类"""

    def __init__(self):
        self.players: Dict[str, ReActAgent] = {}
        self.roles: Dict[str, str] = {}
        self.moderator = GameModerator()
        self.alive_players: List[ReActAgent] = []
        self.werewolves: List[ReActAgent] = []
        self.villagers: List[ReActAgent] = []
        self.seer: List[ReActAgent] = []
        self.witch: List[ReActAgent] = []
        self.hunter: List[ReActAgent] = []
        self.model_name = os.getenv("LLM_MODEL_ID")
        self.api_key = os.getenv("LLM_API_KEY")
        self.base_url = os.getenv("LLM_BASE_URL")
        self.enable_llm_debug = os.getenv("LLM_DEBUG", "0").strip().lower() in {
            "1",
            "true",
            "yes",
        }
        self.enable_thinking = os.getenv(
            "LLM_ENABLE_THINKING", "0"
        ).strip().lower() in {"1", "true", "yes"}
        if not self.enable_llm_debug:
            logging.getLogger("_openai_formatter").setLevel(logging.ERROR)
            logging.getLogger("_openai_model").setLevel(logging.ERROR)

        # 女巫道具状态
        self.witch_has_antidote = True
        self.witch_has_poison = True

    async def create_player(self, role: str, character: str) -> ReActAgent:
        """创建具有三国背景的玩家"""
        name = get_chinese_name(character)
        self.roles[name] = role

        agent = ReActAgent(
            name=name,
            sys_prompt=ChinesePrompts.get_role_prompt(role, character),
            model=DebugOpenAIChatModel(
                model_name=self.model_name,
                api_key=self.api_key,
                client_kwargs={"base_url": self.base_url} if self.base_url else None,
                generate_kwargs={
                    "extra_body": {"enable_thinking": self.enable_thinking}
                },
                debug_enabled=self.enable_llm_debug,
            ),
            formatter=OpenAIMultiAgentFormatter(),
        )

        # 角色身份确认
        await agent.observe(
            await self.moderator.announce(
                f"【{name}】你在这场三国狼人杀中扮演{GameRoles.get_role_desc(role)}，"
                f"你的角色是{character}。{GameRoles.get_role_ability(role)}"
            )
        )

        self.players[name] = agent
        return agent

    async def setup_game(self, player_count: int = 6):
        """设置游戏"""
        print("🎮 开始设置三国狼人杀游戏...")

        # 获取角色配置
        roles = GameRoles.get_standard_setup(player_count)
        characters = random.sample(
            [
                "刘备",
                "关羽",
                "张飞",
                "诸葛亮",
                "赵云",
                "曹操",
                "司马懿",
                "周瑜",
                "孙权",
            ],
            player_count,
        )

        # 创建玩家
        for i, (role, character) in enumerate(zip(roles, characters)):
            agent = await self.create_player(role, character)
            self.alive_players.append(agent)

            # 分配到对应阵营
            if role == "狼人":
                self.werewolves.append(agent)
            elif role == "预言家":
                self.seer.append(agent)
            elif role == "女巫":
                self.witch.append(agent)
            elif role == "猎人":
                self.hunter.append(agent)
            else:
                self.villagers.append(agent)

        # 游戏开始公告
        await self.moderator.announce(
            f"三国狼人杀游戏开始！参与者：{format_player_list(self.alive_players)}"
        )

        print(f"✅ 游戏设置完成，共{len(self.alive_players)}名玩家")

    async def werewolf_phase(self, round_num: int):
        """狼人阶段"""
        if not self.werewolves:
            return None

        await self.moderator.announce(f"🐺 狼人请睁眼，选择今晚要击杀的目标...")

        # 狼人讨论
        async with MsgHub(
            self.werewolves,
            enable_auto_broadcast=True,
            announcement=await self.moderator.announce(
                f"狼人们，请讨论今晚的击杀目标。存活玩家：{format_player_list(self.alive_players)}"
            ),
        ) as werewolves_hub:
            # 讨论阶段
            for _ in range(MAX_DISCUSSION_ROUND):
                for wolf in self.werewolves:
                    await wolf(structured_model=DiscussionModelCN)

            # 投票击杀
            werewolves_hub.set_auto_broadcast(False)
            kill_votes = await fanout_pipeline(
                self.werewolves,
                msg=await self.moderator.announce("请选择击杀目标"),
                structured_model=WerewolfKillModelCN,
                enable_gather=False,
            )

            # 统计投票
            votes = {}
            for i, vote_msg in enumerate(kill_votes):
                # 检查vote_msg是否为None或metadata是否存在
                if (
                    vote_msg is not None
                    and hasattr(vote_msg, "metadata")
                    and vote_msg.metadata is not None
                ):
                    votes[self.werewolves[i].name] = vote_msg.metadata.get("target")
                else:
                    # 如果返回无效,随机选择一个目标
                    print(f"⚠️ {self.werewolves[i].name} 的击杀投票无效,随机选择目标")
                    import random

                    valid_targets = [
                        p.name
                        for p in self.alive_players
                        if p.name not in [w.name for w in self.werewolves]
                    ]
                    votes[self.werewolves[i].name] = (
                        random.choice(valid_targets) if valid_targets else None
                    )

            killed_player, _ = majority_vote_cn(votes)
            return killed_player

    async def seer_phase(self):
        """预言家阶段"""
        if not self.seer:
            return

        seer_agent = self.seer[0]
        await self.moderator.announce("🔮 预言家请睁眼，选择要查验的玩家...")

        check_result = await seer_agent(
            structured_model=get_seer_model_cn(self.alive_players)
        )

        # 检查返回结果是否有效
        if (
            check_result is None
            or not hasattr(check_result, "metadata")
            or check_result.metadata is None
        ):
            print(f"⚠️ 预言家查验失败,跳过此阶段")
            return

        target_name = check_result.metadata.get("target")
        if not target_name:
            print(f"⚠️ 预言家未选择查验目标,跳过此阶段")
            return

        target_role = self.roles.get(target_name, "村民")

        # 告知预言家结果
        result_msg = (
            f"查验结果：{target_name}是{'狼人' if target_role == '狼人' else '好人'}"
        )
        await seer_agent.observe(await self.moderator.announce(result_msg))

    async def witch_phase(self, killed_player: str):
        """女巫阶段"""
        if not self.witch:
            return killed_player, None

        witch_agent = self.witch[0]
        await self.moderator.announce("🧙‍♀️ 女巫请睁眼...")

        # 告知女巫死亡信息
        death_info = (
            f"今晚{killed_player}被狼人击杀" if killed_player else "今晚平安无事"
        )
        await witch_agent.observe(await self.moderator.announce(death_info))

        # 女巫行动
        witch_action = await witch_agent(structured_model=WitchActionModelCN)

        saved_player = None
        poisoned_player = None

        # 检查返回结果是否有效
        if (
            witch_action is None
            or not hasattr(witch_action, "metadata")
            or witch_action.metadata is None
        ):
            print(f"⚠️ 女巫行动失败,视为不使用技能")
        else:
            if witch_action.metadata.get("use_antidote") and self.witch_has_antidote:
                if killed_player:
                    saved_player = killed_player
                    self.witch_has_antidote = False
                    await witch_agent.observe(
                        await self.moderator.announce(f"你使用解药救了{killed_player}")
                    )

            if witch_action.metadata.get("use_poison") and self.witch_has_poison:
                poisoned_player = witch_action.metadata.get("target_name")
                if poisoned_player:
                    self.witch_has_poison = False
                    await witch_agent.observe(
                        await self.moderator.announce(
                            f"你使用毒药毒杀了{poisoned_player}"
                        )
                    )

        # 确定最终死亡玩家
        final_killed = killed_player if not saved_player else None

        return final_killed, poisoned_player

    async def hunter_phase(self, shot_by_hunter: str):
        """猎人阶段"""
        if not self.hunter:
            return None

        hunter_agent = self.hunter[0]
        if hunter_agent.name == shot_by_hunter:
            await self.moderator.announce("🏹 猎人发动技能，可以带走一名玩家...")

            hunter_action = await hunter_agent(
                structured_model=get_hunter_model_cn(self.alive_players)
            )

            # 检查返回结果是否有效
            if (
                hunter_action is None
                or not hasattr(hunter_action, "metadata")
                or hunter_action.metadata is None
            ):
                print(f"⚠️ 猎人技能使用失败,视为放弃开枪")
                return None

            if hunter_action.metadata.get("shoot"):
                target = hunter_action.metadata.get("target")
                if target:
                    await self.moderator.announce(
                        f"猎人{hunter_agent.name}开枪带走了{target}"
                    )
                    return target
                else:
                    print(f"⚠️ 猎人选择开枪但未指定目标,视为放弃")
                    return None

        return None

    def update_alive_players(self, dead_players: List[str]):
        """更新存活玩家列表"""
        for dead_name in dead_players:
            if dead_name:
                # 从存活列表移除
                self.alive_players = [
                    p for p in self.alive_players if p.name != dead_name
                ]
                # 从各阵营移除
                self.werewolves = [p for p in self.werewolves if p.name != dead_name]
                self.villagers = [p for p in self.villagers if p.name != dead_name]
                self.seer = [p for p in self.seer if p.name != dead_name]
                self.witch = [p for p in self.witch if p.name != dead_name]
                self.hunter = [p for p in self.hunter if p.name != dead_name]

    async def day_phase(self, round_num: int):
        """白天阶段"""
        await self.moderator.day_announcement(round_num)

        # 讨论阶段
        async with MsgHub(
            self.alive_players,
            enable_auto_broadcast=True,
            announcement=await self.moderator.announce(
                f"现在开始自由讨论。存活玩家：{format_player_list(self.alive_players)}"
            ),
        ) as all_hub:
            # 每人发言一轮
            await sequential_pipeline(self.alive_players)

            # 投票阶段
            all_hub.set_auto_broadcast(False)
            vote_msgs = await fanout_pipeline(
                self.alive_players,
                await self.moderator.announce("请投票选择要淘汰的玩家"),
                structured_model=get_vote_model_cn(self.alive_players),
                enable_gather=False,
            )

            # 统计投票
            votes = {}
            for i, vote_msg in enumerate(vote_msgs):
                # 检查vote_msg是否为None或metadata是否存在
                if (
                    vote_msg is not None
                    and hasattr(vote_msg, "metadata")
                    and vote_msg.metadata is not None
                ):
                    votes[self.alive_players[i].name] = vote_msg.metadata.get("vote")
                else:
                    # 如果返回无效,默认弃票
                    print(f"⚠️ {self.alive_players[i].name} 的投票无效,视为弃票")
                    votes[self.alive_players[i].name] = None

            voted_out, vote_count = majority_vote_cn(votes)
            await self.moderator.vote_result_announcement(voted_out, vote_count)

            return voted_out

    async def run_game(self):
        """运行游戏主循环"""
        try:
            await self.setup_game()

            for round_num in range(1, MAX_GAME_ROUND + 1):
                print(f"\n🌙 === 第{round_num}轮游戏开始 ===")

                # 夜晚阶段
                await self.moderator.night_announcement(round_num)

                # 狼人击杀
                killed_player = await self.werewolf_phase(round_num)

                # 预言家查验
                await self.seer_phase()

                # 女巫行动
                final_killed, poisoned_player = await self.witch_phase(killed_player)

                # 更新死亡玩家
                night_deaths = [p for p in [final_killed, poisoned_player] if p]
                self.update_alive_players(night_deaths)

                # 死亡公告
                await self.moderator.death_announcement(night_deaths)

                # 检查胜利条件
                winner = check_winning_cn(self.alive_players, self.roles)
                if winner:
                    await self.moderator.game_over_announcement(winner)
                    return

                # 白天阶段
                voted_out = await self.day_phase(round_num)

                # 猎人技能
                hunter_shot = await self.hunter_phase(voted_out)

                # 更新死亡玩家
                day_deaths = [p for p in [voted_out, hunter_shot] if p]
                self.update_alive_players(day_deaths)

                # 检查胜利条件
                winner = check_winning_cn(self.alive_players, self.roles)
                if winner:
                    await self.moderator.game_over_announcement(winner)
                    return

                print(
                    f"第{round_num}轮结束，存活玩家：{format_player_list(self.alive_players)}"
                )

        except Exception as e:
            print(f"❌ 游戏运行出错：{e}")
            import traceback

            traceback.print_exc()


async def main():
    """主函数"""
    # 检查环境变量
    required_env_keys = ["LLM_API_KEY", "LLM_MODEL_ID", "LLM_BASE_URL"]
    missing = [key for key in required_env_keys if not os.getenv(key)]
    if missing:
        print(f"❌ 请设置环境变量: {', '.join(missing)}")
        return

    print("🎮 欢迎来到三国狼人杀！")

    # 创建并运行游戏
    game = ThreeKingdomsWerewolfGame()
    await game.run_game()


def _safe_get(data: Any, key: str, default: Any = None) -> Any:
    if isinstance(data, dict):
        return data.get(key, default)
    return getattr(data, key, default)


def _preview(text: Any, limit: int = 120) -> str:
    if text is None:
        return ""
    s = str(text).replace("\n", "\\n")
    return s if len(s) <= limit else f"{s[:limit]}..."


class DebugOpenAIChatModel(OpenAIChatModel):
    def __init__(self, *args: Any, debug_enabled: bool = True, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._debug_enabled = debug_enabled

    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        result = await super().__call__(*args, **kwargs)
        if hasattr(result, "__aiter__"):

            async def _wrapped_stream():
                async for chunk in result:
                    self._log_chat_response(chunk)
                    yield chunk

            return _wrapped_stream()
        self._log_chat_response(result)
        return result

    def _log_chat_response(self, response: Any) -> None:
        if not self._debug_enabled:
            return
        blocks = _safe_get(response, "content", []) or []
        summary: list[str] = []
        for block in blocks:
            block_type = _safe_get(block, "type", "unknown")
            if block_type == "thinking":
                summary.append(f"thinking:{_preview(_safe_get(block, 'thinking'))}")
            elif block_type == "text":
                summary.append(f"text:{_preview(_safe_get(block, 'text'))}")
            elif block_type == "tool_use":
                name = _safe_get(block, "name", "")
                inputs = _safe_get(block, "input", {})
                summary.append(f"tool_use:{name}({_preview(inputs, 80)})")
            else:
                summary.append(f"{block_type}:{_preview(block)}")
        print(
            f"[LLM DEBUG] model={self.model_name} id={_safe_get(response, 'id', '-')}"
        )
        print(f"[LLM DEBUG] blocks={summary}")


if __name__ == "__main__":
    asyncio.run(main())
