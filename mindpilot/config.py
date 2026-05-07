"""
MindPilot 全局配置
==================
支持任意 OpenAI 兼容接口（百炼、OpenAI、DeepSeek、Ollama 等）。

百炼 Coding Plan 可用模型（2026-01）：
  千问系列：qwen3.5-plus / qwen3-max-2026-01-23 / qwen3-coder-next / qwen3-coder-plus
  智谱系列：glm-5 / glm-4.7
  Kimi：    kimi-k2.5
  MiniMax： MiniMax-M2.5
"""

import os
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


def _is_mock() -> bool:
    key = os.getenv("LLM_API_KEY", "")
    return not key or key.startswith("sk-your-") or key == "mock"


@dataclass
class LLMConfig:
    api_key: str = field(default_factory=lambda: os.getenv("LLM_API_KEY", "mock"))
    base_url: str = field(default_factory=lambda: os.getenv(
        "LLM_BASE_URL", "https://coding.dashscope.aliyuncs.com/v1"))

    # 通用对话模型（规划、文献摘要、评估等）
    model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "glm-5"))

    # 代码生成专用模型
    code_model: str = field(default_factory=lambda: os.getenv("CODE_MODEL", "qwen3-coder-plus"))

    # 通用对话参数
    temperature: float = field(default_factory=lambda: float(os.getenv("LLM_TEMPERATURE", "0.3")))
    max_tokens: int = field(default_factory=lambda: int(os.getenv("LLM_MAX_TOKENS", "2048")))

    # 代码生成专用参数
    code_temperature: float = field(default_factory=lambda: float(os.getenv("CODE_TEMPERATURE", "0.1")))
    code_max_tokens: int = field(default_factory=lambda: int(os.getenv("CODE_MAX_TOKENS", "4096")))

    # timeout 设为 180s：glm-5 等深度思考模型单次响应可能需要 60~120s，
    # 用 60s 会导致调用被误判超时后静默降级到 Mock 响应。
    # 这是保证 Agent 能力的关键参数：超时太短 = 实际上没有调用真实模型。
    timeout: int = 180

    proxy_url: Optional[str] = field(
        default_factory=lambda: os.getenv("LLM_PROXY_URL", "") or None)


@dataclass
class PlanningConfig:
    """
    模块① 任务规划配置

    ToT（Tree of Thought）参数说明：
      branching_factor = 每轮生成的候选研究路径数
        → 越大 = 搜索空间越广 = 规划质量越高 = 消耗 LLM 调用次数越多
        → 推荐保持 3（完整能力），时间充裕时可调到 4~5
      max_depth = ToT 搜索深度（当前实现为单层评分，此参数控制路径生成轮数）
        → 保持 3，与 branching_factor 配合

    注意：branching_factor=3 时，规划阶段需要约 6~9 次 LLM 调用。
    使用 glm-5（单次约 30~60s）时，规划耗时预计 4~8 分钟，这是正常现象。
    """
    max_depth: int = 3            # 保持 3，不降低搜索深度
    branching_factor: int = 3     # 保持 3，保证 ToT 完整搜索能力
    memory_top_k: int = 5
    max_subtasks: int = 8


@dataclass
class LiteratureConfig:
    arxiv_max_results: int = 10
    embedding_model: str = "all-MiniLM-L6-v2"
    retrieval_top_k: int = 5
    summary_max_len: int = 300


@dataclass
class CodeConfig:
    max_debug_rounds: int = 5
    execution_timeout: int = 90
    forbidden_modules: list = field(default_factory=lambda: [
        "os.system", "subprocess", "shutil.rmtree", "__import__('os').system"
    ])


@dataclass
class AnalysisConfig:
    significance_level: float = 0.05
    default_chart_format: str = "png"
    report_formats: list = field(default_factory=lambda: ["markdown", "html", "pdf"])


@dataclass
class CommunicationConfig:
    max_concurrent_agents: int = 4
    message_timeout: int = 300   # 同步调整：单任务最长等待时间
    retry_attempts: int = 3
    log_dir: str = "logs"


@dataclass
class EvaluationConfig:
    score_threshold: float = 0.65
    max_reflection_rounds: int = 3
    judge_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "glm-5"))


@dataclass
class MindPilotConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    planning: PlanningConfig = field(default_factory=PlanningConfig)
    literature: LiteratureConfig = field(default_factory=LiteratureConfig)
    code: CodeConfig = field(default_factory=CodeConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    communication: CommunicationConfig = field(default_factory=CommunicationConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    output_dir: str = "outputs"
    memory_dir: str = "memory/store"
    mock_mode: bool = field(default_factory=_is_mock)
    verbose: bool = True
    # 改进④：HumanInTheLoop 由配置驱动，而非 hardcode
    # 设为 True 时，关键步骤（如代码执行、实验设计）会暂停等待用户确认
    # 通过环境变量 HUMAN_IN_THE_LOOP=true 开启
    human_in_the_loop: bool = field(
        default_factory=lambda: os.getenv("HUMAN_IN_THE_LOOP", "").lower() in ("true", "1", "yes")
    )


CONFIG = MindPilotConfig()
