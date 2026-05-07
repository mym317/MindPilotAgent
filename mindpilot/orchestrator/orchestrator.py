"""
中央编排器（Orchestrator）
===========================
完整流程：
  Step 1 → 任务规划（ToT + ReAct）
  Step 2 → 文献检索（ArXiv + 知识图谱）
  Step 3 → 实验设计（基于文献输出）
  Step 4 → 代码实现（沙箱 + 自动调试）
  Step 5 → 数据分析（统计 + 可视化）
  Step 6 → 评估反思 + 报告生成（Word/MD/HTML）

改进点（v2）：
  1. Step 2→3 串行传递：文献检索结果完整传入实验设计，baseline/指标选取有据可依
  2. 错误降级策略：每个 Step 失败时提供合理的降级结果，而非空 dict
  3. 全局超时控制：每个 Step 有独立超时，防止单个 Agent 卡死整个流程
  4. HumanInTheLoop 改由配置驱动，不再 hardcode enabled=False
  5. aggregated 字段去重整理，字段语义清晰
  6. 新增进度回调 hook（on_step_done），支持外部监听执行进度
"""

import time
import uuid
import concurrent.futures
from typing import Optional, Callable

from config import CONFIG
from framework.logger import MindPilotLogger
from framework.communication import ErrorCode, MessageBus, HumanInTheLoop
from framework.scheduler import SyncScheduler, Task
from memory.memory_store import MemoryStore
from tools.llm_client import LLMClient
from tools.arxiv_search import ArXivSearchTool
from tools.code_executor import CodeExecutor
from tools.visualizer import AutoVisualizer
from tools.report_generator import ReportGenerator
from agents.planning_agent import PlanningAgent
from agents.literature_agent import LiteratureAgent
from agents.code_agent import CodeAgent
from agents.analysis_agent import AnalysisAgent
from agents.evaluation_agent import EvaluationAgent


# 每个 Step 的独立超时（秒），超时后使用降级结果继续流程
_STEP_TIMEOUTS = {
    "planning":    300,   # Step 1：规划，含多次 LLM 调用，给足时间
    "literature":  1024,   # Step 2：文献检索
    "experiment":  1024,   # Step 3：实验设计（在文献检索完成后执行）
    "code":        1024,   # Step 4：代码生成+调试
    "analysis":    240,   # Step 5：数据分析
    "evaluation":  1024,   # Step 6：评估+报告生成
}


class MindPilotOrchestrator:
    def __init__(
        self,
        config=None,
        session_id: Optional[str] = None,
        on_step_done: Optional[Callable[[str, dict], None]] = None,  # ← 改进⑥
    ):
        self.config     = config or CONFIG
        self.session_id = session_id or str(uuid.uuid4())[:12]
        # 改进⑥：进度回调，每个 Step 完成后调用 on_step_done(step_name, result)
        self.on_step_done = on_step_done

        self.logger = MindPilotLogger(
            session_id=self.session_id,
            log_dir=self.config.communication.log_dir,
            verbose=self.config.verbose,
        )
        self.bus = MessageBus()
        self._register_message_endpoints()
        # 改进④：HumanInTheLoop 由配置驱动（字段定义在 config.py MindPilotConfig 中）
        self.human_loop = HumanInTheLoop(
            enabled=self.config.human_in_the_loop
        )

        self.llm    = LLMClient(self.config)
        self.memory = MemoryStore(
            store_dir=self.config.memory_dir,
            session_id=self.session_id,
            logger=self.logger,
        )
        self.arxiv = ArXivSearchTool(
            llm_client=self.llm,
            max_results=self.config.literature.arxiv_max_results,
            logger=self.logger,
        )
        self.executor   = CodeExecutor(timeout=self.config.code.execution_timeout, logger=self.logger)
        self.visualizer = AutoVisualizer(output_dir=self.config.output_dir, logger=self.logger)
        self.report_gen = ReportGenerator(output_dir=self.config.output_dir, logger=self.logger)

        self.planner        = PlanningAgent(self.config, self.llm, self.memory, self.logger)
        self.lit_agent      = LiteratureAgent(self.config, self.llm, self.arxiv, self.memory, self.logger)
        self.code_agent     = CodeAgent(self.config, self.llm, self.executor, self.memory, self.logger)
        self.analysis_agent = AnalysisAgent(
            self.config, self.llm, self.visualizer, self.report_gen, self.memory, self.logger)
        self.eval_agent     = EvaluationAgent(
            self.config, self.llm, self.report_gen, self.memory, self.logger)

        self._print_banner()

    # ── 工具方法：带超时的 Step 执行 ─────────────────────────────────────
    def _run_step(
        self,
        step_name: str,
        fn: Callable,
        fallback: dict,
        timeout: Optional[int] = None,
    ) -> dict:
        """
        改进②③：统一的 Step 执行框架。
          - 捕获异常：记录日志，返回 fallback 而非空 dict
          - 超时控制：超过 timeout 秒后中断，返回 fallback
          - 进度回调：完成后触发 on_step_done
        """
        effective_timeout = timeout or _STEP_TIMEOUTS.get(step_name, 300)
        result = fallback.copy()
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(fn)
                try:
                    result = future.result(timeout=effective_timeout)
                    if result is None:
                        result = fallback.copy()
                except concurrent.futures.TimeoutError:
                    self.logger.error(
                        "Orchestrator",
                        f"【{step_name}】超时（>{effective_timeout}s），使用降级结果继续流程"
                    )
                    result = {**fallback, "_timeout": True}
        except Exception as e:
            self.logger.error("Orchestrator", f"【{step_name}】执行失败: {e}")
            result = {**fallback, "_error": str(e)}

        # 进度回调
        if self.on_step_done:
            try:
                self.on_step_done(step_name, result)
            except Exception:
                pass  # 回调失败不影响主流程

        return result

    # ── 主流程 ────────────────────────────────────────────────────────────
    def run(self, query: str) -> dict:
        self.logger.info("Orchestrator", "="*52)
        self.logger.info("Orchestrator", f"收到科研问题: {query[:60]}")
        self.logger.info("Orchestrator", "="*52)
        start_time = time.time()

        # ── Step 1: 任务规划 ──────────────────────────────────
        self.logger.info("Orchestrator", "【Step 1/6】 任务规划（ToT + ReAct）...")
        plan = None

        def _do_planning():
            self._send_request("PlanningAgent", "planning", {"query": query})
            p = self.planner.run(query)
            self._send_response("PlanningAgent", "planning", {"plan_id": p.plan_id, "tasks": len(p.tasks)})
            self.planner.print_plan(p)
            return p

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_do_planning)
                plan = future.result(timeout=_STEP_TIMEOUTS["planning"])
        except concurrent.futures.TimeoutError:
            self.logger.error("Orchestrator", "规划超时，使用默认任务模板")
        except Exception as e:
            self.logger.error("Orchestrator", f"规划失败: {e}")

        # 规划失败时使用空计划，后续步骤用 fallback 描述
        if plan is None:
            from agents.planning_agent import ResearchPlan
            plan = ResearchPlan(
                plan_id="fallback",
                query=query,
                tasks=[],
                reasoning="规划失败，使用各步骤默认描述",
            )

        if self.on_step_done:
            try:
                self.on_step_done("planning", plan.to_dict())
            except Exception:
                pass

        # ── 审核点 1：规划方案人工审核 ────────────────────────
        if self.human_loop.enabled:
            review = self.human_loop.review_plan(plan)
            if review["action"] == "abort":
                self.logger.info("Orchestrator", "User aborted at plan review")
                return self._make_abort_result(query, start_time, "User aborted at plan review")
            if review["action"] == "select_path" and review.get("path"):
                # 用户选择了不同路径，重新分解任务
                self.logger.info("Orchestrator",
                    f"User selected path: {review['path'][:50]}")
                new_tasks, new_reasoning = self.planner.react.decompose(
                    query, review["path"])
                new_tasks, _ = self.planner._validate_dag(new_tasks)
                plan.tasks = new_tasks
                plan.reasoning = new_reasoning
                plan.selected_path = review["path"]
                self.planner.print_plan(plan)
            if review["action"] == "skip_tasks" and review.get("skip_ids"):
                skip_ids = set(review["skip_ids"])
                plan.tasks = [t for t in plan.tasks if t.task_id not in skip_ids]
                self.logger.info("Orchestrator",
                    f"User skipped tasks: {skip_ids}")

        # 辅助函数：从计划中提取某个 Agent 的任务描述
        def _task_desc(agent_name: str, default: str) -> str:
            t = next((t for t in plan.tasks if t.agent == agent_name), None)
            return t.description if t else default

        # ── Step 2: 文献检索 ──────────────────────────────────
        # 文献检索先行完成，其输出结果（论文列表、综述、知识图谱）将直接
        # 传递给 Step 3 实验设计，使实验设计能够参考领域现有工作，从而
        # 提出更有针对性、更具创新性的实验方案（对照组、指标选取等）。
        self.logger.info("Orchestrator", "【Step 2/6】 文献检索与知识图谱构建...")

        lit_fallback = {
            "top_papers": [],
            "literature_review": f"文献检索未完成，请手动补充关于「{query[:30]}」的相关文献",
            "knowledge_graph": {},
            "_fallback": True,
        }
        lit_desc = _task_desc(
            "LiteratureAgent", f"检索关于「{query}」的学术文献"
        )
        lit_result = self._run_step(
            "literature",
            lambda: self.lit_agent.run(lit_desc, query),
            lit_fallback,
        )

        # ── Step 3: 实验设计（以文献结果为上下文）────────────────────
        # 将文献检索输出（top_papers、literature_review、knowledge_graph）
        # 完整传入 design_experiment，让实验设计 Agent 能够：
        #   ① 参考已有方法选取合适的 baseline
        #   ② 根据文献中常用指标确定评估维度
        #   ③ 识别文献空白，提出差异化的实验假设
        self.logger.info("Orchestrator", "【Step 3/6】 实验设计方案生成（基于文献结果）...")

        exp_fallback = {
            "research_hypothesis": f"针对「{query[:40]}」进行对比实验",
            "baselines": ["Baseline A", "Baseline B"],
            "metrics": ["Accuracy", "F1", "Runtime"],
            "full_description": "实验设计未完成，请根据文献结果手动补充",
            "_fallback": True,
        }
        exp_design = self._run_step(
            "experiment",
            lambda: self.eval_agent.design_experiment(query, lit_result, research_path=plan.selected_path),
            exp_fallback,
        )

        self._print_exp_design(exp_design)

        # ── 审核点 2：实验设计人工审核 ────────────────────────
        if self.human_loop.enabled:
            review = self.human_loop.review_experiment(query, exp_design)
            if review["action"] == "abort":
                self.logger.info("Orchestrator", "User aborted at experiment review")
                return self._make_abort_result(query, start_time, "User aborted at experiment review")
            if review["action"] == "modify" and review.get("modifications"):
                for key, val in review["modifications"].items():
                    exp_design[key] = val
                self.logger.info("Orchestrator",
                    f"User modified experiment: {list(review['modifications'].keys())}")
                self._print_exp_design(exp_design)

        # ── Step 4: 代码实现 ──────────────────────────────────
        self.logger.info("Orchestrator", "【Step 4/6】 代码生成与自动调试...")
        code_desc = _task_desc("CodeAgent", f"为「{query}」实现核心算法")
        context = {
            "top_papers": lit_result.get("top_papers", [])[:3],
            "exp_design":  exp_design,
            "baselines":   exp_design.get("baselines", []),
            "metrics":     exp_design.get("metrics", []),
        }
        code_fallback = {
            "final_code": f"# 代码生成失败，请手动实现\n# 问题：{query[:60]}\n",
            "stdout": "",
            "success": False,
            "_fallback": True,
        }
        code_result = self._run_step(
            "code",
            lambda: self.code_agent.run(code_desc, context=context),
            code_fallback,
        )

        # ── 审核点 3：代码执行结果人工审核 ────────────────────
        _skip_analysis = False
        if self.human_loop.enabled:
            review = self.human_loop.review_code(code_result)
            if review["action"] == "abort":
                self.logger.info("Orchestrator", "User aborted at code review")
                return self._make_abort_result(query, start_time, "User aborted at code review")
            if review["action"] == "retry":
                self.logger.info("Orchestrator", "User requested code retry")
                code_result = self._run_step(
                    "code",
                    lambda: self.code_agent.run(code_desc, context=context),
                    code_fallback,
                )
            if review["action"] == "skip":
                self.logger.info("Orchestrator", "User skipped analysis step")
                _skip_analysis = True

        # ── Step 5: 数据分析 ──────────────────────────────────
        if _skip_analysis:
            analysis_result = {
                "conclusion": "Analysis skipped by user",
                "charts": [],
                "_skipped": True,
            }
        else:
            self.logger.info("Orchestrator", "【Step 5/6】 数据分析与可视化...")
            ana_desc = _task_desc("AnalysisAgent", f"分析「{query}」的实验结果")
            code_stdout = code_result.get("stdout", "")
            analysis_fallback = {
                "conclusion": "数据分析未完成，代码执行结果为空或分析失败",
                "charts": [],
                "_fallback": True,
            }
            analysis_result = self._run_step(
                "analysis",
                lambda: self.analysis_agent.run(ana_desc, code_output=code_stdout),
                analysis_fallback,
            )

        # ── Step 6: 评估反思 + 报告生成 ──────────────────────
        self.logger.info("Orchestrator", "【Step 6/6】 评估反思与报告生成（Word/MD/HTML）...")
        # 改进⑤：字段整理，去除重复，语义清晰
        aggregated = {
            # 原始各模块完整输出（供评估 Agent 全面访问）
            "literature_result":  lit_result,
            "experiment_design":  exp_design,
            "code_result":        code_result,
            "analysis_result":    analysis_result,
            # 摘要字段（供报告生成快速引用）
            "literature_review":  lit_result.get("literature_review", ""),
            "papers":             lit_result.get("top_papers", []),
            "code":               code_result.get("final_code", ""),
            "stdout":             code_result.get("stdout", ""),
            "analysis":           analysis_result.get("conclusion", ""),
            "charts":             analysis_result.get("charts", []),
            # 标记哪些步骤使用了降级结果（供评估 Agent 在报告中注明）
            "fallback_steps": [
                name for name, res in [
                    ("literature", lit_result),
                    ("experiment", exp_design),
                    ("code", code_result),
                    ("analysis", analysis_result),
                ]
                if res.get("_fallback") or res.get("_timeout") or res.get("_error")
            ],
        }
        eval_fallback = {
            "final_score": {"overall": "N/A"},
            "report_files": {},
            "reflection": "评估未完成",
            "_fallback": True,
        }
        eval_result = self._run_step(
            "evaluation",
            lambda: self.eval_agent.run(query, aggregated),
            eval_fallback,
        )

        # ── 收尾 ──────────────────────────────────────────────
        self.memory.save_long_term()
        self.logger.save_summary()
        self.logger.print_call_chain()

        total_time   = round(time.time() - start_time, 2)
        final_result = {
            "query":        query,
            "session_id":   self.session_id,
            "plan":         plan.to_dict(),
            "literature":   lit_result,
            "experiment":   exp_design,
            "code":         code_result,
            "analysis":     analysis_result,
            "evaluation":   eval_result,
            "report_files": eval_result.get("report_files", {}),
            "total_time_s": total_time,
            "session_log":  str(self.logger.log_file),
            # "message_bus":  bus_stats,
        }
        self._print_final_summary(final_result, total_time)
        return final_result

    def _register_message_endpoints(self):
        for agent in [
            "Orchestrator",
            "PlanningAgent",
            "LiteratureAgent",
            "CodeAgent",
            "AnalysisAgent",
            "EvaluationAgent",
        ]:
            self.bus.register(agent)

    def _send_request(self, receiver: str, task_id: str, payload: dict):
        msg = self.bus.request(
            sender="Orchestrator",
            receiver=receiver,
            task_id=task_id,
            payload=payload,
            session_id=self.session_id,
        )
        self.logger.log_message(msg)
        return msg

    def _send_response(self, sender: str, task_id: str, payload: dict):
        msg = self.bus.response(
            sender=sender,
            receiver="Orchestrator",
            task_id=task_id,
            payload=payload,
            session_id=self.session_id,
        )
        self.logger.log_message(msg)
        return msg

    def _send_error(self, sender: str, task_id: str, error: Exception):
        msg = self.bus.error(
            sender=sender,
            receiver="Orchestrator",
            task_id=task_id,
            detail=str(error),
            code=ErrorCode.EXECUTION_ERROR,
            session_id=self.session_id,
        )
        self.logger.log_message(msg)
        return msg

    def _print_exp_design(self, exp: dict):
        print(f"\n{'='*60}")
        print(f"  Experiment Design{'  [DEGRADED]' if exp.get('_fallback') else ''}")
        print(f"{'='*60}")
        hyp = exp.get("research_hypothesis", "")
        if hyp:
            print(f"  Hypothesis: {hyp[:70]}")
        baselines = exp.get("baselines", [])
        if baselines:
            print(f"  Baselines:  {' | '.join(str(b)[:25] for b in baselines[:3])}")
        metrics = exp.get("metrics", [])
        if metrics:
            print(f"  Metrics:    {' | '.join(str(m)[:20] for m in metrics[:3])}")
        desc = exp.get("full_description", "")
        if desc:
            print(f"  Description: {desc[:80]}...")
        print(f"{'='*60}\n")

    def _make_abort_result(self, query: str, start_time: float, reason: str) -> dict:
        """Generate a result dict when user aborts the run via Human-in-the-Loop."""
        self.memory.save_long_term()
        self.logger.save_summary()
        total_time = round(time.time() - start_time, 2)
        print(f"\n  [ABORTED] {reason}")
        print(f"  Elapsed: {total_time}s\n")
        return {
            "query":        query,
            "session_id":   self.session_id,
            "aborted":      True,
            "abort_reason":  reason,
            "plan":         {},
            "literature":   {},
            "experiment":   {},
            "code":         {},
            "analysis":     {},
            "evaluation":   {},
            "report_files": {},
            "total_time_s": total_time,
            "session_log":  str(self.logger.log_file),
            "human_decisions": self.human_loop.decisions if self.human_loop.enabled else [],
        }

    def _print_banner(self):
        mode = "Mock 模式" if self.config.mock_mode else f"API [{self.config.llm.model}]"
        hitl = "开启" if self.config.human_in_the_loop else "关闭"
        print(f"""
╔══════════════════════════════════════════════════════╗
║         MindPilot  多模态智能科研助手 Agent           ║
║         六步全流程：规划→文献→实验→代码→分析→报告    ║
╠══════════════════════════════════════════════════════╣
║  会话 ID   : {self.session_id:<40s}║
║  LLM 模式  : {mode:<40s}║
║  人工审核  : {hitl:<40s}║
╚══════════════════════════════════════════════════════╝
""")

    def _print_final_summary(self, result: dict, total_time: float):
        score    = result["evaluation"].get("final_score", {}).get("overall", "N/A")
        reports  = result.get("report_files", {})
        fallback = result.get("evaluation", {}).get("_fallback")
        steps_fb = result.get("evaluation", {}).get("fallback_steps", [])

        print(f"""
╔══════════════════════════════════════════════════════╗
║                  ✅ 全流程完成！                      ║
╠══════════════════════════════════════════════════════╣
║  总耗时   : {total_time}s""")
        print(f"║  质量评分 : {score}{'  (评估模块降级)' if fallback else ''}")
        if steps_fb:
            print(f"║  降级步骤 : {', '.join(steps_fb)}（结果可能不完整）")
        print(f"║  报告文件 :")
        for fmt, path in reports.items():
            print(f"║    [{fmt.upper():8s}] {path}")
        print(f"║  会话日志 : {result['session_log']}")
        print(f"╚══════════════════════════════════════════════════════╝")
        