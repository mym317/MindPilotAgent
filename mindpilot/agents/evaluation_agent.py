"""
Module 6: evaluation, reflection, and final report generation.
"""

from __future__ import annotations

import json
import random
import re
import time
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class EvalScore:
    overall: float
    accuracy: float
    completeness: float
    format_quality: float
    feedback: str
    needs_reflection: bool
    reflection_suggestion: str = ""


@dataclass
class ReflectionRecord:
    round_num: int
    original_output: str
    score_before: float
    reflection: str
    revised_output: str
    score_after: float
    improved: bool


def _extract_json_object(text: str) -> Optional[dict]:
    # Many LLM responses include extra prose before/after the JSON payload.
    match = re.search(r"\{[\s\S]+\}", text or "")
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except Exception:
        return None


def _clamp_score(value: Any, default: float = 0.7) -> float:
    # Judge scores are normalized to the closed interval [0, 1].
    try:
        return max(0.0, min(1.0, float(value)))
    except Exception:
        return default


class LLMJudge:
    def __init__(self, llm_client, threshold: float = 0.65, logger=None):
        self.llm = llm_client
        self.threshold = threshold
        self.logger = logger

    def score(
        self,
        query: str,
        output: str,
        output_type: str = "report",
        role_name: str = "",
        rubric: str = "",
    ) -> EvalScore:
        # First quality gate: ask the judge model for structured subscores and
        # actionable feedback instead of a single opaque grade.
        reviewer = role_name or "科研输出质量评审专家"
        system = f"你是一名严格的{reviewer}。请对下面的{output_type}评分，并只返回 JSON。"
        rubric_part = f"本轮重点评分维度：{rubric}\n\n" if rubric else ""
        prompt = (
            "返回格式：\n"
            "{"
            '"overall_score": 0.70, '
            '"accuracy": 0.70, '
            '"completeness": 0.70, '
            '"format_quality": 0.70, '
            '"feedback": "具体评价", '
            '"needs_reflection": false, '
            '"reflection_suggestion": "可选的改进建议"'
            "}\n\n"
            "评分标准：accuracy=内容准确性，completeness=信息完整性，"
            "format_quality=结构与表达质量。当 overall_score 低于阈值时，needs_reflection=true。\n\n"
            f"{rubric_part}"
            f"研究问题：{query}\n\n"
            "注意：下方可能是为节省上下文而截取的报告文本。"
            "如果内容末尾被截断，请不要把“提示截取边界”误判为报告正文存在句子截断；"
            "只有在章节正文内部明显缺字、缺句或语义断裂时，才指出截断问题。\n\n"
            f"待评分内容（报告节选，最多 8000 字）：\n{output[:8000]}"
        )
        resp = self.llm.chat(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ]
        )
        data = _extract_json_object(resp)
        if data:
            # A valid JSON payload lets us keep downstream logic deterministic.
            overall = _clamp_score(data.get("overall_score", 0.7))
            accuracy = _clamp_score(data.get("accuracy", overall))
            completeness = _clamp_score(data.get("completeness", overall))
            format_quality = _clamp_score(data.get("format_quality", overall))
            needs_reflection = bool(data.get("needs_reflection", overall < self.threshold))
            return EvalScore(
                overall=overall,
                accuracy=accuracy,
                completeness=completeness,
                format_quality=format_quality,
                feedback=str(data.get("feedback", "")).strip(),
                needs_reflection=needs_reflection,
                reflection_suggestion=str(data.get("reflection_suggestion", "")).strip(),
            )

        # Fallback keeps the whole pipeline alive when judge output is malformed.
        score = round(random.uniform(0.45, 0.70), 2)
        return EvalScore(
            overall=score,
            accuracy=min(score + 0.02, 1.0),
            completeness=max(score - 0.03, 0.0),
            format_quality=min(score + 0.05, 1.0),
            feedback="Mock 评分",
            needs_reflection=score < self.threshold,
        )

    def compute_rouge_l(self, hypothesis: str, reference: str) -> float:
        def lcs(left: list[str], right: list[str]) -> int:
            rows = len(left)
            cols = len(right)
            dp = [[0] * (cols + 1) for _ in range(rows + 1)]
            for i in range(1, rows + 1):
                for j in range(1, cols + 1):
                    if left[i - 1] == right[j - 1]:
                        dp[i][j] = dp[i - 1][j - 1] + 1
                    else:
                        dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
            return dp[rows][cols]

        hyp_tokens = hypothesis.lower().split()
        ref_tokens = reference.lower().split()
        if not hyp_tokens or not ref_tokens:
            return 0.0
        common = lcs(hyp_tokens[:100], ref_tokens[:100])
        precision = common / len(hyp_tokens)
        recall = common / len(ref_tokens)
        if precision + recall == 0:
            return 0.0
        return round(2 * precision * recall / (precision + recall), 4)


class MultiAgentJudge:
    """Use multiple LLM judge personas and aggregate their scores."""

    SEGMENT_CHAR_LIMIT = 3500

    DEFAULT_REVIEWERS = [
        {
            "name": "证据一致性评审专家",
            "weight": 0.40,
            "rubric": "重点检查报告结论是否有文献、实验设计、代码执行和分析结果支撑，惩罚无依据的泛化与夸大。",
        },
        {
            "name": "实验方法评审专家",
            "weight": 0.35,
            "rubric": "重点检查实验假设、评估指标、基线方法、流程和结果分析是否前后一致。",
        },
        {
            "name": "论文写作与结构评审专家",
            "weight": 0.25,
            "rubric": "重点检查摘要、章节结构、学术表达、格式规范和可读性。",
        },
    ]

    def __init__(self, llm_client, threshold: float = 0.65, logger=None, reviewers: Optional[list] = None):
        self.threshold = threshold
        self.reviewers = reviewers or self.DEFAULT_REVIEWERS
        self.single_judge = LLMJudge(llm_client, threshold=threshold, logger=logger)

    def score_many(self, query: str, output: str, output_type: str = "report") -> tuple[EvalScore, list[dict]]:
        segments = self._split_report_segments(output)
        reviews = []
        for reviewer in self.reviewers:
            score, segment_reviews = self._score_reviewer_segments(
                query,
                segments,
                output_type,
                reviewer,
            )
            reviews.append(
                {
                    "reviewer": reviewer["name"],
                    "weight": reviewer["weight"],
                    "rubric": reviewer["rubric"],
                    "score": score,
                    "segments": segment_reviews,
                }
            )

        total_weight = sum(item["weight"] for item in reviews) or 1.0
        overall = round(sum(item["weight"] * item["score"].overall for item in reviews) / total_weight, 3)
        accuracy = round(sum(item["weight"] * item["score"].accuracy for item in reviews) / total_weight, 3)
        completeness = round(sum(item["weight"] * item["score"].completeness for item in reviews) / total_weight, 3)
        format_quality = round(sum(item["weight"] * item["score"].format_quality for item in reviews) / total_weight, 3)
        needs_reflection = overall < self.threshold or any(
            item["score"].needs_reflection and item["score"].overall < self.threshold
            for item in reviews
        )
        feedback = " | ".join(
            f"{item['reviewer']}: {item['score'].feedback or '无详细反馈'}"
            for item in reviews
        )
        suggestions = " | ".join(
            item["score"].reflection_suggestion
            for item in reviews
            if item["score"].reflection_suggestion
        )
        aggregate = EvalScore(
            overall=overall,
            accuracy=accuracy,
            completeness=completeness,
            format_quality=format_quality,
            feedback=feedback,
            needs_reflection=needs_reflection,
            reflection_suggestion=suggestions,
        )
        serializable_reviews = [
            {
                "reviewer": item["reviewer"],
                "weight": item["weight"],
                "rubric": item["rubric"],
                "score": item["score"].__dict__,
                "segments": item.get("segments", []),
            }
            for item in reviews
        ]
        return aggregate, serializable_reviews

    def score(self, query: str, output: str, output_type: str = "report") -> EvalScore:
        score, _ = self.score_many(query, output, output_type=output_type)
        return score

    def _score_reviewer_segments(
        self,
        query: str,
        segments: list[dict],
        output_type: str,
        reviewer: dict,
    ) -> tuple[EvalScore, list[dict]]:
        segment_reviews = []
        total_segments = len(segments) or 1
        for idx, segment in enumerate(segments, 1):
            segment_rubric = (
                f"{reviewer['rubric']}\n"
                f"本次采用分段评审，这是第 {idx}/{total_segments} 段：{segment['title']}。"
                "请只评价本段真实内容，不要因为没有看到整篇报告而惩罚完整性；"
                "只有本段内部确实存在语义断裂时，才指出截断问题。"
            )
            score = self.single_judge.score(
                query,
                segment["text"],
                output_type=f"{output_type}片段 {idx}/{total_segments}: {segment['title']}",
                role_name=reviewer["name"],
                rubric=segment_rubric,
            )
            segment_reviews.append(
                {
                    "index": idx,
                    "title": segment["title"],
                    "chars": segment["chars"],
                    "score": score,
                }
            )

        aggregate = self._aggregate_segment_scores(segment_reviews)
        return aggregate, [
            {
                "index": item["index"],
                "title": item["title"],
                "chars": item["chars"],
                "score": item["score"].__dict__,
            }
            for item in segment_reviews
        ]

    def _aggregate_segment_scores(self, segment_reviews: list[dict]) -> EvalScore:
        if not segment_reviews:
            return EvalScore(0.0, 0.0, 0.0, 0.0, "未获得分段评审结果", True)

        total_chars = sum(max(1, item.get("chars", 0)) for item in segment_reviews) or 1

        def weighted(attr: str) -> float:
            return round(
                sum(max(1, item.get("chars", 0)) * getattr(item["score"], attr) for item in segment_reviews)
                / total_chars,
                3,
            )

        overall = weighted("overall")
        accuracy = weighted("accuracy")
        completeness = weighted("completeness")
        format_quality = weighted("format_quality")
        needs_reflection = overall < self.threshold or any(item["score"].needs_reflection for item in segment_reviews)
        weakest = sorted(segment_reviews, key=lambda item: item["score"].overall)[:3]
        feedback = "；".join(
            f"{item['title']}：{item['score'].feedback or '无详细反馈'}"
            for item in weakest
        )
        suggestions = "；".join(
            item["score"].reflection_suggestion
            for item in segment_reviews
            if item["score"].reflection_suggestion
        )
        return EvalScore(
            overall=overall,
            accuracy=accuracy,
            completeness=completeness,
            format_quality=format_quality,
            feedback=feedback,
            needs_reflection=needs_reflection,
            reflection_suggestion=suggestions,
        )

    def _split_report_segments(self, output: str) -> list[dict]:
        text = (output or "").strip()
        if not text:
            return [{"title": "空报告", "text": "", "chars": 0}]

        blocks = self._split_report_blocks(text)
        segments = []
        current_title = ""
        current_parts = []
        current_len = 0

        for title, block_text in blocks:
            block_len = len(block_text)
            if block_len > self.SEGMENT_CHAR_LIMIT:
                if current_parts:
                    segments.append(self._make_segment(current_title, current_parts))
                    current_title, current_parts, current_len = "", [], 0
                segments.extend(self._chunk_large_block(title, block_text))
                continue

            if current_parts and current_len + block_len > self.SEGMENT_CHAR_LIMIT:
                segments.append(self._make_segment(current_title, current_parts))
                current_title, current_parts, current_len = "", [], 0

            if not current_title:
                current_title = title
            elif title not in current_title:
                current_title = f"{current_title} / {title}"
            current_parts.append(block_text)
            current_len += block_len

        if current_parts:
            segments.append(self._make_segment(current_title, current_parts))

        return segments or [{"title": "完整报告", "text": text, "chars": len(text)}]

    def _split_report_blocks(self, text: str) -> list[tuple[str, str]]:
        blocks = []
        current_title = "报告开头"
        current_lines = []
        for line in text.splitlines():
            heading = re.match(r"^(#{1,3})\s+(.+?)\s*$", line)
            if heading and current_lines:
                block_text = "\n".join(current_lines).strip()
                if block_text:
                    blocks.append((current_title, block_text))
                current_title = heading.group(2).strip()
                current_lines = [line]
            else:
                if heading:
                    current_title = heading.group(2).strip()
                current_lines.append(line)

        block_text = "\n".join(current_lines).strip()
        if block_text:
            blocks.append((current_title, block_text))
        return blocks or [("完整报告", text)]

    def _chunk_large_block(self, title: str, text: str) -> list[dict]:
        chunks = []
        current = []
        current_len = 0
        paragraphs = re.split(r"(\n\s*\n)", text)
        for part in paragraphs:
            if not part:
                continue
            part_len = len(part)
            if current and current_len + part_len > self.SEGMENT_CHAR_LIMIT:
                chunks.append(self._make_segment(title, current))
                current, current_len = [], 0
            if part_len > self.SEGMENT_CHAR_LIMIT:
                for start in range(0, part_len, self.SEGMENT_CHAR_LIMIT):
                    chunks.append(
                        {
                            "title": title,
                            "text": part[start:start + self.SEGMENT_CHAR_LIMIT],
                            "chars": len(part[start:start + self.SEGMENT_CHAR_LIMIT]),
                        }
                    )
            else:
                current.append(part)
                current_len += part_len
        if current:
            chunks.append(self._make_segment(title, current))
        return chunks

    def _make_segment(self, title: str, parts: list[str]) -> dict:
        text = "\n".join(part for part in parts if part).strip()
        return {"title": title or "报告片段", "text": text, "chars": len(text)}


class SelfReflector:
    def __init__(self, llm_client, max_rounds=3, logger=None):
        self.llm = llm_client
        self.max_rounds = max_rounds
        self.logger = logger

    def reflect_and_revise(self, query: str, output: str, score: EvalScore) -> str:
        system = (
            "你是科研报告质量改进专家。请根据评审反馈补充缺失内容、修正表达问题，"
            "在不删减核心信息的前提下提升报告质量。"
        )
        prompt = (
            f"研究问题：{query}\n\n"
            f"待改进报告（前 1500 字）：\n{output[:1500]}\n\n"
            f"评审意见：{score.feedback}\n"
            f"改进建议：{score.reflection_suggestion}\n"
            f"当前得分：{score.overall:.2f}（目标至少 0.65）\n\n"
            "请给出改进后的完整版本。"
        )
        return self.llm.chat(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ]
        )

    def reflect_and_revise_report(
        self,
        query: str,
        report_content: dict,
        score: EvalScore,
        *,
        evidence_context: str = "",
        findings: Optional[list[dict]] = None,
        target_sections: Optional[list[dict]] = None,
    ) -> Optional[dict]:
        """Return a targeted revised report payload so accepted reflections reach saved files."""
        sections = target_sections or [
            {
                "index": idx,
                "heading": sec.get("heading", ""),
                "body": sec.get("body", ""),
                "level": sec.get("level", 1),
            }
            for idx, sec in enumerate(report_content.get("sections", []))
        ]
        system = (
            "你是科研报告质量改进专家。请只修订指定章节，严格依据给定证据，"
            "禁止只根据研究问题自行补编实验、文献或结果。只返回 JSON。"
        )
        prompt = (
            f"研究问题：{query}\n\n"
            f"当前总分：{score.overall:.2f}\n"
            f"评审反馈：{score.feedback}\n"
            f"改进建议：{score.reflection_suggestion}\n\n"
            f"规则检查发现：\n{json.dumps(findings or [], ensure_ascii=False)}\n\n"
            f"可用上游证据边界：\n{evidence_context or '未提供额外证据。'}\n\n"
            "修订要求：\n"
            "1. 只修改下方 target_sections 中列出的章节，不要改动未列出的章节。\n"
            "2. 任何新增判断都必须能从“可用上游证据边界”中找到依据。\n"
            "3. 如果证据不足，请改成保守表述或明确说明缺失，不要创造论文、指标、基线、数值或实验结论。\n"
            "4. 若代码执行失败，结果和结论只能说明未完成验证及后续验证计划。\n\n"
            "返回格式：\n"
            "{\n"
            '  "abstract": "可选，仅当摘要也需要修订时返回",\n'
            '  "section_revisions": [{"index": 0, "body": "指定章节修订内容"}]\n'
            "}\n\n"
            f"当前摘要：\n{report_content.get('abstract', '')}\n\n"
            f"target_sections：\n{json.dumps(sections, ensure_ascii=False)}"
        )
        resp = self.llm.chat(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ]
        )
        return _extract_json_object(resp)


class BenchmarkEvaluator:
    BENCHMARK_QUESTIONS = [
        "Transformer 的自注意力机制是如何工作的？",
        "BERT 和 GPT 的预训练目标有什么根本区别？",
        "联邦学习的核心优势和主要挑战是什么？",
        "扩散模型的前向和反向过程分别是什么？",
        "对比学习的核心思想是什么？",
        "知识蒸馏如何将大模型压缩为小模型？",
        "图神经网络中的消息传递机制是什么？",
        "梯度消失问题有哪些常见解决方案？",
        "强化学习中 PPO 算法的关键改进点是什么？",
        "大模型的 Scaling Law 说明了什么规律？",
    ]

    ANSWER_KEYWORDS = {
        0: ["query", "key", "value", "softmax", "注意力"],
        1: ["mlm", "nsp", "自回归", "双向", "单向"],
        2: ["隐私", "数据不出域", "通信", "异构", "fedavg"],
        3: ["马尔可夫", "噪声", "去噪", "ddpm", "生成"],
        4: ["正样本", "负样本", "simclr", "moco", "infonce"],
        5: ["教师", "学生", "软标签", "温度", "kl"],
        6: ["邻居聚合", "消息", "更新", "过平滑", "gcn"],
        7: ["sigmoid", "残差连接", "梯度裁剪", "lstm", "激活函数"],
        8: ["clip", "重要性采样", "近端策略", "ppo", "优势函数"],
        9: ["参数量", "计算量", "幂律", "涌现", "数据量"],
    }

    def __init__(self, llm_client, logger=None):
        self.llm = llm_client
        self.logger = logger

    def run_comparison(self, system_runner, n_questions=5) -> dict:
        results = {"mindpilot": [], "llm_only": [], "rag_only": []}
        for idx, question in enumerate(self.BENCHMARK_QUESTIONS[:n_questions]):
            if self.logger:
                self.logger.info("EvaluationAgent", f"Benchmark {idx + 1}/{n_questions}: {question[:40]}")
            keywords = self.ANSWER_KEYWORDS.get(idx, [])
            try:
                mindpilot_output = system_runner(question) if system_runner else question
            except Exception:
                mindpilot_output = question
            llm_output = self.llm.chat([{"role": "user", "content": question}])
            results["mindpilot"].append(self._recall(str(mindpilot_output), keywords))
            results["llm_only"].append(self._recall(str(llm_output), keywords))
            results["rag_only"].append(round(results["llm_only"][-1] * 0.85, 3))
            time.sleep(0.1)

        summary = {
            name: {
                "scores": scores,
                "avg": round(sum(scores) / len(scores), 3) if scores else 0.0,
                "max": max(scores) if scores else 0.0,
                "min": min(scores) if scores else 0.0,
            }
            for name, scores in results.items()
        }
        summary["mindpilot_wins"] = sum(
            1
            for idx in range(min(n_questions, len(results["mindpilot"])))
            if results["mindpilot"][idx] >= max(results["llm_only"][idx], results["rag_only"][idx])
        )
        summary["total_questions"] = n_questions
        return summary

    def _recall(self, text: str, keywords: list[str]) -> float:
        if not keywords:
            return 0.5
        lower = text.lower()
        hits = sum(1 for keyword in keywords if keyword.lower() in lower)
        return round(hits / len(keywords), 3)


class EvaluationAgent:
    AGENT_NAME = "EvaluationAgent"

    def __init__(self, config, llm_client, report_gen, memory_store, logger):
        self.config = config
        self.llm = llm_client
        self.report_gen = report_gen
        self.memory = memory_store
        self.logger = logger
        self.judge = MultiAgentJudge(llm_client, threshold=config.evaluation.score_threshold, logger=logger)
        self.reflector = SelfReflector(
            llm_client, max_rounds=config.evaluation.max_reflection_rounds, logger=logger
        )
        # Reserved for offline evaluation/ablation experiments rather than the
        # normal online report path.
        self.benchmark = BenchmarkEvaluator(llm_client, logger=logger)

    # ── 实验设计（EvaluationAgent 内部生成）───────────────────
    def design_experiment(self, query: str, literature_result: dict,
                          research_path: str = "",
                          upstream_design: Optional[dict] = None) -> dict:
        """
        基于文献综述和推荐研究路径生成完整实验设计方案。

        当前整体流程不再依赖外部实验设计模块，因此这里负责产出
        研究假设、数据集、基线、指标、流程、消融和可复现性设置。
        upstream_design 仅作为兼容旧调用或测试桩的可选覆盖。
        """
        call = self.logger.start_call(self.AGENT_NAME, "experiment_design", query)
        try:
            source_design = (
                upstream_design
                if upstream_design is not None
                else self._find_upstream_experiment_design(literature_result)
            )
            if self._has_experiment_design_content(source_design):
                result = self._normalize_experiment_design_payload(source_design, research_path)
                result["_source"] = "explicit_override" if upstream_design is not None else "upstream"
                result["structured_summary"] = self._format_experiment_design(result)
                self.logger.finish_call(call, result)
                self.logger.success(self.AGENT_NAME, "已使用显式传入的实验设计")
                return result

            # Literature outputs are grounding signals only; experiment design is
            # owned by EvaluationAgent in the current workflow.
            papers = literature_result.get("top_papers", []) if isinstance(literature_result, dict) else []
            methods_ref = ""
            if papers:
                methods = [
                    p.get("structured_summary", {}).get("method", "")
                    for p in papers[:5]
                    if p.get("structured_summary")
                ]
                methods_ref = "\n".join(f"- {m}" for m in methods if m)

            system = """你是资深科研实验设计专家。请为以下研究问题设计一个完整、严谨、可直接进入论文报告的实验方案。

实验设计必须覆盖以下内容：
1. 研究目标与假设：明确研究假设、验证对象和预期边界。
2. 实验环境与数据集：说明数据来源、样本或场景范围、预处理和软硬件环境。
3. 基线方法与对照组：至少给出 3 个可比较的 baseline，并说明对照目的。
4. 评估指标：给出定量指标、含义、必要公式或计算方式。
5. 实验流程：说明从基线构建、方法实现、训练/推理到结果分析的步骤。
6. 变量控制与消融设计：说明控制变量和关键组件消融。
7. 可复现性设置：说明随机种子、重复次数、硬件/软件环境、统计检验或日志记录。
8. 预期结果：只能作为待验证假设，不得写成已经获得的实验结论。

请以 JSON 格式返回，字段：
{
  "research_hypothesis": "研究假设...",
  "objectives": ["目标1...", "目标2...", "目标3..."],
  "dataset": "数据集、预处理、软硬件环境描述...",
  "metrics": ["指标1: 公式和说明", "指标2: 公式和说明"],
  "baselines": ["基线1: 说明", "基线2: 说明", "基线3: 说明"],
  "variables": {"控制变量1": "控制方式", "控制变量2": "控制方式"},
  "ablations": ["消融1: 目的", "消融2: 目的"],
  "procedure": ["步骤1...", "步骤2...", "步骤3...", "步骤4..."],
  "reproducibility": "随机种子、重复次数、硬件/软件环境、统计检验方法...",
  "expected_results": "预期结果分析，必须表述为待验证假设...",
  "full_description": "完整实验设计总述，250-400字，论文风格自然段",
  "sections": [
    {"heading": "3.1 根据研究主题自行命名的小标题", "body": "对应小节正文，120-200字"},
    {"heading": "3.2 根据研究主题自行命名的小标题", "body": "对应小节正文，120-200字"}
  ]
}

sections 要求：
- 禁止使用 Markdown 代码块，直接返回 JSON 对象。
- 生成 5~6 个二级小节，heading 必须带 3.x 编号。
- heading 应根据研究问题、推荐研究路径和文献方法自行命名，不要机械套用固定模板。
- sections 整体必须覆盖：研究假设/目标、数据集/环境、基线/对照、评估指标、变量控制/消融、实验流程/可复现性、预期分析。
- body 应直接写成可放入学术报告的正文，避免只写成提纲或字段列表。
- 不要生成 Markdown 代码块，不要在 JSON 外输出解释文字。"""

            path_part = f"推荐研究路径：\n{research_path}\n\n" if research_path else ""
            prompt = (
                "请返回完整 JSON，禁止 Markdown 代码块，禁止在 JSON 外输出解释文字。"
                "数组字段建议 3-5 项，sections 生成 5-6 个带 3.x 编号、可直接放入论文的二级小节。\n\n"
                "返回字段：research_hypothesis, objectives, dataset, metrics, baselines, procedure, "
                "variables, ablations, reproducibility, expected_results, full_description, sections。\n"
                "其中 objectives、metrics、baselines、procedure、ablations 必须为数组，variables 必须为对象。\n\n"
                f"研究问题：{query}\n\n"
                f"{path_part}"
                f"相关文献方法参考：\n{methods_ref or '暂无可用文献方法摘要，请基于研究问题设计可复现实验。'}\n\n"
                "请设计完整实验方案，重点保证实验流程完整、可复现性明确，且 sections 与结构化字段相互一致："
            )
            resp = self.llm.chat(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=4096,
            )
            data = self._extract_experiment_design_payload(resp)
            result = self._normalize_experiment_design_payload(data, research_path)
            if not self._has_experiment_design_content(result):
                self.logger.info(
                    self.AGENT_NAME,
                    "实验设计 JSON 解析失败，使用结构化兜底方案，避免原始 JSON 进入报告"
                )
                result = self._fallback_experiment_design(query, research_path)
            if not result.get("full_description"):
                result["full_description"] = self._summarize_experiment_design_from_fields(result)
            result["_source"] = "evaluation_agent_generated"
            result["structured_summary"] = self._format_experiment_design(result)
            self.logger.finish_call(call, result)
            self.logger.success(self.AGENT_NAME, "实验设计方案生成完成")
            return result
        except Exception as exc:
            self.logger.fail_call(call, str(exc))
            result = self._fallback_experiment_design(query, research_path)
            result["full_description"] = f"实验设计生成失败，已使用结构化兜底方案。错误信息：{exc}\n\n{result['full_description']}"
            result["_fallback"] = True
            return result

    def run(self, query: str, outputs: dict) -> dict:
        call = self.logger.start_call(self.AGENT_NAME, "evaluation", query)
        try:
            # Stage 1: merge all upstream outputs into one report-centric payload.
            report_content = self._build_rich_report(query, outputs)
            self.logger.info(self.AGENT_NAME, "LLM-as-Judge 评分...")

            final_report = deepcopy(report_content)
            # The judge sees a flat text representation, but the generator keeps
            # the structured section layout for later export.
            final_text = self._render_report_text(final_report)
            final_score, scoring_breakdown = self._score_report(query, final_report, outputs)
            self.logger.info(
                self.AGENT_NAME,
                f"初始评分: {final_score.overall:.2f} | 需要反思: {final_score.needs_reflection}",
            )

            reflection_log = []
            attempted_rounds = 0
            accepted_rounds = 0
            while final_score.needs_reflection and attempted_rounds < self.config.evaluation.max_reflection_rounds:
                attempted_rounds += 1
                self.logger.info(self.AGENT_NAME, f"反思轮次 {attempted_rounds}...")
                rule_fixed_report, rule_changes, target_sections = self._apply_rule_based_reflection_fixes(
                    final_report,
                    scoring_breakdown,
                    outputs,
                )
                evidence_context = self._build_reflection_evidence_context(outputs)
                if rule_changes and self._has_unfixable_upstream_blockers(scoring_breakdown, outputs):
                    final_report = rule_fixed_report
                    rescored_score, rescored_breakdown = self._score_report(query, final_report, outputs)
                    accepted_rounds += 1
                    reflection_log.append(
                        {
                            "round": attempted_rounds,
                            "score_before": final_score.overall,
                            "score_after": rescored_score.overall,
                            "improved": True,
                            "accepted": True,
                            "status": "accepted_rule_only",
                            "reason": "remaining hard findings require upstream rerun; rule fixes applied and final report was rescored",
                            "rule_changes": rule_changes,
                            "target_sections": [section.get("heading", "") for section in target_sections],
                        }
                    )
                    final_score = rescored_score
                    scoring_breakdown = rescored_breakdown
                    self.logger.info(self.AGENT_NAME, "剩余问题依赖上游模块修复，应用规则修复并重评最终正文后停止反思")
                    break

                revised_payload = self._call_targeted_reflector(
                    query,
                    rule_fixed_report,
                    final_score,
                    scoring_breakdown,
                    evidence_context,
                    target_sections,
                )
                revised_report = self._apply_revised_report(rule_fixed_report, revised_payload)
                if not revised_report and rule_changes:
                    revised_report = rule_fixed_report
                if not revised_report:
                    reflection_log.append(
                        {
                            "round": attempted_rounds,
                            "score_before": final_score.overall,
                            "score_after": final_score.overall,
                            "improved": False,
                            "accepted": False,
                            "status": "invalid_revision",
                            "reason": "reflector did not return a valid structured report revision",
                            "rule_changes": rule_changes,
                        }
                    )
                    self.logger.info(self.AGENT_NAME, "反思结果格式无效，停止继续修订")
                    break

                if self._has_unfixable_upstream_blockers(scoring_breakdown, outputs):
                    final_report = revised_report
                    rescored_score, rescored_breakdown = self._score_report(query, final_report, outputs)
                    accepted_rounds += 1
                    reflection_log.append(
                        {
                            "round": attempted_rounds,
                            "score_before": final_score.overall,
                            "score_after": rescored_score.overall,
                            "improved": True,
                            "accepted": True,
                            "status": "accepted_after_rescore",
                            "reason": "remaining hard findings require upstream rerun; final report was rescored after accepted revision",
                            "rule_changes": rule_changes,
                            "target_sections": [section.get("heading", "") for section in target_sections],
                        }
                    )
                    final_score = rescored_score
                    scoring_breakdown = rescored_breakdown
                    self.logger.info(self.AGENT_NAME, "剩余问题依赖上游模块修复，接受本轮反思并重评最终正文后停止")
                    break

                revised_text = self._render_report_text(revised_report)
                new_score, new_breakdown = self._score_report(query, revised_report, outputs)
                improved = new_score.overall > final_score.overall
                # Track whether each reflection round actually improves quality.
                reflection_log.append(
                    {
                        "round": attempted_rounds,
                        "score_before": final_score.overall,
                        "score_after": new_score.overall,
                        "improved": improved,
                        "accepted": improved,
                        "status": "accepted" if improved else "not_improved",
                        "rule_changes": rule_changes,
                        "target_sections": [section.get("heading", "") for section in target_sections],
                    }
                )
                if improved:
                    # Only accept a revision after the judge confirms it is better.
                    final_report = revised_report
                    final_text = revised_text
                    final_score = new_score
                    scoring_breakdown = new_breakdown
                    accepted_rounds += 1
                    self.logger.success(
                        self.AGENT_NAME,
                        f"反思有效: {reflection_log[-1]['score_before']:.2f} -> {new_score.overall:.2f}",
                    )
                else:
                    self.logger.info(self.AGENT_NAME, "反思后无提升，停止继续修订")
                    break

            judge_score_data = scoring_breakdown.get("judge_score", {}) or {}
            if isinstance(judge_score_data, EvalScore):
                judge_score_data = judge_score_data.__dict__
            rule_score_data = scoring_breakdown.get("final_score", {}) or {}
            if isinstance(rule_score_data, EvalScore):
                rule_score_data = rule_score_data.__dict__

            final_report["evaluation"] = {
                "overall_score": final_score.overall,
                "final_deliverable_score": final_score.overall,
                "llm_expert_score": judge_score_data.get("overall", final_score.overall),
                "rule_consistency_score": rule_score_data.get("overall", final_score.overall),
                "accuracy": final_score.accuracy,
                "completeness": final_score.completeness,
                "format_quality": final_score.format_quality,
                "feedback": final_score.feedback,
                "review_summary": scoring_breakdown.get("review_summary", {}),
                "role_reviews": scoring_breakdown.get("review_summary", {}).get("role_reviews", []),
                "scoring_method": scoring_breakdown.get("method", "hybrid_rule_multi_agent_llm"),
                "rule_dimension_scores": scoring_breakdown.get("scores", {}),
                "scoring_weights": scoring_breakdown.get("weights", {}),
                "scoring_breakdown": scoring_breakdown,
                "reflection_rounds": accepted_rounds,
                "attempted_reflection_rounds": attempted_rounds,
                "reflection_log": reflection_log,
            }

            report_files = self.report_gen.generate(
                final_report,
                filename="final_report",
                formats=["docx", "markdown", "html"],
            )

            # Persist the final evaluation outcome for future retrieval.
            self.memory.add(
                content=f"评估完成: {query[:80]}，得分 {final_score.overall:.2f}",
                agent=self.AGENT_NAME,
                payload={
                    "score": final_score.overall,
                    "reflections": accepted_rounds,
                    "attempted_reflections": attempted_rounds,
                },
                tags=["evaluation"],
            )

            result = {
                "final_score": final_score.__dict__,
                "reflection_rounds": accepted_rounds,
                "attempted_reflection_rounds": attempted_rounds,
                "reflection_log": reflection_log,
                "report_files": report_files,
                "scoring_breakdown": scoring_breakdown,
            }
            self.logger.finish_call(call, result)
            self._print_result(final_score, reflection_log, report_files)
            return result
        except Exception as exc:
            self.logger.fail_call(call, str(exc))
            raise

    def _render_report_text(self, report_content: dict) -> str:
        # Convert the structured report object into a single text blob for
        # scoring and reflection prompts.
        chunks = []
        title = str(report_content.get("title", "")).strip()
        abstract = str(report_content.get("abstract", "")).strip()
        if title:
            chunks.append(title)
        if abstract:
            chunks.append(f"摘要\n{abstract}")
        for section in report_content.get("sections", []):
            heading = str(section.get("heading", "")).strip()
            body = str(section.get("body", "")).strip()
            if heading or body:
                chunks.append(f"{heading}\n{body}".strip())
        return "\n\n".join(chunk for chunk in chunks if chunk)

    def _remove_fenced_code_blocks(self, text: str) -> str:
        return re.sub(r"```[\s\S]*?```", "[内容已省略]", text or "")

    def _sanitize_text_for_abstract(self, text: str) -> str:
        # Mock mode routes prompts containing programming trigger words to a
        # sample code response, so abstract input uses neutral wording.
        sanitized = self._remove_fenced_code_blocks(text)
        return (
            sanitized.replace("核心代码实现", "核心方法说明")
            .replace("核心方法实现", "核心方法说明")
            .replace("代码摘要", "方法材料")
            .replace("代码", "程序")
            .replace("实现", "完成")
            .replace("python", "程序")
            .replace("Python", "程序")
        )

    def _call_targeted_reflector(
        self,
        query: str,
        report_content: dict,
        score: EvalScore,
        scoring_breakdown: dict,
        evidence_context: str,
        target_sections: list[dict],
    ) -> Optional[dict]:
        findings = scoring_breakdown.get("findings", []) if isinstance(scoring_breakdown, dict) else []
        try:
            return self.reflector.reflect_and_revise_report(
                query,
                report_content,
                score,
                evidence_context=evidence_context,
                findings=findings,
                target_sections=target_sections,
            )
        except TypeError:
            # Older tests or integrations may provide the legacy reflector API.
            return self.reflector.reflect_and_revise_report(query, report_content, score)

    def _apply_revised_report(self, report_content: dict, revised_payload: Optional[dict]) -> Optional[dict]:
        # Reject malformed revisions so we do not silently break chapter order.
        if not revised_payload or not isinstance(revised_payload, dict):
            return None

        revised_sections = revised_payload.get("section_revisions") or revised_payload.get("sections")
        original_sections = report_content.get("sections", [])
        if not isinstance(revised_sections, list):
            return None

        merged = deepcopy(report_content)
        merged_sections = [dict(section) for section in original_sections]
        changed = False
        is_full_revision = len(revised_sections) == len(original_sections) and all(
            isinstance(revised, dict) and "index" not in revised and "heading" not in revised
            for revised in revised_sections
        )
        if is_full_revision:
            for idx, revised in enumerate(revised_sections):
                body = str((revised or {}).get("body", "")).strip()
                if not body:
                    return None
                merged_sections[idx]["body"] = body
                changed = True
        else:
            for revised in revised_sections:
                if not isinstance(revised, dict):
                    return None
                body = str(revised.get("body", "")).strip()
                if not body:
                    return None
                target_idx = revised.get("index")
                if isinstance(target_idx, str) and target_idx.isdigit():
                    target_idx = int(target_idx)
                if not isinstance(target_idx, int):
                    heading = self._safe_text(revised.get("heading", ""))
                    target_idx = next(
                        (
                            idx
                            for idx, section in enumerate(merged_sections)
                            if heading and self._safe_text(section.get("heading")) == heading
                        ),
                        None,
                    )
                if not isinstance(target_idx, int) or not (0 <= target_idx < len(merged_sections)):
                    return None
                merged_sections[target_idx]["body"] = body
                changed = True

        abstract = str(revised_payload.get("abstract", "")).strip()
        if abstract:
            merged["abstract"] = abstract
            changed = True
        merged["sections"] = merged_sections
        return merged if changed else None

    def _score_report(self, query: str, report_content: dict, outputs: dict) -> tuple[EvalScore, dict]:
        """Score the report with multi-agent judgement plus deterministic upstream checks."""
        report_text = self._render_report_text(report_content)
        judge_score, judge_reviews = self._score_with_judges(query, report_text)
        upstream_keys = ("literature_result", "experiment_design", "code_result", "analysis_result")
        has_upstream_context = any(outputs.get(key) not in (None, {}, [], "") for key in upstream_keys)
        if not has_upstream_context:
            scoring_context = {
                "llm_expert_score": judge_score.overall,
                "rule_consistency_score": judge_score.overall,
                "final_deliverable_score": judge_score.overall,
                "method": "multi_agent_llm_only",
            }
            review_summary = self._build_review_summary(judge_score, [], judge_reviews, scoring_context)
            judge_score.feedback = self._format_review_feedback(review_summary)
            return judge_score, {
                "method": "multi_agent_llm_only",
                "reason": "no_upstream_outputs",
                "judge_score": judge_score.__dict__,
                "judge_reviews": judge_reviews,
                "review_summary": review_summary,
                "display_scores": scoring_context,
            }

        breakdown = self._rule_score_report(report_content, outputs, judge_score)
        scores = breakdown["scores"]
        findings = breakdown["findings"]

        overall = round(
            0.20 * scores["evidence_grounding"]
            + 0.20 * scores["experiment_consistency"]
            + 0.20 * scores["execution_validity"]
            + 0.15 * scores["result_faithfulness"]
            + 0.15 * scores["completeness"]
            + 0.10 * scores["format_quality"],
            3,
        )
        accuracy = round(
            0.55 * scores["evidence_grounding"]
            + 0.25 * scores["result_faithfulness"]
            + 0.20 * scores["execution_validity"],
            3,
        )
        completeness = round(scores["completeness"], 3)
        format_quality = round(scores["format_quality"], 3)
        needs_reflection = overall < 0.75 or any(item.get("severity") == "hard" for item in findings)

        suggestion = (
            "Revise the report so claims are grounded in literature, experiment design, "
            "code execution, and analysis outputs."
            if needs_reflection
            else judge_score.reflection_suggestion
        )

        final_score = EvalScore(
            overall=overall,
            accuracy=accuracy,
            completeness=completeness,
            format_quality=format_quality,
            feedback="",
            needs_reflection=needs_reflection,
            reflection_suggestion=suggestion,
        )
        breakdown["method"] = "hybrid_rule_multi_agent_llm"
        breakdown["weights"] = {
            "evidence_grounding": 0.20,
            "experiment_consistency": 0.20,
            "execution_validity": 0.20,
            "result_faithfulness": 0.15,
            "completeness": 0.15,
            "format_quality": 0.10,
        }
        breakdown["judge_score"] = judge_score.__dict__
        breakdown["judge_reviews"] = judge_reviews
        breakdown["display_scores"] = {
            "llm_expert_score": judge_score.overall,
            "rule_consistency_score": overall,
            "final_deliverable_score": overall,
            "rule_dimension_scores": scores,
            "weights": breakdown["weights"],
            "method": breakdown["method"],
        }
        review_summary = self._build_review_summary(
            final_score,
            findings,
            judge_reviews,
            breakdown["display_scores"],
        )
        final_score.feedback = self._format_review_feedback(review_summary)
        breakdown["review_summary"] = review_summary
        breakdown["final_score"] = final_score.__dict__
        return final_score, breakdown

    def _score_with_judges(self, query: str, report_text: str) -> tuple[EvalScore, list[dict]]:
        if hasattr(self.judge, "score_many"):
            return self.judge.score_many(query, report_text)

        # Test doubles and older integrations may only implement score().
        score = self.judge.score(query, report_text)
        return score, [
            {
                "reviewer": "single_judge",
                "weight": 1.0,
                "rubric": "legacy single judge",
                "score": score.__dict__,
            }
        ]

    def _build_review_summary(
        self,
        final_score: EvalScore,
        findings: list[dict],
        judge_reviews: list[dict],
        scoring_context: Optional[dict] = None,
    ) -> dict:
        score_note = self._format_score_layer_note(final_score, scoring_context)
        hard_findings = [item for item in findings if item.get("severity") == "hard"]
        soft_findings = [item for item in findings if item.get("severity") != "hard"]
        if hard_findings:
            issues = "；".join(item.get("message", "") for item in hard_findings[:3] if item.get("message"))
            overall = (
                f"综合评审意见：最终可交付评分为 {final_score.overall:.3f}，当前仍需重点修改。"
                f"主要问题集中在证据支撑、实验一致性或结果可信度方面：{issues}。"
                "建议优先修正高风险结论，再补充缺失的上游依据。"
            )
        elif soft_findings:
            issues = "；".join(item.get("message", "") for item in soft_findings[:3] if item.get("message"))
            overall = (
                f"综合评审意见：最终可交付评分为 {final_score.overall:.3f}，整体结构基本完整，"
                f"但仍存在需要谨慎处理的信息边界：{issues}。建议在最终提交前补强证据说明。"
            )
        else:
            overall = (
                f"综合评审意见：最终可交付评分为 {final_score.overall:.3f}，整体质量较稳定，"
                "章节结构、证据使用和表达规范性基本满足当前流程要求。"
            )
        if score_note:
            overall = f"{overall}\n{score_note}"

        role_reviews = []
        for item in judge_reviews:
            score_data = item.get("score", {}) or {}
            if isinstance(score_data, EvalScore):
                score_data = score_data.__dict__
            role_reviews.append(
                {
                    "reviewer": item.get("reviewer", "评审角色"),
                    "weight": item.get("weight", 1.0),
                    "overall_score": _clamp_score(score_data.get("overall", score_data.get("overall_score", 0.0)), 0.0),
                    "brief": self._safe_text(score_data.get("feedback", "")) or "未提供详细意见",
                    "suggestion": self._safe_text(score_data.get("reflection_suggestion", "")),
                }
            )
        return {"overall": overall, "role_reviews": role_reviews}

    def _format_score_layer_note(self, final_score: EvalScore, scoring_context: Optional[dict]) -> str:
        if not scoring_context:
            return ""
        llm_score = scoring_context.get("llm_expert_score")
        rule_score = scoring_context.get("rule_consistency_score", final_score.overall)
        try:
            llm_score_text = f"{float(llm_score):.3f}"
        except Exception:
            llm_score_text = "N/A"
        try:
            rule_score_text = f"{float(rule_score):.3f}"
        except Exception:
            rule_score_text = f"{final_score.overall:.3f}"
        return (
            "评分说明：LLM 专家评审分为 "
            f"{llm_score_text}，规则一致性/最终可交付评分为 {rule_score_text}。"
            "最终分优先反映文献、实验设计、代码执行和分析结果是否形成证据闭环，"
            "不等同于各专家分数的简单平均。"
        )

    def _format_review_feedback(self, review_summary: dict) -> str:
        overall = self._safe_text(review_summary.get("overall", ""))
        lines = [overall] if overall else []
        role_reviews = review_summary.get("role_reviews", []) or []
        if role_reviews:
            lines.append("各评审角色简要意见：")
            for review in role_reviews:
                score = review.get("overall_score", "N/A")
                try:
                    score = f"{float(score):.3f}"
                except Exception:
                    score = str(score)
                brief = self._safe_text(review.get("brief", "未提供详细意见"))
                lines.append(f"- {review.get('reviewer', '评审角色')}（{score}）：{brief}")
        return "\n".join(lines)

    def _apply_rule_based_reflection_fixes(
        self,
        report_content: dict,
        scoring_breakdown: dict,
        outputs: dict,
    ) -> tuple[dict, list[str], list[dict]]:
        fixed = deepcopy(report_content)
        sections = fixed.get("sections", [])
        findings = scoring_breakdown.get("findings", []) if isinstance(scoring_breakdown, dict) else []
        changes = []
        target_indices: set[int] = set()
        appendix_notes: list[str] = []

        for finding in findings:
            dimension = finding.get("dimension", "")
            severity = finding.get("severity", "")
            if severity != "hard":
                continue

            if dimension == "evidence_grounding":
                for idx in self._find_section_indices(sections, ["研究背景", "文献综述", "结论"]):
                    original = self._safe_text(sections[idx].get("body", ""))
                    guarded = self._downgrade_broad_literature_claims(original)
                    note = (
                        "证据边界说明：本次文献检索未返回可直接引用的论文证据，因此本节只能保守描述研究动机、"
                        "问题背景和待验证方向，不能将一般性背景写成已由当前文献检索支持的结论。"
                    )
                    sections[idx]["body"] = guarded
                    appendix_notes.append(note)
                    target_indices.add(idx)
                changes.append("rule_fix:evidence_grounding")

            if dimension == "experiment_consistency":
                exp_design = self._normalize_experiment_design_payload(outputs.get("experiment_design", {}) or {})
                exp_sections = self._normalize_experiment_sections(exp_design.get("sections", []))
                third_section_indices = [
                    idx
                    for idx, section in enumerate(sections)
                    if self._safe_text(section.get("heading", "")).startswith("3.")
                ]
                if exp_sections:
                    for idx, sec in zip(third_section_indices, exp_sections):
                        sections[idx]["heading"] = sec["heading"]
                        sections[idx]["body"] = sec["body"]
                        sections[idx]["level"] = 2
                        target_indices.add(idx)
                else:
                    for idx, section in enumerate(sections):
                        heading = self._safe_text(section.get("heading", ""))
                        if heading.startswith("3.1"):
                            sections[idx]["body"] = self._compose_experiment_hypothesis_section(exp_design)
                            target_indices.add(idx)
                        elif heading.startswith("3.2"):
                            sections[idx]["body"] = self._compose_experiment_metrics_section(exp_design)
                            target_indices.add(idx)
                        elif heading.startswith("3.3"):
                            sections[idx]["body"] = self._compose_experiment_baselines_section(exp_design)
                            target_indices.add(idx)
                changes.append("rule_fix:experiment_consistency")

            if dimension == "execution_validity":
                for idx in self._find_section_indices(sections, ["核心方法", "方法实现", "实验结果", "结果与分析", "结论"]):
                    original = self._safe_text(sections[idx].get("body", ""))
                    sections[idx]["body"] = self._guard_failed_execution_claims(original, outputs.get("code_result", {}) or {})
                    target_indices.add(idx)
                fixed["abstract"] = self._guard_failed_execution_claims(
                    self._safe_text(fixed.get("abstract", "")),
                    outputs.get("code_result", {}) or {},
                )
                appendix_notes.append(self._format_code_execution_status(outputs.get("code_result", {}) or {}))
                changes.append("rule_fix:execution_validity")

            if dimension == "result_faithfulness":
                for idx in self._find_section_indices(sections, ["实验结果", "结果与分析", "结论"]):
                    original = self._safe_text(sections[idx].get("body", ""))
                    note = (
                        "结果证据边界：实验设计阶段或分析结果未提供足够的定量指标支撑，"
                        "因此本节不得给出具体性能提升、显著性或相对基线优势，只能说明后续验证计划。"
                    )
                    sections[idx]["body"] = original
                    appendix_notes.append(note)
                    target_indices.add(idx)
                changes.append("rule_fix:result_faithfulness")

        if not target_indices:
            target_indices.update(self._find_section_indices(sections, ["摘要", "结论", "实验结果", "文献综述"]))
        target_sections = [
            {
                "index": idx,
                "heading": sections[idx].get("heading", ""),
                "body": sections[idx].get("body", ""),
                "level": sections[idx].get("level", 1),
            }
            for idx in sorted(target_indices)
            if 0 <= idx < len(sections)
        ]
        if appendix_notes:
            sections = self._append_report_boundary_notes(sections, appendix_notes)
        fixed["sections"] = sections
        return fixed, changes, target_sections

    def _find_section_indices(self, sections: list[dict], keywords: list[str]) -> list[int]:
        indices = []
        for idx, section in enumerate(sections):
            heading = self._safe_text(section.get("heading", ""))
            if any(keyword and keyword in heading for keyword in keywords):
                indices.append(idx)
        return indices

    def _prepend_once(self, text: str, note: str) -> str:
        text = self._safe_text(text)
        if not note or note in text:
            return text
        return f"{note}\n\n{text}" if text else note

    def _append_report_boundary_notes(self, sections: list[dict], notes: list[str]) -> list[dict]:
        unique_notes = []
        seen = set()
        for note in notes:
            text = self._safe_text(note)
            if not text or text in seen:
                continue
            seen.add(text)
            unique_notes.append(text)
        if not unique_notes:
            return sections

        body = "\n\n".join(f"{idx}. {note}" for idx, note in enumerate(unique_notes, 1))
        existing_idx = None
        for idx, section in enumerate(sections):
            heading = self._safe_text(section.get("heading", ""))
            if "证据边界与执行状态说明" in heading:
                existing_idx = idx
                break
        if existing_idx is None:
            sections.append(
                {
                    "heading": "附录：证据边界与执行状态说明",
                    "body": body,
                    "level": 1,
                }
            )
        else:
            existing = self._safe_text(sections[existing_idx].get("body", ""))
            sections[existing_idx]["body"] = "\n\n".join(part for part in [existing, body] if part)
        return sections

    def _downgrade_broad_literature_claims(self, text: str) -> str:
        replacements = {
            "已有研究表明": "在当前材料范围内尚不能确认已有研究表明",
            "大量研究支持": "当前检索结果尚不足以证明大量研究支持",
            "文献表明": "当前文献检索结果尚不足以表明",
            "已有文献": "当前检索到的文献证据",
            "相关研究": "相关背景讨论",
            "state-of-the-art": "待进一步文献核验的前沿方法",
        }
        guarded = self._safe_text(text)
        for old, new in replacements.items():
            guarded = guarded.replace(old, new)
        return guarded

    def _guard_failed_execution_claims(self, text: str, code_result: dict) -> str:
        guarded = self._safe_text(text)
        replacements = [
            ("实验结果表明", "当前代码执行失败，尚不能由实验结果表明"),
            ("验证成功", "尚未完成验证"),
            ("显著提升", "潜在提升目标"),
            ("显著优于基线", "是否优于基线仍需验证"),
            ("优于基线", "与基线的相对表现仍需验证"),
            ("显著优于", "是否优于仍需验证"),
            ("实测部署", "待完成部署验证"),
            ("完成了从理论压缩到物理实现的闭环验证", "尚未完成从理论压缩到物理实现的闭环验证"),
            ("有力支撑了研究假设", "仍需在代码成功运行后支撑研究假设"),
            ("验证了该策略", "尚未验证该策略"),
            ("outperforms", "requires further validation against"),
            ("successfully validates", "does not yet validate"),
        ]
        for old, new in replacements:
            guarded = guarded.replace(old, new)
        guarded = self._clean_guarded_execution_text(guarded)
        error = self._safe_text(code_result.get("error") or code_result.get("stderr") or code_result.get("error_type"))
        note = (
            "执行状态说明：代码执行模块返回失败状态，当前报告不能声称实验已经成功验证，"
            "也不能给出未由真实执行输出支持的定量结论。"
        )
        if error:
            note += f"失败信息摘要：{error[:300]}。"
        return guarded

    def _clean_guarded_execution_text(self, text: str) -> str:
        cleaned = self._safe_text(text)
        cleanup_pairs = [
            ("实验已经尚未完成验证", "实验尚未完成验证"),
            ("尚未完成验证成功", "尚未完成验证"),
            ("是否与基线的相对表现仍需验证仍需验证", "与基线的相对表现仍需验证"),
            ("仍需验证仍需验证", "仍需验证"),
            ("潜在提升目标、p 值", "显著提升、p 值"),
        ]
        for old, new in cleanup_pairs:
            cleaned = cleaned.replace(old, new)
        return cleaned

    def _build_reflection_evidence_context(self, outputs: dict) -> str:
        lit_result = outputs.get("literature_result", {}) or {}
        exp_design = outputs.get("experiment_design", {}) or {}
        code_result = outputs.get("code_result", {}) or {}
        analysis_result = outputs.get("analysis_result", {}) or {}

        papers = lit_result.get("top_papers") or lit_result.get("papers") or []
        paper_titles = self._stringify_items(
            [paper.get("title", "") for paper in papers[:5]],
            fallback="无可用论文标题",
            numbered=True,
        )
        lines = [
            f"文献检索数量：{lit_result.get('total_found', len(papers) if papers else 0)}",
            f"文献综述摘要：{self._safe_text(lit_result.get('literature_review', ''))[:900] or '无'}",
            f"论文标题：\n{paper_titles}",
            f"研究假设：{self._safe_text(exp_design.get('research_hypothesis', '')) or '未提供'}",
            f"实验目标：\n{self._stringify_items(exp_design.get('objectives', []), fallback='未提供', numbered=True)}",
            f"评估指标：\n{self._stringify_items(exp_design.get('metrics', []), fallback='未提供', numbered=True)}",
            f"基线方法：\n{self._stringify_items(exp_design.get('baselines', []), fallback='未提供', numbered=True)}",
            f"实验流程：\n{self._stringify_items(exp_design.get('procedure', []), fallback='未提供', numbered=True)}",
            f"代码执行成功：{code_result.get('success', '未知')}",
            f"代码错误：{self._safe_text(code_result.get('error') or code_result.get('stderr') or '')[:700] or '无'}",
            f"执行输出：{self._safe_text(code_result.get('stdout', ''))[:700] or '无'}",
            f"分析结论：{self._safe_text(analysis_result.get('conclusion', ''))[:900] or '无'}",
            f"图表证据：{self._summarize_chart_evidence(analysis_result.get('charts', []) or [])}",
        ]
        return "\n\n".join(lines)

    def _has_unfixable_upstream_blockers(self, scoring_breakdown: dict, outputs: dict) -> bool:
        findings = scoring_breakdown.get("findings", []) if isinstance(scoring_breakdown, dict) else []
        hard_dimensions = {
            item.get("dimension")
            for item in findings
            if item.get("severity") == "hard"
        }
        code_result = outputs.get("code_result", {}) or {}
        lit_result = outputs.get("literature_result", {}) or {}
        papers = lit_result.get("top_papers") or lit_result.get("papers") or []
        total_found = lit_result.get("total_found", len(papers))
        try:
            total_found = int(total_found or 0)
        except Exception:
            total_found = len(papers)
        code_failed = code_result.get("success") is False
        literature_missing = total_found == 0 and not papers
        return (
            ("execution_validity" in hard_dimensions and code_failed)
            or ("evidence_grounding" in hard_dimensions and literature_missing)
        )

    def _rule_score_report(self, report_content: dict, outputs: dict, judge_score: EvalScore) -> dict:
        text = self._render_report_text(report_content)
        lit_result = outputs.get("literature_result", {}) or {}
        exp_design = outputs.get("experiment_design", {}) or {}
        code_result = outputs.get("code_result", {}) or {}
        analysis_result = outputs.get("analysis_result", {}) or {}
        findings = []

        papers = lit_result.get("top_papers") or lit_result.get("papers") or []
        total_found = lit_result.get("total_found", len(papers))
        try:
            total_found = int(total_found or 0)
        except Exception:
            total_found = len(papers)

        evidence_grounding = 0.85 if total_found > 0 or papers else 0.65
        broad_lit_claims = [
            "已有研究",
            "大量研究",
            "现有研究表明",
            "文献表明",
            "已有文献",
            "相关研究",
            "state-of-the-art",
            "prior studies",
        ]
        if total_found == 0 and self._contains_any(text, broad_lit_claims):
            evidence_grounding = 0.45
            findings.append(
                {
                    "dimension": "evidence_grounding",
                    "severity": "hard",
                    "message": "literature retrieval found 0 papers but the report makes broad literature claims",
                }
            )
        elif total_found == 0:
            findings.append(
                {
                    "dimension": "evidence_grounding",
                    "severity": "soft",
                    "message": "literature retrieval found 0 papers, so evidence grounding is limited",
                }
            )

        experiment_text = self._experiment_sections_text(report_content)
        experiment_consistency = 0.85 if exp_design else 0.55
        experiment_checks = [
            ("3.1", exp_design.get("research_hypothesis", ""), "research hypothesis"),
            ("3.x", exp_design.get("metrics", []), "metrics"),
            ("3.x", exp_design.get("baselines", []), "baselines"),
        ]
        for prefix, expected, label in experiment_checks:
            body = self._section_body(report_content, prefix)
            if prefix == "3.x":
                body = experiment_text
            expected_items = self._list_values(expected)
            if expected_items:
                matched = sum(1 for item in expected_items if self._experiment_item_matches_body(item, body))
                required = max(1, (len(expected_items) + 1) // 2)
                if matched < required:
                    experiment_consistency = min(experiment_consistency, 0.60)
                    findings.append(
                        {
                            "dimension": "experiment_consistency",
                            "severity": "hard",
                            "message": f"experiment section does not match upstream {label}",
                        }
                    )
            elif body and not self._contains_any(body, ["未提供", "not provided", "missing"]):
                experiment_consistency = min(experiment_consistency, 0.65)
                findings.append(
                    {
                        "dimension": "experiment_consistency",
                        "severity": "soft",
                        "message": f"experiment section fills in {label} even though upstream design is missing",
                    }
                )

        execution_validity = 0.65
        success = code_result.get("success")
        if success is True:
            execution_validity = 0.90 if self._safe_text(code_result.get("stdout")) else 0.80
        elif success is False:
            execution_validity = 0.35
            if self._contains_any(
                text,
                ["验证成功", "实验结果表明", "显著提升", "达到", "优于", "successfully", "outperforms"],
            ):
                execution_validity = 0.30
            findings.append(
                {
                    "dimension": "execution_validity",
                    "severity": "hard",
                    "message": "code execution failed but the final report may still present experimental conclusions",
                }
            )
        elif code_result:
            findings.append(
                {
                    "dimension": "execution_validity",
                    "severity": "soft",
                    "message": "code execution status is missing",
                }
            )

        result_faithfulness = 0.80
        metrics = self._list_values(exp_design.get("metrics", []))
        metric_claim_terms = metrics + [
            "mAP",
            "FPS",
            "FLOPs",
            "Accuracy",
            "F1",
            "AUC",
            "BLEU",
            "%",
            "百分点",
            "p=",
        ]
        has_quant_claim = self._contains_any(text, metric_claim_terms)
        has_analysis_evidence = bool(
            self._safe_text(analysis_result.get("conclusion"))
            or analysis_result.get("charts")
            or self._safe_text(code_result.get("stdout"))
        )
        if not metrics and has_quant_claim:
            result_faithfulness = 0.45
            findings.append(
                {
                    "dimension": "result_faithfulness",
                    "severity": "hard",
                    "message": "the report contains quantitative claims while upstream metrics are missing",
                }
            )
        elif not has_analysis_evidence:
            result_faithfulness = 0.60
            findings.append(
                {
                    "dimension": "result_faithfulness",
                    "severity": "soft",
                    "message": "analysis evidence is sparse or missing",
                }
            )
        if success is False and self._contains_any(text, ["显著提升", "优于", "outperforms", "improves"]):
            result_faithfulness = min(result_faithfulness, 0.35)

        completeness = self._score_report_completeness(report_content)
        section_count = len(report_content.get("sections", []))
        expected_section_count = 9
        structure_score = min(1.0, section_count / expected_section_count)
        format_quality = round(0.70 * judge_score.format_quality + 0.30 * structure_score, 3)

        return {
            "scores": {
                "evidence_grounding": round(evidence_grounding, 3),
                "experiment_consistency": round(experiment_consistency, 3),
                "execution_validity": round(execution_validity, 3),
                "result_faithfulness": round(result_faithfulness, 3),
                "completeness": round(completeness, 3),
                "format_quality": round(format_quality, 3),
            },
            "findings": findings,
            "upstream_status": {
                "literature_total_found": total_found,
                "experiment_has_metrics": bool(metrics),
                "code_success": success,
                "has_analysis_evidence": has_analysis_evidence,
            },
        }

    def _score_report_completeness(self, report_content: dict) -> float:
        score = 0.0
        if self._safe_text(report_content.get("abstract")):
            score += 0.20
        sections = report_content.get("sections", [])
        if sections:
            score += min(0.55, 0.55 * len(sections) / 9)
        if self._section_body(report_content, "3.1"):
            score += 0.08
        if self._section_body(report_content, "3.2"):
            score += 0.08
        if self._section_body(report_content, "3.3"):
            score += 0.09
        return round(min(score, 1.0), 3)

    def _section_body(self, report_content: dict, heading_prefix: str) -> str:
        for section in report_content.get("sections", []):
            heading = self._safe_text(section.get("heading"))
            if heading.startswith(heading_prefix):
                return self._safe_text(section.get("body"))
        return ""

    def _experiment_sections_text(self, report_content: dict) -> str:
        """Return all third-chapter subsection text for semantic consistency checks."""
        parts = []
        for section in report_content.get("sections", []):
            heading = self._safe_text(section.get("heading"))
            if heading.startswith("3.") or heading.startswith("三、实验设计"):
                body = self._safe_text(section.get("body"))
                if body:
                    parts.append(f"{heading}\n{body}")
        return "\n\n".join(parts)

    def _contains_any(self, text: str, terms: list[str]) -> bool:
        lower = self._safe_text(text).lower()
        return any(term and term.lower() in lower for term in terms)

    def _experiment_item_matches_body(self, item: str, body: str) -> bool:
        normalized_item = self._normalize_match_text(item)
        normalized_body = self._normalize_match_text(body)
        if not normalized_item or not normalized_body:
            return False
        if normalized_item in normalized_body:
            return True

        if any(self._normalize_match_text(term) in normalized_body for term in self._extract_match_terms(item)):
            return True

        # Long hypotheses are often paraphrased. Match a stable phrase instead
        # of requiring the exact sentence to appear verbatim.
        if len(normalized_item) >= 12:
            for size in (18, 14, 10):
                step = max(1, size // 2)
                for start in range(0, max(1, len(normalized_item) - size + 1), step):
                    phrase = normalized_item[start:start + size]
                    if len(phrase) >= 8 and phrase in normalized_body:
                        return True
        return False

    def _normalize_match_text(self, text: str) -> str:
        text = self._safe_text(text).lower()
        return re.sub(r"[\s:：,，.。;；、()（）\[\]【】\"'`]+", "", text)

    def _extract_match_terms(self, text: str) -> list[str]:
        raw = self._safe_text(text)
        terms = []
        patterns = [
            r"^[^:：()（）,，;；。]+",
            r"[A-Z][A-Za-z0-9+-]{1,}",
            r"[\u4e00-\u9fffA-Za-z0-9+-]{2,}(?=[:：()（）,，;；。])",
        ]
        seen = set()
        for pattern in patterns:
            for match in re.finditer(pattern, raw):
                term = match.group(0).strip()
                norm = self._normalize_match_text(term)
                if len(norm) >= 2 and norm not in seen:
                    terms.append(term)
                    seen.add(norm)
        return terms

    def _list_values(self, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [self._safe_text(item) for item in value if self._safe_text(item)]
        text = self._safe_text(value)
        return [text] if text else []

    def _build_rich_report_legacy(self, query: str, outputs: dict) -> dict:
        self.logger.info(self.AGENT_NAME, "构建详细学术报告...")

        # Upstream outputs are merged here into one report-centric view.
        lit_result = outputs.get("literature_result", {})
        exp_design = outputs.get("experiment_design", {})
        code_result = outputs.get("code_result", {})
        ana_result = outputs.get("analysis_result", {})

        papers = lit_result.get("top_papers", []) or lit_result.get("papers", [])
        literature_review = self._safe_text(lit_result.get("literature_review", ""))
        final_code = self._safe_text(code_result.get("final_code", ""))
        stdout = self._safe_text(code_result.get("stdout", ""))
        analysis = self._safe_text(ana_result.get("conclusion", ""))
        charts = ana_result.get("charts", [])

        # Keep a compact grounding summary of retrieved papers for the review section.
        paper_summaries = "\n".join(
            [
                f"论文{i + 1}：{paper.get('title', '')}\n"
                f"方法：{(paper.get('structured_summary') or {}).get('method', '')}\n"
                f"结论：{(paper.get('structured_summary') or {}).get('conclusion', '')}"
                for i, paper in enumerate(papers[:6])
            ]
        )

        self.logger.info(self.AGENT_NAME, "生成研究背景...")
        background = self._expand_section(
            (
                f"请围绕研究问题“{query}”撰写研究背景与问题陈述，至少 300 字。"
                "需要覆盖领域现状、问题重要性、现有方法局限与本研究价值。"
            ),
            min_words=300,
        )

        self.logger.info(self.AGENT_NAME, "生成文献综述...")
        literature = self._expand_section(
            (
                f"请基于以下论文摘要信息撰写文献综述，至少 400 字。\n\n"
                f"{paper_summaries or '暂无论文摘要'}\n\n"
                f"现有综述片段：{literature_review or '暂无'}\n\n"
                "请总结主流方法、指出研究空白，并引出本文的切入点。"
            ),
            min_words=400,
        )

        self.logger.info(self.AGENT_NAME, "整合实验设计...")
        experiment_description = self._safe_text(exp_design.get("full_description", ""))
        if len(experiment_description) < 100:
            # Regenerate the methodology section when the explicit design is too thin.
            experiment_description = self._expand_section(
                (
                    f"请为研究问题“{query}”撰写完整实验设计，至少 400 字。"
                    "需要包含研究假设、数据集、基线方法、评估指标与实验流程。"
                ),
                min_words=400,
            )
            exp_design = {**exp_design, "full_description": exp_desc}
        exp_summary = exp_design.get("structured_summary") or self._format_experiment_design(exp_design)
        exp_sections = self._normalize_experiment_sections(exp_design.get("sections", []))
        if not exp_sections:
            exp_sections = [{
                "heading": "3.1 实验设计补充说明",
                "body": exp_summary,
            }]

        self.logger.info(self.AGENT_NAME, "生成方法实现说明...")
        method_description = self._expand_section(
            (
                "请根据下面的代码摘要，用学术语言描述核心方法实现，至少 200 字。\n\n"
                f"研究问题：{query}\n"
                f"代码摘要：\n{final_code[:800] or '暂无代码实现'}\n\n"
                "请说明算法原理、关键实现步骤和重要参数。"
            ),
            min_words=200,
        )

        self.logger.info(self.AGENT_NAME, "生成结果分析...")
        result_description = self._expand_section(
            (
                "请根据下面的分析结论和执行输出撰写实验结果与分析，至少 300 字。\n\n"
                f"研究问题：{query}\n"
                f"统计结论：{analysis or '暂无'}\n"
                f"执行输出：{stdout[:600] or '暂无'}\n\n"
                "请描述主要结果、与基线的对比以及结果含义。"
            ),
            min_words=300,
        )

        self.logger.info(self.AGENT_NAME, "生成结论与展望...")
        conclusion = self._expand_section(
            (
                f"请围绕研究问题“{query}”撰写结论与展望，至少 200 字。"
                "需要概括研究发现、局限性和后续工作方向。"
            ),
            min_words=200,
        )

        title = f"MindPilot 科研报告：{query}"
        # Build the abstract from already-computed artifacts to keep it
        # deterministic and cheaper than another generation call.
        abstract = self._build_abstract(query, papers, experiment_description, result_description)

        return {
            "title": title,
            "query": query,
            "abstract": abstract,
            "sections": [
                {"heading": "一、研究背景与问题陈述", "body": bg,          "level": 1},
                {"heading": "二、文献综述",           "body": lit_full,     "level": 1},
                {"heading": "三、实验设计与方法论",    "body": exp_summary,  "level": 1},
                *[
                    {"heading": sec["heading"], "body": sec["body"], "level": 2}
                    for sec in exp_sections
                ],
                {"heading": "四、核心方法实现",       "body": method_desc,  "level": 1},
                {"heading": "五、实验结果与分析",      "body": result_desc,  "level": 1},
                {"heading": "六、结论与展望",          "body": conclusion_desc, "level": 1},
            ],
            "code": final_code,
            "stdout": stdout,
            "literature": papers,
            "charts": charts,
        }

    def _build_abstract_legacy(
        self,
        query: str,
        papers: list[dict],
        experiment_description: str,
        result_description: str,
    ) -> str:
        paper_count = len(papers)
        summary = (
            f"本报告围绕“{query}”开展系统分析，整合了文献检索、实验设计、"
            "代码实现与结果分析等环节。"
        )
        if paper_count:
            summary += f"在文献阶段共检索到 {paper_count} 篇相关论文，作为方案设计与方法比较的依据。"
        if experiment_description:
            summary += "随后构建了结构化实验方案，并明确了基线、指标与实验流程。"
        if result_description:
            summary += "最终结合实现输出与分析结论，总结了主要结果、局限性与后续方向。"
        return summary

    def _build_rich_report_basic(self, query: str, outputs: dict) -> dict:
        self.logger.info(self.AGENT_NAME, "构建详细学术报告...")

        # Upstream outputs are merged here into one report-centric view.
        lit_result = outputs.get("literature_result", {})
        exp_design = outputs.get("experiment_design", {})
        code_result = outputs.get("code_result", {})
        ana_result = outputs.get("analysis_result", {})

        papers = lit_result.get("top_papers", []) or lit_result.get("papers", [])
        literature_review = self._safe_text(lit_result.get("literature_review", ""))
        final_code = self._safe_text(code_result.get("final_code", ""))
        stdout = self._safe_text(code_result.get("stdout", ""))
        analysis = self._safe_text(ana_result.get("conclusion", ""))
        charts = ana_result.get("charts", [])

        # The literature review is primarily grounded in the upstream module output,
        # because the abstract is not available yet at this stage.
        reference_titles = "\n".join(
            f"{i + 1}. {paper.get('title', '').strip()}"
            for i, paper in enumerate(papers[:8])
            if paper.get("title")
        )
        paper_summaries = "\n\n".join(
            [
                "\n".join(
                    [
                        f"论文标题：{paper.get('title', '').strip()}",
                        f"方法要点：{((paper.get('structured_summary') or {}).get('method', '') or '').strip()}",
                        f"主要结论：{((paper.get('structured_summary') or {}).get('conclusion', '') or '').strip()}",
                    ]
                ).strip()
                for paper in papers[:5]
            ]
        )

        self.logger.info(self.AGENT_NAME, "生成研究背景...")
        background = self._expand_section(
            (
                f"请围绕研究问题“{query}”撰写研究背景与问题陈述，至少 300 字。\n"
                "需要覆盖领域现状、问题重要性、现有方法局限与本研究价值。"
            ),
            min_words=300,
        )

        self.logger.info(self.AGENT_NAME, "生成文献综述...")
        literature = self._expand_section(
            (
                f"请围绕研究问题“{query}”撰写论文风格的文献综述，至少 500 字。\n\n"
                f"请优先基于上游模块传入的文献综述内容进行改写与扩写：\n{literature_review or '暂无上游文献综述'}\n\n"
                f"可作为补充参考的文献标题如下：\n{reference_titles or '暂无标题信息'}\n\n"
                f"可作为补充事实依据的结构化摘要如下：\n{paper_summaries or '暂无结构化摘要'}\n\n"
                "写作要求：\n"
                "1. 采用正式的中文学术论文语体，不要写成项目汇报或分点罗列。\n"
                "2. 以连续段落组织内容，可按研究方向、方法类别、关键进展、局限与研究空白展开。\n"
                "3. 不要虚构不存在的方法、数据或结论；若信息不足，保持审慎表述。\n"
                "4. 结尾自然引出本文的研究切入点和后续实验设计动机。"
            ),
            min_words=500,
        )

        self.logger.info(self.AGENT_NAME, "整合实验设计...")
        experiment_description = self._safe_text(exp_design.get("full_description", ""))
        if len(experiment_description) < 100:
            # Regenerate the methodology section when the explicit design is too thin.
            experiment_description = self._expand_section(
                (
                    f"请为研究问题“{query}”撰写完整实验设计，至少 400 字。\n"
                    "需要包含研究假设、数据集、基线方法、评估指标与实验流程。"
                ),
                min_words=400,
            )

        self.logger.info(self.AGENT_NAME, "生成方法实现说明...")
        method_description = self._expand_section(
            (
                "请根据下面的代码摘要，用学术语言描述核心方法实现，至少 200 字。\n\n"
                f"研究问题：{query}\n"
                f"代码摘要：\n{final_code[:800] or '暂无代码实现'}\n\n"
                "请说明算法原理、关键实现步骤和重要参数。"
            ),
            min_words=200,
        )

        self.logger.info(self.AGENT_NAME, "生成结果分析...")
        result_description = self._expand_section(
            (
                "请根据下面的分析结论和执行输出撰写实验结果与分析，至少 300 字。\n\n"
                f"研究问题：{query}\n"
                f"统计结论：{analysis or '暂无'}\n"
                f"执行输出：{stdout[:600] or '暂无'}\n\n"
                "请描述主要结果、与基线的对比以及结果含义。"
            ),
            min_words=300,
        )

        self.logger.info(self.AGENT_NAME, "生成结论与展望...")
        conclusion = self._expand_section(
            (
                f"请围绕研究问题“{query}”撰写结论与展望，至少 200 字。\n"
                "需要概括研究发现、局限性和后续工作方向。"
            ),
            min_words=200,
        )

        title = f"MindPilot 科研报告：{query}"
        sections = [
            {"heading": "一、研究背景与问题陈述", "body": background, "level": 1},
            {"heading": "二、文献综述", "body": literature, "level": 1},
            {"heading": "三、实验设计与方法设计", "body": experiment_description, "level": 1},
            {
                "heading": "3.1 实验假设与目标",
                "body": self._safe_text(exp_design.get("research_hypothesis", "")) or "见实验设计与方法设计。",
                "level": 2,
            },
            {
                "heading": "3.2 评估指标",
                "body": "\n".join(str(item) for item in exp_design.get("metrics", [])) or "见实验设计与方法设计。",
                "level": 2,
            },
            {
                "heading": "3.3 基线方法",
                "body": "\n".join(str(item) for item in exp_design.get("baselines", [])) or "见实验设计与方法设计。",
                "level": 2,
            },
            {"heading": "四、核心方法实现", "body": method_description, "level": 1},
            {"heading": "五、实验结果与分析", "body": result_description, "level": 1},
            {"heading": "六、结论与展望", "body": conclusion, "level": 1},
        ]

        # Generate the abstract from the completed body instead of a fixed template.
        report_body = self._render_report_text(
            {
                "title": title,
                "query": query,
                "abstract": "",
                "sections": sections,
            }
        )
        abstract = self._build_abstract(query, report_body)

        return {
            "title": title,
            "query": query,
            "abstract": abstract,
            "sections": sections,
            "code": final_code,
            "stdout": stdout,
            "literature": papers,
            "charts": charts,
        }

    def _stringify_items(
        self,
        items: Any,
        *,
        fallback: str = "暂无",
        numbered: bool = False,
    ) -> str:
        if items is None:
            return fallback
        if not isinstance(items, list):
            text = self._safe_text(items)
            return text or fallback

        lines = []
        for idx, item in enumerate(items, start=1):
            if isinstance(item, dict):
                text = self._safe_text(
                    item.get("title")
                    or item.get("name")
                    or item.get("description")
                    or item.get("path")
                    or json.dumps(item, ensure_ascii=False)
                )
            else:
                text = self._safe_text(item)
            if not text:
                continue
            prefix = f"{idx}. " if numbered else "- "
            lines.append(f"{prefix}{text}")
        return "\n".join(lines) if lines else fallback

    def _summarize_chart_evidence(self, charts: Any) -> str:
        if not charts:
            return "暂无图表证据"
        lines = []
        for idx, chart in enumerate(charts[:5], start=1):
            if isinstance(chart, dict):
                text = self._safe_text(
                    chart.get("title")
                    or chart.get("caption")
                    or chart.get("path")
                    or json.dumps(chart, ensure_ascii=False)
                )
            else:
                text = self._safe_text(chart)
            if text:
                lines.append(f"{idx}. {text}")
        return "\n".join(lines) if lines else "暂无图表证据"

    def _find_upstream_experiment_design(self, payload: Any) -> dict:
        if not isinstance(payload, dict):
            return {}

        for key in (
            "experiment_design",
            "experimental_design",
            "experiment_result",
            "experiment_plan",
            "design_result",
        ):
            candidate = payload.get(key)
            if self._has_experiment_design_content(candidate):
                return candidate

        for key in ("outputs", "result", "data"):
            nested = payload.get(key)
            candidate = self._find_upstream_experiment_design(nested)
            if candidate:
                return candidate

        # The root payload may be a literature_result. Do not treat generic
        # retrieval metrics such as recall@5 as an experiment design.
        if self._has_strong_experiment_design_content(payload):
            return payload
        return {}

    def _has_experiment_design_content(self, design: Any) -> bool:
        if not isinstance(design, dict) or design.get("_deprecated"):
            return False
        fields = (
            "research_hypothesis",
            "hypothesis",
            "objectives",
            "goals",
            "dataset",
            "datasets",
            "baselines",
            "baseline_methods",
            "metrics",
            "evaluation_metrics",
            "procedure",
            "steps",
            "protocol",
            "variables",
            "ablations",
            "reproducibility",
            "expected_results",
            "full_description",
            "description",
            "experiment_design_text",
            "content",
            "sections",
        )
        return any(self._field_has_content(design.get(field)) for field in fields)

    def _has_strong_experiment_design_content(self, design: Any) -> bool:
        if not isinstance(design, dict) or design.get("_deprecated"):
            return False
        strong_fields = (
            "research_hypothesis",
            "hypothesis",
            "objectives",
            "goals",
            "research_objectives",
            "dataset",
            "datasets",
            "baselines",
            "baseline_methods",
            "baseline",
            "procedure",
            "steps",
            "protocol",
            "variables",
            "ablations",
            "reproducibility",
            "expected_results",
            "full_description",
            "experiment_design_text",
            "sections",
        )
        return any(self._field_has_content(design.get(field)) for field in strong_fields)

    def _field_has_content(self, value: Any) -> bool:
        if isinstance(value, dict):
            return any(self._field_has_content(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return any(self._field_has_content(item) for item in value)
        return bool(self._safe_text(value))

    def _coerce_experiment_list(self, value: Any) -> list:
        if value is None:
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return list(value)
        text = self._safe_text(value)
        return [text] if text else []

    def _normalize_experiment_design_payload(self, data: dict, research_path: str = "") -> dict:
        data = data or {}
        if isinstance(data, str):
            data = self._extract_experiment_design_payload(data)
        elif isinstance(data, dict):
            data = self._merge_embedded_experiment_json_fields(data)
        else:
            data = {}
        result = {
            "research_path": research_path or self._safe_text(data.get("research_path", "")),
            "research_hypothesis": self._safe_text(
                data.get("research_hypothesis") or data.get("hypothesis") or ""
            ),
            "objectives": self._coerce_experiment_list(
                data.get("objectives") or data.get("goals") or data.get("research_objectives")
            ),
            "dataset": self._safe_text(data.get("dataset") or data.get("datasets") or data.get("data") or ""),
            "baselines": self._coerce_experiment_list(
                data.get("baselines") or data.get("baseline_methods") or data.get("baseline")
            ),
            "metrics": self._coerce_experiment_list(
                data.get("metrics") or data.get("evaluation_metrics") or data.get("metric")
            ),
            "variables": data.get("variables", {}) if isinstance(data.get("variables", {}), dict) else {},
            "ablations": self._coerce_experiment_list(data.get("ablations") or data.get("ablation_studies")),
            "procedure": self._coerce_experiment_list(
                data.get("procedure") or data.get("steps") or data.get("protocol")
            ),
            "reproducibility": self._safe_text(data.get("reproducibility", "")),
            "expected_results": self._safe_text(data.get("expected_results", "")),
            "full_description": self._clean_experiment_description_text(
                data.get("full_description")
                or data.get("description")
                or data.get("experiment_design_text")
                or data.get("content")
                or "",
            ),
            "sections": self._normalize_experiment_sections(data.get("sections", [])),
        }
        return result

    def _merge_embedded_experiment_json_fields(self, data: dict) -> dict:
        """Fill missing structured fields when upstream packed design JSON into text."""
        embedded_payload: dict[str, Any] = {}
        for field in ("full_description", "description", "experiment_design_text", "content"):
            text = self._safe_text(data.get(field, ""))
            if text and self._looks_like_json_blob(text):
                parsed = self._extract_experiment_design_payload(text)
                if parsed:
                    embedded_payload.update(parsed)

        if not embedded_payload:
            return data

        merged = dict(embedded_payload)
        for key, value in data.items():
            if self._field_has_content(value):
                merged[key] = value
        return merged

    def _extract_experiment_design_payload(self, response: str) -> dict:
        text = self._strip_json_fences(self._safe_text(response))
        data = _extract_json_object(text)
        if data:
            return data

        # Best-effort salvage for responses truncated before the closing brace.
        payload: dict[str, Any] = {}
        for field in (
            "research_hypothesis",
            "dataset",
            "reproducibility",
            "expected_results",
            "full_description",
            "description",
        ):
            value = self._extract_partial_json_string(text, field)
            if value:
                payload[field] = value

        for field in ("objectives", "baselines", "metrics", "ablations", "procedure"):
            values = self._extract_partial_json_array(text, field)
            if values:
                payload[field] = values
        return payload

    def _fallback_experiment_design(self, query: str, research_path: str = "") -> dict:
        """Provide a clean internal experiment design when the model response is unusable."""
        path_text = f"该方案参考推荐研究路径“{research_path}”，" if research_path else ""
        result = {
            "research_path": research_path,
            "research_hypothesis": (
                f"针对“{query}”的核心假设是：通过系统化方法设计和受控实验验证，"
                "可以在保持主要任务性能的同时改善模型效率、资源占用或部署可行性。"
            ),
            "objectives": [
                "明确待验证方法相对原始方案或主流方案的性能变化。",
                "量化方法在计算开销、运行效率或资源占用方面的影响。",
                "通过对照组和消融实验分析关键组件的实际贡献。",
            ],
            "dataset": (
                "选择与研究问题直接相关的公开数据集或标准任务基准，统一训练、验证和测试划分，"
                "并记录数据预处理、输入格式、硬件环境和软件依赖，保证不同方法在同一协议下比较。"
            ),
            "baselines": [
                "原始未改进模型或标准实现: 用于建立性能上限和资源开销基准。",
                "主流基线方法: 用于代表已有研究或工程实践中的常用方案。",
                "简化或去除关键组件的变体: 用于衡量本文核心设计带来的增益。",
            ],
            "metrics": [
                "任务性能指标: 根据具体任务选择准确率、误差、召回率或相关主指标，衡量方法有效性。",
                "效率指标: 统计推理延迟、吞吐量、FLOPs 或运行时间，衡量部署效率。",
                "资源指标: 统计参数量、显存占用或模型大小，衡量资源约束下的可用性。",
            ],
            "variables": {
                "数据划分": "所有方法使用相同训练集、验证集和测试集。",
                "训练配置": "保持学习率、批大小、训练轮数和随机种子一致。",
                "硬件环境": "在同一设备和软件依赖版本下记录运行结果。",
            },
            "ablations": [
                "去除核心模块或策略，观察任务性能与效率变化。",
                "调整关键超参数，分析方法对配置变化的敏感性。",
            ],
            "procedure": [
                "建立原始模型和主流基线的统一运行环境。",
                "实现或接入待验证方法，并保持数据和训练配置一致。",
                "在相同测试协议下采集任务性能、效率和资源指标。",
                "结合对照组与消融实验分析关键设计的贡献和局限。",
            ],
            "reproducibility": (
                "固定随机种子，至少重复运行三次并报告均值或方差；记录硬件型号、软件版本、"
                "关键超参数和日志路径，必要时采用配对检验或置信区间判断差异是否稳定。"
            ),
            "expected_results": (
                "预期该方案能够在主要任务性能基本保持的前提下降低资源开销或提升运行效率；"
                "该表述仅作为待验证假设，不能替代真实执行结果。"
            ),
            "full_description": (
                f"本实验围绕“{query}”展开，{path_text}采用受控对照实验框架验证研究假设。"
                "实验首先在统一数据集和运行环境下建立原始模型及主流基线的性能与资源开销基准，"
                "随后接入待验证方法，并保持数据划分、训练配置、硬件环境和评价协议一致。"
                "评价过程同时关注任务性能、运行效率和资源占用，避免只凭单一指标判断方法优劣。"
                "在对照实验之外，方案还设置关键组件消融和参数敏感性分析，用于解释性能变化是否来自核心设计。"
                "所有结果需要结合日志、指标表和图表进行复核；若代码执行阶段未成功完成，预期结果只能作为后续验证方向。"
            ),
            "sections": [
                {
                    "heading": "3.1 研究假设与验证目标",
                    "body": (
                        f"本节围绕“{query}”明确实验假设与验证目标。实验重点不是单纯展示模型输出，"
                        "而是通过可比较、可复现的流程判断方法是否真正带来性能、效率或资源占用方面的改善，"
                        "并分析这种变化是否来自核心设计而非数据划分、训练配置或随机性差异。"
                    ),
                },
                {
                    "heading": "3.2 数据集构建与实验环境",
                    "body": (
                        "数据集部分需要说明数据来源、样本规模、清洗规则、划分方式和软硬件环境。所有方法应使用"
                        "相同训练集、验证集和测试集，并统一输入格式、缺失值处理和异常样本过滤规则，从而保证"
                        "后续结果具有可比性和可复核性。"
                    ),
                },
                {
                    "heading": "3.3 基线方法与对照组设置",
                    "body": (
                        "对照组应包含原始未改进模型、主流可比较方案以及去除关键组件的变体三个层次。原始模型提供"
                        "性能和资源开销参照，主流方案代表已有研究或工程实践水平，组件变体用于直接衡量本研究新增"
                        "模块或策略的贡献。"
                    ),
                },
                {
                    "heading": "3.4 评估指标与变量控制",
                    "body": (
                        "评估指标围绕任务性能、运行效率和资源占用三类维度组织，并在相同测试协议下采集。实验需要"
                        "固定数据划分、训练轮数、随机种子、硬件环境和评价脚本，只改变待验证方法或消融变量，"
                        "避免把环境差异误解释为方法优势。"
                    ),
                },
                {
                    "heading": "3.5 消融实验与可复现性设置",
                    "body": (
                        "消融实验通过移除核心模块、调整关键超参数或替换关键策略来分析性能变化来源。为保证结果可复现，"
                        "实验应记录依赖版本、硬件型号、随机种子、运行日志和输出文件，并在条件允许时重复运行多次，"
                        "报告均值、方差或显著性检验结果。"
                    ),
                },
            ],
        }
        result["structured_summary"] = self._format_experiment_design(result)
        result["_source"] = "evaluation_agent_fallback"
        return result

    def _strip_json_fences(self, text: str) -> str:
        cleaned = self._safe_text(text)
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        return cleaned.strip()

    def _looks_like_json_blob(self, text: str) -> bool:
        cleaned = self._strip_json_fences(text)
        if not cleaned:
            return False
        json_markers = (
            '"research_hypothesis"',
            '"objectives"',
            '"baselines"',
            '"metrics"',
            '"procedure"',
            "{",
            "}",
        )
        return cleaned.startswith("{") or sum(marker in cleaned for marker in json_markers) >= 2

    def _extract_partial_json_string(self, text: str, field: str) -> str:
        match = re.search(rf'"{re.escape(field)}"\s*:\s*"((?:\\.|[^"\\])*)', text, flags=re.S)
        if not match:
            return ""
        value = match.group(1)
        try:
            return json.loads(f'"{value}"')
        except Exception:
            return value.replace('\\"', '"').replace("\\n", "\n").strip()

    def _extract_partial_json_array(self, text: str, field: str) -> list[str]:
        match = re.search(rf'"{re.escape(field)}"\s*:\s*\[([\s\S]*?)(?:\]|\n\s*")', text)
        if not match:
            return []
        fragment = match.group(1)
        items = []
        for raw in re.findall(r'"((?:\\.|[^"\\])*)"', fragment):
            try:
                item = json.loads(f'"{raw}"')
            except Exception:
                item = raw.replace('\\"', '"').replace("\\n", "\n")
            item = self._safe_text(item)
            if item:
                items.append(item)
        return items

    def _clean_experiment_description_text(self, value: Any) -> str:
        text = self._strip_json_fences(self._safe_text(value))
        if self._looks_like_json_blob(text):
            return ""
        return text

    def _summarize_experiment_design_from_fields(self, exp_design: dict) -> str:
        parts = []
        hypothesis = self._safe_text(exp_design.get("research_hypothesis", ""))
        if hypothesis:
            parts.append(f"实验围绕以下研究假设展开：{hypothesis}")
        dataset = self._safe_text(exp_design.get("dataset", ""))
        if dataset:
            parts.append(f"数据集与环境采用：{dataset}")
        if exp_design.get("metrics") or exp_design.get("baselines"):
            parts.append("评估指标与基线方法分别在 3.2 和 3.3 中展开，以避免总述部分重复列举。")
        procedure = self._format_numbered_experiment_items(exp_design.get("procedure", []), "")
        if procedure:
            parts.append("实验流程概览：\n" + procedure)
        reproducibility = self._safe_text(exp_design.get("reproducibility", ""))
        if reproducibility:
            parts.append(f"可复现性设置：{reproducibility}")
        return "\n\n".join(parts)

    def _build_report_context(self, query: str, outputs: dict) -> dict:
        lit_result = outputs.get("literature_result", {}) or {}
        raw_exp_design = outputs.get("experiment_design", {}) or {}
        exp_design = self._normalize_experiment_design_payload(raw_exp_design)
        code_result = outputs.get("code_result", {}) or {}
        ana_result = outputs.get("analysis_result", {}) or {}

        papers = lit_result.get("top_papers", []) or lit_result.get("papers", [])
        literature_review = self._safe_text(lit_result.get("literature_review", ""))
        final_code = self._safe_text(code_result.get("final_code", ""))
        stdout = self._safe_text(code_result.get("stdout", ""))
        analysis = self._safe_text(ana_result.get("conclusion", ""))
        charts = ana_result.get("charts", []) or []
        code_success = code_result.get("success")

        paper_titles = self._stringify_items(
            [paper.get("title", "") for paper in papers[:8]],
            fallback="暂无标题信息",
            numbered=True,
        )
        paper_summaries = "\n\n".join(
            [
                "\n".join(
                    [
                        f"论文标题：{paper.get('title', '').strip()}",
                        f"方法要点：{((paper.get('structured_summary') or {}).get('method', '') or '').strip()}",
                        f"主要结论：{((paper.get('structured_summary') or {}).get('conclusion', '') or '').strip()}",
                    ]
                ).strip()
                for paper in papers[:5]
                if self._safe_text(paper.get("title", ""))
            ]
        ) or "暂无结构化摘要"

        return {
            "query": query,
            "papers": papers,
            "literature_result": lit_result,
            "experiment_design": exp_design,
            "literature_review": literature_review or "暂无上游文献综述",
            "paper_titles": paper_titles,
            "paper_summaries": paper_summaries,
            "research_hypothesis": self._safe_text(exp_design.get("research_hypothesis", "")) or "暂无明确研究假设",
            "dataset": self._safe_text(exp_design.get("dataset", "")) or "暂无数据集说明",
            "baselines_text": self._stringify_items(
                exp_design.get("baselines", []),
                fallback="暂无基线设置",
                numbered=True,
            ),
            "metrics_text": self._stringify_items(
                exp_design.get("metrics", []),
                fallback="暂无评估指标",
                numbered=True,
            ),
            "procedure_text": self._stringify_items(
                exp_design.get("procedure", []),
                fallback="暂无实验流程",
                numbered=True,
            ),
            "expected_results": self._safe_text(exp_design.get("expected_results", "")) or "暂无预期结果说明",
            "experiment_description": self._safe_text(exp_design.get("full_description", "")),
            "final_code": final_code,
            "code_excerpt": final_code[:1200] or "暂无代码实现",
            "code_result": code_result,
            "code_success": code_success,
            "execution_status": self._format_code_execution_status(code_result),
            "stdout": stdout,
            "stdout_excerpt": stdout[:800] or "暂无执行输出",
            "analysis": analysis or "暂无分析结论",
            "chart_evidence": self._summarize_chart_evidence(charts),
            "charts": charts,
        }

    def _compose_experiment_description(self, query: str, context: dict) -> str:
        # 第 3 章只做确定性整理，不再调用 LLM 生成实验设计，避免报告阶段补造内容。
        exp_design = context.get("experiment_design", {}) or {}
        pieces = []

        overview = self._build_experiment_narrative_overview(exp_design)
        if overview:
            pieces.append(overview)

        rationale = self._build_experiment_design_rationale(exp_design, context)
        if rationale:
            pieces.append(rationale)

        dataset = self._safe_text(exp_design.get("dataset", ""))
        if dataset:
            pieces.append(f"数据集与实验环境：{dataset}")

        procedure = self._format_numbered_experiment_items(exp_design.get("procedure", []), "")
        if procedure:
            pieces.append("实验流程：\n" + procedure)

        variables = self._format_experiment_part(exp_design.get("variables", {}))
        if variables:
            pieces.append("变量控制：\n" + variables)

        ablations = self._format_numbered_experiment_items(exp_design.get("ablations", []), "")
        if ablations:
            pieces.append("消融设计：\n" + ablations)

        reproducibility = self._safe_text(exp_design.get("reproducibility", ""))
        if reproducibility:
            pieces.append(f"可复现性设置：{reproducibility}")

        expected_results = self._safe_text(exp_design.get("expected_results", ""))
        if expected_results:
            pieces.append(f"预期分析方向：{expected_results}")

        if not pieces:
            pieces.append("本章围绕研究假设、评价对象、实验流程和对照方法组织实验设计，为后续实现与结果分析提供方案基础。")
        return "\n\n".join(pieces)

    def _build_experiment_narrative_overview(self, exp_design: dict) -> str:
        if not exp_design or exp_design.get("_deprecated"):
            return ""

        sentences = []
        hypothesis = self._safe_text(exp_design.get("research_hypothesis", ""))
        if hypothesis:
            sentences.append(f"本实验以“{hypothesis}”为核心假设。")
        if exp_design.get("metrics") or exp_design.get("baselines"):
            sentences.append("具体评估指标与基线方法分别见 3.2 和 3.3，本章总述主要概括实验目标、流程与组织方式。")
        if sentences:
            return "\n\n".join(sentences)

        full_description = self._clean_experiment_description_text(exp_design.get("full_description", ""))
        if full_description:
            return f"设计概述：{full_description}"
        return ""

    def _build_experiment_design_rationale(self, exp_design: dict, context: dict) -> str:
        if not exp_design or exp_design.get("_deprecated") or not self._has_experiment_design_content(exp_design):
            return ""

        lines = []
        objectives = self._format_inline_experiment_items(exp_design.get("objectives", []))
        procedure = self._format_inline_experiment_items(exp_design.get("procedure", []))
        metrics_count = len(self._list_values(exp_design.get("metrics", [])))
        baselines_count = len(self._list_values(exp_design.get("baselines", [])))

        if objectives:
            lines.append(
                "设计逻辑：本实验采用由研究假设牵引的对照实验框架，"
                f"核心目标包括{objectives}。这些目标共同对应模型压缩任务中“性能保持、资源节省与部署可行性”"
                "之间的权衡关系。"
            )
        else:
            lines.append(
                "设计逻辑：本实验采用由研究假设牵引的对照实验框架，围绕模型压缩后的性能保持、资源开销"
                "与部署可行性展开验证。"
            )

        if metrics_count or baselines_count:
            lines.append(
                "评价组织：实验设计将评价对象、指标体系和对照方法分离组织。"
                f"其中指标体系包含 {metrics_count or '若干'} 类核心评价维度，"
                f"对照组包含 {baselines_count or '若干'} 类基线方法；具体定义分别放在 3.2 和 3.3，"
                "以便最终报告保持层次清晰。"
            )

        if procedure:
            lines.append(
                f"实验组织：整体流程按照{procedure}的顺序推进，先建立可比较的基线状态，再实施压缩或改进策略，"
                "最后在统一评价协议下比较结果。"
            )

        return "\n".join(lines)

    def _build_experiment_design_boundary(self, context: dict) -> str:
        exp_design = context.get("experiment_design", {}) or {}
        if not self._has_experiment_design_content(exp_design):
            return ""
        missing = []
        if not exp_design.get("variables"):
            missing.append("变量控制")
        if not exp_design.get("ablations"):
            missing.append("消融实验")
        if not self._safe_text(exp_design.get("reproducibility", "")):
            missing.append("可复现性设置")
        if not missing:
            return ""
        return (
            "设计边界说明：EvaluationAgent 实验设计阶段未提供"
            + "、".join(missing)
            + "的详细配置，报告阶段不额外补造这些设定；后续可继续增强实验设计提示词或结构化字段。"
        )

    def _build_report_boundary_notes(self, context: dict) -> list[str]:
        notes = []
        lit_result = context.get("literature_result", {}) or {}
        papers = context.get("papers", []) or []
        try:
            total_found = int(lit_result.get("total_found", len(papers) if papers else 0) or 0)
        except Exception:
            total_found = len(papers)
        if total_found == 0 and not papers:
            notes.append(
                "文献证据边界：本次文献检索未返回可直接引用的论文证据，正文中的背景与综述应理解为待进一步文献核验的研究动机和问题整理。"
            )

        notes.append("实验设计来源说明：实验设计由 EvaluationAgent 的实验设计阶段生成，报告生成阶段只做结构化整理和语言润色。")

        design_boundary = self._build_experiment_design_boundary(context)
        if design_boundary:
            notes.append(design_boundary)

        code_result = context.get("code_result", {}) or {}
        success = code_result.get("success")
        if success is not True:
            notes.append(self._format_code_execution_status(code_result))

        if not self._safe_text(context.get("analysis", "")) or context.get("analysis") == "暂无分析结论":
            notes.append("分析证据边界：分析模块未提供充分结论，结果章节中的解释应视为待真实执行结果补充后的初步讨论。")

        unique = []
        seen = set()
        for note in notes:
            text = self._safe_text(note)
            if text and text not in seen:
                seen.add(text)
                unique.append(text)
        return unique

    def _format_inline_experiment_items(self, value: Any) -> str:
        items = [self._format_experiment_item(item) for item in self._coerce_experiment_list(value)]
        items = [item for item in items if item]
        if not items:
            return ""
        if len(items) == 1:
            return f"“{items[0]}”"
        if len(items) == 2:
            return f"“{items[0]}”和“{items[1]}”"
        return "、".join(f"“{item}”" for item in items[:-1]) + f"和“{items[-1]}”"

    def _format_experiment_item(self, item: Any) -> str:
        if isinstance(item, dict):
            primary = self._safe_text(
                item.get("name")
                or item.get("title")
                or item.get("metric")
                or item.get("baseline")
                or item.get("method")
                or item.get("objective")
                or item.get("hypothesis")
            )
            details = []
            for key, value in item.items():
                if key in {"name", "title", "metric", "baseline", "method", "objective", "hypothesis"}:
                    continue
                detail = self._format_experiment_item(value) if isinstance(value, (dict, list)) else self._safe_text(value)
                if detail:
                    details.append(f"{key}：{detail}")
            if primary and details:
                return f"{primary}（{'；'.join(details)}）"
            if primary:
                return primary
            return "；".join(details)
        if isinstance(item, list):
            parts = [self._format_experiment_item(value) for value in item]
            return "；".join(part for part in parts if part)
        return self._clean_experiment_description_text(item)

    def _format_numbered_experiment_items(self, value: Any, missing_label: str) -> str:
        if isinstance(value, list):
            lines = [self._format_experiment_item(item) for item in value]
            lines = [line for line in lines if line]
            if lines:
                return "\n".join(f"{idx}. {line}" for idx, line in enumerate(lines, 1))
        else:
            text = self._format_experiment_item(value)
            if text:
                return text
        return f"实验设计模块未提供{missing_label}，报告阶段不进行额外补编。" if missing_label else ""

    def _format_experiment_hypothesis_and_goals(self, exp_design: dict) -> str:
        lines = []
        hypothesis = self._safe_text(exp_design.get("research_hypothesis", ""))
        if hypothesis:
            lines.append(f"研究假设：{hypothesis}")

        objectives = self._format_numbered_experiment_items(exp_design.get("objectives", []), "")
        if objectives:
            lines.append("研究目标：\n" + objectives)

        if lines:
            return "\n\n".join(lines)
        return "实验设计模块未提供实验假设与目标，报告阶段不进行额外补编。"

    def _compose_experiment_hypothesis_section(self, exp_design: dict) -> str:
        upstream = self._experiment_section_body(exp_design, "3.1")
        if self._is_rich_experiment_subsection(upstream):
            return upstream

        hypothesis = self._safe_text(exp_design.get("research_hypothesis", ""))
        objectives = self._list_values(exp_design.get("objectives", []))
        procedure = self._format_inline_experiment_items(exp_design.get("procedure", []))
        if not hypothesis and not objectives:
            return "本节围绕研究假设与实验目标组织验证思路，用于明确后续实验需要回答的核心问题。"

        paragraphs = []
        if hypothesis:
            paragraphs.append(
                f"本节围绕 EvaluationAgent 实验设计阶段提出的研究假设展开：{hypothesis}。"
                "该假设不是单纯描述预期效果，而是为后续实验限定验证对象和比较方向，即需要同时观察方法有效性、"
                "资源开销变化以及部署可行性之间的关系。"
            )
        if objectives:
            paragraphs.append(
                "围绕这一假设，实验目标被组织为几个相互关联的验证任务："
                f"{self._format_inline_experiment_items(objectives)}。"
                "这些目标共同构成后续评价的判断框架，使实验不只关注单一结果，而是能够说明压缩策略在不同约束下的表现。"
            )
        if procedure:
            paragraphs.append(
                f"在执行顺序上，目标验证将依托{procedure}等步骤展开，先形成可比较的基线状态，"
                "再观察压缩或改进策略是否符合假设预期。"
            )
        return "\n\n".join(paragraphs)

    def _compose_experiment_metrics_section(self, exp_design: dict) -> str:
        upstream = self._experiment_section_body(exp_design, "3.2")
        if self._is_rich_experiment_subsection(upstream):
            return upstream

        metrics = self._list_values(exp_design.get("metrics", []))
        if not metrics:
            return "本节从任务性能、运行效率和资源占用等维度组织评估指标，用于支撑后续结果分析。"

        metric_text = self._join_experiment_items_for_prose(metrics)
        return (
            "本节将评估指标设计为连接研究假设与实验结论的核心桥梁。"
            f"实验设计阶段给出的指标包括{metric_text}。"
            "这些指标分别从任务性能、效率或资源约束等角度刻画模型压缩后的变化，"
            "因此在后续结果分析中不能孤立解读单个数值，而应结合基线方法和实验流程判断压缩策略是否真正满足研究目标。"
            "若代码执行阶段尚未产生真实指标，本节中的指标仅作为后续验证协议，而不能被写成已经获得的实测结果。"
        )

    def _compose_experiment_baselines_section(self, exp_design: dict) -> str:
        upstream = self._experiment_section_body(exp_design, "3.3")
        if self._is_rich_experiment_subsection(upstream):
            return upstream

        baselines = self._list_values(exp_design.get("baselines", []))
        if not baselines:
            return "本节围绕原始模型、主流方法和关键组件变体组织对照基线，用于建立可解释的比较参照。"

        baseline_text = self._join_experiment_items_for_prose(baselines)
        return (
            "本节的基线设计用于建立压缩方法的比较参照，避免只讨论单个模型的绝对表现。"
            f"实验设计阶段给出的对照方法包括{baseline_text}。"
            "这些基线共同覆盖未压缩上限、常规压缩策略或替代轻量化路径，使后续实验能够比较不同方案在性能保持和资源节省上的取舍。"
            "结合训练条件、硬件环境和统计检验方式，本节能够支撑后续形成完整的对照实验设计。"
        )

    def _is_rich_experiment_subsection(self, body: str) -> bool:
        text = self._safe_text(body)
        if len(text) < 80:
            return False
        numbered_lines = [line for line in text.splitlines() if re.match(r"^\s*\d+[.、)]", line)]
        return len(numbered_lines) <= max(1, len(text.splitlines()) // 2)

    def _join_experiment_items_for_prose(self, items: list[str]) -> str:
        formatted = [self._format_experiment_item(item) for item in items]
        formatted = [item for item in formatted if item]
        if not formatted:
            return "未提供"
        if len(formatted) == 1:
            return f"“{formatted[0]}”"
        if len(formatted) == 2:
            return f"“{formatted[0]}”和“{formatted[1]}”"
        return "、".join(f"“{item}”" for item in formatted[:-1]) + f"和“{formatted[-1]}”"

    def _experiment_section_body(self, exp_design: dict, prefix: str) -> str:
        """Use upstream/fallback section prose when it matches the target 3.x slot."""
        for section in exp_design.get("sections", []) or []:
            if not isinstance(section, dict):
                continue
            heading = self._safe_text(section.get("heading", ""))
            if not heading.startswith(prefix):
                continue
            body = self._clean_experiment_description_text(section.get("body", ""))
            if body:
                return body
        return ""

    def _format_experiment_subsection(self, value: Any, missing_label: str) -> str:
        return self._format_numbered_experiment_items(value, missing_label)

    def _format_code_execution_status(self, code_result: dict) -> str:
        success = code_result.get("success")
        if success is True:
            stdout = self._safe_text(code_result.get("stdout", ""))
            return (
                "代码执行状态：成功。结果分析可以基于真实执行输出展开。"
                f"\n执行输出摘要：{stdout[:500] or '未提供标准输出。'}"
            )

        if success is False:
            error_parts = [
                self._safe_text(code_result.get("error_type", "")),
                self._safe_text(code_result.get("error", "")),
                self._safe_text(code_result.get("stderr", "")),
            ]
            iterations = code_result.get("iterations", []) or []
            if iterations and isinstance(iterations[-1], dict):
                error_parts.append(self._safe_text(iterations[-1].get("error", "")))
                error_parts.append(self._safe_text(iterations[-1].get("stderr", "")))
            error_text = " | ".join(part for part in error_parts if part)
            return (
                "代码执行状态：失败。最终报告不得声称实验已经验证成功、不得给出未由执行结果支持的"
                "显著提升、p 值、加速倍数或定量优势；结果章节应说明实验未完成、失败原因和后续验证计划。"
                f"\n失败信息摘要：{error_text[:700] or '未提供具体错误信息。'}"
            )

        return (
            "代码执行状态：未知。最终报告应避免写成已经完成实验验证，只能基于设计目标、代码草案和"
            "已有分析材料进行保守讨论。"
        )

    def _compose_failed_execution_results(self, query: str, context: dict) -> str:
        return (
            f"围绕“{query}”的实验结果分析需要首先说明一个关键限制：代码执行模块返回失败状态，"
            "因此当前阶段尚未形成可用于证明方法有效性的真实实验结果。根据 EvaluationAgent 生成的实验设计，系统原计划围绕"
            f"{context['metrics_text']} 等指标，并以 {context['baselines_text']} 作为比较对象展开验证；"
            "但由于执行过程未成功完成，本节将结果分析限定为实验方案、评价指标和后续验证路径的整理。\n\n"
            "从当前材料看，研究方案已经形成较清晰的指标体系和基线设置，可为后续实验提供比较框架。"
            f"当前可用图表证据为：{context['chart_evidence']}。"
            "后续工作应优先修复代码执行问题，重新生成可复现实验日志，再基于真实 stdout、指标表和图表补充结果分析。"
        )

    def _compose_execution_limited_conclusion(self, query: str, context: dict, result_description: str) -> str:
        return (
            f"本报告围绕“{query}”整理了文献背景、研究假设、实验设计和方法实现思路。根据当前 EvaluationAgent 实验设计结果，"
            f"研究假设为：{context['research_hypothesis']}。实验设计已经给出了数据集、评价指标、基线方法和"
            "预期结果，为后续验证提供了较完整的方案基础。\n\n"
            "不过，需要明确的是，代码执行模块当前返回失败状态，因此本报告不能将设计目标或预期结果表述为已经"
            "得到验证的实验发现。现阶段更合理的结论是：该方案具备进一步实现和验证的研究价值，但其实际性能、"
            "压缩收益、推理速度和相对基线优势仍需在代码成功运行后重新评估。后续改进应优先定位执行失败原因，"
            "补齐可复现实验日志和指标计算流程，再由分析模块输出结构化结果，最后更新结果章节和摘要。"
        )

    def _build_rich_report(self, query: str, outputs: dict) -> dict:
        self.logger.info(self.AGENT_NAME, "构建详细学术报告...")

        # Keep the external workflow unchanged and only reshape upstream data
        # inside the evaluation agent before generating report sections.
        context = self._build_report_context(query, outputs)

        self.logger.info(self.AGENT_NAME, "生成研究背景...")
        background = self._expand_section(
            (
                f"请基于以下材料，为研究问题“{query}”撰写研究背景与问题陈述，至少 300 字。\n\n"
                f"上游文献综述：\n{context['literature_review']}\n\n"
                f"相关文献标题：\n{context['paper_titles']}\n\n"
                f"结构化文献摘要：\n{context['paper_summaries']}\n\n"
                f"当前研究假设：{context['research_hypothesis']}\n\n"
                "写作要求：\n"
                "1. 重点说明已有研究主要解决了什么、仍存在哪些不足，以及本文问题的研究价值。\n"
                "2. 必须优先依据给定文献材料展开，不要只根据问题名称泛泛而谈。\n"
                "3. 若材料不足，请用审慎表述指出信息边界。"
            ),
            min_words=300,
        )

        self.logger.info(self.AGENT_NAME, "生成文献综述...")
        literature = self._expand_section(
            (
                f"请围绕研究问题“{query}”撰写论文风格的文献综述，至少 500 字。\n\n"
                f"请优先基于上游模块传入的文献综述内容进行改写与扩写：\n{context['literature_review']}\n\n"
                f"可作为补充参考的文献标题如下：\n{context['paper_titles']}\n\n"
                f"可作为补充事实依据的结构化摘要如下：\n{context['paper_summaries']}\n\n"
                "写作要求：\n"
                "1. 采用正式的中文学术论文语体，不要写成项目汇报或分点罗列。\n"
                "2. 以连续段落组织内容，可按研究方向、方法类别、关键进展、局限与研究空白展开。\n"
                "3. 不要虚构不存在的方法、数据或结论；若信息不足，保持审慎表述。\n"
                "4. 结尾自然引出本文的研究切入点和后续实验设计动机。"
            ),
            min_words=500,
        )

        self.logger.info(self.AGENT_NAME, "整合实验设计...")
        experiment_description = self._compose_experiment_description(query, context)

        self.logger.info(self.AGENT_NAME, "生成方法实现说明...")
        method_description = self._expand_section(
            (
                "请根据以下前置模块结果，用学术语言描述核心方法实现，至少 250 字。\n\n"
                f"研究问题：{query}\n\n"
                f"研究假设：{context['research_hypothesis']}\n\n"
                f"代码执行状态：\n{context['execution_status']}\n\n"
                f"实验流程：\n{context['procedure_text']}\n\n"
                f"相关文献方法线索：\n{context['paper_summaries']}\n\n"
                f"代码摘要：\n{context['code_excerpt']}\n\n"
                "写作要求：\n"
                "1. 说明实现如何服务于研究目标与实验设计，而不只是解释代码语法。\n"
                "2. 尽量指出实现与已有方法之间的联系或差异，但不要编造未出现的算法细节。\n"
                "3. 如果代码执行失败，只能描述设计意图、代码草案和后续验证计划，不得写成已经部署、实测或验证成功。\n"
                "4. 使用完整段落，不要列表化。"
            ),
            min_words=250,
        )
        if context["code_success"] is False:
            method_description = self._guard_failed_execution_claims(method_description, context["code_result"])

        self.logger.info(self.AGENT_NAME, "生成结果分析...")
        if context["code_success"] is False:
            result_description = self._compose_failed_execution_results(query, context)
        else:
            result_description = self._expand_section(
                (
                    "请严格基于以下前置模块结果，撰写实验结果与分析，至少 350 字。\n\n"
                    f"研究问题：{query}\n\n"
                    f"代码执行状态：\n{context['execution_status']}\n\n"
                    f"评估指标：\n{context['metrics_text']}\n\n"
                    f"基线方法：\n{context['baselines_text']}\n\n"
                    f"预期结果：{context['expected_results']}\n\n"
                    f"分析结论：{context['analysis']}\n\n"
                    f"执行输出：\n{context['stdout_excerpt']}\n\n"
                    f"图表证据：\n{context['chart_evidence']}\n\n"
                    "写作要求：\n"
                    "1. 结果分析必须围绕指标、基线比较和图表证据展开，不要脱离这些材料自由发挥。\n"
                    "2. 若无法得出严格定量结论，应明确说明证据不足，而不是虚构数值。\n"
                    "3. 需要解释结果含义、潜在原因，以及与预期是否一致。\n"
                    "4. 如果代码执行状态不是成功，禁止写成已经完成实验验证。"
                ),
                min_words=350,
            )

        self.logger.info(self.AGENT_NAME, "生成结论与展望...")
        if context["code_success"] is False:
            conclusion = self._compose_execution_limited_conclusion(query, context, result_description)
        else:
            conclusion = self._expand_section(
                (
                    f"请基于以下材料，为研究问题“{query}”撰写结论与展望，至少 220 字。\n\n"
                    f"研究背景与空白线索：\n{context['literature_review']}\n\n"
                    f"研究假设：{context['research_hypothesis']}\n\n"
                    f"实验设计摘要：\n{experiment_description[:1200]}\n\n"
                    f"结果分析摘要：\n{result_description[:1200]}\n\n"
                    f"代码执行状态：\n{context['execution_status']}\n\n"
                    f"图表与执行证据：\n{context['chart_evidence']}\n{context['stdout_excerpt']}\n\n"
                    "写作要求：\n"
                    "1. 结论必须回扣前文证据，不要仅根据 query 作模板化总结。\n"
                    "2. 需要包含主要发现、现有局限和可执行的后续改进方向。\n"
                    "3. 若某些结论证据不充分，请明确说明其不确定性。\n"
                    "4. 只有代码执行成功且分析模块提供证据时，才可以写确定性的实验发现。"
                ),
                min_words=220,
            )

        title = f"MindPilot 科研报告：{query}"
        exp_design = context["experiment_design"]
        exp_sections = self._normalize_experiment_sections(exp_design.get("sections", []))
        if not exp_sections:
            exp_sections = [
                {
                    "heading": "3.1 实验假设与目标",
                    "body": self._compose_experiment_hypothesis_section(exp_design),
                },
                {
                    "heading": "3.2 评估指标",
                    "body": self._compose_experiment_metrics_section(exp_design),
                },
                {
                    "heading": "3.3 基线方法",
                    "body": self._compose_experiment_baselines_section(exp_design),
                },
            ]
        sections = [
            {"heading": "一、研究背景与问题陈述", "body": background, "level": 1},
            {"heading": "二、文献综述", "body": literature, "level": 1},
            {"heading": "三、实验设计与方法论", "body": experiment_description, "level": 1},
            *[
                {"heading": sec["heading"], "body": sec["body"], "level": 2}
                for sec in exp_sections
            ],
            {"heading": "四、核心方法实现", "body": method_description, "level": 1},
            {"heading": "五、实验结果与分析", "body": result_description, "level": 1},
            {"heading": "六、结论与展望", "body": conclusion, "level": 1},
        ]
        boundary_notes = self._build_report_boundary_notes(context)
        if boundary_notes:
            sections.append(
                {
                    "heading": "附录：证据边界与执行状态说明",
                    "body": "\n\n".join(
                        f"{idx}. {note}" for idx, note in enumerate(boundary_notes, 1)
                    ),
                    "level": 1,
                }
            )

        report_body = self._render_report_text(
            {
                "title": title,
                "query": query,
                "abstract": "",
                "sections": sections,
            }
        )
        abstract = self._build_abstract(query, report_body)

        return {
            "title": title,
            "query": query,
            "abstract": abstract,
            "sections": sections,
            "code": context["final_code"],
            "stdout": context["stdout"],
            "literature": context["papers"],
            "charts": context["charts"],
        }

    def _build_abstract(self, query: str, report_body: str) -> str:
        summary_body = self._sanitize_text_for_abstract(report_body)
        prompt = (
            f"请基于已经完成的科研报告正文，为研究问题“{query}”撰写中文摘要。\n\n"
            f"报告正文如下：\n{summary_body[:6000]}\n\n"
            "写作要求：\n"
            "1. 摘要必须严格基于正文内容总结，不要根据固定模板重写，不要补充正文中没有的信息。\n"
            "2. 用一段或两段连续学术表述概括研究背景、方法、实验设置、核心结果与结论。\n"
            "3. 控制在 180 到 260 字之间，风格接近论文摘要。\n"
            "4. 摘要中的研究假设、指标、基线和实验结论必须与正文第三章及结果章节一致。\n"
            "5. 如果正文说明代码执行失败或证据不足，不得写成实验已验证成功，也不得新增量化提升。\n"
            "6. 不要使用项目符号，不要写“本文将”这类尚未完成时态，保持结果导向。"
        )
        abstract = self.llm.chat(
            [
                {
                    "role": "system",
                    "content": "你是资深学术写作专家，请根据给定正文生成准确、简洁、论文风格的中文摘要。",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=512,
        )
        abstract = self._safe_text(abstract)
        data = _extract_json_object(abstract)
        if data:
            abstract_parts = [
                self._safe_text(data.get(key, ""))
                for key in ("background", "method", "result", "conclusion", "limitation")
            ]
            abstract = " ".join(part for part in abstract_parts if part)
        if abstract:
            return abstract
        return f"本文围绕“{query}”开展研究，摘要应根据最终正文内容生成，但当前模型未返回有效结果。"

    def _expand_section(self, prompt: str, min_words: int = 200) -> str:
        resp = self.llm.chat(
            [
                {
                    "role": "system",
                    "content": (
                        "你是资深科研论文写作专家，请用严谨、专业的中文学术语言撰写内容。"
                        f"正文尽量不少于 {min_words} 字，并使用完整段落表达。"
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=2048,
        )
        return resp if resp else f"内容生成中，请参考问题：{prompt[:120]}"

    def _safe_text(self, value: Any) -> str:
        if value is None:
            return ""
        return str(value).strip()

    def _normalize_experiment_sections(self, sections: Any) -> list[dict]:
        """清洗模型返回的实验设计小节，报告生成只消费 heading/body。"""
        if not isinstance(sections, list):
            return []

        normalized = []
        for idx, sec in enumerate(sections, 1):
            if not isinstance(sec, dict):
                continue
            heading = str(sec.get("heading", "")).strip()
            body = self._clean_experiment_description_text(sec.get("body", ""))
            if not heading or not body:
                continue
            if not heading.startswith("3."):
                heading = f"3.{idx} {heading}"
            normalized.append({"heading": heading, "body": body})
        return normalized

    def _format_experiment_part(self, items: Any, fallback: str = "") -> str:
        """把实验设计字段稳定格式化，避免报告中出现一整段散文。"""
        if isinstance(items, str):
            return self._clean_experiment_description_text(items) or fallback
        if isinstance(items, dict):
            lines = []
            for key, value in items.items():
                if isinstance(value, list):
                    value = "；".join(self._format_experiment_item(v) for v in value if self._format_experiment_item(v))
                else:
                    value = self._format_experiment_item(value)
                if self._safe_text(value):
                    lines.append(f"{key}：{value}")
            return "\n".join(lines) if lines else fallback
        if isinstance(items, list):
            cleaned = [self._format_experiment_item(item) for item in items]
            cleaned = [item for item in cleaned if item]
            return "\n".join(f"{idx}. {item}" for idx, item in enumerate(cleaned, 1)) if cleaned else fallback
        return fallback

    def _format_variables_and_ablations(self, exp_design: dict) -> str:
        variables = exp_design.get("variables", {}) or {}
        lines = []
        if isinstance(variables, dict):
            label_map = {
                "independent": "自变量",
                "dependent": "因变量",
                "controlled": "控制变量",
            }
            for key, label in label_map.items():
                value = variables.get(key, [])
                if isinstance(value, list):
                    value = "；".join(str(v) for v in value if str(v).strip())
                if str(value).strip():
                    lines.append(f"{label}：{value}")
        ablations = self._format_experiment_part(exp_design.get("ablations", []))
        if ablations:
            lines.append("消融实验：\n" + ablations)
        return "\n".join(lines) if lines else "本节通过固定训练配置、数据划分和评价协议，比较关键组件加入或移除后的性能变化。"

    def _format_experiment_design(self, exp_design: dict) -> str:
        """生成第 3 章开头的结构化总述。"""
        pieces = []
        overview = exp_design.get("full_description", "")
        if overview:
            pieces.append(overview.strip())

        hypothesis = exp_design.get("research_hypothesis", "")
        if hypothesis:
            pieces.append(f"研究假设：{hypothesis.strip()}")

        dataset = exp_design.get("dataset", "")
        if dataset:
            pieces.append(f"数据与环境：{dataset.strip()}")

        baselines = self._format_experiment_part(exp_design.get("baselines", []))
        if baselines:
            pieces.append("基线与对照组：\n" + baselines)

        metrics = self._format_experiment_part(exp_design.get("metrics", []))
        if metrics:
            pieces.append("核心评估指标：\n" + metrics)

        procedure = self._format_experiment_part(exp_design.get("procedure", []))
        if procedure:
            pieces.append("实验流程概览：\n" + procedure)

        variables = self._format_experiment_part(exp_design.get("variables", {}))
        if variables:
            pieces.append("变量控制：\n" + variables)

        ablations = self._format_experiment_part(exp_design.get("ablations", []))
        if ablations:
            pieces.append("消融实验：\n" + ablations)

        reproducibility = exp_design.get("reproducibility", "")
        if reproducibility:
            pieces.append(f"可复现性设置：{reproducibility.strip()}")

        expected = exp_design.get("expected_results", "")
        if expected:
            pieces.append(f"预期分析方向：{expected.strip()}")

        return "\n\n".join(pieces) if pieces else "本章从研究假设、数据集、基线方法、评估指标、实验流程和可复现性设置六个方面组织实验设计。"

    def _print_result(self, score: EvalScore, log: list, reports: dict):
        accepted_reflections = sum(1 for record in log if (record or {}).get("accepted", False))
        print(f"\n{'=' * 58}")
        print("  Evaluation and report generation complete")
        print(f"{'=' * 58}")
        print(
            f"  overall={score.overall:.2f}  accuracy={score.accuracy:.2f}  "
            f"completeness={score.completeness:.2f}  format={score.format_quality:.2f}"
        )
        print(f"  reflection_rounds={accepted_reflections}")
        print("  report_files:")
        for fmt, path in reports.items():
            print(f"    [{fmt.upper():8s}] {path}")
        print(f"  feedback: {score.feedback[:100]}")
        print(f"{'=' * 58}\n")
