"""
Focused tests for Module 6 EvaluationAgent behavior.

This file keeps EvaluationAgent-specific checks out of test_code_eval.py so
shared code-generation tests stay stable for other contributors.

Run with:
    python mindpilot/tests/test_evaluation_agent.py -q
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.evaluation_agent import EvalScore, EvaluationAgent, LLMJudge, MultiAgentJudge
from config import CONFIG
from tools.report_generator import ReportGenerator


class FakeLLM:
    def __init__(self, response=None):
        self.response = response or (
            '{"overall_score": 0.70, "accuracy": 0.70, "completeness": 0.70, '
            '"format_quality": 0.70, "feedback": "ok", "needs_reflection": false}'
        )
        self.messages = []

    def chat(self, messages, max_tokens=None):
        self.messages.append(messages)
        return self.response


class FakeLLMWithKwargs(FakeLLM):
    def __init__(self, response=None):
        super().__init__(response)
        self.kwargs = []

    def chat(self, messages, max_tokens=None):
        self.messages.append(messages)
        self.kwargs.append({"max_tokens": max_tokens})
        return self.response


class ExplodingLLM(FakeLLM):
    def chat(self, messages, max_tokens=None):
        raise AssertionError("EvaluationAgent should not call LLM for experiment design")


class FakeLogger:
    def start_call(self, *args, **kwargs):
        return {"args": args, "kwargs": kwargs}

    def finish_call(self, *args, **kwargs):
        return None

    def fail_call(self, *args, **kwargs):
        return None

    def info(self, *args, **kwargs):
        return None

    def success(self, *args, **kwargs):
        return None


class FakeMemory:
    def add(self, *args, **kwargs):
        return None


class FakeReportGenerator:
    def __init__(self):
        self.last_content = None

    def generate(self, content, filename="report", formats=None):
        self.last_content = content
        return {"markdown": "fake.md"}


class FakeJudge:
    def __init__(self, scores):
        self.scores = list(scores)
        self.index = 0

    def score(self, query, output, output_type="report"):
        score = self.scores[min(self.index, len(self.scores) - 1)]
        self.index += 1
        return score


class FakeReflector:
    def __init__(self, payload):
        self.payload = payload

    def reflect_and_revise_report(self, query, report_content, score):
        return self.payload


def build_agent():
    return EvaluationAgent(
        CONFIG,
        llm_client=FakeLLM(),
        report_gen=FakeReportGenerator(),
        memory_store=FakeMemory(),
        logger=FakeLogger(),
    )


def build_agent_with_llm(llm):
    return EvaluationAgent(
        CONFIG,
        llm_client=llm,
        report_gen=FakeReportGenerator(),
        memory_store=FakeMemory(),
        logger=FakeLogger(),
    )


def section_map(report):
    return {section["heading"]: section["body"] for section in report["sections"]}


def section_body_contains(report, heading_text):
    for section in report["sections"]:
        if heading_text in section["heading"]:
            return section["body"]
    raise AssertionError(f"Missing section containing: {heading_text}")


class TestLLMJudge(unittest.TestCase):
    def test_score_returns_eval_score(self):
        judge = LLMJudge(FakeLLM(), threshold=0.65, logger=FakeLogger())

        score = judge.score("test query", "test output")

        self.assertIsInstance(score, EvalScore)
        self.assertEqual(score.overall, 0.70)

    def test_needs_reflection_defaults_from_score_threshold(self):
        response = (
            '{"overall_score": 0.50, "accuracy": 0.50, "completeness": 0.50, '
            '"format_quality": 0.50, "feedback": "weak"}'
        )
        judge = LLMJudge(FakeLLM(response), threshold=0.65, logger=FakeLogger())

        score = judge.score("test query", "test output")

        self.assertTrue(score.needs_reflection)

    def test_rouge_l_identical_and_empty(self):
        judge = LLMJudge(FakeLLM(), threshold=0.65, logger=FakeLogger())

        self.assertAlmostEqual(judge.compute_rouge_l("hello world", "hello world"), 1.0, places=1)
        self.assertEqual(judge.compute_rouge_l("", ""), 0.0)


class TestMultiAgentJudge(unittest.TestCase):
    def test_score_many_returns_weighted_reviews(self):
        judge = MultiAgentJudge(FakeLLM(), threshold=0.65, logger=FakeLogger())

        score, reviews = judge.score_many("test query", "test report")

        self.assertIsInstance(score, EvalScore)
        self.assertEqual(len(reviews), 3)
        self.assertAlmostEqual(sum(review["weight"] for review in reviews), 1.0, places=2)
        self.assertEqual(score.overall, 0.70)
        self.assertTrue(all("reviewer" in review and "score" in review for review in reviews))

    def test_long_report_is_reviewed_by_segments(self):
        llm = FakeLLM()
        judge = MultiAgentJudge(
            llm,
            threshold=0.65,
            logger=FakeLogger(),
            reviewers=[
                {
                    "name": "实验方法评审专家",
                    "weight": 1.0,
                    "rubric": "重点检查实验设计。",
                }
            ],
        )
        long_report = "\n\n".join(
            [
                "# Test Report",
                "## 摘要\n" + "summary " * 300,
                "## 三、实验设计与方法论\n" + "experiment design " * 500,
                "### 3.1 实验假设与目标\n" + "hypothesis " * 350,
                "### 3.2 评估指标\n" + "metrics " * 350,
                "### 3.3 基线方法\n" + "baselines " * 350,
            ]
        )

        score, reviews = judge.score_many("query", long_report)

        self.assertIsInstance(score, EvalScore)
        self.assertEqual(len(reviews), 1)
        self.assertGreater(len(reviews[0]["segments"]), 1)
        self.assertEqual(len(llm.messages), len(reviews[0]["segments"]))
        self.assertTrue(
            all(segment["chars"] <= judge.SEGMENT_CHAR_LIMIT for segment in reviews[0]["segments"])
        )
        self.assertTrue(any("第 1/" in call[1]["content"] for call in llm.messages))
        self.assertTrue(any("三、实验设计与方法论" in segment["title"] for segment in reviews[0]["segments"]))


class TestEvaluationAgentReflection(unittest.TestCase):
    def test_reflection_updates_saved_report_content(self):
        agent = build_agent()
        agent._build_rich_report = lambda query, outputs: {
            "title": "Test Report",
            "query": query,
            "abstract": "old abstract",
            "sections": [
                {"heading": "一、背景", "body": "old body", "level": 1},
                {"heading": "二、结论", "body": "old ending", "level": 1},
            ],
        }
        agent.judge = FakeJudge(
            [
                EvalScore(0.50, 0.50, 0.50, 0.50, "needs work", True, "revise"),
                EvalScore(0.90, 0.90, 0.90, 0.90, "much better", False, ""),
            ]
        )
        agent.reflector = FakeReflector(
            {
                "abstract": "new abstract",
                "sections": [
                    {"body": "new body"},
                    {"body": "new ending"},
                ],
            }
        )

        result = agent.run("test query", {})

        self.assertEqual(result["reflection_rounds"], 1)
        self.assertEqual(result["attempted_reflection_rounds"], 1)
        self.assertEqual(agent.report_gen.last_content["abstract"], "new abstract")
        self.assertEqual(agent.report_gen.last_content["sections"][0]["body"], "new body")
        self.assertEqual(agent.report_gen.last_content["evaluation"]["overall_score"], 0.90)

    def test_invalid_reflection_attempt_is_not_counted_as_accepted_round(self):
        agent = build_agent()
        agent._build_rich_report = lambda query, outputs: {
            "title": "Test Report",
            "query": query,
            "abstract": "old abstract",
            "sections": [
                {"heading": "一、背景", "body": "old body", "level": 1},
                {"heading": "二、结论", "body": "old ending", "level": 1},
            ],
        }
        agent.judge = FakeJudge([EvalScore(0.50, 0.50, 0.50, 0.50, "needs work", True, "revise")])
        agent.reflector = FakeReflector({"sections": []})

        result = agent.run("test query", {})

        self.assertEqual(result["reflection_rounds"], 0)
        self.assertEqual(result["attempted_reflection_rounds"], 1)
        self.assertEqual(result["reflection_log"][0]["status"], "invalid_revision")
        self.assertFalse(result["reflection_log"][0]["accepted"])
        self.assertEqual(agent.report_gen.last_content["evaluation"]["reflection_rounds"], 0)
        self.assertEqual(agent.report_gen.last_content["evaluation"]["attempted_reflection_rounds"], 1)

    def test_rule_based_reflection_fixes_high_risk_claims_before_llm(self):
        agent = build_agent()
        report = {
            "abstract": "实验结果表明显著提升。",
            "sections": [
                {"heading": "一、研究背景", "body": "已有研究表明该方向成熟。", "level": 1},
                {"heading": "3.1 实验假设与目标", "body": "invented", "level": 2},
                {"heading": "3.2 评估指标", "body": "mAP", "level": 2},
                {"heading": "五、实验结果与分析", "body": "实验结果表明显著提升 10%。", "level": 1},
            ],
        }
        breakdown = {
            "findings": [
                {"dimension": "evidence_grounding", "severity": "hard", "message": "no papers"},
                {"dimension": "experiment_consistency", "severity": "hard", "message": "bad design"},
                {"dimension": "execution_validity", "severity": "hard", "message": "code failed"},
            ]
        }
        outputs = {
            "literature_result": {"total_found": 0, "top_papers": []},
            "experiment_design": {
                "research_hypothesis": "H1 from upstream.",
                "metrics": ["Accuracy"],
                "baselines": ["Baseline-A"],
            },
            "code_result": {"success": False, "error": "RuntimeError"},
            "analysis_result": {},
        }

        fixed, changes, targets = agent._apply_rule_based_reflection_fixes(report, breakdown, outputs)

        self.assertIn("rule_fix:evidence_grounding", changes)
        self.assertIn("rule_fix:experiment_consistency", changes)
        self.assertIn("rule_fix:execution_validity", changes)
        self.assertNotIn("证据边界说明", fixed["sections"][0]["body"])
        self.assertIn("研究假设展开", fixed["sections"][1]["body"])
        self.assertIn("H1 from upstream.", fixed["sections"][1]["body"])
        self.assertIn("评估指标设计为连接研究假设与实验结论", fixed["sections"][2]["body"])
        self.assertIn("Accuracy", fixed["sections"][2]["body"])
        self.assertNotIn("执行状态说明", fixed["sections"][3]["body"])
        self.assertNotIn("执行状态说明", fixed["abstract"])
        appendix = section_body_contains(fixed, "证据边界与执行状态说明")
        self.assertIn("证据边界说明", appendix)
        self.assertIn("代码执行状态：失败", appendix)
        self.assertTrue(any(target["heading"] == "五、实验结果与分析" for target in targets))

    def test_rule_based_experiment_fix_preserves_upstream_section_prose(self):
        agent = build_agent()
        report = {
            "abstract": "",
            "sections": [
                {"heading": "3.1 实验假设与目标", "body": "invented", "level": 2},
                {"heading": "3.2 评估指标", "body": "invented", "level": 2},
                {"heading": "3.3 基线方法", "body": "invented", "level": 2},
            ],
        }
        breakdown = {
            "findings": [
                {"dimension": "experiment_consistency", "severity": "hard", "message": "bad design"},
            ]
        }
        outputs = {
            "experiment_design": {
                "research_hypothesis": "H1 from upstream.",
                "metrics": ["Accuracy"],
                "baselines": ["Baseline-A"],
                "sections": [
                    {
                        "heading": "3.1 实验假设与目标",
                        "body": (
                            "论文式假设与目标正文。该段详细说明研究假设、实验目标与验证路径之间的关系，"
                            "能够直接作为报告小节使用，而不是简单字段列表。它进一步说明目标之间如何形成递进关系，"
                            "并交代后续实验将围绕这些目标判断方案是否成立。"
                        ),
                    },
                    {
                        "heading": "3.2 评估指标",
                        "body": (
                            "论文式指标说明正文。该段解释指标如何服务于实验判断，并说明不同指标之间的互补关系，"
                            "能够直接作为报告小节使用。它还说明这些指标如何共同覆盖性能、效率和资源约束，"
                            "避免后续结果分析只依赖单一数字作判断。"
                        ),
                    },
                    {
                        "heading": "3.3 基线方法",
                        "body": (
                            "论文式基线说明正文。该段解释基线方法的对照意义，以及这些基线如何帮助判断实验方案的有效性，"
                            "能够直接作为报告小节使用。它还说明不同基线覆盖的比较维度，"
                            "从而支撑后续对方法收益和局限的讨论。"
                        ),
                    },
                ],
            }
        }

        fixed, changes, _ = agent._apply_rule_based_reflection_fixes(report, breakdown, outputs)

        self.assertIn("rule_fix:experiment_consistency", changes)
        self.assertIn("论文式假设与目标正文", fixed["sections"][0]["body"])
        self.assertIn("论文式指标说明正文", fixed["sections"][1]["body"])
        self.assertIn("论文式基线说明正文", fixed["sections"][2]["body"])

    def test_experiment_consistency_uses_key_terms_not_exact_sentences(self):
        agent = build_agent()
        report = {
            "title": "Test report",
            "abstract": "summary",
            "sections": [
                {"heading": "3.1 实验假设与目标", "body": "本研究假设通过结合结构化剪枝与量化感知训练，可降低计算开销。"},
                {"heading": "3.2 评估指标", "body": "实验采用 NDS（nuScenes Detection Score）、mAP 与推理延迟评估模型。"},
                {"heading": "3.3 基线方法", "body": "基线包括原始模型、均匀量化和知识蒸馏。"},
            ],
        }
        outputs = {
            "experiment_design": {
                "research_hypothesis": "结合结构化剪枝与量化感知训练的混合压缩策略，能在保持感知精度的前提下降低计算复杂度。",
                "metrics": [
                    "NDS (nuScenes Detection Score): 综合评估检测精度",
                    "mAP (mean Average Precision): 平均检测精度",
                    "Latency (ms): 单帧推理耗时",
                ],
                "baselines": [
                    "原始模型: 未压缩模型",
                    "均匀量化: INT8量化",
                    "知识蒸馏: 教师-学生压缩",
                ],
            },
            "literature_result": {"total_found": 1, "top_papers": [{"title": "Paper A"}]},
            "code_result": {"success": True, "stdout": "NDS=0.5"},
            "analysis_result": {"conclusion": "analysis"},
        }

        breakdown = agent._rule_score_report(report, outputs, EvalScore(0.8, 0.8, 0.8, 0.8, "", False))

        self.assertGreaterEqual(breakdown["scores"]["experiment_consistency"], 0.85)
        self.assertFalse(
            any(finding["dimension"] == "experiment_consistency" for finding in breakdown["findings"])
        )

    def test_failed_execution_guard_cleans_repeated_replacements(self):
        agent = build_agent()

        guarded = agent._guard_failed_execution_claims(
            "最终报告不得声称实验已经验证成功，也不能写显著优于基线或优于基线。",
            {"success": False},
        )

        self.assertIn("实验尚未完成验证", guarded)
        self.assertIn("与基线的相对表现仍需验证", guarded)
        self.assertNotIn("尚未完成验证成功", guarded)
        self.assertNotIn("仍需验证仍需验证", guarded)

    def test_sparse_reflection_patch_updates_only_target_section(self):
        agent = build_agent()
        report = {
            "abstract": "old abstract",
            "sections": [
                {"heading": "一、背景", "body": "keep", "level": 1},
                {"heading": "五、结果", "body": "old result", "level": 1},
            ],
        }

        revised = agent._apply_revised_report(
            report,
            {"section_revisions": [{"index": 1, "body": "grounded result"}]},
        )

        self.assertEqual(revised["sections"][0]["body"], "keep")
        self.assertEqual(revised["sections"][1]["body"], "grounded result")

    def test_targeted_reflector_prompt_contains_evidence_and_target_sections(self):
        llm = FakeLLM(
            '{"section_revisions": [{"index": 0, "body": "revised with evidence"}]}'
        )
        reflector = build_agent_with_llm(llm).reflector

        payload = reflector.reflect_and_revise_report(
            "query",
            {
                "abstract": "old",
                "sections": [
                    {"heading": "一、背景", "body": "old background", "level": 1},
                    {"heading": "五、结果", "body": "old result", "level": 1},
                ],
            },
            EvalScore(0.5, 0.5, 0.5, 0.5, "bad", True, "fix"),
            evidence_context="代码执行成功：False\n评估指标：未提供",
            findings=[{"dimension": "execution_validity", "severity": "hard"}],
            target_sections=[{"index": 1, "heading": "五、结果", "body": "old result"}],
        )

        prompt = llm.messages[-1][1]["content"]
        self.assertIn("可用上游证据边界", prompt)
        self.assertIn("代码执行成功：False", prompt)
        self.assertIn("target_sections", prompt)
        self.assertIn('"index": 1', prompt)
        self.assertNotIn('"index": 0, "heading": "一、背景"', prompt)
        self.assertEqual(payload["section_revisions"][0]["body"], "revised with evidence")

    def test_reflection_stops_without_rescore_when_upstream_blocker_remains(self):
        agent = build_agent()
        agent._build_rich_report = lambda query, outputs: {
            "title": "Test Report",
            "query": query,
            "abstract": "实验结果表明显著提升。",
            "sections": [
                {"heading": "五、实验结果与分析", "body": "实验结果表明显著提升。", "level": 1},
            ],
        }
        agent.judge = FakeJudge([EvalScore(0.50, 0.50, 0.50, 0.50, "needs work", True, "revise")])
        agent.reflector = FakeReflector(
            {"section_revisions": [{"index": 0, "body": "代码执行失败，尚不能验证实验效果。"}]}
        )

        result = agent.run(
            "query",
            {
                "literature_result": {"total_found": 0, "top_papers": []},
                "experiment_design": {"metrics": ["mAP"]},
                "code_result": {"success": False, "error": "RuntimeError"},
                "analysis_result": {},
            },
        )

        self.assertEqual(result["reflection_rounds"], 1)
        self.assertEqual(result["attempted_reflection_rounds"], 1)
        self.assertEqual(result["reflection_log"][0]["status"], "accepted_rule_only")
        self.assertEqual(agent.judge.index, 2)


class TestEvaluationAgentReportGeneration(unittest.TestCase):
    def test_design_experiment_uses_upstream_design_without_llm_call(self):
        agent = build_agent_with_llm(ExplodingLLM())

        result = agent.design_experiment(
            "model compression",
            {
                "experiment_design": {
                    "hypothesis": "H1 from upstream module.",
                    "goals": ["keep accuracy"],
                    "evaluation_metrics": ["mAP"],
                    "baseline_methods": ["Vanilla student"],
                }
            },
        )

        self.assertEqual(result["_source"], "upstream")
        self.assertEqual(result["research_hypothesis"], "H1 from upstream module.")
        self.assertEqual(result["objectives"], ["keep accuracy"])
        self.assertEqual(result["metrics"], ["mAP"])
        self.assertEqual(result["baselines"], ["Vanilla student"])

    def test_design_experiment_falls_back_to_llm_when_upstream_is_missing(self):
        response = (
            '{"research_hypothesis": "H1 fallback.", '
            '"objectives": ["objective A"], '
            '"dataset": "Dataset-X", '
            '"baselines": ["Baseline-A"], '
            '"metrics": ["Accuracy"], '
            '"procedure": ["train", "evaluate"], '
            '"expected_results": "better accuracy", '
            '"full_description": "Fallback generated experiment design."}'
        )
        agent = build_agent_with_llm(FakeLLM(response))

        result = agent.design_experiment("model compression", {"top_papers": []})

        self.assertEqual(result["_source"], "evaluation_agent_generated")
        self.assertEqual(result["research_hypothesis"], "H1 fallback.")
        self.assertEqual(result["metrics"], ["Accuracy"])
        self.assertEqual(result["baselines"], ["Baseline-A"])

    def test_literature_retrieval_metrics_do_not_count_as_upstream_experiment_design(self):
        response = (
            '{"research_hypothesis": "H1 fallback.", '
            '"metrics": ["mAP"], '
            '"baselines": ["Teacher model"], '
            '"full_description": "Fallback design."}'
        )
        agent = build_agent_with_llm(FakeLLM(response))

        result = agent.design_experiment(
            "model compression",
            {
                "total_found": 0,
                "metrics": {"recall@5": 0.0, "precision@5": 0.0},
                "literature_review": "No papers found.",
            },
        )

        self.assertEqual(result["_source"], "evaluation_agent_generated")
        self.assertEqual(result["metrics"], ["mAP"])
        self.assertEqual(result["baselines"], ["Teacher model"])

    def test_generated_experiment_design_uses_complete_json_prompt_and_source_section_shape(self):
        response = (
            '{"research_hypothesis": "H1 fallback.", '
            '"metrics": ["mAP"], '
            '"baselines": ["Teacher model"], '
            '"full_description": "Fallback design."}'
        )
        llm = FakeLLMWithKwargs(response)
        agent = build_agent_with_llm(llm)

        result = agent.design_experiment("model compression", {"top_papers": []})

        prompt = llm.messages[-1][1]["content"]
        self.assertEqual(result["_source"], "evaluation_agent_generated")
        self.assertIn("禁止 Markdown 代码块", prompt)
        self.assertIn("数组字段建议 3-5 项", prompt)
        self.assertIn("5-6 个带 3.x 编号", prompt)
        self.assertIn("sections 与结构化字段相互一致", prompt)
        self.assertIn("full_description", prompt)
        self.assertIn("reproducibility", prompt)
        self.assertEqual(llm.kwargs[-1]["max_tokens"], 4096)

    def test_fallback_experiment_sections_can_drive_report_subsections(self):
        response = '''{
  "research_hypothesis": "H1: compressed model preserves planning quality.",
  "objectives": ["verify planning quality", "measure inference speed"],
  "metrics": ["Planning accuracy", "Latency"],
  "baselines": ["Teacher model", "Uncompressed student"],
  "procedure": ["train", "compress", "evaluate"],
  "sections": [
    {"heading": "3.1 实验假设与目标", "body": "本节提出假设 H1，并围绕规划质量保持和推理速度提升两个目标展开验证。"},
    {"heading": "3.2 评估指标", "body": "评估采用 Planning accuracy 衡量规划质量，并采用 Latency 衡量部署效率。"},
    {"heading": "3.3 基线方法", "body": "基线包括 Teacher model 与 Uncompressed student，用于比较压缩前后的性能变化。"}
  ]
}'''
        agent = build_agent_with_llm(FakeLLM(response))
        result = agent.design_experiment("model compression", {"top_papers": []})
        agent._expand_section = lambda prompt, min_words=200: "generated body"
        agent._build_abstract = lambda query, report_body: "final abstract"

        report = agent._build_rich_report(
            "model compression",
            {
                "literature_result": {"literature_review": "lit review", "top_papers": []},
                "experiment_design": result,
                "code_result": {},
                "analysis_result": {},
            },
        )
        sections = section_map(report)

        self.assertIn("H1", sections["3.1 实验假设与目标"])
        self.assertIn("规划质量保持", sections["3.1 实验假设与目标"])
        self.assertIn("Planning accuracy", sections["3.2 评估指标"])
        self.assertIn("Latency", sections["3.2 评估指标"])
        self.assertIn("Teacher model", sections["3.3 基线方法"])
        self.assertIn("Uncompressed student", sections["3.3 基线方法"])

    def test_short_upstream_experiment_sections_are_enriched_from_fields(self):
        agent = build_agent()
        agent._expand_section = lambda prompt, min_words=200: "generated body"
        agent._build_abstract = lambda query, report_body: "final abstract"

        report = agent._build_rich_report(
            "model compression",
            {
                "literature_result": {"literature_review": "lit review", "top_papers": []},
                "experiment_design": {
                    "research_hypothesis": "H1: compressed model keeps accuracy.",
                    "objectives": ["Compare accuracy", "Measure latency"],
                    "metrics": ["mAP", "FPS"],
                    "baselines": ["Teacher", "INT8"],
                    "sections": [
                        {"heading": "3.1 实验假设与目标", "body": "H1."},
                        {"heading": "3.2 评估指标", "body": "1. mAP\n2. FPS"},
                        {"heading": "3.3 基线方法", "body": "1. Teacher\n2. INT8"},
                    ],
                },
                "code_result": {},
                "analysis_result": {},
            },
        )
        sections = section_map(report)

        self.assertEqual("H1.", sections["3.1 实验假设与目标"])
        self.assertIn("mAP", sections["3.2 评估指标"])
        self.assertIn("Teacher", sections["3.3 基线方法"])

    def test_truncated_experiment_json_is_salvaged_without_raw_json_in_report(self):
        response = '''```json
{
  "research_hypothesis": "H1: compressed model keeps planning accuracy.",
  "objectives": ["compare accuracy", "measure latency"],
  "baselines": ["Teacher model", "INT8 quantization"],
  "metrics": ["mAP", "FPS"],
  "procedure": ["train teacher", "compress student"
'''
        agent = build_agent_with_llm(FakeLLM(response))

        result = agent.design_experiment("model compression", {"top_papers": []})
        description = agent._compose_experiment_description(
            "model compression",
            {"experiment_design": result},
        )

        self.assertEqual(result["_source"], "evaluation_agent_generated")
        self.assertIn("compressed model keeps planning accuracy", result["research_hypothesis"])
        self.assertEqual(result["metrics"], ["mAP", "FPS"])
        self.assertEqual(result["baselines"], ["Teacher model", "INT8 quantization"])
        self.assertNotIn("```json", description)
        self.assertNotIn('"research_hypothesis"', description)
        self.assertIn("本实验以", description)
        self.assertIn("具体评估指标与基线方法分别见 3.2 和 3.3", description)

    def test_report_generation_prompts_reuse_upstream_outputs(self):
        agent = build_agent()
        prompts = []

        def fake_expand_section(prompt, min_words=200):
            prompts.append(prompt)
            return f"section-{len(prompts)}"

        agent._expand_section = fake_expand_section
        agent._build_abstract = lambda query, report_body: "final abstract"

        outputs = {
            "literature_result": {
                "literature_review": "上游综述指出现有方法在鲁棒性和泛化性方面仍存在明显不足。",
                "top_papers": [
                    {
                        "title": "Paper A",
                        "structured_summary": {
                            "method": "使用对比学习增强表征",
                            "conclusion": "在小样本场景下效果更稳定",
                        },
                    }
                ],
            },
            "experiment_design": {
                "research_hypothesis": "引入结构化先验后可提升模型稳定性。",
                "dataset": "Dataset-X，含训练集、验证集和测试集。",
                "baselines": ["Baseline-1", "Baseline-2"],
                "metrics": ["Accuracy", "F1-score"],
                "procedure": ["训练主模型", "与基线比较", "误差分析"],
                "expected_results": "预期在 Accuracy 和 F1-score 上均优于基线。",
                "full_description": "",
            },
            "code_result": {
                "success": True,
                "final_code": "def train_model():\n    return 'ok'\n",
                "stdout": "Accuracy=0.91; F1=0.88",
            },
            "analysis_result": {
                "conclusion": "实验结果表明该方法在核心指标上优于基线。",
                "charts": ["reports/chart_accuracy.png"],
            },
        }

        report = agent._build_rich_report("test query", outputs)

        self.assertEqual(report["abstract"], "final abstract")
        self.assertEqual(len(prompts), 5)
        self.assertTrue(any("上游综述指出现有方法" in prompt for prompt in prompts))
        self.assertTrue(any("Paper A" in prompt for prompt in prompts))
        self.assertTrue(any("Baseline-1" in prompt and "F1-score" in prompt for prompt in prompts))
        self.assertTrue(any("reports/chart_accuracy.png" in prompt for prompt in prompts))
        self.assertTrue(any("def train_model" in prompt for prompt in prompts))
        self.assertTrue(any("结构化先验" in prompt for prompt in prompts))

    def test_experiment_subsections_are_rendered_from_design_fields_only(self):
        agent = build_agent()
        agent._expand_section = lambda prompt, min_words=200: "generated body"
        agent._build_abstract = lambda query, report_body: "final abstract"

        report = agent._build_rich_report(
            "model compression",
            {
                "literature_result": {"literature_review": "lit review", "top_papers": []},
                "experiment_design": {
                    "research_hypothesis": "H1: distilled student keeps perception accuracy.",
                    "objectives": ["Compare student accuracy with teacher.", "Measure inference speed."],
                    "metrics": ["mAP", "FPS"],
                    "baselines": ["Vanilla student", "FitNets"],
                    "full_description": "Detailed experiment design from the experiment stage.",
                },
                "code_result": {},
                "analysis_result": {},
            },
        )
        sections = section_map(report)

        self.assertIn("研究假设展开", sections["3.1 实验假设与目标"])
        self.assertIn("Compare student accuracy with teacher.", sections["3.1 实验假设与目标"])
        self.assertIn("Measure inference speed.", sections["3.1 实验假设与目标"])
        self.assertIn("评估指标设计为连接研究假设与实验结论", sections["3.2 评估指标"])
        self.assertIn("mAP", sections["3.2 评估指标"])
        self.assertIn("FPS", sections["3.2 评估指标"])
        self.assertIn("基线设计用于建立压缩方法的比较参照", sections["3.3 基线方法"])
        self.assertIn("Vanilla student", sections["3.3 基线方法"])
        self.assertIn("FitNets", sections["3.3 基线方法"])

    def test_experiment_overview_is_richer_without_repeating_metric_and_baseline_lists(self):
        agent = build_agent()
        agent._expand_section = lambda prompt, min_words=200: "generated body"
        agent._build_abstract = lambda query, report_body: "final abstract"

        report = agent._build_rich_report(
            "model compression",
            {
                "literature_result": {"literature_review": "lit review", "top_papers": []},
                "experiment_design": {
                    "research_hypothesis": "H1: compressed model keeps perception accuracy.",
                    "objectives": ["Compare accuracy", "Measure latency", "Analyze compression ratio"],
                    "dataset": "nuScenes validation split.",
                    "metrics": ["mAP", "FPS"],
                    "baselines": ["Teacher", "INT8"],
                    "procedure": ["train teacher", "compress student", "evaluate on validation split"],
                },
                "code_result": {"success": False, "error": "RuntimeError"},
                "analysis_result": {},
            },
        )
        overview = section_map(report)["三、实验设计与方法论"]

        self.assertIn("设计逻辑：", overview)
        self.assertIn("评价组织：", overview)
        self.assertIn("实验组织：", overview)
        self.assertNotIn("当前边界：", overview)
        self.assertNotIn("设计边界说明：", overview)
        appendix = section_body_contains(report, "证据边界与执行状态说明")
        self.assertIn("代码执行状态：失败", appendix)
        self.assertIn("设计边界说明", appendix)
        self.assertNotIn("评估指标包括：\n1. mAP", overview)
        self.assertNotIn("基线方法包括：\n1. Teacher", overview)

    def test_experiment_overview_converts_json_blob_to_natural_language(self):
        agent = build_agent()
        agent._expand_section = lambda prompt, min_words=200: "generated body"
        agent._build_abstract = lambda query, report_body: "final abstract"

        report = agent._build_rich_report(
            "model compression",
            {
                "literature_result": {"literature_review": "lit review", "top_papers": []},
                "experiment_design": {
                    "research_hypothesis": "H1: compressed model keeps accuracy.",
                    "metrics": ["mAP", "FPS"],
                    "baselines": ["Teacher", "INT8"],
                    "full_description": '```json\n{"research_hypothesis": "bad raw json"}\n```',
                },
                "code_result": {},
                "analysis_result": {},
            },
        )

        overview = section_map(report)["三、实验设计与方法论"]
        self.assertNotIn("```json", overview)
        self.assertNotIn('"research_hypothesis"', overview)
        self.assertIn("本实验以", overview)
        self.assertIn("具体评估指标与基线方法分别见 3.2 和 3.3", overview)

    def test_report_context_salvages_embedded_experiment_json(self):
        agent = build_agent()
        agent._expand_section = lambda prompt, min_words=200: "generated body"
        agent._build_abstract = lambda query, report_body: "final abstract"

        report = agent._build_rich_report(
            "model compression",
            {
                "literature_result": {"literature_review": "lit review", "top_papers": []},
                "experiment_design": {
                    "full_description": '''```json
{
  "research_hypothesis": "H1: compressed model preserves task accuracy.",
  "metrics": ["mAP", "FPS"],
  "baselines": ["Teacher", "INT8"],
  "procedure": ["train", "evaluate"
'''
                },
                "code_result": {},
                "analysis_result": {},
            },
        )

        sections = section_map(report)
        self.assertNotIn("```json", sections["三、实验设计与方法论"])
        self.assertNotIn('"research_hypothesis"', sections["三、实验设计与方法论"])
        self.assertIn("H1: compressed model preserves task accuracy.", sections["3.1 实验假设与目标"])
        self.assertIn("评估指标设计为连接研究假设与实验结论", sections["3.2 评估指标"])
        self.assertIn("mAP", sections["3.2 评估指标"])
        self.assertIn("FPS", sections["3.2 评估指标"])
        self.assertIn("基线设计用于建立压缩方法的比较参照", sections["3.3 基线方法"])
        self.assertIn("Teacher", sections["3.3 基线方法"])
        self.assertIn("INT8", sections["3.3 基线方法"])

    def test_report_stage_does_not_generate_missing_experiment_design_with_llm(self):
        agent = build_agent()
        prompts = []
        agent._expand_section = lambda prompt, min_words=200: prompts.append(prompt) or "generated body"
        agent._build_abstract = lambda query, report_body: "final abstract"

        report = agent._build_rich_report(
            "model compression",
            {
                "literature_result": {},
                "experiment_design": {},
                "code_result": {},
                "analysis_result": {},
            },
        )

        sections = section_map(report)
        self.assertIn("研究假设、评价对象、实验流程和对照方法", sections["三、实验设计与方法论"])
        self.assertIn("实验设计来源说明", sections["附录：证据边界与执行状态说明"])
        self.assertFalse(any("完整实验方案" in prompt or "实验设计与方法设计章节" in prompt for prompt in prompts))

    def test_missing_experiment_subsections_are_not_generated_in_report_stage(self):
        agent = build_agent()
        agent._expand_section = lambda prompt, min_words=200: "generated body"
        agent._build_abstract = lambda query, report_body: "final abstract"

        report = agent._build_rich_report(
            "model compression",
            {
                "literature_result": {},
                "experiment_design": {"full_description": "Detailed experiment design from the experiment stage."},
                "code_result": {},
                "analysis_result": {},
            },
        )
        sections = section_map(report)

        self.assertIn("研究假设与实验目标", sections["3.1 实验假设与目标"])
        self.assertIn("任务性能、运行效率和资源占用", sections["3.2 评估指标"])
        self.assertIn("原始模型、主流方法和关键组件变体", sections["3.3 基线方法"])

    def test_failed_code_execution_changes_result_and_conclusion_tone(self):
        agent = build_agent()
        agent._expand_section = lambda prompt, min_words=200: "generated body"
        agent._build_abstract = lambda query, report_body: "final abstract"

        report = agent._build_rich_report(
            "autonomous driving model compression",
            {
                "literature_result": {"literature_review": "literature context", "top_papers": []},
                "experiment_design": {
                    "research_hypothesis": "H1: the compressed student remains accurate.",
                    "metrics": ["mAP", "FPS"],
                    "baselines": ["Vanilla student", "FitNets"],
                    "procedure": ["train", "evaluate"],
                    "expected_results": "The student should be smaller and faster.",
                    "full_description": "Detailed experiment design from the experiment stage.",
                },
                "code_result": {
                    "success": False,
                    "final_code": "print('experiment')",
                    "error": "RuntimeError: dataset not found",
                    "stdout": "",
                },
                "analysis_result": {"charts": []},
            },
        )

        result_body = section_body_contains(report, "实验结果")
        conclusion_body = section_body_contains(report, "结论")
        method_body = section_body_contains(report, "核心方法")
        appendix = section_body_contains(report, "证据边界与执行状态说明")

        self.assertNotIn("执行状态说明", method_body)
        self.assertIn("代码执行模块返回失败状态", result_body)
        self.assertNotIn("不能被解释为已经获得的实验结论", result_body)
        self.assertIn("代码执行模块当前返回失败状态", conclusion_body)
        self.assertIn("不能将设计目标或预期结果表述为已经得到验证", conclusion_body)
        self.assertIn("代码执行状态：失败", appendix)


class TestEvaluationAgentHybridScoring(unittest.TestCase):
    def test_hybrid_scoring_penalizes_ungrounded_report_claims(self):
        agent = build_agent()
        agent.judge = FakeJudge([EvalScore(0.92, 0.92, 0.92, 0.92, "LLM liked it", False, "")])
        report = {
            "title": "Test report",
            "abstract": "summary",
            "sections": [
                {"heading": "1. Background", "body": "已有研究表明该方向已经取得显著进展。"},
                {"heading": "2. Literature", "body": "大量研究支持本文方法。"},
                {"heading": "3. Experiment", "body": "experiment overview"},
                {"heading": "3.1 Hypothesis", "body": "invented hypothesis"},
                {"heading": "3.2 Metrics", "body": "mAP\nFPS"},
                {"heading": "3.3 Baselines", "body": "invented baseline"},
                {"heading": "4. Method", "body": "method"},
                {"heading": "5. Results", "body": "实验结果表明显著提升 10% mAP and FPS."},
                {"heading": "6. Conclusion", "body": "conclusion"},
            ],
        }
        outputs = {
            "literature_result": {"total_found": 0, "top_papers": []},
            "experiment_design": {
                "research_hypothesis": "H1 from upstream design",
                "metrics": [],
                "baselines": [],
            },
            "code_result": {"success": False, "stdout": "", "error": "runtime error"},
            "analysis_result": {},
        }

        score, breakdown = agent._score_report("query", report, outputs)

        self.assertEqual(breakdown["method"], "hybrid_rule_multi_agent_llm")
        self.assertLess(score.overall, 0.75)
        self.assertTrue(score.needs_reflection)
        self.assertIn("综合评审意见", score.feedback)
        self.assertIn("各评审角色简要意见", score.feedback)
        self.assertIn("LLM 专家评审分", score.feedback)
        self.assertIn("不等同于各专家分数的简单平均", score.feedback)
        self.assertIn("review_summary", breakdown)
        self.assertEqual(len(breakdown["review_summary"]["role_reviews"]), 1)
        self.assertEqual(breakdown["display_scores"]["llm_expert_score"], 0.92)
        self.assertEqual(breakdown["display_scores"]["rule_consistency_score"], score.overall)
        self.assertEqual(breakdown["scores"]["evidence_grounding"], 0.45)
        self.assertLessEqual(breakdown["scores"]["execution_validity"], 0.35)
        self.assertTrue(any(item["severity"] == "hard" for item in breakdown["findings"]))

    def test_hybrid_scoring_rewards_consistent_upstream_usage(self):
        agent = build_agent()
        agent.judge = FakeJudge([EvalScore(0.82, 0.82, 0.82, 0.82, "solid", False, "")])
        report = {
            "title": "Test report",
            "abstract": "summary",
            "sections": [
                {"heading": "1. Background", "body": "grounded background"},
                {"heading": "2. Literature", "body": "Paper A supports the method."},
                {"heading": "3. Experiment", "body": "experiment overview"},
                {"heading": "3.1 Hypothesis", "body": "H1: student keeps accuracy."},
                {"heading": "3.2 Metrics", "body": "mAP\nFPS"},
                {"heading": "3.3 Baselines", "body": "Vanilla student\nFitNets"},
                {"heading": "4. Method", "body": "method"},
                {"heading": "5. Results", "body": "stdout shows mAP and FPS evidence."},
                {"heading": "6. Conclusion", "body": "conclusion"},
            ],
        }
        outputs = {
            "literature_result": {"total_found": 2, "top_papers": [{"title": "Paper A"}]},
            "experiment_design": {
                "research_hypothesis": "H1: student keeps accuracy.",
                "metrics": ["mAP", "FPS"],
                "baselines": ["Vanilla student", "FitNets"],
            },
            "code_result": {"success": True, "stdout": "mAP=0.41 FPS=32"},
            "analysis_result": {"conclusion": "The run produced measurable results.", "charts": ["chart.png"]},
        }

        score, breakdown = agent._score_report("query", report, outputs)

        self.assertGreater(score.overall, 0.80)
        self.assertFalse(score.needs_reflection)
        self.assertEqual(breakdown["upstream_status"]["literature_total_found"], 2)
        self.assertTrue(breakdown["upstream_status"]["code_success"])


class TestEvaluationReportRendering(unittest.TestCase):
    def test_markdown_renders_overall_and_role_review_comments(self):
        generator = ReportGenerator("outputs")

        markdown = generator._to_markdown(
            {
                "title": "Test report",
                "query": "query",
                "abstract": "summary",
                "sections": [],
                "evaluation": {
                    "overall_score": 0.72,
                    "llm_expert_score": 0.88,
                    "rule_consistency_score": 0.72,
                    "rule_dimension_scores": {
                        "evidence_grounding": 0.45,
                        "execution_validity": 0.35,
                    },
                    "accuracy": 0.70,
                    "completeness": 0.80,
                    "format_quality": 0.75,
                    "review_summary": {
                        "overall": "综合评审意见：整体可用，但证据仍需加强。",
                        "role_reviews": [
                            {
                                "reviewer": "证据一致性评审专家",
                                "overall_score": 0.68,
                                "brief": "需要补充文献和执行证据。",
                            }
                        ],
                    },
                },
            }
        )

        self.assertIn("### 综合评审意见", markdown)
        self.assertIn("整体可用，但证据仍需加强", markdown)
        self.assertIn("### 各评审角色简要意见", markdown)
        self.assertIn("证据一致性评审专家", markdown)
        self.assertIn("| 最终可交付评分 | 0.72 |", markdown)
        self.assertIn("| LLM 专家评审分 | 0.88 |", markdown)
        self.assertIn("### 规则评分维度", markdown)
        self.assertIn("| evidence_grounding | 0.45 |", markdown)


if __name__ == "__main__":
    unittest.main()
