"""
模块③ — 代码生成与执行 Agent
================================
「需求 → 代码 → 执行 → 错误 → 修复」闭环自动调试。
AST 静态安全检测 + 受限沙箱执行。
"""

import ast
import csv
import json
import os
import re
import urllib.error
import urllib.request
from datetime import datetime, timezone
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class CodeSession:
    """一次代码生成会话的完整记录"""
    task_id: str
    requirement: str
    iterations: list[dict] = field(default_factory=list)   # 每轮代码 + 执行结果
    final_code: str = ""
    final_result: Optional[dict] = None
    success: bool = False
    total_rounds: int = 0
    pass_at_1: bool = False      # 第一轮是否成功


class CodeAgent:
    """
    模块③ — 代码生成与执行 Agent
    """

    AGENT_NAME = "CodeAgent"

    def __init__(self, config, llm_client, code_executor, memory_store, logger):
        self.config = config
        self.llm = llm_client
        self.executor = code_executor
        self.memory = memory_store
        self.logger = logger
        self.max_rounds = config.code.max_debug_rounds
        self.dataset_dir = Path(__file__).resolve().parent.parent / "dataset"
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.debug_mode = os.getenv("MP_DEBUG", "0") == "1"

    def run(self, task_description: str, context: dict = None) -> dict:
        """
        主入口：生成并执行代码，自动调试
        """
        call = self.logger.start_call(self.AGENT_NAME, "code_generation", task_description)
        session = CodeSession(task_id=call.call_id, requirement=task_description)
        context = context or {}

        try:
            # Step 1: 准备实验数据集，并把路径带给后续代码生成
            self.logger.info(self.AGENT_NAME, "准备实验数据集...")
            dataset_info = self._prepare_dataset(task_description, context, session.task_id)
            self.logger.info(
                self.AGENT_NAME,
                f"数据集已保存: {dataset_info['csv_path']} | 来源: {dataset_info['source']}"
            )

            # Step 2: 初始代码生成（基于任务描述 + 本地数据集路径）
            self.logger.info(self.AGENT_NAME, "生成初始代码...")
            code = self._generate_code(task_description, context, dataset_info)

            quick_issues = self._quick_validate_code(code)
            if quick_issues:
                self.logger.warning(
                    self.AGENT_NAME,
                    f"初始代码快速校验失败: {quick_issues[0]}"
                )
                code = self._regenerate_clean_code(task_description, code)


            # Step 3: 执行 + 自动调试循环
            for round_num in range(1, self.max_rounds + 1):
                self.logger.info(self.AGENT_NAME, f"第 {round_num}/{self.max_rounds} 轮执行...")

                # ── AST 安全检测 ────────────────────────────────
                issues = self.executor.checker.check(code)
                if issues:
                    # 区分"语法错误"（代码提取失败）和"真正的安全问题"
                    syntax_issues   = [i for i in issues if "语法错误" in i or "SyntaxError" in i]
                    security_issues = [i for i in issues if "语法错误" not in i and "SyntaxError" not in i]

                    if syntax_issues:
                        # 语法错误：说明 extract_code 没能剥离 markdown 标记，
                        # 或 LLM 返回了非 Python 内容。
                        # 策略：再次强制提取 + 让 LLM 重新生成纯代码。
                        self.logger.warning(self.AGENT_NAME,
                            f"代码提取异常（语法错误），尝试重新提取并要求 LLM 输出纯代码...")
                        # 先尝试二次提取
                        code = self.executor.extract_code(code)
                        # 二次提取后再检测
                        issues2 = self.executor.checker.check(code)
                        if any("语法错误" in i for i in issues2):
                            # 仍有语法错误，要求 LLM 重新输出
                            code = self._regenerate_clean_code(task_description, code)

                    if security_issues:
                        # 真正的安全问题（危险函数调用等）
                        self.logger.warning(self.AGENT_NAME,
                            f"安全检测：{len(security_issues)} 个安全问题 → {security_issues[0]}")
                        code = self._fix_safety_issues(code, security_issues)

                # 执行代码（子进程模式，支持完整 import）
                exec_result = self.executor.execute_with_subprocess(code)
                iteration = {
                    "round": round_num,
                    "code": code,
                    "stdout": exec_result.stdout[:500],
                    "stderr": exec_result.stderr[:500],
                    "success": exec_result.success,
                    "duration": exec_result.execution_time,
                    "safety_issues": exec_result.safety_issues,
                }
                session.iterations.append(iteration)

                if exec_result.success:
                    session.success = True
                    session.pass_at_1 = (round_num == 1)
                    session.final_code = code
                    session.final_result = exec_result.to_dict()
                    self.logger.success(self.AGENT_NAME,
                        f"✓ 代码执行成功（第{round_num}轮，Pass@1={session.pass_at_1}）")
                    break
                else:
                    # 未到最后一轮才修复
                    if round_num < self.max_rounds:
                        self.logger.info(self.AGENT_NAME,
                            f"执行失败，自动调试... 错误: {exec_result.error_type}")
                        code = self._debug_code(
                            code, exec_result.stderr, exec_result.error_type, task_description
                        )
                    else:
                        self.logger.warning(self.AGENT_NAME,
                            f"达到最大调试轮数 ({self.max_rounds})，任务未完成")
                        session.final_code = code
                        session.final_result = exec_result.to_dict()

            session.total_rounds = len(session.iterations)

            # 单元测试自动生成
            test_code = self._generate_tests(session.final_code, task_description)

            # 存入记忆
            self.memory.add(
                content=f"代码任务: {task_description[:100]}",
                agent=self.AGENT_NAME,
                payload={"code": session.final_code[:500], "success": session.success},
                tags=["code"],
                importance=1.2 if session.success else 0.8,
            )

            result = {
                "success": session.success,
                "final_code": session.final_code,
                "stdout": session.final_result.get("stdout", "") if session.final_result else "",
                "total_rounds": session.total_rounds,
                "pass_at_1": session.pass_at_1,
                "iterations": session.iterations,
                "test_code": test_code,
                "dataset_path": dataset_info.get("csv_path", ""),
                "dataset_json_path": dataset_info.get("json_path", ""),
                "dataset_source": dataset_info.get("source", ""),
                "dataset_rows": dataset_info.get("row_count", 0),
            }
            self.logger.finish_call(call, result)
            self._print_session(session)
            return result

        except Exception as e:
            self.logger.fail_call(call, str(e))
            raise

    def _generate_code(self, requirement: str, context: dict = None, dataset_info: dict = None) -> str:
        """初始代码生成（基于任务描述 + 简要上下文）"""
        context = context or {}

        context_parts = []

        # 1. 文献方法摘要，只取少量，避免 prompt 太长
        top_papers = context.get("top_papers", [])
        if top_papers:
            methods = []
            for p in top_papers[:2]:
                summary = p.get("structured_summary", {}) if isinstance(p, dict) else {}
                method = summary.get("method", "")
                if method:
                    methods.append(method[:300])
            if methods:
                context_parts.append("参考文献方法摘要：\n" + "\n".join(f"- {m}" for m in methods))

        # 2. 实验设计摘要
        exp_design = context.get("exp_design", {})
        if isinstance(exp_design, dict) and exp_design:
            hypothesis = exp_design.get("research_hypothesis", "")
            full_desc = exp_design.get("full_description", "")
            if hypothesis:
                context_parts.append(f"实验假设：{hypothesis[:300]}")
            if full_desc:
                context_parts.append(f"实验设计说明：{full_desc[:500]}")

        # 3. baseline 和 metrics
        baselines = context.get("baselines", [])
        if baselines:
            context_parts.append(
                "实验对照/基线：\n" + "\n".join(f"- {str(b)[:150]}" for b in baselines[:3])
            )

        metrics = context.get("metrics", [])
        if metrics:
            context_parts.append(
                "评价指标：\n" + "\n".join(f"- {str(m)[:120]}" for m in metrics[:5])
            )

        context_text = "\n\n".join(context_parts)

        system = (
            "你是资深 Python 科研工程师。请根据任务描述和给定上下文生成高质量、可直接运行的 Python 代码。\n"
            "要求：\n"
            "1. 任务描述是主要依据，上下文只作为参考，不要输出文献综述或解释文字。\n"
            "2. 添加中文注释。\n"
            "3. 包含完整的错误处理。\n"
            "4. 代码必须是轻量级可运行示例，不要下载外部模型、数据集或权重文件。\n"
            "5. 不要依赖本地不存在的数据路径。\n"
            "6. 如果任务涉及 YOLO、Transformer、CLIP 等大型模型，请用合成数据或简化模型模拟核心优化思想。\n"
            "7. 最后必须 print 可供后续分析的关键数值结果，例如 loss、accuracy、latency、model_size、speedup 等。\n"
            "8. 建议按如下格式打印：\n"
            "print('ANALYSIS_RESULT_START')\n"
            "print('accuracy=0.91')\n"
            "print('loss=0.12')\n"
            "print('latency_ms=12.5')\n"
            "print('ANALYSIS_RESULT_END')\n"
            "9. 只输出纯 Python 代码，不要 markdown，不要解释。"
        )

        if context_text:
            prompt = f"任务描述：{requirement}\n\n参考上下文：\n{context_text}"
        else:
            prompt = f"任务描述：{requirement}"
        prompt = f"{prompt}{self._format_dataset_hint(dataset_info)}"
        resp = self.llm.chat_code([
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ])

        extracted = self.executor.extract_code(resp)

        self._save_debug_file("generate_raw.txt", str(resp))
        self._save_debug_file("generate_extracted.py", extracted)

        return extracted

    def _prepare_dataset(self, requirement: str, context: dict, task_id: str) -> dict:
        """优先使用真实公开数据，失败后让 LLM 生成本地 CSV/JSON 数据集。"""
        task_dir = self.dataset_dir / self._safe_slug(task_id)
        task_dir.mkdir(parents=True, exist_ok=True)

        dataset_spec = self._try_get_real_dataset(requirement, context)
        if dataset_spec:
            self.logger.info(
                self.AGENT_NAME,
                f"数据集来源: {dataset_spec.get('source')} ({dataset_spec.get('dataset_name')})"
            )
        else:
            self.logger.info(self.AGENT_NAME, "数据集来源: llm_generated")
            dataset_spec = self._generate_dataset_with_llm(requirement, context)

        if not dataset_spec:
            self.logger.info(self.AGENT_NAME, "数据集来源: local_fallback")
            dataset_spec = self._build_fallback_dataset(requirement)

        dataset_name = self._safe_slug(str(dataset_spec.get("dataset_name") or "mindpilot_dataset"))
        dataset_spec["dataset_name"] = dataset_name
        dataset_spec["generated_at"] = datetime.now(timezone.utc).isoformat()
        dataset_spec["requirement"] = requirement
        dataset_spec.setdefault("source", "unknown")

        rows = dataset_spec.get("rows") if isinstance(dataset_spec.get("rows"), list) else []
        if not rows:
            dataset_spec = self._build_fallback_dataset(requirement)
            dataset_name = dataset_spec["dataset_name"]
            rows = dataset_spec["rows"]

        json_path = task_dir / f"{dataset_name}.json"
        csv_path = task_dir / f"{dataset_name}.csv"
        self._write_dataset_files(dataset_spec, json_path, csv_path)

        column_names = self._dataset_columns(dataset_spec, rows)
        return {
            "dataset_name": dataset_name,
            "json_path": str(json_path),
            "csv_path": str(csv_path),
            "row_count": len(rows),
            "column_names": column_names,
            "summary": dataset_spec.get("description", ""),
            "preview": rows[:3],
            "source": dataset_spec.get("source", "unknown"),
            "source_url": dataset_spec.get("source_url", ""),
        }

    def _try_get_real_dataset(self, requirement: str, context: dict) -> dict:
        query_text = self._dataset_query_text(requirement, context)
        for dataset in self._ranked_public_datasets(query_text):
            try:
                rows = self._download_csv_rows(dataset["url"])
            except (OSError, TimeoutError, UnicodeError, urllib.error.URLError, csv.Error) as exc:
                self.logger.warning(self.AGENT_NAME, f"真实数据集下载失败({dataset['name']}): {exc}")
                continue
            if rows:
                return {
                    "dataset_name": dataset["name"],
                    "description": dataset["description"],
                    "columns": self._infer_columns(rows),
                    "rows": rows,
                    "source": "downloaded_public_dataset",
                    "source_url": dataset["url"],
                }

        builtin = self._load_builtin_dataset(query_text)
        if builtin:
            return builtin
        return {}

    def _generate_dataset_with_llm(self, requirement: str, context: dict) -> dict:
        dataset_desc = ""
        exp_design = context.get("exp_design") if isinstance(context, dict) else {}
        if isinstance(exp_design, dict):
            dataset_desc = exp_design.get("dataset", "")

        system = (
            "你是科研实验数据集生成器。请根据需求生成一个可直接保存为 CSV 和 JSON 的本地数据集。\n"
            "要求：\n"
            "1. 只输出严格 JSON，不要 Markdown，不要解释。\n"
            "2. JSON 需要包含 dataset_name、description、columns、rows。\n"
            "3. rows 中每一项都应是扁平键值对，至少包含 12 条样本。\n"
            "4. 数据必须自包含，不要依赖网络或外部下载。"
        )
        prompt = f"需求：{requirement}"
        if dataset_desc:
            prompt += f"\n实验设计中的数据集要求：{dataset_desc}"
        prompt += "\n请生成与上述需求直接相关的本地表格数据集。"

        try:
            resp = self.llm.chat([
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ])
        except Exception as exc:
            self.logger.warning(self.AGENT_NAME, f"LLM 数据集生成失败: {exc}")
            return {}

        data = self._parse_dataset_spec(resp)
        if data:
            data["source"] = "llm_generated"
        return data

    def _format_dataset_hint(self, dataset_info: dict) -> str:
        if not dataset_info:
            return ""
        preview = json.dumps(dataset_info.get("preview", []), ensure_ascii=False)
        return (
            "\n\n本地实验数据集已经准备完成，请优先读取该数据集，而不是假设外部输入或重新下载数据：\n"
            f"- 数据集名称：{dataset_info.get('dataset_name', '')}\n"
            f"- CSV 路径：{dataset_info.get('csv_path', '')}\n"
            f"- JSON 路径：{dataset_info.get('json_path', '')}\n"
            f"- 来源：{dataset_info.get('source', '')}\n"
            f"- 列名：{', '.join(dataset_info.get('column_names', []))}\n"
            f"- 样本数：{dataset_info.get('row_count', 0)}\n"
            f"- 数据摘要：{dataset_info.get('summary', '')}\n"
            f"- 样例：{preview}\n"
        )

    def _dataset_query_text(self, requirement: str, context: dict) -> str:
        parts = [requirement]
        exp_design = context.get("exp_design") if isinstance(context, dict) else {}
        if isinstance(exp_design, dict):
            parts.append(exp_design.get("dataset", ""))
        for paper in (context.get("top_papers", []) if isinstance(context, dict) else [])[:3]:
            if isinstance(paper, dict):
                parts.append(str(paper.get("title", "")))
        return " ".join(part for part in parts if part).lower()

    def _ranked_public_datasets(self, query_text: str) -> list[dict]:
        catalog = [
            {
                "name": "iris_public_dataset",
                "url": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv",
                "description": "Iris 鸢尾花分类公开数据集。",
                "keywords": ["iris", "flower", "鸢尾", "分类", "classification", "species"],
            },
            {
                "name": "titanic_public_dataset",
                "url": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv",
                "description": "Titanic 乘客生存分析公开数据集。",
                "keywords": ["titanic", "survival", "泰坦尼克", "生存", "分类"],
            },
            {
                "name": "tips_public_dataset",
                "url": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv",
                "description": "餐厅消费与小费公开数据集。",
                "keywords": ["tips", "restaurant", "bill", "小费", "餐厅", "回归"],
            },
            {
                "name": "mpg_public_dataset",
                "url": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv",
                "description": "汽车燃油效率公开数据集。",
                "keywords": ["mpg", "car", "fuel", "汽车", "油耗", "回归"],
            },
        ]
        scored = []
        for dataset in catalog:
            score = sum(1 for key in dataset["keywords"] if key.lower() in query_text)
            if score:
                scored.append((score, dataset))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [dataset for _, dataset in scored[:2]]

    def _download_csv_rows(self, url: str, max_rows: int = 200) -> list[dict]:
        req = urllib.request.Request(url, headers={"User-Agent": "MindPilotAgent/1.0"})
        with urllib.request.urlopen(req, timeout=4) as resp:
            text = resp.read(2_000_000).decode("utf-8-sig")
        rows = []
        for row in csv.DictReader(text.splitlines()):
            clean = {str(k).strip(): self._coerce_cell(v) for k, v in row.items() if k}
            if any(value != "" for value in clean.values()):
                rows.append(clean)
            if len(rows) >= max_rows:
                break
        return rows

    def _load_builtin_dataset(self, query_text: str) -> dict:
        try:
            from sklearn import datasets as sklearn_datasets
        except Exception:
            return {}

        builtin = [
            ("iris_sklearn_dataset", "load_iris", "Iris 鸢尾花分类真实数据集。", ["iris", "鸢尾", "classification", "分类"]),
            ("wine_sklearn_dataset", "load_wine", "Wine 葡萄酒分类真实数据集。", ["wine", "葡萄酒", "classification", "分类"]),
            ("diabetes_sklearn_dataset", "load_diabetes", "Diabetes 糖尿病回归真实数据集。", ["diabetes", "糖尿病", "regression", "回归"]),
        ]
        for name, loader_name, desc, keywords in builtin:
            if not any(key.lower() in query_text for key in keywords):
                continue
            data = getattr(sklearn_datasets, loader_name)()
            rows = self._sklearn_rows(data)
            if rows:
                return {
                    "dataset_name": name,
                    "description": desc,
                    "columns": self._infer_columns(rows),
                    "rows": rows,
                    "source": "builtin_public_dataset",
                    "source_url": f"sklearn.datasets.{loader_name}",
                }
        return {}

    def _sklearn_rows(self, data, max_rows: int = 200) -> list[dict]:
        raw_feature_names = getattr(data, "feature_names", None)
        feature_names = list(raw_feature_names) if raw_feature_names is not None else []
        raw_target_names = getattr(data, "target_names", None)
        target_names = list(raw_target_names) if raw_target_names is not None else []
        values = getattr(data, "data", None)
        if values is None:
            return []
        rows = []
        for idx, vector in enumerate(values[:max_rows]):
            row = {}
            for col_idx, value in enumerate(vector):
                name = feature_names[col_idx] if col_idx < len(feature_names) else f"feature_{col_idx + 1}"
                row[self._safe_slug(str(name))] = self._python_scalar(value)
            target = getattr(data, "target", None)
            if target is not None:
                target_value = self._python_scalar(target[idx])
                row["target"] = target_value
                if isinstance(target_value, int) and 0 <= target_value < len(target_names):
                    row["target_name"] = str(target_names[target_value])
            rows.append(row)
        return rows

    def _parse_dataset_spec(self, text: str) -> dict:
        if not text:
            return {}
        cleaned = self.executor.extract_code(text)
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        raw = match.group(0) if match else cleaned
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        if not isinstance(data, dict):
            return {}
        rows = data.get("rows") or data.get("records") or []
        data["rows"] = rows if isinstance(rows, list) else []
        return data

    def _build_fallback_dataset(self, requirement: str) -> dict:
        topic = requirement.strip()[:40] or "mindpilot"
        rows = []
        for idx in range(1, 13):
            rows.append({
                "sample_id": idx,
                "task_text": f"{topic} - 样本 {idx}",
                "feature_a": round(idx * 1.25, 3),
                "feature_b": round((idx % 5) * 2.5 + 0.5, 3),
                "label": "train" if idx <= 8 else "test",
            })
        return {
            "dataset_name": "mindpilot_fallback_dataset",
            "description": f"用于任务「{topic}」的本地回退数据集。",
            "columns": self._infer_columns(rows),
            "rows": rows,
            "source": "local_fallback",
        }

    def _write_dataset_files(self, dataset_spec: dict, json_path: Path, csv_path: Path):
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(dataset_spec, f, ensure_ascii=False, indent=2)

        rows = dataset_spec.get("rows", [])
        fieldnames = self._dataset_columns(dataset_spec, rows)
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({name: self._csv_value(row.get(name)) for name in fieldnames})

    def _dataset_columns(self, dataset_spec: dict, rows: list[dict]) -> list[str]:
        columns = dataset_spec.get("columns") or []
        names = [col.get("name", "") if isinstance(col, dict) else str(col) for col in columns]
        names = [name for name in names if name]
        for row in rows:
            if isinstance(row, dict):
                for key in row.keys():
                    if key not in names:
                        names.append(key)
        return names

    def _infer_columns(self, rows: list[dict]) -> list[dict]:
        return [{"name": name, "type": "str", "description": f"数据字段 {name}"} for name in self._dataset_columns({}, rows)]

    def _coerce_cell(self, value):
        text = "" if value is None else str(value).strip()
        if text == "":
            return ""
        try:
            if re.fullmatch(r"[-+]?\d+", text):
                return int(text)
            if re.fullmatch(r"[-+]?(?:\d+\.\d*|\d*\.\d+)(?:[eE][-+]?\d+)?", text):
                return float(text)
        except ValueError:
            return text
        return text

    def _csv_value(self, value):
        if value is None:
            return ""
        if isinstance(value, (str, int, float, bool)):
            return value
        return json.dumps(value, ensure_ascii=False)

    def _python_scalar(self, value):
        return value.item() if hasattr(value, "item") else value

    @staticmethod
    def _safe_slug(text: str) -> str:
        slug = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip()).strip("._-")
        return slug[:80] or "mindpilot_dataset"

    def _debug_code(self, code: str, error: str, error_type: str, requirement: str) -> str:
        """根据错误信息自动修复代码"""
        system = (
            "你是 Python 调试专家。根据错误信息修复以下代码。"
            "只输出修复后的纯 Python 完整代码，不要 markdown，不要解释。"
        )
        prompt = (
            f"原始需求：{requirement}\n\n"
            f"错误类型：{error_type}\n"
            f"错误信息：{error[:600]}\n\n"
            f"当前代码：\n```python\n{code}\n```"
        )
        resp = self.llm.chat_code([
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ])

        extracted = self.executor.extract_code(resp)

        self._save_debug_file("debug_raw.txt", str(resp))
        self._save_debug_file("debug_extracted.py", extracted)

        return extracted

    def _fix_safety_issues(self, code: str, issues: list[str]) -> str:
        """修复安全问题"""
        system = (
            "你是代码安全专家。以下代码存在安全问题，请移除危险操作，"
            "替换为安全的等价实现。可以保留正常科研计算库（包括 PyTorch、NumPy、Pandas 等），"
            "只输出修复后的纯 Python 完整代码，不要 markdown，不要解释。"
        )
        prompt = f"安全问题：{'; '.join(issues)}\n\n代码：\n```python\n{code}\n```"
        resp = self.llm.chat_code([
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ])
        return self.executor.extract_code(resp)

    def _regenerate_clean_code(self, requirement: str, bad_code: str) -> str:
        """
        当 LLM 返回了带 markdown 标记或非 Python 内容时，
        明确要求 LLM 重新输出纯 Python 代码（无任何 markdown 格式）。
        """
        system = (
            "你是 Python 专家。请直接输出可运行的 Python 代码，"
            "【绝对不要】包含任何 markdown 标记（不要有 ```python 或 ``` ），"
            "不要有任何解释文字，只输出纯 Python 代码本身。"
            "第一行必须是 import 语句或注释，不能是 ``` 或其他标记。"
        )
        prompt = (
            f"需求：{requirement}\n\n"
            f"之前的输出包含了格式错误，请重新输出纯 Python 代码（不含任何 markdown）：\n"
            f"前次输出片段：{bad_code[:200]}"
        )
        resp = self.llm.chat_code([
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt}
        ])
        # 再次提取，双重保险
        return self.executor.extract_code(resp)

    def _generate_tests(self, code: str, requirement: str) -> str:
        """自动生成单元测试"""
        if not code or not code.strip():
            return ""
        system = (
            "为以下 Python 代码生成简单的单元测试（使用 unittest）。"
            "只测试核心逻辑，可使用代码本身依赖的常见科学计算/深度学习库（如 NumPy、Pandas、PyTorch）。"
            "只输出纯 Python 代码，不要 markdown，不要解释。"
        )
        resp = self.llm.chat_code([
            {"role": "system", "content": system},
            {"role": "user", "content": f"需求：{requirement}\n\n代码：\n```python\n{code[:800]}\n```"}
        ])
        return self.executor.extract_code(resp)

    def _print_session(self, session: CodeSession):
        print(f"\n{'━'*58}")
        print(f"  💻 代码生成会话 [{session.task_id}]")
        print(f"{'━'*58}")
        print(f"  需求: {session.requirement[:55]}")
        print(f"  状态: {'✓ 成功' if session.success else '✗ 失败'} | "
            f"轮次: {session.total_rounds} | Pass@1: {session.pass_at_1}")
        for it in session.iterations:
            icon = "✓" if it["success"] else "✗"
            print(f"  轮{it['round']}: {icon} [{it['duration']}s] "
                f"{'OK' if it['success'] else it['stderr'][:40]}")
        print(f"{'━'*58}\n")

    def _save_debug_file(self, filename: str, content: str):
        if not self.debug_mode:
            return

        debug_dir = "debug_outputs"
        os.makedirs(debug_dir, exist_ok=True)

        path = os.path.join(debug_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content if content is not None else "")

    def _quick_validate_code(self, code: str) -> list[str]:
        """对生成代码做轻量校验，返回问题列表"""
        issues = []

        if not code or not code.strip():
            issues.append("生成代码为空")
            return issues

        # 长度过短，通常说明生成失败或提取失败
        if len(code.strip()) < 50:
            issues.append("生成代码过短，疑似不完整")

        # 先检查语法
        try:
            ast.parse(code)
        except SyntaxError as e:
            issues.append(f"语法错误: {e}")
            return issues

        return issues
