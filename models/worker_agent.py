from typing import Optional

from transformers import PreTrainedModel, PreTrainedTokenizer

class WorkerAgent:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        max_new_tokens: int = 256,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
    def generate_summary(
        self,
        prev_summary: str,
        current_chunk: str,
        query: str = "",
    ) -> str:
        """
        Args
        ----
        prev_summary : str
            上一个 Worker 生成的累积摘要（第一段时为空串）。
        current_chunk : str
            当前要阅读的 section 原文。
        query : str
            如果是 query-based summarization，这里给出终端问题；
            若为空，则生成通用摘要。
        Returns
        -------
        str
            更新后的“累积摘要”，将传递给下一位 Worker（或 Manager）。
        """

        # ===== 通用系统指令（角色 + 风格）========================
        system_instruction = (
            "You are a **specialised academic reviewer agent**. "
            "Your task is to read exactly one section of a scientific paper, "
            "merge its key information with the cumulative summary provided so far, "
            "and output an updated, self-contained cumulative summary.\n\n"
            "### Style requirements\n"
            "1. Write in concise, formal academic English.\n"
            "2. Preserve important technical terminology exactly as in the original.\n"
            "3. Use bullet points where helpful, but keep the whole summary under "
            f"{self.max_new_tokens // 2} words.\n"
            "4. Avoid repetition—each fact should appear only once in the growing summary.\n"
            "5. Always keep evidence relevant to answering the final query (if any).\n"
        )

        # ===== 针对是否 Query-based 追加不同指令 ================
        if query.strip():
            # Query-based 场景
            task_instruction = (
                "### Your objective\n"
                f"- **Ultimate question**: {query}\n"
                "- Extract or paraphrase evidence (facts, results, equations, citations) "
                "from the current section if they help answer the question.\n"
                "- Merge that evidence with the previous summary so that downstream "
                "agents can answer the question without seeing the original text.\n"
            )
        else:
            # 通用摘要场景
            task_instruction = (
                "### Your objective\n"
                "- Produce an **updated cumulative summary** of the entire paper up to "
                "this section, emphasising new contributions introduced here.\n"
            )

        # ===== 组装完整 prompt ==================================
        prompt = (
            f"{system_instruction}"
            f"{task_instruction}\n"
            "### Input you must process\n"
            f"**Current section text:**\n{current_chunk}\n\n"
            f"**Cumulative summary so far (if any):**\n{prev_summary}\n\n"
            "### Your output\n"
            "Return ONLY the updated cumulative summary. Do not add explanations.\n"
        )

        # ===== LLM 推理 =========================================
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        generated = self.model.generate(
            inputs["input_ids"],
            max_new_tokens=self.max_new_tokens,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        # 仅截取新增部分
        new_tokens = generated[0][inputs["input_ids"].shape[1] :]
        new_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        return new_text.strip()

    # -----------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"WorkerAgent(model={self.model.__class__.__name__}, "
            f"tokenizer={self.tokenizer.__class__.__name__})"
        )
