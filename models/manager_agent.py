# models/manager_agent.py
from transformers import PreTrainedModel, PreTrainedTokenizer


class ManagerAgent:
    """
    A professional, experienced academic reviewer.
    It never sees the original paper; it relies exclusively on the
    cumulative summary produced by the chain of Worker agents and
    outputs an *official conference-style review* in a fixed template.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        max_new_tokens: int = 512,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
    def generate_response(
        self,
        final_communication_unit: str,
        task_specific_requirement: str = "",
    ) -> str:
        """
        Parameters
        ----------
        final_communication_unit : str
            The cumulative summary returned by the last Worker agent.
        query : str
            (Optional) A downstream question – if provided, the review should
            explicitly address it in the *Questions* or *Strengths* block.
        query_based : bool
            Kept for compatibility; not required by this template.
        task_specific_requirement : str
            Extra high-level constraints (e.g. length limits) injected by caller.
        """

        # ===== 1)  system-level instruction =================================
        system_inst = (
            "You are an **experienced peer reviewer** for top-tier ML conferences "
            "(NeurIPS / ICLR / ICML). "
            "Based *only* on the summary below, produce an official review.\n"
        )

        # ===== 2)  fixed template the model must fill =======================
        # We embed the template verbatim in the prompt so the LLM copies it.
        template = (
            "Official Review\n"
            "Summary: {summary}\n"
            "Soundness: {soundness}\n"
            "Presentation: {presentation}\n"
            "Contribution: {contribution}\n"
            "Strengths: {strengths}\n"
            "Weaknesses: {weaknesses}\n"
            "Questions: {questions}\n"
            "Rating: {rating}\n"
            "Confidence: {confidence}\n"
        )

        # ===== 3)  build the full prompt ====================================
        prompt = (
            f"{system_inst}"
            f"{task_specific_requirement}\n\n"
            "### Paper summary from Worker Agents\n"
            f"{final_communication_unit}\n\n"
            "### Your task\n"
            "Fill in every field in the template below. "
            "Keep each field concise but specific (≈ 2–3 sentences, "
            "bullet lists allowed for Strengths / Weaknesses). "
            "Use numbers where appropriate (e.g. *Rating: 7 / 10*).\n\n"
            f"{template}"
        )

        # ===== 4)  LLM inference ============================================
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            inputs["input_ids"],
            max_new_tokens=self.max_new_tokens,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        new_tokens = outputs[0][inputs["input_ids"].shape[1] :]
        review_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

