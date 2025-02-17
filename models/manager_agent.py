"""Manager Agent Module for Query-Based or General Answering Tasks"""

from transformers import PreTrainedModel, PreTrainedTokenizer


class ManagerAgent:
    """A manager agent that processes summarized text and generates responses."""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        max_new_tokens: int = 256
    ):
        """Initializes the ManagerAgent.

        Args:
            model (PreTrainedModel): The transformer model used for text generation.
            tokenizer (PreTrainedTokenizer): Tokenizer corresponding to the model.
            max_new_tokens (int, optional): Maximum tokens for generated output. Defaults to 256.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens

    def generate_response(
        self,
        final_communication_unit: str,
        query: str,
        query_based: bool,
        task_specific_requirement: str = ""
    ) -> str:
        """Generates a response based on summarized text.

        Args:
            final_communication_unit (str): The summarized text from worker agents.
            query (str): The query related to the text.
            query_based (bool): Whether the response should be query-specific.
            task_specific_requirement (str, optional): Additional task instructions. Defaults to "".

        Returns:
            str: The generated response.
        """
        if query_based:
            manager_prompt = (
                "The following are given passages. However, the source text is too long and "
                "has been summarized. You need to answer based on the summary:\n"
                f"{final_communication_unit}\n"
                f"Question: {query}\n"
                "Answer: "
            )
        else:
            manager_prompt = (
                "The following are given passages. However, the source text is too long and "
                "has been summarized. You need to answer based on the summary:\n"
                f"{final_communication_unit}\n"
                "Answer: "
            )

        full_prompt = f"{task_specific_requirement}\n{manager_prompt}"
        # Tokenize the prompt
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.model.device)
        # Generate response
        generated_outputs = self.model.generate(
            inputs["input_ids"],
            max_new_tokens=self.max_new_tokens
        )
        # Extract the generated text
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = generated_outputs[0][input_length:]
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        return generated_text

    def __repr__(self) -> str:
        """Returns a string representation of the ManagerAgent instance."""
        return (
            f"ManagerAgent(model={self.model.__class__.__name__}, "
            f"tokenizer={self.tokenizer.__class__.__name__})"
        )
    