"""Worker Agent Module for Summarization Tasks"""

from transformers import PreTrainedModel, PreTrainedTokenizer


class WorkerAgent:
    """A worker agent that processes text chunks and generates summaries."""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        max_new_tokens: int = 256
    ):
        """Initializes the WorkerAgent.

        Args:
            model (PreTrainedModel): The transformer model used for text generation.
            tokenizer (PreTrainedTokenizer): Tokenizer corresponding to the model.
            max_new_tokens (int, optional): Maximum tokens for generated output. Defaults to 256.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens

    def generate_summary(self, prev_summary: str, current_chunk: str, query: str = "") -> str:
        """Processes a text chunk and generates a summary using a given model.

        Args:
            prev_summary (str): Summary of the previous text chunk.
            current_chunk (str): The current text chunk.
            query (str, optional): If empty, generates general summary.

        Returns:
            str: The generated summary.
        """
        instruction = (
            "You are a Worker agent. You read a chunk of text and produce a summary. "
            "Then pass it to the next agent."
        )

        query_based = bool(query.strip())

        if query_based:
            prompt = (
                f"{current_chunk}\n"
                f"Here is the summary of the previous source text: {prev_summary}\n"
                f"Question: {query}\n"
                "You need to read the current source text and the summary of the previous text "
                "(if any) and generate a summary that includes both. Later, this summary will "
                "be used by other agents to answer the query. Ensure the summary includes "
                "evidence for answering the query."
            )
        else:
            prompt = (
                f"{current_chunk}\n"
                f"Here is the summary of the previous source text: {prev_summary}\n"
                "You need to read the current source text and the summary of the previous text "
                "(if any) and generate a summary for the entire text. "
                "The generated summary should be relatively long."
            )

        full_prompt = f"{instruction}\n{prompt}"
        # Tokenize the prompt
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.model.device)
        # Generate summary
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
        """Returns a string representation of the WorkerAgent instance."""
        return (
            f"WorkerAgent(model={self.model.__class__.__name__}, "
            f"tokenizer={self.tokenizer.__class__.__name__})"
        )
    