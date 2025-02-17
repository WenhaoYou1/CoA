"""Utility functions for text processing and tokenization."""

import re


def chunk_text_by_sentence(text: str, query: str, k: int, instruction: str, tokenizer) -> list:
    """Splits text into sentence-based chunks while respecting token constraints.

    Implementation: pp. 18, Appendix B, Algorithm 2.

    Args:
        text (str): The input text to be chunked.
        query (str): The query text.
        k (int): The total token budget for a chunk.
        instruction (str): The initial instruction prompt.
        tokenizer: The tokenizer used to count tokens.

    Returns:
        list: A list of text chunks that fit within the token constraints.
    """
    sentences = re.split(r'(?<=[。！？\.\!\?])\s*', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    token_budget = (
        k - count_tokens(instruction, tokenizer) - count_tokens(query, tokenizer)
    )

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        new_token_count = count_tokens(sentence, tokenizer) + count_tokens(current_chunk, tokenizer)
        if new_token_count > token_budget:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            current_chunk = f"{current_chunk} {sentence}".strip() if current_chunk else sentence

    if count_tokens(current_chunk, tokenizer) != 0:
        chunks.append(current_chunk.strip())

    return chunks


def count_tokens(text: str, tokenizer) -> int:
    """Counts the number of tokens in a given text using the specified tokenizer.

    Args:
        text (str): The input text to tokenize.
        tokenizer: The tokenizer used to count tokens.

    Returns:
        int: The number of tokens in the text.
    """
    return len(tokenizer(text)["input_ids"])
