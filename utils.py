import re

def chunk_text_by_sentence(x: str, query: str, k: int,  I_w: str, tokenizer) -> list:
    """
    Implementation: pp. 18, Appendix B, Algorithm 2.
    """
    sentences = re.split(r'(?<=[。！？\.\!\?])\s*', x.strip()) 
    sentences = [s.strip() for s in sentences if s.strip()]
    B = k - count_tokens(I_w, tokenizer) - count_tokens(query, tokenizer)

    C = []           
    c = ""            
    for s in sentences:
        if count_tokens(s, tokenizer) + count_tokens(c, tokenizer) > B:
            if c.strip():
                C.append(c.strip())
            c = s 
        else:
            if not c:
                c = s
            else:
                c = c + " " + s
    if count_tokens(c, tokenizer) != 0:
        C.append(c.strip())
    return C

def count_tokens(text: str, tokenizer) -> int:
    return len(tokenizer(text)["input_ids"])