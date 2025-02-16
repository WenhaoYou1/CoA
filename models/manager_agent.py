def worker_agent(worker_id: int, prev_communication_unit: str, current_chunk: str, query: str, query_based: bool, model, tokenizer, max_new_tokens=256) -> str:
    """
    Implementation: pp. 19, Appendix B, Table 11 & Table 12.
    """
    if query_based:  
        prompt = (
            f"{current_chunk}\n"
            f"Here is the summary of the previous source text: {prev_communication_unit}\n"
            f"Question: {query}\n"
            "You need to read the current source text and summary of previous source text (if any) "
            "and generate a summary to include them both. Later, this summary will be used "
            "for other agents to answer the Query. So please write the summary that can include "
            "the evidence for answering the Query:\n"
        )
    else:  
        prompt = (
            f"{current_chunk}\n"
            f"Here is the summary of the previous source text:{prev_communication_unit}\n"
            "You need to read the current source text and summary of previous source text (if any) "
            "and generate a summary for the whole text. Thus, your generated summary should be "
            "relatively long.\n"
        )

    instruction_iw = ""  #TODO: replace with correct prompts    
    prompt = instruction_iw + "\n" + prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    generated_outputs = model.generate(
        inputs["input_ids"], 
        max_new_tokens=max_new_tokens, 
    )
    input_length = inputs["input_ids"].shape[1]
    generated_tokens = generated_outputs[0][input_length:] 
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return generated_text
