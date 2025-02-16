def manager_agent(final_communication_unit: str, query: str, query_based: bool, model, tokenizer, max_new_tokens=256, task_specific_requirement="") -> str:
    """
    Implementation: pp. 19, Appendix B, Table 11 & Table 12.
    """
    if query_based:
        manager_prompt = (
            "The following are given passages. However, the source text is too long and has been summarized. "
            "You need to answer based on the summary:\n"
            f"{final_communication_unit}\n"
            f"Question: {query}\n"
            "Answer: "
        )
    else:
        manager_prompt = (
            "The following are given passages. However, the source text is too long and has been summarized. "
            "You need to answer based on the summary:\n"
            f"{final_communication_unit}\n"
            "Answer:"
        )

    manager_prompt = f"{task_specific_requirement}\n" + manager_prompt
    inputs = tokenizer(manager_prompt, return_tensors="pt").to(model.device)
    generated_outputs = model.generate(
        inputs["input_ids"], 
        max_new_tokens=max_new_tokens, 
    )
    input_length = inputs["input_ids"].shape[1]
    generated_tokens = generated_outputs[0][input_length:] 
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return generated_text
