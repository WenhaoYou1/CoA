import torch
import json
import argparse
from models.manager_agent import manager_agent
from models.worker_agent import worker_agent
from utils import chunk_text_by_sentence
from transformers import AutoTokenizer, AutoModelForCausalLM

INSTRUCTION_IW = (
    "You are a Worker agent. You read a chunk of the text and produce a summary. "
    "Then pass it to the next agent."
)

def chain_of_agents_pipeline(source_text: str, query: str, k: int, query_based: bool, model, tokenizer) -> str:
    chunks = chunk_text_by_sentence(
        x=source_text,
        query=query,
        k=k,
        I_w=INSTRUCTION_IW,
        tokenizer=tokenizer
    )

    # Stage 1: Worker Agents
    cu_prev = "" 
    for i, chunk_text in enumerate(chunks, start=1):
        # print(f"Worker {i}")
        cu_curr = worker_agent(
            worker_id=i,
            prev_communication_unit=cu_prev,
            current_chunk=chunk_text,
            query=query,
            query_based=query_based,
            model=model,
            tokenizer=tokenizer
        )
        # print(f"-游chunk: {chunk_text}")
        # print(f"-文query: {query}")
        # print(f"-灏answer: {cu_curr}")
        cu_prev = cu_curr  

    # Stage 2: Manager
    final_answer = manager_agent(
        final_communication_unit=cu_prev,
        query=query,
        query_based=query_based,
        model=model,
        tokenizer=tokenizer
    )
    return final_answer

def main():
    tokenizer = AutoTokenizer.from_pretrained(args.llm_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.llm_name, device_map="auto", trust_remote_code=True)

    # DATA...
    with open(args.json_file, "r", encoding="utf-8") as f:
        data = json.load(f) 
    # SAMPLE ...
    sample = data[1]
    question = sample["question"] 
    context_list = sample["context"] 

    source_text = ""
    for item in context_list:
        title, sentences_list = item
        source_text += title + ": "
        for sentence in sentences_list:
            source_text += sentence
       
    final_answer = chain_of_agents_pipeline(
        source_text=source_text,
        query=question,
        k=args.window_size,
        query_based=args.query_based,
        model=model,
        tokenizer=tokenizer
    )

    print("Source Text:", source_text[:200], "...")
    print("-------------------------")
    print("Question:", question)
    print("Answer", final_answer)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Chain-of-Agents on HotPotQA dataset using Qwen-VL-Chat")
    parser.add_argument("--json_file", type=str, default="./datasets/hotpot_test_fullwiki_v1.json", help="Path to the HotPotQA JSON file")
    parser.add_argument("--llm_name", type=str, default="Qwen/Qwen-VL-Chat", help="Name of the pretrained model")
    parser.add_argument("--window_size", type=int, default=1024, help="Window size for chunking text")
    parser.add_argument("--query_based", type=bool, dafult=True)
    
    args = parser.parse_args()
    main(args)