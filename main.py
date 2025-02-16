from models.manager_agent import manager_agent
from models.worker_agent import worker_agent
from utils import chunk_text_by_sentence

MODEL_NAME = "Qwen/Qwen-VL-Chat"
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
