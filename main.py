import argparse
import json
import pathlib

from transformers import AutoTokenizer, AutoModelForCausalLM

from models.worker_agent import WorkerAgent
from models.manager_agent import ManagerAgent


# ---------------------------------------------------------------------------
# Core helper: run Worker chain on one paper
# ---------------------------------------------------------------------------
def run_on_sections(
    sections: list[str],
    worker: WorkerAgent,
    manager: ManagerAgent,
) -> str:
    cumulative_summary = ""
    for section_text in sections:
        print(f"Here is summary: {cumulative_summary}\n")
        cumulative_summary = worker.generate_summary(
            prev_summary=cumulative_summary,
            current_chunk=section_text,
        )
    return manager.generate_response(
        final_communication_unit=cumulative_summary,
    )

def main(args):
    # 1) 载入 LLM
    print(f"Loading model [{args.llm_name}] …")
    tokenizer = AutoTokenizer.from_pretrained(args.llm_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.llm_name,
        device_map=args.device_map,      # "auto" / "cpu" / "cuda:0" …
        trust_remote_code=True,
    )

    worker = WorkerAgent(model=model, tokenizer=tokenizer)
    manager = ManagerAgent(model=model, tokenizer=tokenizer)

    folder = pathlib.Path(args.folder).expanduser()
    json_files = sorted(folder.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found under {folder}")
    print(f"Found {len(json_files)} files under {folder}\n")

    for jf in json_files:
        with open(jf, "r", encoding="utf-8") as f:
            data = json.load(f)

        paper_id = data.get("id", jf.stem)
        title = data.get("title")
        sections = [v for k, v in data.items() if k not in ("id", "title")]

        review = run_on_sections(sections, worker, manager)

        print("=" * 100)
        print(f"Paper ID : {paper_id}")
        print(f"Title    : {title}")
        print("Official Review\n")
        print(review)
        print("\n") 


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chain-of-Agents academic reviewer")
    parser.add_argument("--folder", default="./processed_jsons", help="Folder containing per-paper JSON files")
    parser.add_argument("--llm_name", default="Qwen/Qwen-VL-Chat", help="HF model name or local path")
    parser.add_argument("--device_map", default="cpu", help='Device placement for model, e.g. "cpu", "auto", "cuda:0"')
    args = parser.parse_args()
    main(args)
