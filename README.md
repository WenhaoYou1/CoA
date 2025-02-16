## CoA

Implementation of [Chain of Agents](https://openreview.net/pdf?id=LuCLf4BJsr).

### Settings

The [paper](https://openreview.net/pdf?id=LuCLf4BJsr) uses commercial products such as Claude 3, while we conduct tests using the open-source pretrained large language model, [Qwen-VL-Chat](https://huggingface.co/Qwen/Qwen-VL-Chat).

### Structure

```sh
./
│
├── main.py
├── datasets
│   ├── <json, csv, ...>
│   └── README.md
├── utils.py
├── models/
│   ├── worker_agent.py
│   └── manager_agent.py
└── README.md
```

### Instruction

1. Download sample data.

   ```bash
    cd datasets
    wget http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_test_fullwiki_v1.json
    cd ..
   ```

2. Run the main section.

   ```bash
   python3 main.py \
       --json_file "./datasets/hotpot_test_fullwiki_v1.json" \
       --llm_name "Qwen/Qwen-VL-Chat" \
       --window_size 1024 \
       --query_based True
   ```
