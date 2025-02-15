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
