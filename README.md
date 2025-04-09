# Mental-Health-Counselor-Qwen2.5-7B-Instruct

This is a fine-tuned language model built on top of [`Qwen2.5-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct), specifically trained for mental health counseling conversations. The model is designed to provide compassionate, informative, and safe responses to users seeking support for stress, anxiety, and other emotional challenges.

## 🔗 Model on Hugging Face

👉 [View on Hugging Face](https://huggingface.co/arafatanam/Mental-Health-Counselor-Qwen2.5-7B-Instruct)

## 🧠 Model Details

- **Base model:** `unsloth/Qwen2.5-7B-Instruct`
- **Fine-tuned on:** [Amod/mental_health_counseling_conversations](https://huggingface.co/datasets/Amod/mental_health_counseling_conversations)
- **Finetuning technique:** PEFT (Parameter-Efficient Fine-Tuning)
- **Language support:** Multilingual
- **License:** Apache 2.0

## 💬 Intended Use

This model is ideal for:

- Building empathetic mental health chatbots
- Conversational AI for self-care support
- Educational tools and university counseling assistants

**⚠️ Important:** This model is not a replacement for professional mental health care or medical diagnosis.

## 🚀 Quick Start

```python

!pip install -U torch transformers peft bitsandbytes

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig

# Base and adapter models
base_model = "Qwen/Qwen2.5-7B-Instruct"
adapter = "arafatanam/Mental-Health-Counselor-Qwen2.5-7B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True, padding_side='left')
config = PeftConfig.from_pretrained(adapter)
model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    load_in_4bit=True,
    device_map="auto",
    torch_dtype=torch.float16
)
model = PeftModel.from_pretrained(model, adapter)

# Inference
user_message = [{"role": "user", "content": "I'm feeling overwhelmed and anxious. How do I manage this?"}]
input_ids = tokenizer.apply_chat_template(user_message, tokenize=True, return_tensors='pt').to(model.device)
output = model.generate(input_ids, max_new_tokens=512, temperature=0.7)
response = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
print(response)
```

## 📁 Files Included

- `adapter_model.safetensors` – PEFT weights
- `adapter_config.json` – PEFT configuration
- Tokenizer files (`tokenizer.json`, `vocab.json`, `merges.txt`, etc.)

## 📜 License

This repository is licensed under the [Apache 2.0 License](./LICENSE).

## 🙋‍♂️ Author

Created by **Arafat Anam Chowdhury**

For questions or collaborations, feel free to connect!
