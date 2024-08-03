import json
from typing import Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


class LLMInference:
    def __init__(self,
                 config: Optional[str] = None,
                 device: Optional[str] = None) -> None:
        # 创建一个空的字典，用于存放局部变量
        local_vars = {}

        # 执行文件
        with open(config, 'r', encoding="utf-8") as file:
            exec(file.read(), {}, local_vars)

        # 从局部变量中提取 model
        cfg = local_vars['model']
        tokenizer_config = cfg.get("tokenizer_config", {})
        tokenizer_path = tokenizer_config.get("tokenizer_path", "")
        if not tokenizer_path:
            raise f"123"
        use_fast = tokenizer_config.get("use_fast", False)
        trust_remote_code = tokenizer_config.get("use_fast", True)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path,
                                                       use_fast=use_fast,
                                                       trust_remote_code=trust_remote_code)
        model_config = cfg.get("model_config", {})
        model_path = model_config.get("model_path", "")
        if not model_path:
            raise f"123"
        device_map = model_config.get("device_map", "auto")
        torch_dtype = model_config.get("torch_dtype", torch.bfloat16)
        model = AutoModelForCausalLM.from_pretrained(model_path,
                                                     device_map=device_map,
                                                     torch_dtype=torch_dtype)
        model_id = model_config.get("model_id", "")
        if model_id:
            # 加载训练好的Lora模型，将下面的checkpointXXX替换为实际的checkpoint文件名名称
            self.model = PeftModel.from_pretrained(model, model_id=model_id)
        else:
            self.model = model
        self.device = device
        self.inference_config = cfg.get("inference_config", {})

    def __call__(self, corpus: Optional[str] = None) -> None:
        with open(corpus, 'r', encoding='utf-8') as file:
            for line in file:
                # 解析每一行的JSON数据
                data = json.loads(line)
                instruction = data['instruction']
                input_value = data['input']

                messages = [
                    {"role": "system", "content": f"{instruction}"},
                    {"role": "user", "content": f"{input_value}"}
                ]

                response = self.predict(messages, self.model, self.tokenizer)
                print(response)

    def predict(self, messages, model, tokenizer):
        tokenize = self.inference_config.get("tokenize", False)
        add_generation_prompt = self.inference_config.get("add_generation_prompt", True)
        return_tensors = self.inference_config.get("return_tensors", "pt")
        max_new_tokens = self.inference_config.get("max_new_tokens", 512)
        skip_special_tokens = self.inference_config.get("skip_special_tokens", True)

        text = tokenizer.apply_chat_template(messages, tokenize=tokenize, add_generation_prompt=add_generation_prompt)
        model_inputs = tokenizer([text], return_tensors=return_tensors).to(self.device)

        generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=max_new_tokens)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in
                         zip(model_inputs.input_ids, generated_ids)]
        answer = tokenizer.batch_decode(generated_ids, skip_special_tokens=skip_special_tokens)[0]

        return answer
