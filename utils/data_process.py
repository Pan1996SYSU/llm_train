import json

from loguru import logger
from torch.utils.data import Dataset


class CommonDataProcess(Dataset):
    def __init__(self, file, tokenizer, max_length, template):
        self.tokenizer = tokenizer
        self.template_name = template.template_name
        self.system_format = template.system_format
        self.user_format = template.user_format
        self.assistant_format = template.assistant_format
        self.system = template.system
        self.max_length = max_length

        logger.info(f'数据文件路径: {file}')
        with open(file, 'r', encoding='utf8') as f:
            data_list = f.readlines()
        logger.info(f'本次训练模板: {self.template_name}')
        logger.info(f"本次训练数量: {len(data_list)}")
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        data = self.data_list[item]
        data = json.loads(data)
        input_ids, target_mask = [], []

        if self.system_format is not None:
            system = self.system
            if system is not None:
                system_text = self.system_format.format(content=system)
                input_ids = self.tokenizer.encode(system_text, add_special_tokens=False)
                target_mask = [0] * len(input_ids)
        instruction = data['instruction']
        output = data['output']

        instruction_text = self.user_format.format(content=instruction, stop_token=self.tokenizer.eos_token)
        input_tokens = self.tokenizer.encode(instruction_text, add_special_tokens=False)
        output_text = self.assistant_format.format(content=output, stop_token=self.tokenizer.eos_token)
        output_tokens = self.tokenizer.encode(output_text, add_special_tokens=False)

        input_ids += input_tokens + output_tokens
        target_mask += [0] * len(input_tokens) + [1] * len(output_tokens)

        if len(input_ids) != len(target_mask):
            raise Exception("请检查训练数据")

        input_ids = input_ids[:self.max_length]
        target_mask = target_mask[:self.max_length]
        attention_mask = [1] * len(input_ids)
        if not len(input_ids) == len(target_mask) == len(attention_mask):
            raise Exception("请检查训练数据")
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "target_mask": target_mask
        }

        return inputs


class DpoDataset(Dataset):
    def __init__(self, file, tokenizer, max_length, max_prompt_length, template):
        self.tokenizer = tokenizer
        self.template_name = template.template_name
        self.system_format = template.system_format
        self.user_format = template.user_format
        self.assistant_format = template.assistant_format
        self.system = template.system

        self.max_length = max_length
        self.max_prompt_length = max_prompt_length
        self.max_seq_length = max_length

        logger.info(f'数据文件路径: {file}')
        with open(file, 'r', encoding='utf8') as f:
            data_list = f.readlines()
        logger.info(f'本次训练模板: {self.template_name}')
        logger.info(f"本次训练数量: {len(data_list)}")
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        data = self.data_list[item]
        data = json.loads(data)
        prompt = data['prompt']
        chosen = data['chosen']
        rejected = data['rejected']
        prompt = self.user_format.format(content=prompt, stop_token=self.tokenizer.eos_token)
        prompt_input_ids = []
        if self.system_format is not None:
            system = self.system
            if system is not None:
                system_text = self.system_format.format(content=system)
                input_ids = self.tokenizer.encode(system_text, add_special_tokens=False)
                prompt_input_ids = input_ids + self.tokenizer.encode(prompt, add_special_tokens=False)
        else:
            prompt_input_ids = self.tokenizer.encode(prompt, add_special_tokens=False)

        chosen = self.assistant_format.format(content=chosen, stop_token=self.tokenizer.eos_token)
        rejected = self.assistant_format.format(content=rejected, stop_token=self.tokenizer.eos_token)
        chosen_input_ids = self.tokenizer.encode(chosen, add_special_tokens=False)
        rejected_input_ids = self.tokenizer.encode(rejected, add_special_tokens=False)

        longer_response_length = max(len(chosen_input_ids), len(rejected_input_ids))
        if len(prompt_input_ids) + longer_response_length > self.max_seq_length:
            max_prompt_length = max(self.max_prompt_length, self.max_seq_length - longer_response_length)
            prompt_input_ids = prompt_input_ids[-max_prompt_length:]
        if len(prompt_input_ids) + longer_response_length > self.max_seq_length:
            chosen_input_ids = chosen_input_ids[: self.max_seq_length - len(prompt_input_ids)]
            rejected_input_ids = rejected_input_ids[: self.max_seq_length - len(prompt_input_ids)]

        chosen_labels = [-100] * len(prompt_input_ids) + chosen_input_ids
        chosen_input_ids = prompt_input_ids + chosen_input_ids
        rejected_labels = [-100] * len(prompt_input_ids) + rejected_input_ids
        rejected_input_ids = prompt_input_ids + rejected_input_ids
        if len(chosen_labels) != len(chosen_input_ids):
            raise Exception("请检查训练数据")
        if len(rejected_labels) != len(rejected_input_ids):
            raise Exception("请检查训练数据")

        inputs = dict(
            prompt_input_ids=prompt_input_ids,
            prompt_attention_mask=[1] * len(prompt_input_ids),
            chosen_input_ids=chosen_input_ids,
            chosen_attention_mask=[1] * len(chosen_input_ids),
            chosen_labels=chosen_labels,
            rejected_input_ids=rejected_input_ids,
            rejected_attention_mask=[1] * len(rejected_input_ids),
            rejected_labels=rejected_labels,
        )
        return inputs

    def map(self, func, **kwargs):
        return self
