from dataclasses import dataclass, field
from typing import Dict


@dataclass
class Template:
    template_name: str = field(metadata={"help": ""})
    system_format: str = field(metadata={"help": ""})
    user_format: str
    assistant_format: str
    system: str
    stop_word: str


template_dict: Dict[str, Template] = {}


def register_template(template_name, system_format, user_format, assistant_format, system, stop_word=None):
    template_dict[template_name] = Template(
        template_name=template_name,
        system_format=system_format,
        user_format=user_format,
        assistant_format=assistant_format,
        system=system,
        stop_word=stop_word
    )


register_template(
    template_name='default',
    system_format='System: {content}\n',
    user_format='User: {content}\nAssistant: ',
    assistant_format='{content} {stop_token}',
    system=None,
    stop_word=None
)

register_template(
    template_name='Qwen2',
    system_format='<|im_start|>system\n{content}<|im_end|>\n',
    user_format='<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n',
    assistant_format='{content}<|im_end|>\n',
    system="You are a helpful assistant.",
    stop_word='<|im_end|>'
)

register_template(
    template_name='Llama3',
    system_format='<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>',
    user_format='<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|><|start_header_id|>'
                'assistant<|end_header_id|>\n\n',
    assistant_format='{content}<|eot_id|>',
    system=None,
    stop_word='<|eot_id|>'
)
