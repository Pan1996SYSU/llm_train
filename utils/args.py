from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class TemplateName(Enum):
    """
    目前支持的 template 类型
    """
    Qwen2 = 'Qwen2'
    Llama3 = 'Llama3'


class TrainMode(Enum):
    """
    目前支持的训练模式
    """
    # 量化LoRA微调
    Q_LoRA = 'Q_LoRA'
    # LoRA微调
    LoRA = 'LoRA'
    # 全量微调
    FULL = 'FULL'


class TrainArgPath(Enum):
    """
    目前支持的训练策略
    """
    SFT_LORA_Q_LORA_BASE = 'train_args/sft/sft_config.py'
    DPO_LORA_Q_LORA_BASE = 'train_args/dpo/dpo_config.py'


@dataclass
class CommonArgs:
    """
    常用参数
    """
    # --------------------------------------------------- 训练参数 ---------------------------------------------------
    train_args_path: TrainArgPath = field(default=TrainArgPath.SFT_LORA_Q_LORA_BASE.value,
                                          metadata={"help": "当前模式的训练策略: [SFT, DPO]"})
    max_len: int = field(default=1024, metadata={"help": "最大输入长度"})
    max_prompt_length: int = field(default=512, metadata={"help": "prompt最大长度"})
    train_data_path: Optional[str] = field(default=r"./data/sft_train.jsonl", metadata={"help": "训练集路径"})
    model_name_or_path: str = field(default=r"./model/qwen/1.5B", metadata={"help": "模型路径"})
    template_name: TemplateName = field(default=TemplateName.Qwen2.value, metadata={"help": "指定模型数据输入格式"})
    train_mode: TrainMode = field(default=TrainMode.Q_LoRA.value, metadata={"help": "训练模式: [Q_LoRA, LoRA, FULL]"})
    use_dora: bool = field(default=False, metadata={"help": "是否使用DoRA"})
    task_type: str = field(default="SFT", metadata={"help": "训练类型：[PRETRAIN, SFT, DPO_MULTI, DPO_SINGLE]"})
    # --------------------------------------------------- LoRA相关 ---------------------------------------------------
    lora_rank: Optional[int] = field(default=64, metadata={"help": "The rank size of lora intermediate linear"})
    lora_alpha: Optional[int] = field(default=16, metadata={"help": "The alpha parameter for Lora scaling"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "The dropout probability for Lora layers"})
    # --------------------------------------------------- 推理参数 ---------------------------------------------------
    # --------------------------------------------------- 合并参数 ---------------------------------------------------
