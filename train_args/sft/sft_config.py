from dataclasses import dataclass, field
from typing import Optional, Union, List

from transformers import TrainingArguments, SchedulerType, IntervalStrategy
from transformers.training_args import OptimizerNames


@dataclass
class TrainArgument(TrainingArguments):
    """
    SFT 训练参数
    """
    output_dir: str = field(default='./output')
    num_train_epochs: int = field(default=1)
    per_device_train_batch_size: int = field(default=2)
    gradient_checkpointing: bool = field(default=True)
    gradient_accumulation_steps: int = field(default=16)
    learning_rate: float = field(default=2e-4)
    logging_steps: int = field(default=100)
    save_steps: int = field(default=500)
    eval_strategy: Union[IntervalStrategy, str] = field(default="no")
    save_strategy: Union[IntervalStrategy, str] = field(default="epoch")
    save_total_limit: Optional[int] = field(default=2, )
    lr_scheduler_type: Union[SchedulerType, str] = field(default="constant_with_warmup", )
    warmup_steps: int = field(default=10)
    optim: Union[OptimizerNames, str] = field(default='paged_adamw_32bit')
    seed: int = field(default=42)
    report_to: Optional[List[str]] = field(default='tensorboard')
    weight_decay: float = field(default=0.0)
    max_grad_norm: float = field(default=1.0)
    remove_unused_columns: Optional[bool] = field(default=False)
    bf16: bool = field(default=True)
    fp16: bool = field(default=False)
