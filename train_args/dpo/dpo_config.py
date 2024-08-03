from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict

from transformers import TrainingArguments, SchedulerType, IntervalStrategy
from transformers.training_args import OptimizerNames


@dataclass
class TrainArgument(TrainingArguments):
    """
    DPO 训练参数
    """
    output_dir: str = field(default='./output')
    num_train_epochs: int = field(default=1)
    per_device_train_batch_size: int = field(default=1)
    gradient_checkpointing: bool = field(default=True)
    gradient_accumulation_steps: int = field(default=2)
    learning_rate: float = field(default=2e-4)
    logging_steps: int = field(default=100)
    save_steps: int = field(default=500)
    eval_strategy: Union[IntervalStrategy, str] = field(default="no")
    save_strategy: Union[IntervalStrategy, str] = field(default="epoch")
    save_total_limit: Optional[int] = field(default=2)
    lr_scheduler_type: Union[SchedulerType, str] = field(default="constant_with_warmup")
    warmup_steps: int = field(default=10)
    optim: Union[OptimizerNames, str] = field(default='paged_adamw_32bit')
    report_to: Optional[List[str]] = field(default='tensorboard')
    weight_decay: float = field(default=0.0)
    max_grad_norm: float = field(default=1.0)
    remove_unused_columns: Optional[bool] = field(default=False)
    bf16: bool = field(default=True)
    fp16: bool = field(default=False)
    beta: float = field(default=0.1)
    label_smoothing: float = field(default=0)
    loss_type: str = field(default="sigmoid")
    label_pad_token_id: int = field(default=-100)
    padding_value: int = field(default=0)
    truncation_mode: str = field(default="keep_end")
    max_length: Optional[int] = field(default=None)
    max_prompt_length: Optional[int] = field(default=None)
    max_target_length: Optional[int] = field(default=None)
    is_encoder_decoder: Optional[bool] = field(default=None)
    disable_dropout: bool = field(default=True)
    generate_during_eval: bool = field(default=False)
    precompute_ref_log_probs: bool = field(default=False)
    dataset_num_proc: Optional[int] = field(default=None)
    model_init_kwargs: Optional[Dict] = field(default=None)
    ref_model_init_kwargs: Optional[Dict] = field(default=None)
    model_adapter_name: Optional[str] = field(default=None)
    ref_adapter_name: Optional[str] = field(default=None)
    reference_free: bool = field(default=False)
    force_use_ref_model: bool = field(default=False)
    sync_ref_model: bool = field(default=False)
    ref_model_mixup_alpha: float = field(default=0.9)
    ref_model_sync_steps: int = field(default=64)
    rpo_alpha: Optional[float] = field(default=None)
    f_divergence_type: str = field(default="reverse_kl")
    f_alpha_divergence_coef: Optional[float] = field(default=1.0)
