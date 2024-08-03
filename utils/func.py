import importlib
import os
from os.path import join

import bitsandbytes as bnb
import torch
import torch.nn as nn
from datasets import load_dataset
from loguru import logger
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, Trainer, \
    BitsAndBytesConfig, HfArgumentParser, set_seed
from trl import DPOTrainer

from utils.args import CommonArgs
from utils.data_collator import SftDataCollator
from utils.data_process import CommonDataProcess, DpoDataset
from utils.template import template_dict


def load_config(train_args_path):
    """
    根据训练配置文件获取训练配置对象
    @param train_args_path: 训练配置文件路径
    @return: 训练配置对象
    """
    module_path = train_args_path.replace("/", ".").rstrip(".py")
    # 动态导入训练模块
    module = importlib.import_module(module_path)
    class_name = "TrainArgument"
    train_argument = getattr(module, class_name)()
    return train_argument


def initial_args():
    """
    初始化参数配置
    @return: 常规参数和训练参数
    """
    parser = HfArgumentParser((CommonArgs,))
    args = parser.parse_args_into_dataclasses()[0]
    # 加载训练配置参数
    train_args = load_config(args.train_args_path)
    if not os.path.exists(train_args.output_dir):
        os.mkdir(train_args.output_dir)
    logger.add(join(train_args.output_dir, 'train.log'))
    logger.info(f"训练配置参数: \n{train_args}")
    logger.info(f"常规配置参数: \n{args}")
    set_seed(train_args.seed)
    if not sum([train_args.fp16, train_args.bf16]) == 1:
        raise Exception("请检查训练精度类型设置")
    return args, train_args


def find_all_linear_names(model, train_mode):
    """
    找出所有全连接层，为所有全连接添加adapter
    @param model: 模型
    @param train_mode: 训练模式: [Q_LoRA, LoRA, FULL]
    @return: LoRA目标模块名称
    """
    if train_mode not in ['LoRA', 'Q_LoRA']:
        raise Exception("请检查训练模式")
    cls = bnb.nn.Linear4bit if train_mode == 'Q_LoRA' else nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[-1])
    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')
    lora_module_names = list(lora_module_names)
    logger.info(f'LoRA目标模块名称: {lora_module_names}')
    return lora_module_names


def create_tokenizer(args):
    """
    初始化分词器对象
    @param args: 配置管理对象
    @return: 分词器对象
    """
    config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True,
                                              use_fast=False if config.model_type == 'llama' else True)
    if tokenizer.__class__.__name__ == 'QWenTokenizer':
        tokenizer.pad_token_id = tokenizer.eod_id
        tokenizer.bos_token_id = tokenizer.eod_id
        tokenizer.eos_token_id = tokenizer.eod_id
    if tokenizer.bos_token is None:
        tokenizer.add_special_tokens({"bos_token": tokenizer.eos_token})
        tokenizer.bos_token_id = tokenizer.eos_token_id
    if tokenizer.pad_token_id is None:
        raise Exception("请检查 tokenizer.pad_token_id")
    if tokenizer.eos_token_id is None:
        raise Exception("请检查 tokenizer.eos_token_id")
    logger.info(f'分词器词汇表大小: {tokenizer.vocab_size}')
    return tokenizer


def load_model(model_kwargs, model_name_or_path):
    """
    加载模型
    @param model_kwargs: 模型配置
    @param model_name_or_path: 模型路径
    @return: 模型
    """
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
    return model


def create_model(args, train_args):
    """
    初始化模型
    @param args: 常规配置
    @param train_args: 训练配置
    @return: 模型和配置
    """
    logger.info(f'基础模型地址: {args.model_name_or_path}')
    logger.info(f'训练模式: {args.train_mode}')
    torch_dtype = torch.float16 if train_args.fp16 else torch.bfloat16
    model_kwargs = dict(
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        use_cache=False if train_args.gradient_checkpointing else True,
    )

    if args.train_mode == 'Q_LoRA':
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
        model_kwargs["quantization_config"] = quantization_config
        model = load_model(model_kwargs, args.model_name_or_path)
        if args.task_type in ['PRETRAIN', 'SFT']:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=train_args.gradient_checkpointing)

    elif args.train_mode == 'Q_LoRA':
        model = load_model(model_kwargs, args.model_name_or_path)
        if hasattr(model, 'enable_input_require_grads'):
            model.enable_input_require_grads()
    elif args.train_mode == 'FULL':
        model = load_model(model_kwargs, args.model_name_or_path)
    else:
        raise Exception("请检查模型训练模式")

    if args.train_mode == 'FULL':
        peft_config = None
    else:
        target_modules = find_all_linear_names(model, args.train_mode)
        peft_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=args.lora_dropout,
            task_type=TaskType.CAUSAL_LM,
            use_dora=args.use_dora
        )

    if args.train_mode in ['LoRA', 'Q_LoRA'] and args.task_type in ['PRETRAIN', 'SFT']:
        model = get_peft_model(model, peft_config)
        logger.info(f'模型内存占用: {round(model.get_memory_footprint() / (1024 * 1024 * 1024), 2)} GB')
        model.print_trainable_parameters()

    total = sum(p.numel() for p in model.parameters())
    logger.info(f"总模型参数量: {round((total / 1e6), 2)}")

    return {
        'model': model,
        'peft_config': peft_config,
    }


def load_sft_dataset(args, tokenizer):
    """
    加载SFT训练数据
    @param args: 配置参数
    @param tokenizer: 分词器
    @return: 训练数据集
    """
    if args.template_name not in template_dict.keys():
        raise Exception(f"对话模版名称不存在，请检查")
    template = template_dict[args.template_name]
    train_dataset = CommonDataProcess(args.train_data_path, tokenizer, args.max_len, template)
    return train_dataset


def load_dpo_dataset(args, tokenizer):
    """
    加载DPO训练数据
    @param args: 配置参数
    @param tokenizer: 分词器
    @return: 训练数据集
    """
    if args.template_name not in template_dict.keys():
        raise Exception(f"对话模版名称不存在，请检查")
    template = template_dict[args.template_name]
    if args.task_type == 'DPO_MULTI':
        if tokenizer.chat_template is None:
            tokenizer.chat_template = ("{% for message in messages %}{{message['role'] + ': ' + message['content'] + "
                                       "'\n\n'}}{% endfor %}{{ eos_token }}")
        train_dataset = load_dataset(data_files=args.train_data_path, path='json')

        def process(row):
            row["chosen"] = tokenizer.apply_chat_template(row["chosen"], tokenize=False)
            row["rejected"] = tokenizer.apply_chat_template(row["rejected"], tokenize=False)
            return row

        train_dataset = train_dataset.map(process)
        train_dataset = train_dataset['train']
        return train_dataset
    elif args.task_type == 'DPO_SINGLE':
        train_dataset = DpoDataset(args.train_data_path, tokenizer, args.max_len, args.max_prompt_length, template)
        return train_dataset


def create_trainer(args, train_args):
    """
    加载训练器
    @param args: 常规配置参数
    @param train_args: 训练配置参数
    @return: 训练器
    """
    tokenizer = create_tokenizer(args)
    model_dict = create_model(args, train_args)
    model = model_dict['model']
    peft_config = model_dict['peft_config']
    logger.info("正在加载训练数据")
    if args.task_type == 'SFT':
        train_dataset = load_sft_dataset(args, tokenizer)
        data_collator = SftDataCollator(tokenizer, args.max_len)
    elif args.task_type == 'DPO_MULTI' or args.task_type == 'DPO_SINGLE':
        train_dataset = load_dpo_dataset(args, tokenizer)
        data_collator = None
    else:
        raise Exception("请检查模型训练类型")
    logger.info("训练数据加载完成")
    logger.info(f'模型训练类型: {args.task_type}')

    if args.task_type == 'SFT' or args.task_type == 'PRETRAIN':
        trainer = Trainer(
            model=model,
            args=train_args,
            train_dataset=train_dataset,
            data_collator=data_collator
        )
    else:
        trainer = DPOTrainer(
            model,
            ref_model=None,
            args=train_args,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            peft_config=peft_config
        )
    return trainer
