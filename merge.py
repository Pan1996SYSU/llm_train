from loguru import logger
from transformers import HfArgumentParser

from utils.args import CommonArgs
from utils.constant import SEPARATOR
from utils.func import merge_model


def main():
    parser = HfArgumentParser((CommonArgs,))
    args = parser.parse_args_into_dataclasses()[0]
    tokenizer, lora_model = merge_model(args)
    # 开始合并
    logger.info(f"{SEPARATOR} 开始合并 {SEPARATOR}")
    model = lora_model.merge_and_unload()
    merge_output_dir = args.merge_output_dir
    if merge_output_dir:
        model.save_pretrained(merge_output_dir)
        tokenizer.save_pretrained(merge_output_dir)
    logger.info(f"{SEPARATOR} 合并结束 {SEPARATOR}")


if __name__ == "__main__":
    main()
