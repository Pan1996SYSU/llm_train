from loguru import logger
from transformers import HfArgumentParser

from utils.args import CommonArgs
from utils.constant import SEPARATOR
from utils.func import get_inference_model, get_messages, predict


def main():
    parser = HfArgumentParser((CommonArgs,))
    args = parser.parse_args_into_dataclasses()[0]
    model, tokenizer = get_inference_model(args)
    messages_list = get_messages(args)
    # 开始推理
    logger.info(f"{SEPARATOR} 开始推理 {SEPARATOR}")
    for i, messages in enumerate(messages_list):
        answer = predict(args, messages, model, tokenizer)
        logger.info(f"回答【{i}】: {answer}")
    logger.info(f"{SEPARATOR} 推理结束 {SEPARATOR}")


if __name__ == "__main__":
    main()
