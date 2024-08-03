from os.path import join

from loguru import logger

from utils.constant import TRAIN, SEPARATOR
from utils.func import initial_args, create_trainer


def main():
    args, train_args = initial_args()
    # 加载trainer
    trainer = create_trainer(args, train_args)
    # 开始训练
    logger.info(f"{SEPARATOR} 开始训练 {SEPARATOR}")
    train_result = trainer.train()
    # 保存最好的checkpoint
    final_save_path = join(train_args.output_dir)
    trainer.save_model(final_save_path)
    logger.info(f"{SEPARATOR} 训练结束 {SEPARATOR}")
    # 保存训练指标
    metrics = train_result.metrics
    trainer.log_metrics(TRAIN, metrics)
    trainer.save_metrics(TRAIN, metrics)
    trainer.save_state()


if __name__ == "__main__":
    main()
