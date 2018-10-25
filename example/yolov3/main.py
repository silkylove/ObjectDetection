# -*- coding: utf-8 -*-
import os
import logging
from trainer import Trainer
from config import get_config

logger = logging.getLogger('InfoLog')

def main(config):
    logger.info('***START TRAINING YOLOV3 MODEL***')
    trainer = Trainer(config)
    trainer.train_and_test()


if __name__ == '__main__':
    config = get_config()
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpuid
    main(config)
