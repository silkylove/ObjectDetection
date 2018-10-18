# -*- coding: utf-8 -*-
import os
import logging
import argparse

parse = argparse.ArgumentParser(description='ObjectDetection SSD')

parse.add_argument('--lr', default=1e-3, type=float)
parse.add_argument('--lr_decay', default=[100, 120, 140], type=list)
parse.add_argument('--lr_decay_rate', default=0.1, type=float)
parse.add_argument('--epochs', default=200, type=int)
parse.add_argument('--batch_size', default=16 * 3, type=int)
parse.add_argument('--distributed', default=True)
parse.add_argument('--gpuid', default='0,1,2')
parse.add_argument('--ckpt_dir', default='./checkpoint/')
parse.add_argument('--resume', default=False, help='resume from checkpoint')

parse.add_argument('--img_root', default='/home/yhuangcc/data/voc(07+12)/JPEGImages/')
parse.add_argument('--train_list', default=["/home/yhuangcc/ObjectDetection/datasets/voc/voc07_trainval.txt",
                                            "/home/yhuangcc/ObjectDetection/datasets/voc/voc12_trainval.txt"])
parse.add_argument('--test_list', default="/home/yhuangcc/ObjectDetection/datasets/voc/voc07_test.txt")
parse.add_argument('--label_file', default='/home/yhuangcc/ObjectDetection/datasets/voc/labels.txt')

log_dir = './log/'
parse.add_argument('--log_dir', default=log_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
logger = logging.getLogger("InfoLog")
logger.setLevel(level=logging.INFO)
handler = logging.FileHandler(log_dir + 'log.txt')
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(formatter)

logger.addHandler(handler)
logger.addHandler(console)


def get_config():
    config, unparsed = parse.parse_known_args()
    return config
