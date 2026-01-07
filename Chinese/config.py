# config.py 全局配置（三脚本共用，仅需修改此处路径）
import torch
import os
from util import args
#DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu")
BERT_MODEL = "bert-base-chinese"
MAX_SEQ_LEN = 512
SEED = args.seed
CACHE_STEP = 1000
# ======== 替换为你的路径 ========
BASE_DATA_PATH = "fraud_base_data.pkl"  # 原始基础pkl路径
TYPO_FILE_PATH = "形近字.txt"    # 自定义形近字文件
SAVE_DIR = "./"                           # 保存在当前目录