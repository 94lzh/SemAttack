import joblib
import torch
import jieba
import warnings
import re
from torch.utils.data import Dataset
from tqdm import tqdm

from util import logger, root_dir, args
from pytorch_transformers import BertTokenizer
from config import *

# ===================== 全局配置 =====================
warnings.filterwarnings('ignore')
import nltk
import ssl

ssl._create_default_https_context = ssl._create_unverified_context
try:
    nltk.data.find('omw-1.4')
except LookupError:
    nltk.download('omw-1.4', quiet=True)
from nltk.corpus import wordnet as wn
from collections import Counter

# ✅ 关键配置：取消截断+文本清洗
NO_MAX_LEN = True  # 关闭长度限制，动态适配文本
CLEAN_PATTERN = r'(left:|right:|\n|\s+|:)'  # 清洗格式符
# ✅ 标点符号黑名单（强制标记为0）
PUNCTUATIONS = {"，", "。", "？", "！", "：", "；", "、", "“", "”", "（", "）", "《", "》", ".", "?", "!", ";", ":"}


# ===================== 数据集类（精准修复版） =====================
class FraudAttackDataset(Dataset):
    def __init__(self, path):
        cache_path = 'FT_FC_' + path
        save_path = 'all_' + path
        self.data = joblib.load(cache_path)
        knowledge_data = []
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
        self.word_list = joblib.load(args.word_list)
        self.word_list_set = set(self.word_list)

        print("=" * 80)
        print("启动【精准修复版】数据处理 | 修复start_mark标记错位 | 共{}条样本".format(len(self.data)))
        print("=" * 80)

        for i, data in enumerate(tqdm(self.data, desc="数据处理进度", unit="条")):
            data['knowledge_dict'] = {}
            data['start_mark'] = []
            data['seq'] = []
            data['seq_len'] = 0

            try:
                if 'raw_text' in data and data['raw_text'].strip():
                    raw_text = data['raw_text'].strip()
                    clean_text = self.clean_raw_text(raw_text)  # 清洗文本
                    # ✅ 生成【无截断seq + 精准start_mark + 纯净词典】
                    knowledge_dict, bert_seq, seq_len, start_mark = self.process_raw_text(clean_text)
                    data['knowledge_dict'] = knowledge_dict
                    data['seq'] = bert_seq
                    data['seq_len'] = seq_len
                    data['start_mark'] = start_mark
                    # 清洗词典无效键
                    if 'similar_dict' in data:
                        data['similar_dict'] = self.clean_dict(data['similar_dict'])
                else:
                    if 'seq' in data and len(data['seq']) > 0:
                        data['start_mark'] = self.gen_start_mark_perfect(data['seq'])
                        data['seq_len'] = len(data['seq'])

                if len(data['seq']) > 0 and len(data['start_mark']) > 0:
                    knowledge_data.append(data)

                if i % 500 == 0 and i > 0:
                    joblib.dump(knowledge_data, save_path)
                    print("进度保存：{}条样本已存入 {}".format(i, save_path))

            except Exception as e:
                print("样本{}处理异常：{} → 跳过".format(i, str(e)[:50]))
                continue

        joblib.dump(knowledge_data, save_path)
        print("=" * 80)
        print("✅ 处理完成！结果保存至 {} | 有效样本数：{}".format(save_path, len(knowledge_data)))
        print("=" * 80)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def clean_raw_text(self, raw_text):
        """清洗原始文本，仅保留纯中文+合法标点"""
        clean_text = re.sub(CLEAN_PATTERN, '', raw_text)
        clean_text = re.sub(r'[^\u4e00-\u9fa5，。！？；：""''（）【】《》、]', '', clean_text)
        return clean_text.strip()

    def clean_dict(self, origin_dict):
        """清洗词典，仅保留纯中文词汇键"""
        clean_dict = {}
        for k, v in origin_dict.items():
            if k.strip() and re.match(r'^[\u4e00-\u9fa5]+$', k.strip()):
                clean_dict[k] = v
        return clean_dict

    def process_raw_text(self, clean_text):
        # 生成无截断seq
        tokens = self.tokenizer.tokenize(clean_text)
        bert_tokens = ['[CLS]'] + tokens + ['[SEP]']
        bert_seq = self.tokenizer.convert_tokens_to_ids(bert_tokens)
        seq_len = len(bert_seq)

        # 生成纯净词典
        jieba_words = [w.strip() for w in jieba.lcut(clean_text) if w.strip() and re.match(r'^[\u4e00-\u9fa5]+$', w)]
        knowledge_dict = {w: self.get_knowledge(w) if len(w) >= 2 else [w] for w in jieba_words}

        # ✅ 调用新标记函数（Jieba多字词粒度）
        start_mark = self.gen_start_mark_jieba_align(bert_seq, clean_text)

        return knowledge_dict, bert_seq, seq_len, start_mark
    def gen_start_mark_jieba_align(self, seq, clean_text):
        """
        ✅ 核心修改：基于Jieba分词结果标记，实现「多字词粒度拆分」
        :param seq: BERT的seq（数字ID列表）
        :param clean_text: 清洗后的原始文本
        :return: start_mark（多字词粒度标记，与Jieba分词对齐）
        """
        # 步骤1：对原始文本做Jieba分词（得到多字词结果）
        jieba_words = jieba.lcut(clean_text)  # 例：["喂", "你好", "是", "张总", "吗"]

        # 步骤2：将seq转为BERT token列表
        token_list = [self.tokenizer._convert_id_to_token(id) for id in seq]
        start_mark = [0] * len(token_list)  # 初始化全0

        # 步骤3：跳过特殊token [CLS]/[SEP]
        valid_token_list = token_list[1:-1]  # 去掉[CLS]和[SEP]
        current_token_idx = 0  # 追踪当前处理到的BERT token下标

        # 步骤4：遍历Jieba分词结果，强制标记多字词粒度
        for word in jieba_words:
            # 跳过标点
            if word in PUNCTUATIONS:
                current_token_idx += len(word)
                continue

            # 多字词的首字 → 标1
            if current_token_idx < len(valid_token_list):
                # 对应到原始seq的下标（+1是因为跳过了[CLS]）
                seq_idx = current_token_idx + 1
                start_mark[seq_idx] = 1

            # 多字词的后续字 → 标0（自动保持0，无需额外操作）
            current_token_idx += len(word)  # 移动到下一个多字词的首字位置

        # 特殊token [CLS]/[SEP]强制标0
        start_mark[0] = 0
        start_mark[-1] = 0

        return start_mark

    def get_knowledge(self, word):
        knowledge = [word]
        try:
            synset = wn.synsets(word, lang='cmn')
            if synset:
                posset = [syn.name().split('.')[1] for syn in synset if '.' in syn.name()]
                if posset:
                    pos = Counter(posset).most_common(1)[0][0]
                    new_synset = [lemma for syn in synset for lemma in syn.lemma_names(lang='cmn')]
                    knowledge = list(set(new_synset + [word]))
        except:
            pass
        return knowledge


# ===================== 配套工具函数 + 自动校验函数 =====================
def transform(seq):
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    if not isinstance(seq, list):
        seq = seq.squeeze().cpu().numpy().tolist()
    tokens = [tokenizer._convert_id_to_token(x) for x in seq if x != 0 and tokenizer._convert_id_to_token(x)]
    return tokenizer.convert_tokens_to_string(tokens)


def verify_start_mark(seq, start_mark):
    """✅ 自动校验函数（校验标准不变）"""
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    token_list = [tokenizer._convert_id_to_token(id) for id in seq]
    report = {"is_pass": True, "error_info": []}
    if len(seq) != len(start_mark):
        report["is_pass"] = False
        report["error_info"].append(f"长度不一致！seq={len(seq)}, start_mark={len(start_mark)}")
        return report

    for idx in range(len(token_list)):
        token = token_list[idx]
        sm_val = start_mark[idx]
        if token in ["[CLS]", "[SEP]"] and sm_val != 0:
            report["is_pass"] = False
            report["error_info"].append(f"下标{idx}：{token} → start_mark={sm_val}（必须为0）")
        elif token in PUNCTUATIONS and sm_val == 1:
            report["is_pass"] = False
            report["error_info"].append(f"下标{idx}：{token}(标点) → start_mark=1（必须为0）")
        elif not token.startswith("##") and token not in ["[CLS]",
                                                          "[SEP]"] and token not in PUNCTUATIONS and sm_val == 0:
            report["is_pass"] = False
            report["error_info"].append(f"下标{idx}：{token}(新词开头) → start_mark=0（必须为1）")
    return report


# ===================== 主程序（运行+校验一体化） =====================
if __name__ == '__main__':
    torch.manual_seed(args.seed)
    # 1. 生成修复后的数据
    test_data = FraudAttackDataset(BASE_DATA_PATH)

    # 2. 随机抽取1条样本，执行自动校验（验证修复效果）
    if len(test_data) > 0:
        sample = test_data[0]
        seq = sample['seq']
        start_mark = sample['start_mark']
        check_report = verify_start_mark(seq, start_mark)

        print("\n" + "=" * 80)
        print("✅ 修复后 start_mark 自动校验结果")
        print("=" * 80)
        if check_report["is_pass"]:
            print(Fore.GREEN + "🎉 校验100%通过！无任何异常，start_mark与词汇精准一一对应！")
        else:
            print(Fore.RED + "❌ 仍有异常：")
            for err in check_report["error_info"]:
                print(f"→ {err}")

        # 打印关键信息
        print(Fore.RESET + "\n📌 样本关键信息：")
        print(f"→ seq长度：{len(seq)}")
        print(f"→ start_mark长度：{len(start_mark)}")
        print(f"→ knowledge_dict有效词条数：{len(sample['knowledge_dict'])}")
        print(f"→ start_mark标记示例（前20位）：{start_mark[:20]}")