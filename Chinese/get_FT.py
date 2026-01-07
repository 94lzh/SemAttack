import random
import joblib
import os
import torch
import re
from torch.utils.data import Dataset
from tqdm import tqdm
from pytorch_transformers import BertTokenizer
from config import *

# ========== 初始化固定 ==========
torch.manual_seed(SEED)
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)

# ========== ✅ 核心新增：加载word_list.pkl 词汇表（必须配置路径） ==========
# ⚠️ 替换为你的word_list.pkl实际路径，与攻击代码中的PRE_WORD_PATH保持一致
WORD_LIST_PATH = args.word_list  # 对应攻击代码的PRE_WORD_PATH
# 校验文件存在性
assert os.path.exists(WORD_LIST_PATH), f" 词汇表文件不存在：{WORD_LIST_PATH}"
# 加载词汇表并去重、净化，构建快速查询集合（O(1)查询效率）
word_list = joblib.load(WORD_LIST_PATH)
PURE_WORD_SET = set()
for word in word_list:
    clean_word = word.strip().replace(' ', '')  # 清理空格、首尾空白
    if clean_word:
        PURE_WORD_SET.add(clean_word)
print(f"✅ 词汇表加载完成 | 有效词汇数：{len(PURE_WORD_SET)}")
print(f"✅ 词汇表示例：{list(PURE_WORD_SET)[:5]}")

# ========== ✅ 核心1：加载形近字+四层过滤（根除乱码/英文/标点/无效字符） ==========
typo_mapping = {}  # 最终仅保留【标准中文+在word_list中】的形近字映射
phrase_mapping = {  # 中文多字词扰动库（BERT分词适配）
    "餐厅": ["饭店", "餐馆"], "预订": ["预约", "订位"], "服务员": ["服务生", "店员"],
    "特色菜": ["招牌菜"], "营业时间": ["营业时段"], "红烧肉": ["东坡肉"]
}

# 严格校验文件存在
assert os.path.exists(TYPO_FILE_PATH), f"❌ 形近字文件不存在：{TYPO_FILE_PATH}"
with open(TYPO_FILE_PATH, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        # ✅ 步骤1：逐字符拆分无间隔行 + 过滤首尾空白
        raw_chars = [c for c in line.strip() if c.strip()]
        if len(raw_chars) < 2:
            continue

        # ✅ 步骤2：核心过滤【四层净化】→ 根除所有异常字符
        clean_chars = []
        for c in raw_chars:
            # 过滤规则1：必须是【单字符】
            if len(c) != 1:
                continue
            # 过滤规则2：必须是【标准中文】(Unicode 4e00-9fff，无扩展/无私有区)
            if not ('\u4e00' <= c <= '\u9fff'):
                continue
            # 过滤规则3：排除【不可见/控制字符】
            if c in ['\t', '\n', '\r', '\f', '\v']:
                continue
            # 过滤规则4：排除【全角/半角符号】
            if c in '！？。，、；：""''（）{}[]【】《》·~@#￥%……&*——+=':
                continue
            clean_chars.append(c)

        # ✅ 步骤3：仅处理净化后的有效中文字符组 + ✔️ 新增：校验是否在word_list中
        if len(clean_chars) < 2:
            continue
        for target_char in clean_chars:
            # 仅保留【在word_list中】的同组其他中文、去重、不追加原字
            typo_chars = list(set([c for c in clean_chars if c != target_char and c in PURE_WORD_SET]))
            if typo_chars:  # 仅保留有有效候选的映射
                typo_mapping[target_char] = typo_chars

print(f"✅ 形近字加载完成 | 纯净中文字数：{len(typo_mapping)}（全部命中word_list）")
print(f"✅ 示例映射：{dict(list(typo_mapping.items())[:3])}")


# ========== ✅ 数据集类（原版结构+纯净校验+候选词word_list校验） ==========
class FraudAttackDataset(Dataset):
    def __init__(self, path):
        cache_path = 'FC_' + path
        save_path = 'FT_FC_' + path
        assert os.path.exists(cache_path), f"❌ 原始数据不存在：{cache_path}"
        self.data = joblib.load(cache_path)
        bug_data = []
        valid_perturb = 0

        print(f"\n✅ 开始处理数据集 | 总样本数：{len(self.data)}")
        for i, data in enumerate(tqdm(self.data)):
            data['typo_dict'] = get_bug_dict(data['seq'])
            if self._check_pure_chinese(data['typo_dict']):
                valid_perturb += 1
            bug_data.append(data)
            if i % CACHE_STEP == 0 and i > 0:
                joblib.dump(bug_data, save_path)

        joblib.dump(bug_data, save_path)
        print(f"\n✅ 处理完成 | 纯中文有效样本：{valid_perturb}/{len(self.data)}")
        print(f"✅ 最终文件保存至：{save_path}")

    def _check_pure_chinese(self, typo_dict):
        """校验：typo_dict全为标准中文，无任何异常字符"""
        for char, typo_list in typo_dict.items():
            if not self._is_standard_chinese(char):
                return False
            for t in typo_list:
                if not self._is_standard_chinese(t):
                    return False
        return len(typo_dict) > 0

    def _is_standard_chinese(self, c):
        """✅ 最终版：严格判断【标准可显示中文字符】"""
        if len(c) != 1:
            return False
        # 仅匹配：常用中文核心区（无扩展、无生僻、无私有区）
        return '\u4e00' <= c <= '\u9fff' and not ('\ue000' <= c <= '\uf8ff')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


# ========== ✅ 工具函数（核心升级：候选词严格命中word_list.pkl + 纯中文+无乱码） ==========
def transform(seq):
    """token_id → 可读纯中文文本"""
    if not isinstance(seq, list):
        seq = seq.squeeze().cpu().numpy().tolist()
    tokens = [tokenizer._convert_id_to_token(x) for x in seq]
    # 过滤token中的异常字符，仅保留中文
    pure_tokens = [t for t in tokens if any('\u4e00' <= c <= '\u9fff' for c in t)]
    return tokenizer.convert_tokens_to_string(pure_tokens)


def get_bug(word):
    """✅ 核心升级：仅返回【标准中文+严格命中word_list.pkl】的扰动词，无乱码/无原字/无无效候选"""
    # 过滤非中文token
    if not any('\u4e00' <= c <= '\u9fff' for c in word):
        return []

    # ✔️ 新增：先校验原词是否在word_list中，不在则直接返回空（无扰动意义）
    if word not in PURE_WORD_SET:
        return []

    # 多字词优先匹配 + ✔️ 过滤：仅保留在word_list中的候选
    if len(word) > 1 and word in phrase_mapping:
        phrase_candidates = [c for c in phrase_mapping[word] if c in PURE_WORD_SET]
        if phrase_candidates:
            return phrase_candidates

    # 单字匹配形近字库 + ✔️ 过滤：仅保留在word_list中的候选
    if len(word) == 1 and word in typo_mapping:
        typo_candidates = [c for c in typo_mapping[word] if c in PURE_WORD_SET]
        if typo_candidates:
            return typo_candidates

    # 中文兜底（仅标准中文+在word_list中）
    force_typo = {
        "你": ["您"], "我": ["俺"], "人": ["入"], "日": ["曰"], "木": ["本"], "口": ["囗"],
        "好": ["郝"], "的": ["地"], "是": ["昰"], "有": ["冇"], "一": ["乙"]
    }
    if word in force_typo:
        force_candidates = [c for c in force_typo[word] if c in PURE_WORD_SET]
        return force_candidates
    return []


def get_bug_dict(indexed_tokens):
    """✅ 生成最终纯净版typo_dict：全标准中文+全部命中word_list+无乱码+无英文+无标点"""
    bug_dict = {}
    tokenized_words = [tokenizer._convert_id_to_token(x) for x in indexed_tokens]
    special_tokens = ["[CLS]", "[SEP]", "[PAD]", "[UNK]"]  # BERT特殊token过滤

    for token in tokenized_words:
        if token in special_tokens:
            continue
        typo_words = get_bug(token)
        if typo_words:  # 仅保留有有效扰动的纯中文token（且候选全在word_list）
            bug_dict[token] = typo_words
    return bug_dict


# ========== ✅ 主函数（原版调用逻辑不变） ==========
if __name__ == '__main__':
    test_data = FraudAttackDataset(BASE_DATA_PATH)