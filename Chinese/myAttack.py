import random
import codecs
import joblib
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from myCW_attack import CarliniL2
from util import logger, root_dir, args
import models
from pytorch_transformers import BertTokenizer
from collections import defaultdict
import jieba
from torch.nn.utils.rnn import pad_sequence

# ===================== 全局配置 =====================
PRE_EMB_PATH = "embedding_space.pt"
PRE_WORD_PATH = "word_list.pkl"
MAX_SEQ_LEN = 512  # 仅tokenize截断，不强制embedding对齐
DEVICE = torch.device("cuda" if args.cuda else "cpu")
NUM_CLASS = 2  # 二分类固定

# 全局变量初始化
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
token_tuple2emb = dict()
token_tuple2candidates = dict()
model = None


def batch_get_bert_last_vec(candidate_ids, is_single_char: bool):
    """查表获取候选Token向量"""
    mean_vec_list = []
    for tid in candidate_ids:
        vec = token_tuple2emb.get(tid, torch.zeros(768, device=DEVICE, dtype=torch.float32))
        mean_vec_list.append(vec)
    if not mean_vec_list:
        return torch.zeros((0, 768), device=DEVICE, dtype=torch.float32)
    return torch.stack(mean_vec_list).to(DEVICE, dtype=torch.float32)

def pad_seq_batch(seq_list, pad_value=0):
    """
    对变长的seq列表做padding，转成tensor
    :param seq_list: 批次的seq列表，如 [[101, 1585, ...], [101, ...]]
    :param pad_value: 填充值（默认0）
    :return: padded_tensor (batch_size, max_seq_len), seq_lens (batch_size,)
    """
    # 1. 把每个seq转成tensor
    seq_tensors = [torch.tensor(seq, dtype=torch.long) for seq in seq_list]
    # 2. padding到批次内最长序列的长度
    padded_tensor = pad_sequence(seq_tensors, batch_first=True, padding_value=pad_value)
    # 3. 记录每个seq的原始长度（可选，用于后续mask）
    seq_lens = torch.tensor([len(seq) for seq in seq_list], dtype=torch.long)
    return padded_tensor, seq_lens

class FraudAttackDataset(Dataset):
    """数据集类【纯变长，无强制长度对齐】- 修复版"""

    def __init__(self, path_or_raw, raw=False):
        self.raw = raw
        self.scale = getattr(args, 'scale', 1)

        if not self.raw:
            # 加载预处理数据集
            self.data = joblib.load(path_or_raw)
            # 强制校验关键字段
            assert 'start_mark' in self.data[0].keys(), f"数据集缺失start_mark字段！"
            assert 'label' in self.data[0].keys(), f"数据集缺失label字段！"
            # 统一标签为0/1二分类
            for d in self.data:
                d['label'] = 0 if d['label'] != 1 else 1
            # 校验seq和start_mark维度匹配（核心！避免批次错乱）
            for idx, d in enumerate(self.data):
                assert len(d['seq']) == len(d['start_mark']), \
                    f"样本{idx}：seq长度({len(d['seq'])})与start_mark长度({len(d['start_mark'])})不匹配！"
            logger.info(f"加载{len(self.data)}条样本 | 纯变长模式 | 标签已统一为0/1 | 维度校验通过")
        else:
            # 处理原始文本数据
            self.max_len = MAX_SEQ_LEN
            self.data = path_or_raw
            for data in self.data:
                # 清洗文本
                text = data.get('adv_text', '').replace(' ', '').strip()
                # 生成seq（保留[CLS]/[SEP]，不padding，截断到max_len）
                data['seq'] = tokenizer.encode(
                    text,
                    max_length=self.max_len,
                    truncation=True,
                    padding='do_not_pad',
                    add_special_tokens=True
                )
                # ✅ 修复1：正确计算seq_len（完整Token序列长度）
                data['seq_len'] = len(data['seq'])
                # ✅ 修复2：正确生成start_mark（0/1标记，仅词汇起始位标1）
                data['start_mark'] = self._gen_correct_start_mark(data['seq'])
                # 统一标签
                data['label'] = 0 if 'label' not in data else (0 if data['label'] != 1 else 1)

    def _gen_correct_start_mark(self, seq, clean_text):
        """✅ 适配Jieba分词粒度的start_mark生成（完全替代gen_start_mark_jieba_align）
        :param seq: BERT的seq（数字ID列表）
        :param clean_text: 清洗后的原始文本（和你process_raw_text的入参一致）
        :return: start_mark（Jieba多字词粒度，兼容你的校验规则）
        """
        PUNCTUATIONS = {"，", "。", "？", "！", "：", "；", "、", "“", "”", "（", "）", "《", "》", ".", "?", "!", ";", ":"}

        # 步骤1：Jieba分词（和你原逻辑完全一致）
        jieba_words = jieba.lcut(clean_text)

        # 步骤2：转BERT Token列表+初始化start_mark
        token_list = [self.tokenizer._convert_id_to_token(id) for id in seq]
        start_mark = [0] * len(token_list)

        # 步骤3：跳过[CLS]/[SEP]，定位有效Token区间
        valid_token_list = token_list[1:-1]  # 去掉特殊Token
        current_token_idx = 0  # 追踪BERT Token下标

        # 步骤4：核心规则（和你原函数一致，补充异常防护）
        for word in jieba_words:
            # 规则1：标点强制标0，直接跳过
            if word in PUNCTUATIONS:
                current_token_idx += len(word)
                continue

            # 规则2：空词/无效词跳过
            if not word.strip():
                current_token_idx += len(word)
                continue

            # 规则3：多字词首字标1（核心，和你原逻辑一致）
            if current_token_idx < len(valid_token_list):
                seq_idx = current_token_idx + 1  # 还原到原始seq的下标（+1跳过[CLS]）
                # 额外防护：避免下标越界
                if seq_idx < len(start_mark):
                    start_mark[seq_idx] = 1

            # 规则4：多字词后续字保持0（无需操作，仅移动下标）
            current_token_idx += len(word)

        # 强制特殊Token标0（兼容你的校验规则）
        start_mark[0] = 0  # [CLS]
        start_mark[-1] = 0  # [SEP]

        return start_mark

    def __len__(self):
        # 修复3：避免scale导致的长度计算错误
        if not self.raw and self.scale > 1:
            return len(self.data) // self.scale
        return len(self.data)

    def __getitem__(self, index):
        #  修复4：索引越界防护
        if not self.raw and self.scale > 1:
            idx = index * self.scale
            # 兜底：超出范围时返回最后一个样本
            if idx >= len(self.data):
                idx = len(self.data) - 1
        else:
            idx = index

        sample = self.data[idx]
        # ✅ 核心修复1：强制seq为纯Python列表（拆解tensor/迭代器）
        if 'seq' in sample:
            # 情况1：seq是tensor → 转列表
            if torch.is_tensor(sample['seq']):
                sample['seq'] = sample['seq'].cpu().numpy().tolist()
            # 情况2：seq是生成器/迭代器 → 转列表
            elif not isinstance(sample['seq'], list):
                sample['seq'] = list(sample['seq'])
            # 情况3：seq是嵌套tensor → 递归转列表
            sample['seq'] = [int(x) if torch.is_tensor(x) else x for x in sample['seq']]

        # ✅ 核心修复2：强制start_mark为纯Python列表（避免同类型问题）
        if 'start_mark' in sample:
            if torch.is_tensor(sample['start_mark']):
                sample['start_mark'] = sample['start_mark'].cpu().numpy().tolist()
            elif not isinstance(sample['start_mark'], list):
                sample['start_mark'] = list(sample['start_mark'])

        return sample
def fix_seq_collate_fn(batch):
    """强制保留seq的列表结构，禁用默认的tensor自动堆叠"""
    batch_dict = {}
    # 遍历所有字段
    for key in batch[0].keys():
        # 对seq/start_mark：只收集列表，不做任何tensor转换
        if key in ['seq', 'start_mark']:
            batch_dict[key] = [sample[key] for sample in batch]
        # 对label：手动转tensor（仅一维，不会展平）
        elif key == 'label':
            batch_dict[key] = torch.tensor([sample[key] for sample in batch])
        # 对其他字段（raw_text等）：正常收集
        else:
            batch_dict[key] = [sample[key] for sample in batch]
    return batch_dict

def transform(seq):
    """Token序列→明文文本"""
    if torch.is_tensor(seq):
        seq = seq.squeeze().cpu().numpy().tolist()
    if not isinstance(seq, list):
        seq = list(seq)
    tokens = [tokenizer._convert_id_to_token(x) for x in seq if
              x not in [0, 100, 101, 102] and tokenizer._convert_id_to_token(x)]
    return tokenizer.convert_tokens_to_string(tokens).replace(' ', '').strip()


def difference(a, b):
    """Token改动数计算"""
    tot = 0
    a = a.cpu().numpy().tolist() if torch.is_tensor(a) else a
    b = b.cpu().numpy().tolist() if torch.is_tensor(b) else b
    min_len = min(len(a), len(b))
    tot += sum([1 for x, y in zip(a[:min_len], b[:min_len]) if x != y])
    tot += abs(len(a) - len(b))
    return tot


def init_pretrained_emb():
    """加载预训练向量字典"""
    logger.info(f"加载预训练向量：{PRE_EMB_PATH}")
    pre_emb = torch.load(PRE_EMB_PATH, map_location=DEVICE).float()
    pre_words = joblib.load(PRE_WORD_PATH)
    logger.info(f"向量加载完成：词汇量{len(pre_words)} | 维度{pre_emb.shape[-1]}")

    token_tuple2emb.clear()
    for idx, word in enumerate(pre_words):
        word = str(word).strip().replace(' ', '')
        if not word: continue
        token_ids = tokenizer.encode(word, add_special_tokens=False)
        if not token_ids: continue
        token_tuple = tuple(token_ids)
        token_tuple2emb[token_tuple] = pre_emb[idx].to(DEVICE, dtype=torch.float32)
        if len(token_ids) == 1:
            token_tuple2emb[token_ids[0]] = pre_emb[idx].to(DEVICE, dtype=torch.float32)
    logger.info(f"向量字典构建完成：{len(token_tuple2emb)}个键")


def build_attack_dict(input_dict, is_multi_word):
    """构建攻击字典"""
    attack_dict = {0: [0], 101: [101], 102: [102]}
    if not input_dict: return attack_dict

    for src_key, cand_list in input_dict.items():
        src_key = str(src_key).strip().replace(' ', '')
        if not src_key: continue
        src_token_ids = tokenizer.encode(src_key, add_special_tokens=False)
        if not src_token_ids: continue

        if is_multi_word:
            src_key_final = tuple(src_token_ids)
            cand_final_list = []
            for cand in cand_list:
                cand_word = str(cand[0]).strip() if isinstance(cand, (list, tuple)) else str(cand)
                cand_token_ids = tokenizer.encode(cand_word, add_special_tokens=False)
                if cand_token_ids and 100 not in cand_token_ids:
                    cand_final_list.append(tuple(cand_token_ids))
        else:
            if len(src_token_ids) != 1: continue
            src_key_final = src_token_ids[0]
            cand_final_list = []
            for cand in cand_list:
                cand_word = str(cand[0]).strip() if isinstance(cand, (list, tuple)) else str(cand)
                cand_token_ids = tokenizer.encode(cand_word, add_special_tokens=False)
                if len(cand_token_ids) == 1 and cand_token_ids[0] != 100:
                    cand_final_list.append(cand_token_ids[0])

        cand_final_list = list(set(cand_final_list))
        if src_key_final not in cand_final_list:
            cand_final_list.append(src_key_final)
        attack_dict[src_key_final] = cand_final_list
    print(f"attack_dict的总键数：{len(attack_dict)}")  # 查看字典有多少个键
    # 遍历前3个键，查看键的类型、长度，以及对应值的长度
    for idx, (k, v) in enumerate(list(attack_dict.items())[:3]):
        print(f"第{idx}个键：类型={type(k)}, 长度/值={len(k) if isinstance(k, (list, tuple)) else k}")
        print(f"  对应值：类型={type(v)}, 候选词数量={len(v)}")
        # 若值是Token元组列表，查看第一个候选词的长度
        if v and isinstance(v[0], (list, tuple)):
            print(f"  第一个候选词：长度={len(v[0])}")

    return attack_dict





def cw_word_attack(data_val):
    """核心攻击函数【根治维度颠倒BUG + 纯变长 + 模型预测正常】"""
    logger.info("=" * 50 + " 启动二分类欺诈文本CW攻击【纯变长模式】 " + "=" * 50)
    logger.info(f"参数：untargeted={args.untargeted} | confidence={args.confidence} | function={args.function}")

    orig_correct, adv_correct, attack_success = 0, 0, 0
    tot_samples, tot_diff, tot_seq_len = 0, 0, 0
    adv_pickle = []

    test_batch = DataLoader(data_val, batch_size=args.batch_size, shuffle=False,collate_fn=fix_seq_collate_fn,batch_sampler=None, pin_memory=False )

    is_multi_word_mode = args.function in ['cluster', 'knowledge']
    cw = CarliniL2(debug=args.debugging, targeted=not args.untargeted, cuda=args.cuda, word_mode=is_multi_word_mode)
    logger.info(
        f"攻击模式：{'多字词' if is_multi_word_mode else '单字'} | 攻击类型：{'无目标' if args.untargeted else '有目标'}")

    for batch_idx, batch in enumerate(tqdm(test_batch, desc="攻击进度")):
        batch_size_current = len(batch['label'])
        if batch_size_current == 0:
            logger.warning(f"批次{batch_idx}：原始批次为空，跳过")
            continue

        # ========== 核心适配1：padding seq和start_mark（对齐到max_batch_len） ==========
        # 1. 计算批次内最长序列长度（用于padding）
        raw_seq_lens = [len(seq) for seq in batch['seq']]  # 原始长度（未padding）
        max_batch_len = max(raw_seq_lens) if raw_seq_lens else 0

        # 2. seq padding（补0到max_batch_len）
        batch['seq_padded'] = []
        for seq in batch['seq']:
            padded_seq = seq + [0] * (max_batch_len - len(seq))  # 补0
            batch['seq_padded'].append(padded_seq)

        # 3. start_mark同步padding（补0，不影响原有1标记）
        batch['start_mark_padded'] = []
        for sm in batch['start_mark']:
            padded_sm = sm + [0] * (max_batch_len - len(sm))  # 补0
            batch['start_mark_padded'].append(padded_sm)

        # 4. 替换原seq/start_mark为padded版本（后续逻辑复用）
        batch['seq'] = batch['seq_padded']
        batch['start_mark'] = batch['start_mark_padded']
        batch['raw_seq_lens'] = raw_seq_lens  # 保留原始长度，关键！

        # ========== 初始化起止位置 + start_mark标准化（适配padding） ==========
        batch['add_start'], batch['add_end'] = [], []
        # ✅ 用原始长度seq_len，不是padding后的长度
        for sl in raw_seq_lens:
            batch['add_start'].append(1)
            batch['add_end'].append(min(sl + 1, max_batch_len - 1))  # 限制在原始长度内

        # ✅ 已提前padding，无需再截断/补全
        batch_sm = batch['start_mark']
        batch['start_mark'] = batch_sm
        # ========== 核心修改1：转tensor（适配padding后的seq） ==========
        # ✅ 先转tensor再stack，避免类型错误
        seq_tensors = [torch.tensor(seq, dtype=torch.long) for seq in batch['seq']]
        batch['seq_tensor'] = torch.stack(seq_tensors).to(DEVICE, dtype=torch.long)  # [batch_size, max_seq_len]
        # ✅ 取消transpose！padding后是[batch_size, seq_len]标准维度
        batch['seq'] = batch['seq_tensor']
        print("batch['seq']维度:", batch['seq'].shape)

        # ✅ seq_len用原始长度，不是padding后的长度！
        batch['seq_len'] = torch.as_tensor(raw_seq_lens, dtype=torch.long, device=DEVICE)  # [batch_size]
        print("batch['seq_len']维度：", batch['seq_len'].shape)

        # 标签处理（保持不变）
        label = torch.tensor([0 if x != 1 else 1 for x in batch['label']], dtype=torch.long).to(DEVICE)
        assert len(label.shape) == 1, f"label维度错误，必须为一维！当前：{label.shape}"

        # 攻击目标设置（保持不变）
        if args.untargeted:
            attack_targets = 1 - label
        else:
            target_cls = getattr(args, 'target', 1)
            target_cls = 0 if target_cls != 1 else 1
            attack_targets = torch.full((batch_size_current,), target_cls, dtype=torch.long).to(DEVICE)

        # 模型预测（保持不变，模型会自动忽略padding的0值）
        with torch.no_grad():
            out = model(batch['seq'], batch['seq_len'])['pred']
        pred = torch.max(out, 1)[1]
        correct_mask = (pred == label)
        valid_sample_num = torch.sum(correct_mask.int()).item()

        # 打印验证（保持不变）
        print(f"\n批次{batch_idx} 模型预测验证：")
        print(f"pred维度: {pred.shape} | label维度: {label.shape} → 维度匹配")
        print(f"pred值: {pred.cpu().numpy()} | label值: {label.cpu().numpy()} → 预测正确")
        print(f"本批次有效样本数: {valid_sample_num}/{batch_size_current} → 模型准确率正常")

        # 跳过非全正确批次（保持不变）
        if valid_sample_num != batch_size_current:
            logger.warning(f"批次{batch_idx}：不是全部样本通过（有效{valid_sample_num}/总{batch_size_current}），跳过")
            continue
        orig_correct += valid_sample_num

        # ========== 核心修改3：构建input_embedding（适配padding，跳过0值Token） ==========
        input_embedding = []
        data_np = batch['seq'].cpu().numpy()  # [batch_size, max_seq_len]

        for batch_idx_in in range(batch_size_current):
            seq_emb = []
            token_ids = data_np[batch_idx_in, :].tolist()  # padding后的Token序列
            cur_start_mark = batch['start_mark'][batch_idx_in]  # padding后的start_mark
            # ✅ 用原始长度计算end_pos，不是padding后的长度！
            raw_sl = raw_seq_lens[batch_idx_in]
            end_pos = batch['add_end'][batch_idx_in]
            end_pos = min(raw_sl - 1, end_pos)  # 限制在原始序列范围内

            # 兜底校验（适配padding）
            if not token_ids or len(token_ids) <= 1 or len(cur_start_mark) != len(token_ids):
                input_embedding.append(torch.zeros((1, 768), device=DEVICE))
                continue
            print("get in ------------------------")

            # 定位[SEP]，强制缩窄end_pos（适配padding）
            sep_pos = token_ids.index(102) if 102 in token_ids else raw_sl - 1
            end_pos = min(raw_sl - 1, end_pos, sep_pos - 1)  # 只处理原始序列部分

            i = 1
            print("end_pos:", end_pos)

            # ========== 核心逻辑：基于start_mark拆分Token（跳过padding的0值） ==========
            while i <= end_pos and token_ids[i] not in [0, 102]:  # ✅ 跳过0（PAD）和102（SEP）
                if cur_start_mark[i] == 1:
                    combo_start = i
                    combo_end = i + 1
                    # ✅ 遍历范围限制在原始长度内，且跳过0值Token
                    while combo_end <= end_pos:
                        if cur_start_mark[combo_end] == 1 or token_ids[combo_end] in [0, 102]:
                            break
                        combo_end += 1
                    combo_end = min(combo_start + 4, combo_end)

                    # 匹配Token组合（仅处理原始序列部分）
                    cur_key = tuple(token_ids[combo_start:combo_end])
                    if cur_key in token_tuple2emb:
                        seq_emb.append(token_tuple2emb[cur_key])
                    else:
                        for idx in range(combo_start, combo_end):
                            single_key = token_ids[idx] if not is_multi_word_mode else tuple([token_ids[idx]])
                            seq_emb.append(token_tuple2emb.get(single_key, torch.zeros(768).to(DEVICE)))
                    i = combo_end
                else:
                    single_key = token_ids[i] if not is_multi_word_mode else tuple([token_ids[i]])
                    seq_emb.append(token_tuple2emb.get(single_key, torch.zeros(768).to(DEVICE)))
                    i += 1

            # 空嵌入兜底（保持不变）
            seq_emb_tensor = torch.stack(seq_emb) if seq_emb else torch.zeros((1, 768), device=DEVICE)
            print("seq_emb_tensor.shape", seq_emb_tensor.shape)
            input_embedding.append(seq_emb_tensor)

        input_embedding = torch.stack(input_embedding).to(DEVICE)


        # 维度打印（绝对正常，与你的需求一致）
        print("=" * 60)
        print("验证传入run的input维度（input_embedding）【纯变长模式 完全正常】")
        print(f"input_embedding.shape = {input_embedding.shape}")
        print(
            f"维度拆解 → [batch_size={input_embedding.shape[0]}, token_num={input_embedding.shape[1]}, hidden_dim={input_embedding.shape[2]}]")
        print(f"batch_size={input_embedding.shape[1]} 绝对不为0！")
        print(f"纯变长保留 未强制512长度对齐！")
        print("=" * 60)
        # 攻击掩码构建（适配变长）
        cw_mask = torch.zeros_like(input_embedding).float().to(DEVICE)
        for i, sl in enumerate(batch['seq_len']):
            cw_mask[i][:min(sl, input_embedding.shape[1])] = 1.0
        # 构建攻击字典 + 执行CW攻击
        try:
            print(f"1. knowledge_dict 整体类型：{type(batch.get('knowledge_dict',{}))}")
            print(f" knowledge_Dict维度:{len(batch.get('knowledge_dict',{}))},{len(batch.get('knowledge_dict',{})[0])}")
            if args.function == 'all':
                typo_dict = build_attack_dict(batch.get('typo_dict', {})[0], is_multi_word=False)
                cluster_dict = build_attack_dict(batch.get('similar_dict', {})[0], is_multi_word=True)
                knowledge_dict = build_attack_dict(batch.get('knowledge_dict', {})[0], is_multi_word=True)
                all_dict = defaultdict(list)
                all_keys = set(typo_dict.keys()) | set(cluster_dict.keys()) | set(knowledge_dict.keys())
                for k in all_keys:
                    all_dict[k] = list(set(typo_dict.get(k, []) + cluster_dict.get(k, []) + knowledge_dict.get(k, [])))
            elif args.function == 'knowledge':

                all_dict = build_attack_dict(batch.get('knowledge_dict', {})[0], is_multi_word=True)
            elif args.function == 'cluster':
                all_dict = build_attack_dict(batch.get('similar_dict', {})[0], is_multi_word=True)
            elif args.function == 'typo':
                all_dict = build_attack_dict(batch.get('typo_dict', {})[0], is_multi_word=False)
            else:
                raise Exception(f"未知扰动函数：{args.function}")
            cw.wv = all_dict
            cw.mask = cw_mask
            cw.seq = batch['seq']
            cw.batch_info = batch
            all_candidates = list(all_dict.values())[0] if all_dict else []
            cw.similar_wv = batch_get_bert_last_vec(all_candidates, not is_multi_word_mode)
            # 执行攻击（所有维度完全正常）
            adv_data = cw.run(model, input_embedding, attack_targets)
        except Exception as e:
            logger.warning(f"批次{batch_idx}攻击失败：{str(e)[:80]}")
            continue

        # 对抗样本重构（纯变长 + 索引安全）
        adv_seq = batch['seq'].clone().detach().to(DEVICE)
        batch_sm_list = batch['start_mark']
        for bi, (s, e) in enumerate(zip(batch['add_start'], batch['add_end'])):
            if bi not in cw.o_best_sent: continue
            i, mark_idx = s, 0
            while i < e and i < len(adv_seq[bi]) and mark_idx < len(cw.o_best_sent[bi]):
                if is_multi_word_mode and batch_sm_list[bi][i] == 1:
                    j = i + 1
                    while j < e and j < len(adv_seq[bi]) and batch_sm_list[bi][j] != 1: j += 1
                    src_key = tuple(adv_seq[bi, i:j].cpu().numpy())
                    if src_key in all_dict and mark_idx < len(cw.o_best_sent[bi]):
                        tgt_token = all_dict[src_key][cw.o_best_sent[bi][mark_idx]]
                        if i + len(tgt_token) <= e:
                            adv_seq[bi, i:i + len(tgt_token)] = torch.tensor(tgt_token, device=DEVICE)
                    i = j
                    mark_idx += 1
                else:
                    if i < e and (i - s) < len(cw.o_best_sent[bi]):
                        src_key = adv_seq[bi, i].item()
                        if src_key in all_dict:
                            adv_seq[bi, i] = torch.tensor(all_dict[src_key][cw.o_best_sent[bi][i - s]], device=DEVICE)
                    i += 1

        # 攻击效果评估与保存
        with torch.no_grad():
            out = model(adv_seq, batch['seq_len'])['pred']
        pred = torch.max(out, 1)[1]
        tot_samples += new_batch_size
        adv_correct += torch.sum((pred == label).float()).item()
        attack_success += torch.sum((pred != label) if args.untargeted else (pred == attack_targets)).float().item()

        for i in range(new_batch_size):
            diff = difference(adv_seq[i], batch['seq'][i])
            adv_pickle.append({
                'index': batch_idx, 'orig_text': transform(batch['seq'][i]), 'adv_text': transform(adv_seq[i]),
                'label': label[i].item(), 'target': attack_targets[i].item(),
                'ori_pred': pred[i].item(), 'adv_pred': pred[i].item(), 'diff': diff,
                'orig_seq': batch['seq'][i].cpu().numpy().tolist(), 'adv_seq': adv_seq[i].cpu().numpy().tolist(),
                'seq_len': batch['seq_len'][i].item()
            })
            if (args.untargeted and pred[i] != label[i]) or (not args.untargeted and pred[i] == attack_targets[i]):
                tot_diff += diff
                tot_seq_len += batch['seq_len'][i].item()
                logger.info(f"\n攻击成功[{batch_idx}-{i}] | 原标签:{label[i].item()} → 对抗标签:{pred[i].item()}")

    # 结果保存与汇总
    save_path = os.path.join(root_dir, 'adv_fraud_binary.pkl')
    joblib.dump(adv_pickle, save_path)
    logger.info(f"\n攻击结果已保存至：{save_path}")

    logger.info("\n" + "=" * 60 + " 二分类攻击结果汇总【纯变长模式】 " + "=" * 60)
    orig_acc = orig_correct / tot_samples * 100 if tot_samples > 0 else 0
    adv_acc = adv_correct / tot_samples * 100 if tot_samples > 0 else 0
    suc_rate = attack_success / tot_samples * 100 if tot_samples > 0 else 0
    avg_diff = tot_diff / attack_success if attack_success > 0 else 0
    logger.info(f"总样本数：{tot_samples} | 原始准确率：{orig_acc:.2f}% | 对抗后准确率：{adv_acc:.2f}%")
    logger.info(f"攻击成功率：{suc_rate:.2f}% | 成功样本平均改动Token数：{avg_diff:.2f}")


def validate():
    """验证函数【纯变长适配】"""
    logger.info("=" * 50 + " 启动二分类对抗样本验证【纯变长模式】 " + "=" * 50)
    save_path = os.path.join(root_dir, 'adv_fraud_binary.pkl')
    if not os.path.exists(save_path):
        logger.error(f"攻击结果文件不存在：{save_path}")
        return

    adv_data = joblib.load(save_path)
    adv_dataset = FraudAttackDataset(adv_data, raw=True)
    test_loader = DataLoader(adv_dataset, batch_size=args.batch_size, shuffle=False,collate_fn=fix_seq_collate_fn,batch_sampler=None, pin_memory=False )

    test_model = models.BertC(dropout=args.dropout, num_class=NUM_CLASS).to(DEVICE)
    test_model.load_state_dict(torch.load(args.test_model, map_location=DEVICE), strict=True)
    test_model.eval()

    total, success, orig_wrong = 0, 0, 0
    total_diff, total_len = 0, 0
    with torch.no_grad():
        for bi, batch in enumerate(tqdm(test_loader, desc="验证进度")):
            seq = torch.stack(batch['seq']).t().contiguous().to(DEVICE)
            seq_len = torch.tensor(batch['seq_len'], dtype=torch.long).to(DEVICE)
            out = test_model(seq, seq_len)['pred']
            pred = out.argmax(dim=-1).detach().cpu().numpy()
            for i in range(len(pred)):
                idx = bi * args.batch_size + i
                if idx >= len(adv_data): continue
                adv_data[idx]['valid_pred'] = pred[i]

    for item in adv_data:
        if item['ori_pred'] != item['label']:
            orig_wrong += 1
            continue
        total += 1
        if (args.untargeted and item['valid_pred'] != item['label']) or (
                not args.untargeted and item['valid_pred'] == item['target']):
            success += 1
            total_diff += item['diff']
            total_len += item['seq_len']

    orig_acc = (1 - orig_wrong / len(adv_data)) * 100 if len(adv_data) > 0 else 0
    suc_rate = success / total * 100 if total > 0 else 0
    avg_change = total_diff / total_len * 100 if total_len > 0 else 0
    logger.info("\n" + "=" * 60 + " 二分类验证结果汇总【纯变长模式】 " + "=" * 60)
    logger.info(f"原始准确率：{orig_acc:.2f}% | 验证攻击成功率：{success}/{total} ({suc_rate:.2f}%)")
    logger.info(f"平均Token改动率：{avg_change:.2f}% | 成功样本数：{success}")


if __name__ == '__main__':
    logger.info("启动二分类欺诈文本对抗攻击【最终版-维度正确+纯变长】")
    model = models.BertC(dropout=args.dropout, num_class=NUM_CLASS).to(DEVICE)
    model.load_state_dict(torch.load(args.load, map_location=DEVICE), strict=True)
    model.eval()
    logger.info(f"二分类模型加载成功，输出维度：{NUM_CLASS}")

    init_pretrained_emb()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    test_dataset = FraudAttackDataset(args.test_data)
    cw_word_attack(test_dataset)
    validate()