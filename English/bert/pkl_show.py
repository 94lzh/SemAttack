import joblib
import os

# ========== 第一步：配置文件路径（替换为你的实际路径） ==========
# 方式1：手动指定路径（推荐）
pkl_path = r"adv_text.pkl"


# ========== 第二步：读取pkl文件 ==========
try:
    adv_text = joblib.load(pkl_path)
    print(f"✅ 成功读取adv_text.pkl，总样本数：{len(adv_text)}")
except FileNotFoundError:
    print(f"❌ 文件不存在：{pkl_path}，请检查路径是否正确")
    exit()
except Exception as e:
    print(f"❌ 读取失败：{e}")
    exit()

# ========== 第三步：快速预览（前3个样本） ==========
print("\n===== 前3个样本的完整结构（快速预览） =====")
for idx, sample in enumerate(adv_text[:3]):
    print(f"\n【样本{idx+1}】")
    # 打印该样本的所有字段（键）
    print(f"字段列表：{list(sample.keys())}")
    # 打印核心字段的内容
    if "orig_text" in sample:
        print(f"原始文本（orig_text）：{sample['orig_text'][:100]}..."  # 只显示前100字符，避免过长
              if len(sample['orig_text'])>100 else sample['orig_text'])
    if "adv_text" in sample:
        print(f"对抗文本（adv_text）：{sample['adv_text'][:100]}..."
              if len(sample['adv_text'])>100 else sample['adv_text'])
    if "label" in sample:
        print(f"真实标签（label）：{sample['label']}")
    if "ori_pred" in sample:
        print(f"原始预测（ori_pred）：{sample['ori_pred']}")
    if "pred" in sample:
        print(f"预测（pred）：{sample['pred']}")
    if "diff" in sample:
        print(f"改动Token数（diff）：{sample['diff']}")
    if "seq_len" in sample:
        print(f"序列长度（seq_len）：{sample['seq_len']}")

# ========== 第四步：统计关键指标（全局） ==========
print("\n===== 全局关键指标统计 =====")
# 1. 字段存在性检查
required_fields = ["orig_text", "adv_text", "label", "diff", "seq_len"]
for field in required_fields:
    has_field = all(field in s for s in adv_text)
    print(f"所有样本是否包含{field}字段：{'✅ 是' if has_field else '❌ 否'}")

# 2. 统计改动率、标签分布等
total_diff = sum(s.get("diff", 0) for s in adv_text)
total_seq_len = sum(s.get("seq_len", 0) for s in adv_text)
avg_diff = total_diff / len(adv_text) if len(adv_text)>0 else 0
avg_diff_rate = (total_diff / total_seq_len) * 100 if total_seq_len>0 else 0

labels = [s.get("label", -1) for s in adv_text]
label_dist = {l: labels.count(l) for l in set(labels)}

print(f"平均改动Token数：{avg_diff:.1f}")
print(f"平均改动率：{avg_diff_rate:.1f}%")
print(f"标签分布：{label_dist}（key=标签值，value=样本数）")

# 3. 攻击成功率统计（无目标攻击）
attack_success = 0
valid_samples = 0
for s in adv_text:
    if "ori_pred" in s and "label" in s and "pred" in s:
        # 原始预测正确的样本才计入统计
        if s["ori_pred"] == s["label"]:
            valid_samples += 1
            # 无目标攻击成功：验证预测≠原标签
            if s["pred"] != s["label"]:
                attack_success += 1

success_rate = (attack_success / valid_samples) * 100 if valid_samples>0 else 0
print(f"有效样本数（原始预测正确）：{valid_samples}")
print(f"无目标攻击成功率：{success_rate:.1f}%")
import pandas as pd

# 转为DataFrame
df = pd.DataFrame(adv_text)
# 保存为CSV
csv_path = os.path.join(os.path.dirname(pkl_path), "adv_text.csv")
df.to_csv(csv_path, index=False, encoding="utf-8")
print(f"✅ 已导出为CSV文件：{csv_path}")