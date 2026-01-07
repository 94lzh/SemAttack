import os
import torch
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# ===================== 1. 强制CPU配置（避免CUDA问题，和训练脚本对齐） =====================
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
torch.set_num_threads(8)
torch.backends.mkldnn.enabled = True
torch.backends.mkldnn.benchmark = True

# ===================== 2. 导入训练脚本的依赖（关键！复用相同的模型/工具） =====================
# 确保train.py和predict.py在同一目录，或把train.py的路径加入sys.path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # 加入当前目录
from pytorch_transformers import BertTokenizer  # 和训练脚本用同一个旧版分词器
import models  # 导入自定义的BertC模型（必须和训练脚本的models.py在同一目录）
from util import logger, args, set_seed  # 复用训练脚本的工具类（如果没有，看下方替代方案）

# ===================== 3. 配置参数（和训练脚本完全对齐） =====================
# 【你需要修改的路径】
BEST_MODEL_PATH = "data/bert/bert/model.pth"  # 最优权重路径
TEST_CSV_PATH = "dataset/test.csv"  # 要预测的测试集路径
MAX_LEN = 512  # 和训练脚本一致
DEVICE = torch.device("cpu")  # 强制CPU，避免CUDA显存/驱动问题
BATCH_SIZE = 32

# ===================== 4. 复用训练脚本的数据集类（保证数据处理一致） =====================
class FraudDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, tokenizer, max_len=512):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data = self._load_and_process_data(csv_path)

    def _load_and_process_data(self, csv_path):
        df = pd.read_csv(csv_path, encoding="utf-8")
        print(f"加载预测数据集 {csv_path} | 数据量：{len(df)}")

        # 标签映射（无标签也能预测，这里保留兼容）
        def label_map(x):
            if pd.isna(x):
                return 0  # 无标签时默认0
            x = str(x).strip().upper()
            return 1 if x == "TRUE" else 0

        df["is_fraud"] = df["is_fraud"].apply(label_map) if "is_fraud" in df.columns else 0
        data_list = df[["specific_dialogue_content", "is_fraud"]].to_dict("records")
        return data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        text = item["specific_dialogue_content"]
        label = item["is_fraud"]

        # 完全复用训练脚本的编码方式（旧版分词器）
        token_ids = self.tokenizer.encode(text, add_special_tokens=True)

        # 手动截断/补0（和训练一致）
        seq_len = len(token_ids)
        if seq_len > self.max_len:
            token_ids = token_ids[:self.max_len]
            seq_len = self.max_len
        elif seq_len < self.max_len:
            token_ids += [0] * (self.max_len - seq_len)

        return {
            "seq": torch.tensor(token_ids, dtype=torch.long),
            "seq_len": torch.tensor(seq_len, dtype=torch.long),
            "class": torch.tensor(label, dtype=torch.long),
            "text": text  # 保留原始文本，方便后续输出
        }

# ===================== 5. 加载模型+权重（关键！适配自定义BertC） =====================
def load_model(model_path, device):
    """加载训练好的BertC模型"""
    # 初始化和训练一致的模型（二分类）
    model = models.BertC(dropout=args.dropout if hasattr(args, 'dropout') else 0.1, num_class=2)
    # 加载最优权重（仅model.state_dict()）
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()  # 评估模式
    print(f"✅ 成功加载模型权重：{model_path}")
    return model

# ===================== 6. 预测+评估（完全复用训练脚本的evaluate逻辑） =====================
def predict_and_evaluate(model, test_loader, device):
    model.eval()
    all_preds = []
    all_trues = []
    all_texts = []
    total_num = 0
    correct_num = 0

    with torch.no_grad():
        for batch in test_loader:
            # 数据转到设备（CPU）
            seq = batch["seq"].to(device)
            seq_len = batch["seq_len"].to(device)
            true_label = batch["class"].to(device)
            texts = batch["text"]

            # 模型预测（和训练一致的输入格式）
            out = model(seq, seq_len)
            logits = out['pred'].detach().cpu()
            pred_label = logits.argmax(dim=-1)

            # 统计准确率
            total_num += pred_label.size(0)
            correct_num += pred_label.eq(true_label.cpu()).sum().item()

            # 收集结果
            all_preds.extend(pred_label.numpy())
            all_trues.extend(true_label.cpu().numpy())
            all_texts.extend(texts)

    # 计算指标
    acc = correct_num / total_num
    print("="*60)
    print(f"分类成功率（准确率）: {acc * 100:.2f}%")
    print(f"正确数: {correct_num} / 总数: {total_num}")
    print("="*60)

    # 生成预测结果DataFrame
    df_result = pd.DataFrame({
        "原始文本": all_texts,
        "真实标签": ["欺诈" if t==1 else "非欺诈" for t in all_trues],
        "预测标签": ["欺诈" if p==1 else "非欺诈" for p in all_preds],
        "是否预测正确": [t==p for t,p in zip(all_trues, all_preds)]
    })
    return df_result, acc

# ===================== 7. 主函数（一键运行） =====================
if __name__ == '__main__':
    # 初始化随机种子（和训练一致）
    set_seed(args) if hasattr(args, 'seed') else torch.manual_seed(42)

    # 加载分词器（和训练一致）
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

    # 加载数据集
    test_dataset = FraudDataset(TEST_CSV_PATH, tokenizer, MAX_LEN)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0  # Windows兼容
    )

    # 加载模型
    model = load_model(BEST_MODEL_PATH, DEVICE)

    # 执行预测+评估
    df_result, acc = predict_and_evaluate(model, test_loader, DEVICE)

    # 保存预测结果
    df_result.to_csv("test_pred_result.csv", index=False, encoding="utf-8-sig")
    print(f"\n✅ 预测结果已保存到 test_pred_result.csv")

# ===================== 【备用方案】如果没有util/args/logger =====================
# 如果你的环境中没有util.py，替换以下代码（放在最顶部）：
# class MockArgs:
#     def __init__(self):
#         self.dropout = 0.1
#         self.lr = 2e-5
#         self.batch_size = 32
#         self.epochs = 10
#         self.cuda = False
# args = MockArgs()
#
# # 简易logger替代
# import logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
#
# # 简易set_seed替代
# def set_seed(args):
#     random.seed(42)
#     np.random.seed(42)
#     torch.manual_seed(42)