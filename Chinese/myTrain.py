import os
import torch
import multiprocessing
# ✅ 1. 强制绑定CPU核心数为8（核心中的核心）
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
torch.set_num_threads(8)

# ✅ 2. 开启CPU多核并行加速（适配PyTorch+NLP计算）
torch.backends.mkldnn.enabled = True  # 开启MKL加速（CPU矩阵运算专用）
torch.backends.mkldnn.benchmark = True


# ==============================================================================================
import random
import time
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from pytorch_transformers import BertTokenizer  # 兼容你的旧库，无需修改

from util import logger, args, set_seed, root_dir
import models


# ===================== 【新增】权重保存路径初始化（关键！确保权重正常保存） =====================
# 创建权重保存目录，防止路径不存在报错
MODEL_SAVE_DIR = os.path.join(root_dir, "fraud_model_weights")
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)  # 目录不存在则创建，存在则跳过
logger.info(f"✅ 权重保存根目录已初始化：{MODEL_SAVE_DIR}")


# ===================== 1. 欺诈数据集加载类（✅ 全量保留原修正，无改动） =====================
class FraudDataset(Dataset):
    # ✅ 固定max_len=512，适配原生BERT，全局生效
    def __init__(self, csv_path, tokenizer, max_len=512):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data = self._load_and_process_data(csv_path)

    def _load_and_process_data(self, csv_path):
        """加载CSV并完成数据预处理、标签映射"""
        df = pd.read_csv(csv_path, encoding="utf-8")
        logger.info(f"加载数据集 {csv_path} | 数据量：{len(df)}")

        # 标签强制映射：TRUE→1，FALSE→0
        def label_map(x):
            x = str(x).strip().upper()
            return 1 if x == "TRUE" else 0

        df["is_fraud"] = df["is_fraud"].apply(label_map)
        data_list = df[["specific_dialogue_content", "is_fraud"]].to_dict("records")
        return data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        text = item["specific_dialogue_content"]
        label = item["is_fraud"]

        # ✅ ✔️ 核心修复：旧库pytorch_transformers专属写法（仅保留支持的参数）
        token_ids = self.tokenizer.encode(text, add_special_tokens=True)

        # ✅ ✔️ 纯手动实现512长度对齐
        seq_len = len(token_ids)
        if seq_len > self.max_len:
            token_ids = token_ids[:self.max_len]
            seq_len = self.max_len
        elif seq_len < self.max_len:
            token_ids += [0] * (self.max_len - seq_len)

        return {
            "seq": torch.tensor(token_ids, dtype=torch.long),
            "seq_len": torch.tensor(seq_len, dtype=torch.long),
            "class": torch.tensor(label, dtype=torch.long)
        }


# ===================== 2. 通用评估函数（✅ 全量保留原修正，无改动） =====================
def evaluate(model, eval_dataloader, device):
    model.eval()
    total_num = 0
    correct_num = 0
    with torch.no_grad():
        for batch in eval_dataloader:
            batch['seq'] = batch['seq'].to(device)
            batch['seq_len'] = batch['seq_len'].to(device)
            batch['class'] = batch['class'].to(device)

            out = model(batch['seq'], batch['seq_len'])
            logits = out['pred'].detach().cpu()
            pred_label = logits.argmax(dim=-1)
            true_label = batch['class'].cpu()

            total_num += pred_label.size(0)
            correct_num += pred_label.eq(true_label).sum().item()

    acc = correct_num / total_num
    logger.info(f"准确率 ACC: {acc:.4f} | 正确数: {correct_num} / 总数: {total_num}")
    return acc

def train_only_test():
    best_ckpt_path = os.path.join(MODEL_SAVE_DIR, "best_model_best_val_acc.pth")
    # 测试集最终评估
    logger.info("\n===== 测试集最终评估 =====")
    # 加载最优验证权重做测试（保证测试结果最优）
    model.load_state_dict(torch.load(best_ckpt_path))
    test_acc = evaluate(model, test_loader, device)
    logger.info(f" 测试集最终准确率: {test_acc:.4f}")

# ===================== 3. 训练主函数（✅ 核心修改：新增完整权重保存+断点续训逻辑） =====================
def train(train_loader, val_loader, test_loader, model, optimizer, device, epochs, resume_epoch=0):
    best_val_acc = 0.0
    # 断点续训：加载历史最优准确率（可选，如需精准恢复可额外保存）
    if resume_epoch > 0:
        logger.info(f"✅ 断点续训模式：从第 {resume_epoch+1} 轮开始训练")
    logger.info(f"开始训练，总轮数: {epochs} | 序列长度: {train_dataset.max_len}")

    torch.backends.mkldnn.enabled = True
    torch.set_num_threads(8)

    for epoch in range(resume_epoch, epochs):
        current_epoch = epoch + 1
        t_start = time.time()
        model.train()
        total_loss = 0.0
        step_num = 0

        for batch in tqdm(train_loader, desc=f"Epoch {current_epoch}/{epochs}"):
            optimizer.zero_grad(set_to_none=True)
            batch['seq'] = batch['seq'].to(device)
            batch['seq_len'] = batch['seq_len'].to(device)
            batch['class'] = batch['class'].to(device)

            out = model(batch['seq'], batch['seq_len'], batch['class'])
            loss = torch.mean(out['loss'])

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            step_num += 1

        avg_loss = total_loss / step_num
        t_end = time.time()
        logger.info(f"Epoch {current_epoch} | 训练损失: {avg_loss:.4f} | 耗时: {t_end - t_start:.2f}s")

        # ✅ 【新增1】保存当前轮次完整权重（每轮都存，永不丢失）
        epoch_ckpt_path = os.path.join(MODEL_SAVE_DIR, f"epoch_{current_epoch}_loss_{avg_loss:.4f}.pth")
        torch.save({
            "epoch": current_epoch,          # 保存当前轮数
            "model_state_dict": model.state_dict(),  # 保存模型权重
            "optimizer_state_dict": optimizer.state_dict(),  # 保存优化器状态（断点续训必备）
            "train_loss": avg_loss,          # 保存当前轮损失
            "best_val_acc": best_val_acc     # 保存历史最优准确率
        }, epoch_ckpt_path)
        logger.info(f"✅ 已保存第{current_epoch}轮权重至：{epoch_ckpt_path}")

        # 验证集评估
        logger.info("===== 验证集评估 =====")
        val_acc = evaluate(model, val_loader, device)

        # ✅ 【新增2】保存最优验证权重（单独命名，方便后续直接调用）
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_ckpt_path = os.path.join(MODEL_SAVE_DIR, "best_model_best_val_acc.pth")
            # 保存最优模型的纯净权重（仅模型参数，体积更小，推理专用）
            torch.save(model.state_dict(), best_ckpt_path)
            logger.info(f"✅ 刷新最优模型 | 最优ACC: {best_val_acc:.4f} | 保存至: {best_ckpt_path}")

    # ✅ 【新增3】训练结束后，保存最终完整权重（汇总所有信息）
    final_ckpt_path = os.path.join(MODEL_SAVE_DIR, f"final_model_epoch_{epochs}_best_acc_{best_val_acc:.4f}.pth")
    torch.save({
        "total_epochs": epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_acc": best_val_acc,
        "final_test_acc": 0.0  # 预留测试集准确率位置
    }, final_ckpt_path)
    logger.info(f"✅ 训练完成！最终完整权重已保存至：{final_ckpt_path}")

    # 测试集最终评估
    logger.info("\n===== 测试集最终评估 =====")
    # 加载最优验证权重做测试（保证测试结果最优）
    model.load_state_dict(torch.load(best_ckpt_path))
    test_acc = evaluate(model, test_loader, device)
    logger.info(f" 测试集最终准确率: {test_acc:.4f}")

    # ✅ 【新增4】更新最终权重的测试集准确率
    final_ckpt = torch.load(final_ckpt_path)
    final_ckpt["final_test_acc"] = test_acc
    torch.save(final_ckpt, final_ckpt_path)
    logger.info(f"✅ 已更新最终权重的测试集准确率：{test_acc:.4f}")


# ===================== 【新增】断点续训加载函数（可选调用，按需启用） =====================
def load_checkpoint(resume_ckpt_path, model, optimizer):
    """加载断点权重，恢复模型+优化器状态"""
    if os.path.exists(resume_ckpt_path):
        checkpoint = torch.load(resume_ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        resume_epoch = checkpoint["epoch"]
        best_val_acc = checkpoint["best_val_acc"]
        logger.info(f"✅ 成功加载断点权重：{resume_ckpt_path}")
        logger.info(f"✅ 恢复至第 {resume_epoch} 轮 | 历史最优ACC：{best_val_acc:.4f}")
        return resume_epoch
    else:
        logger.warning(f"⚠️ 指定的断点权重文件不存在：{resume_ckpt_path}，将从头开始训练")
        return 0


# ===================== 4. 主程序入口（✅ 微调适配权重保存逻辑，路径可直接确认） =====================
if __name__ == '__main__':
    # ===================== 【你仅需确认CSV路径】=====================
    TRAIN_CSV_PATH = "dataset/train_1000.csv"
    VAL_CSV_PATH = "dataset/val_100.csv"
    TEST_CSV_PATH = "dataset/test.csv"
    # ===================== 【可选】断点续训配置（不需要则设为None）=====================
    #RESUME_CKPT_PATH = "results/teacher/all/targeted/fraud_model_weights/epoch_2_loss_0.7446.pth"  # 示例："fraud_model_weights/epoch_5_loss_0.2345.pth"
    RESUME_CKPT_PATH=None
    # ===================================================================

    set_seed(args)
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

    train_dataset = FraudDataset(TRAIN_CSV_PATH, tokenizer)
    val_dataset = FraudDataset(VAL_CSV_PATH, tokenizer)
    test_dataset = FraudDataset(TEST_CSV_PATH, tokenizer)

    # 构建DataLoader（num_workers=0 适配Windows，无内存报错）
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset,batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # 初始化二分类模型
    model = models.BertC(dropout=args.dropout, num_class=2)

    device = torch.device("cuda" if args.cuda else "cpu")
    model.to(device)
    logger.info(f"使用设备: {device} | 模型序列长度: {train_dataset.max_len}")

    # 优化器配置
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-6)

    # ✅ 断点续训逻辑（不需要则跳过，自动从头训练）
    resume_epoch = 0
    if RESUME_CKPT_PATH is not None and os.path.exists(RESUME_CKPT_PATH):
        resume_epoch = load_checkpoint(RESUME_CKPT_PATH, model, optimizer)

    # 启动训练（传入断点续训轮数）
    #train(train_loader, val_loader, test_loader, model, optimizer, device, args.epochs, resume_epoch)
    train_only_test()