# import json

# # JSONL 文件路径（替换成你的实际路径）
# jsonl_path = "/data/new3/ms_vqaresult/ms_vqa_pre.jsonl"

# total = 0
# exact_match = 0

# for line in open(jsonl_path, "r", encoding="utf-8"):
#     item = json.loads(line)

#     # 提取正确答案和预测结果中的选项字母（如 A., B., ...）
#     gt_labels = set(x.strip().split(".")[0] for x in item["answer"].split(","))
#     pred_labels = set(x.strip().split(".")[0] for x in item["pred"].split(","))

#     total += 1
#     if gt_labels == pred_labels:
#         exact_match += 1

# accuracy = exact_match / total if total > 0 else 0
# print(f"Multi-label Exact Match Accuracy: {accuracy:.4f} ({exact_match}/{total})")
# import json
# from sklearn.metrics import precision_recall_fscore_support
# from sklearn.preprocessing import MultiLabelBinarizer

# # 定义标签集合（A 到 S）
# label_options = [chr(i) for i in range(ord('A'), ord('S') + 1)]
# mlb = MultiLabelBinarizer(classes=label_options)

# # 读取数据
# jsonl_path = "/data/new3/ms_vqaresult/ms_vqa_pre.jsonl"  # 替换为你的路径

# y_true = []
# y_pred = []

# for line in open(jsonl_path, "r", encoding="utf-8"):
#     item = json.loads(line)
#     gt_labels = set(x.strip().split(".")[0] for x in item["answer"].split(","))
#     pred_labels = set(x.strip().split(".")[0] for x in item["pred"].split(","))

#     y_true.append(list(gt_labels))
#     y_pred.append(list(pred_labels))

# # 转换为 multi-hot 向量
# y_true_bin = mlb.fit_transform(y_true)
# y_pred_bin = mlb.transform(y_pred)

# # 计算 micro 平均的 precision, recall, f1
# precision, recall, f1, _ = precision_recall_fscore_support(
#     y_true_bin, y_pred_bin, average='macro', zero_division=0
# )

# print(f"Micro Precision: {precision:.4f}")
# print(f"Micro Recall:    {recall:.4f}")
# print(f"Micro F1-score:  {f1:.4f}")



import json
from nltk.tokenize import word_tokenize
from colorama import Fore, init
init(autoreset=True)

# Recall-style soft overlap function
def evaluate_token_recall(reference, candidate):
    reference = reference.strip().lower()
    candidate = candidate.strip().lower()

    ref_tokens = word_tokenize(reference)
    cand_tokens = word_tokenize(candidate)

    common = set(ref_tokens) & set(cand_tokens)
    if not common:
        return 0.0

    recall = len(common) / len(ref_tokens)
    return recall


if __name__ == "__main__":
    jsonl_path = "/data/new3/ms_vqaresult/ms_vqa_pre.jsonl"  # 替换为你的路径

    total = 0
    recall_sum = 0

    for line in open(jsonl_path, "r", encoding="utf-8"):
        item = json.loads(line)

        reference = item["answer"]
        candidate = item["pred"]

        score = evaluate_token_recall(reference, candidate)
        recall_sum += score
        total += 1

    avg_recall = recall_sum / total if total > 0 else 0

    print(f"{Fore.GREEN}📊 Token-overlap-based soft Recall:")
    print(f"{Fore.YELLOW}Avg Recall = {avg_recall:.4f} ({recall_sum:.2f} / {total})")

