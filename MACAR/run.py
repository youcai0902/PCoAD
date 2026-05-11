import json
import sys
from typing import List, Dict, Any

# 导入已有的模块
from tool.load_data import load_data
from agent.Text_Chunking_Agent import text_chunking_agent
from agent.Adaptive_Retrieval_Agent import AdaptiveRetrievalAgent
from agent.Inference_Agent import InferenceAgent


def load_data_2(dataset_path: str):
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("数据集格式应为列表")
        return data
    except FileNotFoundError:
        print(f"错误：文件 {dataset_path} 未找到。")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"错误：文件 {dataset_path} 不是有效的 JSON。")
        sys.exit(1)

def judge(inference_result: dict, ground_truth: dict):

    pred_auth = inference_result.get("Report Authenticity") or inference_result.get("report_authenticity")
    true_auth = ground_truth.get("Report Authenticity")
    auth_correct = (pred_auth == true_auth)


    pred_class = inference_result.get("Error Classification") or inference_result.get("error_classification") or []
    true_class = ground_truth.get("Error Classification") or []
    if isinstance(pred_class, list) and isinstance(true_class, list):
        class_correct = set(pred_class) == set(true_class)
    else:
        class_correct = (pred_class == true_class)


    pred_num = inference_result.get("Number of Errors") or inference_result.get("number_of_errors") or []
    true_num = ground_truth.get("Number of Errors") or []
    if isinstance(pred_num, list) and isinstance(true_num, list):

        pred_set = set(str(x) for x in pred_num)
        true_set = set(str(x) for x in true_num)
        num_correct = (pred_set == true_set)
    else:
        num_correct = (pred_num == true_num)

    return auth_correct, class_correct, num_correct

def main():

    dataset_path = input("请输入数据集文件路径（如 sample.json）: ").strip()
    if not dataset_path:
        dataset_path = "sample.json"


    data = load_data(dataset_path)
    print(f"成功加载 {len(data)} 条记录。\n")


    retrieval_agent = AdaptiveRetrievalAgent()
    inference_agent = InferenceAgent(retrieval_agent=retrieval_agent)


    total = len(data)
    correct_auth = 0
    correct_class = 0
    correct_num = 0

    for idx, item in enumerate(data):
        report_text = item.get("report", "")
        if not report_text:
            print(f"警告：第 {idx+1} 条数据缺少 report 字段，跳过。")
            continue


        try:
            keyword, chunks = text_chunking_agent(report_text)
        except Exception as e:
            print(f"第 {idx+1} 条数据分块失败：{e}")
            continue


        try:
            retrieval_results = retrieval_agent.process(chunks, key=keyword)
        except Exception as e:
            print(f"第 {idx+1} 条数据检索失败：{e}")
            continue


        try:
            inference_result = inference_agent.process(retrieval_results)
        except Exception as e:
            print(f"第 {idx+1} 条数据推理失败：{e}")
            continue


        ground_truth = {
            "Report Authenticity": item.get("Report Authenticity"),
            "Error Classification": item.get("Error Classification", []),
            "Number of Errors": item.get("Number of Errors", [])
        }
        auth_ok, class_ok, num_ok = judge(inference_result, ground_truth)

        if auth_ok:
            correct_auth += 1
        if class_ok:
            correct_class += 1
        if num_ok:
            correct_num += 1


        print(f"[{idx+1}/{total}] 推理完成，认证:{auth_ok} 分类:{class_ok} 数量:{num_ok}")


    print("\n===== 评估结果 =====")
    print(f"Report Authenticity 正确率: {correct_auth}/{total} = {correct_auth/total*100:.2f}%")
    print(f"Error Classification 正确率: {correct_class}/{total} = {correct_class/total*100:.2f}%")
    print(f"Number of Errors 正确率:       {correct_num}/{total} = {correct_num/total*100:.2f}%")

if __name__ == "__main__":
    main()