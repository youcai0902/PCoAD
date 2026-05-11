import json

def load_data(language, data_name):
    datas = []
    if language == "CHN":
        file = "PCoAD/CHN"
    elif language == "USP":
        file = "PCoAD/USP"
    else:
        print("输入药品检验标准错误。")
        return datas
    
    if data_name == "all_report":
        file = file + "/" + data_name + ".json"
        with open(file, "r", encoding="utf-8") as f:
            datas = json.load(f)
    elif data_name == "logic_error":
        file = file + "/" + data_name + ".json"
        with open(file, "r", encoding="utf-8") as f:
            datas = json.load(f)
    elif data_name == "standard_error":
        file = file + "/" + data_name + ".json"
        with open(file, "r", encoding="utf-8") as f:
            datas = json.load(f)
    elif data_name == "procedural_error":
        file = file + "/" + data_name + ".json"
        with open(file, "r", encoding="utf-8") as f:
            datas = json.load(f)
    elif data_name == "step_omission":
        file = file + "/" + data_name + ".json"
        with open(file, "r", encoding="utf-8") as f:
            datas = json.load(f)
    elif data_name == "calculation_error":
        file = file + "/" + data_name + ".json"
        with open(file, "r", encoding="utf-8") as f:
            datas = json.load(f)
    else:
        print("输入的测试数据集名称错误。")
    
    return datas
    
