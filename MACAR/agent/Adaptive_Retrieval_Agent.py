import json
import re
from typing import List, Dict, Any

from tool.prompt import Adaptive_Retrieval_Agent_1, Adaptive_Retrieval_Agent_2
from tool.API import generate
from retrive import BM25, sim


class AdaptiveRetrievalAgent:
    def __init__(self):

        self._drug_db = {}      # { "CHN": list_of_texts, "USA": list_of_texts }
        self._rule_db = {}      # { "CHN": list_of_texts, "USA": list_of_texts }

    def _is_chinese(self, text: str) -> bool:
        return bool(re.search('[\u4e00-\u9fff]', text))

    def _load_drug_database(self, lang: str) -> List[str]:
        if lang == "CHN":
            path = "Database/CHN/drug_database.json"
        else:
            path = "Database/USA/drug_database.json"
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if isinstance(data, list):
            return data

        return []

    def _load_rule_database(self, lang: str) -> List[str]:
        if lang == "CHN":
            path = "Database/CHN/rule_database.json"
        else:
            path = "Database/USA/rule_database.json"
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        return []

    def _get_drug_name(self, doc: str, is_chinese: bool) -> str:
        if is_chinese:

            parts = doc.split('\n', 1)
            name = parts[0].replace(' ', '')
            return name
        else:

            marker = "01\n\n\n\n"
            idx = doc.find(marker)
            if idx == -1:

                return doc.split('\n')[0].strip()
            start = idx + len(marker)
            end = doc.find('\n', start)
            if end == -1:
                return doc[start:].strip()
            return doc[start:end].strip()

    def _drug_search(self, keyword: str, knowledge: List[str], is_chinese: bool) -> str:

        name_to_doc = {}
        docs = []
        for doc in knowledge:
            name = self._get_drug_name(doc, is_chinese)
            name_to_doc[name] = doc
            docs.append(doc)


        if keyword in name_to_doc:
            return name_to_doc[keyword]


        scores = BM25(keyword, docs)  # 返回每篇文档的得分列表
        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        return docs[best_idx]

    def _rule_search(self, keyword: str, text: str, knowledge: List[str]) -> List[str]:

        candidates = [doc for doc in knowledge if keyword in doc]
        if not candidates:

            candidates = knowledge
        if not candidates:
            return []


        s1_scores = BM25(text, candidates)

        s2_scores = [sim(text, cand) for cand in candidates]

        total_scores = [0.6 * s1 + 0.4 * s2 for s1, s2 in zip(s1_scores, s2_scores)]

        idxs = sorted(range(len(total_scores)), key=lambda i: total_scores[i], reverse=True)[:2]
        return [candidates[i] for i in idxs]

    def _extract_json(self, raw: str) -> Dict[str, Any]:

        try:
            return json.loads(raw)
        except json.JSONDecodeError:

            start = raw.find('{')
            end = raw.rfind('}')
            if start != -1 and end != -1 and end > start:
                json_str = raw[start:end+1]
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    pass
            return {}

    def _call_tool(self, tool_name: str, keyword: str, text: str,
                   is_chinese: bool, drug_db: List[str], rule_db: List[str]) -> List[str]:
        results = []
        if tool_name == "具体药品检验标准检索工具":
            result = self._drug_search(keyword, drug_db, is_chinese)
            if result:
                results.append(result)
        elif tool_name == "通则标准检索工具":
            results = self._rule_search(keyword, text, rule_db)
        return results

    def process(self, texts: List[str], key: str) -> List[Dict[str, Any]]:
        output = []

        for text in texts:

            retrieval_knowledge = []
            retrieval_count = 0
            is_chinese = self._is_chinese(text)
            lang = "CHN" if is_chinese else "USA"


            if lang not in self._drug_db:
                self._drug_db[lang] = self._load_drug_database(lang)
            if lang not in self._rule_db:
                self._rule_db[lang] = self._load_rule_database(lang)
            drug_db = self._drug_db[lang]
            rule_db = self._rule_db[lang]


            prompt1 = Adaptive_Retrieval_Agent_1.replace("{input_text}", text)
            resp1 = generate(prompt1)
            decision1 = self._extract_json(resp1)

            need_tool = decision1.get("是否使用工具", [])
            if "是" not in need_tool:
                output.append({"待检索文本": text, "检索知识": retrieval_knowledge})
                continue

            tools_to_use = decision1.get("使用工具", [])

            for tool in tools_to_use:

                new_results = self._call_tool(tool, key, text, is_chinese, drug_db, rule_db)
                for res in new_results:
                    if res not in retrieval_knowledge:
                        retrieval_knowledge.append(res)
                retrieval_count += 1
                if retrieval_count >= 5:
                    break

            if retrieval_count >= 5:
                output.append({"待检索文本": text, "检索知识": retrieval_knowledge})
                continue


            while retrieval_count < 5:

                knowledge_str = "\n\n".join(retrieval_knowledge)
                prompt2 = Adaptive_Retrieval_Agent_2.replace("{input_text}", text).replace(
                    "{retrive_knowledge}", knowledge_str)
                resp2 = generate(prompt2)
                decision2 = self._extract_json(resp2)

                can_answer = decision2.get("是否能解决用户问题", [])
                need_tool_again = decision2.get("是否使用工具", [])
                if "是" in can_answer or "是" not in need_tool_again:
                    break

                tools_again = decision2.get("使用工具", [])
                needed_knowledge_list = decision2.get("需要知识", [])


                for nk in needed_knowledge_list:
                    if retrieval_count >= 5:
                        break
                    for tool in tools_again:
                        if retrieval_count >= 5:
                            break

                        new_results = self._call_tool(tool, nk, text, is_chinese, drug_db, rule_db)
                        for res in new_results:
                            if res not in retrieval_knowledge:
                                retrieval_knowledge.append(res)
                        retrieval_count += 1


                if not needed_knowledge_list:
                    break

            output.append({"待检索文本": text, "检索知识": retrieval_knowledge})

        return output

    
    def inference_retrive(self, text: str, existing_knowledge: List[str], keyword: str) -> Dict[str, Any]:

        retrieval_knowledge = list(existing_knowledge)


        is_chinese = self._is_chinese(text)
        lang = "CHN" if is_chinese else "USA"
        if lang not in self._drug_db:
            self._drug_db[lang] = self._load_drug_database(lang)
        if lang not in self._rule_db:
            self._rule_db[lang] = self._load_rule_database(lang)
        drug_db = self._drug_db[lang]
        rule_db = self._rule_db[lang]


        knowledge_str = "\n\n".join(existing_knowledge)
        prompt = Adaptive_Retrieval_Agent_2.replace("{input_text}", text).replace(
            "{retrive_knowledge}", knowledge_str
        )
        resp = generate(prompt)
        decision = self._extract_json(resp)


        need_tool = decision.get("是否使用工具", [])
        if "是" not in need_tool:
            return {"待检索文本": text, "检索知识": retrieval_knowledge}


        tools_to_use = decision.get("使用工具", [])
        for tool in tools_to_use:
            new_results = self._call_tool(tool, keyword, text, is_chinese, drug_db, rule_db)
            for res in new_results:
                if res not in retrieval_knowledge:
                    retrieval_knowledge.append(res)

        return {"待检索文本": text, "检索知识": retrieval_knowledge}