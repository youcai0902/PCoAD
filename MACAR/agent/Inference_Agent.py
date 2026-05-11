import re
import json
from typing import List, Dict, Any

from tool.prompt import Inference_Agent_1, Inference_Agent_2, Inference_Agent_3
from tool.API import generate
from Adaptive_Retrieval_Agent import AdaptiveRetrievalAgent


class InferenceAgent:
    def __init__(self, retrieval_agent: AdaptiveRetrievalAgent = None):
        self.retrieval_agent = retrieval_agent or AdaptiveRetrievalAgent()

    def _extract_json(self, raw: str) -> Dict[str, Any]:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            start = raw.find('{')
            end = raw.rfind('}')
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(raw[start:end + 1])
                except json.JSONDecodeError:
                    pass
            return {}

    def _extract_needed_knowledge(self, text: str) -> List[str]:

        pattern = r'\*\*\*需要补充的知识：\[(.*?)\]\*\*\*'
        matches = re.findall(pattern, text)

        return matches

    def process(self, retrieval_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        step_results = []  

        for idx, item in enumerate(retrieval_results):
            text = item.get("待检索文本", "")
            knowledge_list = item.get("检索知识", [])

            knowledge_str = "\n\n".join(knowledge_list) if knowledge_list else ""


            prompt1 = Inference_Agent_1.replace("{input_text}", text).replace(
                "{retrive_knowledge}", knowledge_str
            )
            resp1 = generate(prompt1)


            needed = self._extract_needed_knowledge(resp1)

            if needed:

                new_knowledge = self.retrieval_agent.inference_retrive(needed, text)

                combined = knowledge_list.copy()
                for nk in new_knowledge:
                    if nk not in combined:
                        combined.append(nk)
                combined_str = "\n\n".join(combined)


                prompt3 = Inference_Agent_3.replace("{input_text}", text).replace(
                    "{retrive_knowledge}", combined_str
                )
                resp_final = generate(prompt3)
            else:
                resp_final = resp1


            step_result = f"文本{idx + 1}的推理结果为：\n{resp_final}"
            step_results.append(step_result)


        all_steps_text = "\n\n".join(step_results)
        prompt2 = Inference_Agent_2.replace("{input_text}", all_steps_text)
        resp_summary = generate(prompt2)


        final_result = self._extract_json(resp_summary)
        return final_result

