from .openrouter_based_model import OpenRouterBasedModel

import re
import json
from typing import Optional

class DeepSeek(OpenRouterBasedModel):

    def inference(self, query, sources):
        response = super().inference(query, sources)
        r = json.loads(response.content.decode())
        content = r['choices'][0]['message']['content']

        return {
            "answer": self._extract_answer_number(content),
            "reasoning": "{}\n\nОтвет сгенерирован моделью {}".format(content, self._get_model_friendly_name()),
        }

    def _get_model_name(self) -> str:
        return "deepseek/deepseek-r1:free"
    
    def _get_model_friendly_name(self) -> str:
        return "DeepSeek R1"


    @staticmethod
    def _extract_answer_number(text: str) -> Optional[int]:
        match = re.search(r'\b\d+\b', text)
        return int(match.group()) if match else None
