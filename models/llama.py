from .openrouter_based_model import OpenRouterBasedModel

import re
import json
from typing import Optional

class Llama(OpenRouterBasedModel):

    def inference(self, query, sources):
        response = super().inference(query, sources)
        r = json.loads(response.content.decode())
        content = r['choices'][0]['message']['content']

        return {
            "answer": self._extract_answer_number(content),
            "reasoning": "{}\n\nОтвет сгенерирован моделью {}".format(content, self._get_model_friendly_name()),
        }

    def _get_model_name(self) -> str:
        return "meta-llama/llama-3.2-3b-instruct:free"
    
    def _get_model_friendly_name(self) -> str:
        return "Llama 3.2 3B Instruct"


    @staticmethod
    def _extract_answer_number(text: str) -> Optional[int]:
        match = re.search(r'\b\d+\b', text)
        return int(match.group()) if match else None
