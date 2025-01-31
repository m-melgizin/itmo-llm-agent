from .model import Model

import requests
import json
from typing import List, Dict


class OpenRouterBasedModel(Model):
    def __init__(self, api_key):
        self._api_key = api_key

    def inference(self, query: str, sources: List[Dict[str, str]]) -> dict:
        context = "\n".join(
            [f"Источник {i+1}: {source['link']} {source['title']}\n{source['snippet']}" for i, source in enumerate(sources)])
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": "Bearer {}".format(self._api_key),
            },
            data=json.dumps({
                "model": self._get_model_name(),
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": self._get_prompt().format(query, context)
                            }
                        ]
                    }
                ],
                "temperature": 0.1,
            })
        )
        return response

    def _get_model_name(self) -> str:
        raise NotImplementedError

    def _get_model_friendly_name(self) -> str:
        raise NotImplementedError

    def _get_prompt(self) -> str:
        return (
            "Ответь на вопрос, используя только предоставленные источники.\n"
            "Если варианты ответа присутствуют, укажи номер правильного варианта и краткое объяснение ОБЯЗАТЕЛЬНО С УКАЗАНИЕМ ИСТОЧНИКА.\n"
            "СТРОГО СЛЕДУЙ ФОРМАТУ ОТВЕТА\n"
            "Вопрос: {}\n"
            "{}\n\n"
            "Ответ должен содержать:\n"
            "- Строку \"Правильный ответ:\" + Номер правильного варианта (если есть)\n"
            "- ОБЯЗАТЕЛЬНО Краткое объяснение с ОБЯЗАТЕЛЬНО указанием источника\n"
            "Точный формат ответа:\n"
            "Правильный ответ: <НОМЕР ПРАВИЛЬНОГО ВАРИАНТА>\n"
            "<ОБЯЗАТЕЛЬНО краткое объяснение ОБЯЗАТЕЛЬНО С УКАЗАНИЕМ ИСТОЧНИКА>"
        )
