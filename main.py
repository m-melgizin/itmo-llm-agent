import os
import time
from typing import List, Optional
import re

from fastapi import FastAPI, HTTPException, Request, Response
from schemas.request import PredictionRequest, PredictionResponse
from utils.logger import setup_logger
from models import Model, Gemini, DeepSeek, Llama
from searchs import Search, DuckDuckGoSearch


LLM_API_KEY = os.getenv("LLM_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "llama")
SEARCH_NAME = os.getenv("SEARCH_NAME", "duckduckgo")

model_name_to_class = {
    "gemini": Gemini,
    "deepseek": DeepSeek,
    "llama": Llama
}

search_name_to_class = {
    "duckduckgo": DuckDuckGoSearch
}

app = FastAPI()
logger = None
model: Model = model_name_to_class[MODEL_NAME](LLM_API_KEY)


@app.on_event("startup")
async def startup_event():
    global logger
    logger = await setup_logger()


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()

    body = await request.body()
    await logger.info(
        f"Incoming request: {request.method} {request.url}\n"
        f"Request body: {body.decode()}"
    )

    response = await call_next(request)
    process_time = time.time() - start_time

    response_body = b""
    async for chunk in response.body_iterator:
        response_body += chunk

    await logger.info(
        f"Request completed: {request.method} {request.url}\n"
        f"Status: {response.status_code}\n"
        f"Response body: {response_body.decode()}\n"
        f"Duration: {process_time:.3f}s"
    )

    return Response(
        content=response_body,
        status_code=response.status_code,
        headers=dict(response.headers),
        media_type=response.media_type,
    )


@app.post("/api/request", response_model=PredictionResponse)
async def predict(request: PredictionRequest):

    def parse_options(query: str) -> Optional[List[str]]:
        lines = [line.strip() for line in query.split('\n')]
        if len(lines) < 2:
            return None
        return [line for line in lines[1:] if re.match(r'^\d+\.\s+', line)]

    try:
        query = request.query
        options = parse_options(query)

        search = search_name_to_class[SEARCH_NAME]()
        sources = search.search(query, 3) if options else search.search(
            "site:news.itmo.ru " + query, 3)

        await logger.info(f"[{request.id}] search done")

        llm_response = model.inference(query, sources)

        await logger.info(f"[{request.id}] inference done")

        return PredictionResponse(
            id=request.id,
            answer=llm_response['answer'] if options else None,
            reasoning=llm_response['reasoning'],
            sources=[source['link'] for source in sources]
        )

    except ValueError as e:
        error_msg = str(e)
        await logger.error(f"Validation error for request {request.id}: {error_msg}")
        raise HTTPException(status_code=400, detail=error_msg)
    except Exception as e:
        await logger.error(f"Internal error processing request {request.id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
