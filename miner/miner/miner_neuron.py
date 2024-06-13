import pickle
#  The MIT License (MIT)
#  Copyright © 2023 Yuma Rao
#  Copyright © 2024 WOMBO
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
#  documentation files (the “Software”), to deal in the Software without restriction, including without limitation
#  the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
#  and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all copies or substantial portions of
#  the Software.
#
#  THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
#  THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
#  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
#  OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
#  DEALINGS IN THE SOFTWARE.
#
#

import asyncio
from fastapi import FastAPI
from starlette.responses import JSONResponse, Response
from asyncio import Semaphore
from datetime import timedelta
from hashlib import sha256
from uuid import uuid4, UUID
import os

import bittensor as bt
from diffusers import StableDiffusionXLControlNetPipeline

from redis.asyncio import Redis
from gpu_pipeline.pipeline import get_pipeline, get_tao_img
from miner.image_generator import generate
from neuron.defaults import DEFAULT_HEIGHT, DEFAULT_WIDTH, DEFAULT_STEPS, DEFAULT_GUIDANCE
from tensor.protos.inputs_pb2 import GenerationRequestInputs


from urllib.parse import urlparse

from pydantic import BaseModel


class GenerationRequest(BaseModel):
    prompt: str
    prompt_2: str = None
    width: int = 1024
    height: int = 1024
    num_inference_steps: int = 20
    guidance_scale: float = 7.5
    negative_prompt: str = None
    negative_prompt_2: str = None
    seed: int = None
    controlnet_conditioning_scale: float = 2.0

def parse_redis_value(value: str | None, t: type):
    if value is None:
        return t()

    return t(value)


def parse_redis_uri(uri: str):
    url = urlparse(uri)

    if url.scheme == "redis":
        ssl = False
    elif url.scheme == "rediss":
        ssl = True
    else:
        raise RuntimeError(f"Invalid Redis scheme {url.scheme}")

    if url.path:
        path_db = url.path[1:]

        if not path_db:
            db = 0
        else:
            db = int(path_db)
    else:
        db = 0

    if not url.username or url.password:
        username = url.username
        password = url.password
    else:
        username = None
        password = url.username

    return {
        "host": url.hostname,
        "port": url.port,
        "db": db,
        "password": password,
        "ssl": ssl,
        "username": username,
    }

class MinerGenerationService:
    def __init__(self, redis: Redis, gpu_semaphore: Semaphore, pipeline: StableDiffusionXLControlNetPipeline):
        self.redis = redis
        self.gpu_semaphore = gpu_semaphore
        self.pipeline = pipeline

    async def generate(self, data):
        frames = await generate(self.gpu_semaphore, self.pipeline, data)
        return frames

class Miner:
    def __init__(self):
        self.device = 'cuda:0'
        self.redis = Redis(**parse_redis_uri(os.getenv('REDIS_URI')))
        self.gpu_semaphore, self.pipeline = get_pipeline(self.device)
        self.pipeline.vae = None

        print("Running warmup for pipeline")
        self.pipeline(
            prompt="Warmup",
            width=DEFAULT_WIDTH,
            height=DEFAULT_HEIGHT,
            num_inference_steps=DEFAULT_STEPS,
            image=get_tao_img(DEFAULT_WIDTH, DEFAULT_HEIGHT),
            guidance_scale=DEFAULT_GUIDANCE,
            output_type="latent",
        )

        self.app = FastAPI()

        @self.app.post("/generate")
        async def generate_endpoint(request: GenerationRequest):
            print("request received")
            service = MinerGenerationService(self.redis, self.gpu_semaphore, self.pipeline)
            print("service made")
            data = request.dict()
            print(data)
            result = await service.generate(data)
            print("generated")
            pickled_frames = pickle.dumps(result)
            return Response(content=pickled_frames, media_type='application/octet-stream')

        self.port = os.getenv('PORT', '8000')

    async def run(self):
        import uvicorn
        uvicorn.run(self.app, host='0.0.0.0', port=int(self.port))

# Usage example
if __name__ == "__main__":
    miner = Miner()
    import asyncio
