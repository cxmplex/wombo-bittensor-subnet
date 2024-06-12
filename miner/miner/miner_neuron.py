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
from asyncio import Semaphore
from datetime import timedelta
from hashlib import sha256
from uuid import uuid4, UUID
import os

import bittensor as bt
import grpc
from diffusers import StableDiffusionXLControlNetPipeline
from google.protobuf.empty_pb2 import Empty
from neuron.protos.neuron_pb2 import MinerGenerationResponse, MinerGenerationIdentifier
from neuron.protos.neuron_pb2_grpc import MinerServicer, add_MinerServicer_to_server
from redis.asyncio import Redis
from grpc.aio import ServicerContext
from gpu_pipeline.pipeline import get_pipeline, get_tao_img
from miner.image_generator import generate
from neuron.defaults import DEFAULT_HEIGHT, DEFAULT_WIDTH, DEFAULT_STEPS, DEFAULT_GUIDANCE
from neuron.neuron import BaseNeuron
from tensor.protos.inputs_pb2 import GenerationRequestInputs



class MinerGenerationService(MinerServicer):
    def __init__(
        self,
        redis: Redis,
        gpu_semaphore: Semaphore,
        pipeline: StableDiffusionXLControlNetPipeline,
    ):
        super().__init__()
        self.redis = redis
        self.gpu_semaphore = gpu_semaphore
        self.pipeline = pipeline

    async def Generate(self, request: GenerationRequestInputs, context: ServicerContext) -> MinerGenerationResponse:
        
        frames = await generate(self.gpu_semaphore, self.pipeline, request)

        generation_id = uuid4()
        frames_hash = sha256(frames).digest()

        await self.redis.set(generation_id.hex, frames, ex=timedelta(minutes=15))

        return MinerGenerationResponse(
            id=MinerGenerationIdentifier(id=generation_id.bytes),
            hash=frames_hash,
        )


class Miner(BaseNeuron):
    last_metagraph_sync: int

    def __init__(self):
        super().__init__()
        self.gpu_semaphore, self.pipeline = get_pipeline(self.device)

        self.pipeline.vae = None

        bt.logging.info("Running warmup for pipeline")
        self.pipeline(
            prompt="Warmup",
            width=DEFAULT_WIDTH,
            height=DEFAULT_HEIGHT,
            num_inference_steps=DEFAULT_STEPS,
            image=get_tao_img(DEFAULT_WIDTH, DEFAULT_HEIGHT),
            guidance_scale=DEFAULT_GUIDANCE,
            output_type="latent",
        )

        self.server = grpc.aio.server()

        add_MinerServicer_to_server(
            MinerGenerationService(
                self.redis,
                self.gpu_semaphore,
                self.pipeline,
            ),
            self.server
        )

        self.server.add_insecure_port(f"127.0.0.1:{os.getenv('PORT')}")

    async def run(self):
        # Start the miner's gRPC server, making it active on the network.
        await self.server.start()
        bt.logging.info("server started")
        await self.server.wait_for_termination()