from datetime import datetime
from typing import Annotated

import uvicorn

from fastapi import FastAPI, Body

from gpu_pipeline.pipeline import get_pipeline
from validator_api.validator_pipeline import validate_frames
from image_generation_protocol.io_protocol import ValidationInputs, ValidationOutputs


def main():
    app = FastAPI()

    gpu_semaphore, pipeline = get_pipeline()

    @app.post("/api/validate")
    async def validate(inputs: Annotated[ValidationInputs, Body()]) -> ValidationOutputs:
        return await validate_frames(
            gpu_semaphore,
            pipeline,
            inputs.frames,
            inputs.input_parameters,
        )

    @app.get("/")
    def healthcheck():
        return datetime.utcnow()

    uvicorn.run(app, port=8001)


if __name__ == "__main__":
    main()
