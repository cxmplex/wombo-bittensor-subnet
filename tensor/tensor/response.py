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
from asyncio import CancelledError
from time import perf_counter, time_ns
from typing import Literal, TypeVar, Generic, Callable, TypeAlias, Annotated

import bittensor as bt
from bittensor import AxonInfo
from google.protobuf.message import Message
from grpc import StatusCode, RpcError
from grpc.aio import Channel, insecure_channel, UnaryUnaryMultiCallable
from pydantic import BaseModel, ConfigDict, RootModel, Field

from tensor.api_handler import NONCE_HEADER, HOTKEY_HEADER, SIGNATURE_HEADER

ResponseT = TypeVar("ResponseT", bound=Message)
RequestT = TypeVar("RequestT", bound=Message)


class Channels:
    channels: list[Channel]

    def __init__(self, channels: list[Channel]):
        self.channels = channels

    def __aenter__(self) -> list[Channel]:
        for channel in self.channels:
            channel.__aenter__()

        return self.channels

    def __aexit__(self, exc_type, exc_val, exc_tb):
        for channel in self.channels:
            channel.__aexit__(exc_type, exc_val, exc_tb)


class SuccessfulResponseInfo(BaseModel):
    axon: AxonInfo
    process_time: float


class FailedResponseInfo(BaseModel):
    axon: AxonInfo
    status: StatusCode
    detail: str | None


class SuccessfulResponse(SuccessfulResponseInfo, Generic[ResponseT]):
    data: ResponseT
    successful: Literal[True] = True

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def info(self):
        return SuccessfulResponseInfo(axon=self.axon, process_time=self.process_time)


class FailedResponse(FailedResponseInfo):
    successful: Literal[False] = False

    @property
    def info(self):
        return FailedResponseInfo(axon=self.axon, status=self.status, detail=self.detail)


ResponseInfo = SuccessfulResponseInfo | FailedResponseInfo

Response: TypeAlias = RootModel[
    Annotated[
        SuccessfulResponse[ResponseT] | FailedResponse,
        Field(discriminator="successful"),
    ]
]


def axon_address(axon: AxonInfo):
    return f"{axon.ip}:{axon.port}"


def axon_channel(axon: AxonInfo):
    return insecure_channel(axon_address(axon))


def create_metadata(axon: AxonInfo, wallet: bt.wallet):
    nonce = str(time_ns())
    hotkey = wallet.hotkey.ss58_address
    message = f"{nonce}.{hotkey}.{axon.hotkey}"
    signature = f"0x{wallet.hotkey.sign(message).hex()}"

    return [
        (NONCE_HEADER, nonce),
        (HOTKEY_HEADER, hotkey),
        (SIGNATURE_HEADER, signature),
    ]


async def create_request(
    axon: AxonInfo,
    request: RequestT,
    invoker: Callable[[Channel], UnaryUnaryMultiCallable[RequestT, ResponseT]],
    wallet: bt.wallet | None = None,
) -> Response[ResponseT]:
    async with axon_channel(axon) as channel:
        return await call_request(axon, request, invoker(channel), wallet)


async def call_request(
    axon: AxonInfo,
    request: RequestT,
    invoker: UnaryUnaryMultiCallable[RequestT, ResponseT],
    wallet: bt.wallet | None = None,
    timeout: float | None = None,
) -> Response[ResponseT]:
    start = perf_counter()

    method = invoker._method
    metadata = create_metadata(axon, wallet) if wallet else None

    call = invoker(request, timeout=timeout, metadata=metadata)

    try:
        try:
            response = await call
        except CancelledError:
            call.cancel()
            raise

        process_time = perf_counter() - start

        bt.logging.trace(
            f"Successful request {method} -> {axon.hotkey}"
            f"status code {await call.code()} with details {await call.details()}"
        )

        return SuccessfulResponse(
            data=response,
            process_time=process_time,
            axon=axon,
        )
    except RpcError:
        bt.logging.trace(
            f"Failed request {method} -> {axon.hotkey}, "
            f"status code {await call.code()} with error {await call.details()}"
        )

        return FailedResponse(
            axon=axon,
            status=await call.code(),
            detail=await call.details(),
        )
