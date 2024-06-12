# The MIT License (MIT)
# Copyright © 2023 Yuma Rao

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from abc import ABC, abstractmethod

import bittensor as bt
from redis.asyncio import Redis

from neuron.misc import ttl_get_block
from neuron.redis import parse_redis_uri
from neuron.select_endpoint import select_endpoint
# Sync calls set weights and also resyncs the metagraph.
from tensor.config import config


class BaseNeuron(ABC):
    """
    Base class for Bittensor miners. This class is abstract and should be inherited by a subclass. It contains the core logic for all neurons; validators and miners.

    In addition to creating a wallet, subtensor, and metagraph, this class also handles the synchronization of the network state via a basic checkpointing mechanism based on epoch length.
    """

    subtensor: bt.subtensor
    wallet: bt.wallet
    metagraph: bt.metagraph
    config: bt.config
    redis: Redis

    @classmethod
    @abstractmethod
    def check_config(cls, config: bt.config):
        ...

    @classmethod
    @abstractmethod
    def add_args(cls, parser):
        ...

    def __init__(self):
        self.config = config(self.add_args)
        self.check_config(self.config)

        # Set up logging with the provided configuration and directory.
        bt.logging(config=self.config.logging, logging_dir=self.config.full_path)

        # If a gpu is required, set the device to cuda:N (e.g. cuda:0)
        self.device = self.config.neuron.device

        # Log the configuration for reference.
        bt.logging.info(self.config)

        self.redis = Redis(**parse_redis_uri(self.config.neuron.redis_url))
