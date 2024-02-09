#!/bin/bash

set -e

docker build -f Dockerfile -t wombo_subnet:validator_api ../

docker run \
  --network="host" \
  --gpus=all \
  -v ~/.cache:/root/.cache/ \
  -i \
  wombo_subnet:validator_api \
