#!/bin/bash
torch-model-archiver --model-name "blip2" --version 1.0 --handler "ts_handler.py"
mkdir -p model_store & mv blip2.mar model_store
