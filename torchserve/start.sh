#!/bin/bash
export CUDA_VISIBLE_DEVICES="1,2,3,4"
torchserve --start --ts-config config.properties
