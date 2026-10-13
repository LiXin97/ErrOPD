#!/bin/bash
DIR="$1"; SIZE="$2"
cp /path/to/models/Qwen3-${SIZE^^}/tokenizer_config.json "$DIR/tokenizer_config.json"
