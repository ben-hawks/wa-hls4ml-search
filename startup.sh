#!/usr/bin/env bash

whoami
ls -al /venv

source /venv/bin/activate

python python /opt/repo/wa-hls4ml-search/iter_manager.py "$@"