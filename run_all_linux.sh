#!/bin/bash
# Linux 재실험 전체를 순서대로 실행한다. 이미 끝난 실행(logs.pyd 존재)은 건너뛴다.
cd "$(dirname "$0")"
while pgrep -f "run_linux.py base" > /dev/null; do sleep 30; done
for exp in base buffer penalty fisher epoch lwf; do
    .venv/bin/python run_linux.py $exp --workers 4 > raw_data/linux/$exp.out 2>&1
done
echo done > raw_data/linux/ALL_DONE
