#!/bin/bash
# 순서 변경(2026-10-03): 벌점 강도 탐색이 끝나면 Fisher → LwF → 에폭 탐색 순으로 실행한다.
# 이미 끝난 실행(logs.pyd 존재)은 건너뛴다.
cd "$(dirname "$0")"
while pgrep -f "[r]un_linux.py penalty" > /dev/null; do sleep 30; done
for exp in penalty fisher lwf epoch; do
    .venv/bin/python run_linux.py $exp --workers 4 >> raw_data/linux/$exp.out 2>&1
done
echo done > raw_data/linux/ALL_DONE
