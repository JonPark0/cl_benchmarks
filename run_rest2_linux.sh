#!/bin/bash
# 순서 변경(2026-10-04): LwF 격자 → LwF 수치 안정 변형(vdV 3조건 + 기본 조건 대조) → 에폭 탐색.
cd "$(dirname "$0")"
while pgrep -f "[r]un_linux.py lwf" > /dev/null; do sleep 30; done
.venv/bin/python run_linux.py lwf --workers 4 >> raw_data/linux/lwf.out 2>&1
.venv/bin/python run_linux.py lwf_stable --tags vdv,a0.5_t2.0_lr0.001 --workers 4 >> raw_data/linux/lwf_stable.out 2>&1
.venv/bin/python run_linux.py epoch --workers 4 >> raw_data/linux/epoch.out 2>&1
echo done > raw_data/linux/ALL_DONE
