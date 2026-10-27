#!/bin/bash
cd "$(dirname "$0")"
.venv/bin/python run_linux.py lwfep_stable --workers 4 > raw_data/linux/lwfep_stable.out 2>&1
.venv/bin/python run_linux.py lwfep --workers 4 > raw_data/linux/lwfep.out 2>&1
echo done > raw_data/linux/LWFEP_DONE
