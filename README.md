# Continual Learning Benchmark Project

## Overview

The Continual Learning Benchmark Project is designed to evaluate different continual learning algorithms in a standardized way. This project provides tools to measure the performance of various methods under different conditions and datasets.

## Features
- **Standardized Datasets**: Includes popular datasets such as MNIST, CIFAR-10, and others for uniform testing.
- **Modular Framework**: Components can be easily replaced to facilitate experimentation and comparison of different techniques.
- **Performance Metrics**: Automated calculation of various performance metrics including accuracy, memory efficiency, and computational overhead.

## Installation

To get started with the Continual Learning Benchmark Project, follow these steps:
1. Clone the repository:
   ```
   git clone https://github.com/JonPark0/cl_benchmarks.git
   cd cl_benchmarks
   ```
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

After installation, you can run benchmarks using:
```bash
python main.py --config config.yaml
```

Make sure to adjust the `config.yaml` file to set your parameters and choose the algorithms you want to benchmark.

## Submodule 패치

`mammoth`는 `aimagelab/mammoth`를 가리키는 git submodule이라 여기에 직접 커밋할 수
없다. `models_patch/agem_fixed.py`는 `models/agem.py`의 `end_task` 버퍼 크기 미반영
결함(`docs/r1_r2_r3_findings.md` 참조)을 수정한 모델로, 실행 전 아래처럼 서브모듈
안으로 복사해야 `--model agem_fixed`로 인식된다.

```bash
cp models_patch/agem_fixed.py mammoth/models/agem_fixed.py
```

## Contribution

Contributions are welcome! Please submit a pull request or open an issue for suggestions or improvements.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For any questions or inquiries, please contact [JonPark0](https://github.com/JonPark0).