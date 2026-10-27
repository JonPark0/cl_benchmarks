"""Online EWC + 작업별 출력 노드 Fisher 기록 (review4 E3).

`models/ewc_on.py`를 상속하며 학습 로직은 바꾸지 않는다. 각 작업이 끝난 뒤
분류 층(classifier) 가중치의 Fisher를 출력 노드(행)별로 합산해 JSON에 저장한다.
실행 전 `cp models_patch/ewc_on_fisher.py mammoth/models/` 로 서브모듈 안에 복사한다.
"""
import json
from pathlib import Path

import torch

from models.ewc_on import EwcOn


class EwcOnFisher(EwcOn):
    NAME = 'ewc_on_fisher'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il']

    @staticmethod
    def get_parser(parser):
        parser = EwcOn.get_parser(parser)
        parser.add_argument('--fisher_out', type=str, required=True,
                            help='per-task, per-output-node Fisher JSON path')
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        super().__init__(backbone, loss, args, transform, dataset=dataset)
        self.records = []

    def _classifier_slice(self):
        off = 0
        for p in self.net.parameters():
            n = p.numel()
            if p is self.net.classifier.weight:
                return off, n, p.shape
            off += n
        raise RuntimeError('classifier weight not found')

    def end_task(self, dataset):
        prev = None if self.fish is None else self.fish.clone()
        super().end_task(dataset)
        task_fish = self.fish.clone() if prev is None else self.fish - self.args.gamma * prev
        off, n, shape = self._classifier_slice()

        def rows(f):
            return f[off:off + n].view(shape).sum(dim=1).tolist()

        self.records.append({
            'task': len(self.records),
            'classifier_row_fisher_task': rows(task_fish),        # 이번 작업 데이터로 계산한 값
            'classifier_row_fisher_accum': rows(self.fish),       # 페널티에 실제 쓰이는 누적값
            'mean_fisher_per_param_all': task_fish.mean().item(),
        })
        out = Path(self.args.fisher_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({'seed': self.args.seed, 'n_epochs': self.args.n_epochs,
                                   'records': self.records}))
