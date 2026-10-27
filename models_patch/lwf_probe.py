"""LwF + 발산 계측 (교정 1 이후 LwF 심층 분석).

`models/lwf.py`를 상속한다. lwf_stable=0이면 원본과 같은 함수(smooth, modified_kl_div)로
손실을 계산하므로 학습 결과가 원본 LwF와 같다. 계측값은 no_grad로만 구하고 RNG를 쓰지 않는다.
lwf_stable=1이면 증류 항을 수학적으로 같은 식인 -sum softmax(z_old/T) * log_softmax(z_new/T)로
계산한다. smooth(softmax(z), T) = softmax(z/T)이므로 두 식의 값은 같고 수치 안정성만 다르다.

매 갱신 단계의 증류 손실, 이전 클래스 소프트맥스 최솟값, 로짓 간격, 경사의 유한성을 기록하고,
손실이나 경사가 처음 유한하지 않게 된 단계의 값을 즉시 JSON에 저장한다
(Mammoth는 NaN 손실에서 assert로 멈추므로 끝까지 기다리지 않는다).
실행 전 `cp models_patch/lwf_probe.py mammoth/models/` 로 서브모듈 안에 복사한다.
"""
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from models.lwf import Lwf, modified_kl_div, smooth

TINY = torch.finfo(torch.float32).tiny  # 정규화 float32의 최솟값(약 1.18e-38)


class LwfProbe(Lwf):
    NAME = 'lwf_probe'
    COMPATIBILITY = ['class-il', 'task-il']

    @staticmethod
    def get_parser(parser):
        parser = Lwf.get_parser(parser)
        parser.add_argument('--probe_out', type=str, required=True, help='계측 JSON 경로')
        parser.add_argument('--lwf_stable', type=int, default=0, choices=[0, 1],
                            help='1이면 증류 항을 log_softmax로 계산')
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        super().__init__(backbone, loss, args, transform, dataset=dataset)
        self.step = 0
        self.trace = []          # 단계별 요약
        self.first_bad = None    # 손실 또는 경사가 처음 유한하지 않게 된 단계

    def _dump(self, status):
        out = Path(self.args.probe_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            'seed': self.args.seed, 'alpha': self.args.alpha, 'softmax_temp': self.args.softmax_temp,
            'lr': self.args.lr, 'stable': self.args.lwf_stable, 'status': status,
            'first_bad': self.first_bad, 'trace': self.trace}))

    def end_task(self, dataset):
        super().end_task(dataset)
        self._dump(f'task{self.current_task}_done')

    def observe(self, inputs, labels, not_aug_inputs, logits=None, epoch=None):
        self.opt.zero_grad()
        outputs = self.net(inputs)
        ce = self.loss(outputs[:, :self.n_seen_classes], labels)
        loss, kd = ce, None
        rec = {'step': self.step, 'task': self.current_task, 'ce': ce.item()}
        if logits is not None:
            new_z = outputs[:, :self.n_past_classes]
            T = self.args.softmax_temp
            if self.args.lwf_stable:
                old_z = logits[:, :self.n_past_classes].to(self.device)
                kd = -torch.mean(torch.sum(F.softmax(old_z / T, 1) * F.log_softmax(new_z / T, 1), 1))
            else:  # models/lwf.py의 observe와 같은 순서로 계산한다
                kd = modified_kl_div(smooth(self.soft(logits[:, :self.n_past_classes]).to(self.device), T, 1),
                                     smooth(self.soft(new_z), T, 1))
            loss = loss + self.args.alpha * kd
            with torch.no_grad():
                p = self.soft(new_z)
                gap = (new_z.max(1).values - new_z.min(1).values).max().item()
                rec.update({'kd': kd.item(), 'min_p': p.min().item(),
                            'n_zero_p': int((p == 0).sum()), 'n_subnormal_p': int(((p > 0) & (p < TINY)).sum()),
                            'max_gap': gap, 'max_abs_logit': outputs.abs().max().item()})
        loss.backward()
        grad_ok = all(torch.isfinite(q.grad).all() for q in self.net.parameters() if q.grad is not None)
        rec['loss'] = loss.item()
        rec['grad_finite'] = bool(grad_ok)
        self.trace.append(rec)
        if self.first_bad is None and (not grad_ok or not torch.isfinite(loss)):
            self.first_bad = rec
            self._dump('first_bad')
        self.opt.step()
        self.step += 1
        return loss.item()
