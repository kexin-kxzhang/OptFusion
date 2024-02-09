import math
import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union
Params = Union[Iterable[Tensor], Iterable[Dict[str, Any]]]
LossClosure = Callable[[], float]
OptLossClosure = Optional[LossClosure]
Betas2 = Tuple[float, float]
OptFloat = Optional[float]


__all__ = ('Adam_Dev', 'Lamb',)


class Adam_Dev(Optimizer):
    def __init__(self, params: Params, lr: float = 1e-3, betas: Betas2 = (0.9, 0.999),
        eps: float = 1e-8, weight_decay: float = 0) -> None:
        
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(Adam_Dev, self).__init__(params, defaults)

    def step(self, closure: OptLossClosure = None) -> OptFloat:
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    msg = ('can not support sparse gradients, please consider SparseAdam instead')
                    raise RuntimeError(msg)

                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1

                # adam weight decay
                if group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])
                # m_t
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # v_t
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction = math.sqrt(1 - beta2 ** state['step'])
                bias_correction /= 1 - beta1 ** state['step']
                # Apply bias to lr to avoid broadcast.
                step_size = group['lr'] * bias_correction
                adam_step = exp_avg / exp_avg_sq.sqrt().add(group['eps'])
                p.data.add_(adam_step, alpha=-step_size)

        return loss

class Lamb(Optimizer):
    def __init__(
        self,
        params: Params,
        lr: float = 1e-3,
        betas: Betas2 = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        adam: bool = False,
        debias: bool = False,
        clamp_value: float = 10,
    ) -> None:
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, adam=adam, debias=debias)
        self.clamp_value = clamp_value
        super(Lamb, self).__init__(params, defaults)

    def step(self, closure: OptLossClosure = None) -> OptFloat:
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    msg = (
                        'Lamb does not support sparse gradients, '
                        'please consider SparseAdam instead'
                    )
                    raise RuntimeError(msg)

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Paper v3 does not use debiasing.
                if group['debias']:
                    bias_correction = math.sqrt(1 - beta2 ** state['step'])
                    bias_correction /= 1 - beta1 ** state['step']
                else:
                    bias_correction = 1

                # Apply bias to lr to avoid broadcast.
                step_size = group['lr'] * bias_correction

                weight_norm = torch.norm(p.data).clamp(0, self.clamp_value)

                adam_step = exp_avg / exp_avg_sq.sqrt().add(group['eps'])
                if group['weight_decay'] != 0:
                    adam_step.add_(p.data, alpha=group['weight_decay'])

                adam_norm = torch.norm(adam_step)
                if weight_norm == 0 or adam_norm == 0:
                    trust_ratio = 1
                else:
                    trust_ratio = weight_norm / adam_norm
                state['weight_norm'] = weight_norm
                state['adam_norm'] = adam_norm
                state['trust_ratio'] = trust_ratio
                if group['adam']:
                    trust_ratio = 1

                p.data.add_(adam_step, alpha=-step_size * trust_ratio)
        return loss


def counter(data, size, batch_size, type=1):
    ids, cnts = torch.unique(data, return_counts=True)
    all_cnts = torch.ones(size, dtype=cnts.dtype, device=cnts.device)
    if type ==1:
        cnts -=1
        all_cnts[ids] += cnts
    if type==2:
        cnts = cnts/batch_size
        all_cnts[ids] = cnts
    return all_cnts

def cow_clip(w, g , cnt_list, clamp_l=1e-4):
    w_norm = torch.norm(w, dim=1)
    w_norm = torch.max(w_norm, torch.tensor([clamp_l]).cuda())
    g_norm = torch.norm(g, dim=1)
    weight_norm = torch.min(w_norm*cnt_list, g_norm)
    ratio = torch.ones_like(weight_norm)
    ratio = torch.where(g_norm*weight_norm==0, ratio, weight_norm/g_norm).unsqueeze(1)
    return ratio*g

class Cow_Clip(Optimizer):
    def __init__(
        self,
        params: Params,
        lr: float = 1e-3,
        betas: Betas2 = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        adam: bool = False,
        debias: bool = True,
        clamp_u: float = 10,
        clamp_l: float = 1e-5,
    ) -> None:
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        adam=adam, debias=debias, clamp_u=clamp_u, clamp_l=clamp_l)
        super(Cow_Clip, self).__init__(params, defaults)

    def step(self, closure: OptLossClosure = None, cnt_list=None) -> OptFloat:
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    msg = (
                        'Optimizer does not support sparse gradients, '
                        'please consider SparseAdam instead'
                    )
                    raise RuntimeError(msg)

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1

                # L2 
                if group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])
                
                if not group['adam']:
                    grad = cow_clip(p.data, grad, cnt_list, group["clamp_l"])

                # Decay the first and second moment running average coefficient
                # m_t
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # v_t
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Paper v3 does not use debiasing.
                if group["debias"]:
                    bias_correction1 = 1 - beta1 ** state['step']
                    bias_correction2 = 1 - beta2 ** state['step']
                else:
                    bias_correction1 = 1
                    bias_correction2 = 1

                # Apply bias to lr to avoid broadcast.
                exp_avg_hat = exp_avg / bias_correction1
                exp_avg_sq_hat = exp_avg_sq / bias_correction2
                adam_step = (exp_avg_hat) / (exp_avg_sq_hat.sqrt()).add(group['eps'])

                p.data.add_(adam_step, alpha=-group["lr"])

        return loss
