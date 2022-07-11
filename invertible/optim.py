import logging
import torch as th

log = logging.getLogger(__name__)


def grads_all_finite(optimizer):
    for g in optimizer.param_groups:
        for p in g['params']:
            if p.grad is None:
                log.warning("Gradient was none on check of finite grads")
            elif not th.all(th.isfinite(p.grad)).item():
                return False
    return True
