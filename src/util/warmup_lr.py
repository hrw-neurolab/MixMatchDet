# ------------------------------------------------------------------------
# MixMatchDet
# Copyright (c) 2026 Hochschule Ruhr West
# Modified from https://github.com/lehduong/torch-warmup-lr/tree/master.
# Licensed under the Apache License, Version 2.0 [See LICENSE for details]
# ------------------------------------------------------------------------

import math

from torch.optim.lr_scheduler import _LRScheduler


class WarmupLR(_LRScheduler):
    def __init__(self, optimizer, num_warmup=1, warmup_strategy="linear"):
        self._num_warmup = num_warmup
        self._warmup_strategy = warmup_strategy
        self.optimizer = optimizer

        if warmup_strategy not in ["linear", "cos", "constant"]:
            raise ValueError(
                "Expect warmup_strategy to be one of ['linear', 'cos', 'constant'] but got {}".format(
                    warmup_strategy
                )
            )

        # Define the strategy to warm up learning rate
        if warmup_strategy == "cos":
            self._warmup_func = self._warmup_cos
        elif warmup_strategy == "linear":
            self._warmup_func = self._warmup_linear
        else:
            self._warmup_func = self._warmup_const

        # learning rate of each param group will increase
        # from the min_lr to initial_lr
        for group in self.optimizer.param_groups:
            group["warmup_max_lr"] = group["lr"]
            group["warmup_initial_lr"] = group["lr"] * 0.1

        super(WarmupLR, self).__init__(optimizer)

    def _warmup_cos(self, start, end, pct):
        cos_out = math.cos(math.pi * pct) + 1
        return end + (start - end) / 2.0 * cos_out

    def _warmup_const(self, start, end, pct):
        return start if pct < 0.9999 else end

    def _warmup_linear(self, start, end, pct):
        return (end - start) * pct + start

    def get_lr(self):
        step_num = self._step_count

        if step_num > self._num_warmup:
            return [group["lr"] for group in self.optimizer.param_groups]

        # warm up learning rate
        lrs = []
        for group in self.optimizer.param_groups:
            computed_lr = self._warmup_func(
                group["warmup_initial_lr"],
                group["warmup_max_lr"],
                step_num / self._num_warmup,
            )
            lrs.append(computed_lr)

        return lrs

    def step(self, *_):
        if self._step_count > self._num_warmup:
            return

        values = self.get_lr()
        for param_group, lr in zip(self.optimizer.param_groups, values):
            param_group["lr"] = lr
        self._last_lr = values
        self._step_count += 1
