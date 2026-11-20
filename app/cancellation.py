"""后台作业的协作式取消。

取消必须贯穿整个求解流程，而不是只在一次 ``solve_qp`` 调用前后检查：

* 可以在迭代点中断的调用（SLSQP / trust-constr 的回调、IIS 的逐行探测、
  活动集精修循环）在每次迭代前调用 :func:`checkpoint`；
* 无法中断的底层调用（HiGHS ``linprog``）允许自行返回，但返回后立即
  ``checkpoint``，绝不再继续可行性判定、冲突定位或权重求解。

:class:`JobCancelled` 不是 ``SolverError``：作业层捕获后直接退出，不写任何
状态或结果；它也不会被当作“数值失败”或“无解”记录。
"""

from __future__ import annotations

import threading
from contextlib import contextmanager


class JobCancelled(Exception):
    """作业所属输入版本已失效，计算必须在当前检查点放弃。"""


class CancelToken:
    """一次性、线程安全的取消令牌。"""

    __slots__ = ("_event",)

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    def is_cancelled(self) -> bool:
        return self._event.is_set()

    def check(self) -> None:
        if self._event.is_set():
            raise JobCancelled()


# 工作线程会复用：调用求解器期间把“当前作业”的令牌绑定到线程，
# 这样即使求解器被其他代码以旧的两参数签名包了一层，底层仍能感知取消。
_local = threading.local()


@contextmanager
def use_cancel_token(token: CancelToken | None):
    previous = getattr(_local, "token", None)
    _local.token = token
    try:
        yield
    finally:
        _local.token = previous


def checkpoint(token: CancelToken | None = None) -> None:
    """在可中断点调用；令牌已取消时抛出 :class:`JobCancelled`。"""
    if token is None:
        token = getattr(_local, "token", None)
    if token is not None and token.is_cancelled():
        raise JobCancelled()
