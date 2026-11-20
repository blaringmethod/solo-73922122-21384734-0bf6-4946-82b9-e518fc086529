"""后台计算调度。

- 计算在固定线程池中执行，HTTP 请求只负责入队，样本/目标更正从不等待旧作业；
- 每个作业一枚取消令牌。样本或目标更正使 epoch +1 时，旧作业记录被删除、
  令牌被取消，且尚未开工的过期任务连同其排队占位一起被撤销：
    * 仍在排队的旧作业在真正开始前发现已取消，直接退出，不占用计算位；
    * 正在运行的旧作业在下一个可中断迭代点抛出 ``JobCancelled`` 退出；
    * 无法中断的底层调用（HiGHS）允许自行返回，返回后立即检查取消并释放
      计算位，不再做可行性检查、冲突定位或权重求解；
- 旧作业不得写状态或结果：``mark_running``/``finish_job`` 都带 epoch 条件；
- 更正无需等待旧作业退出：腾出的计算位立即可供当前版本的新作业使用。
"""

from __future__ import annotations

import json
import threading
from concurrent.futures import Future, ThreadPoolExecutor

from . import solver
from .cancellation import CancelToken, JobCancelled, use_cancel_token
from .db import Database


class JobManager:
    def __init__(self, db: Database, max_workers: int = 4):
        self.db = db
        self.pool = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="weight-worker"
        )
        self._lock = threading.Lock()
        self._active: dict[str, CancelToken] = {}
        # job_id -> (Future, epoch)。notify_input_changed 只撤销旧 epoch 的
        # 任务，绝不会误伤并发提交的当前版本作业。
        self._futures: dict[str, tuple[Future, int]] = {}

    def submit(self) -> dict | None:
        # 在同一把锁内校验输入、取快照并建作业：避免与并发更正交叉，
        # 产生“新 epoch + 旧数据”的混合快照。输入为空时返回 None。
        created = self.db.create_job_current()
        if created is None:
            return None
        job, token = created
        with self._lock:
            self._active[job["job_id"]] = token
        try:
            future = self.pool.submit(self._run, job["job_id"], token)
        except RuntimeError:
            # 线程池已关闭（进程退出中）
            token.cancel()
            with self._lock:
                self._active.pop(job["job_id"], None)
            raise
        with self._lock:
            self._futures[job["job_id"]] = (future, job["epoch"])
        # 关闭“提交与登记之间发生更正”的竞态窗口：登记后若版本已变，撤销任务。
        if self.db.get_epoch() != job["epoch"]:
            future.cancel()
        return job

    def notify_input_changed(self) -> None:
        """样本/目标更正后调用：撤销旧作业，不等待其退出。

        令牌已在 ``Database`` 的 epoch 变更中统一取消；这里额外撤销旧 epoch
        下尚未开始执行的任务，避免它们日后占用计算位。当前 epoch 的作业
        （并发提交、或更正之后才提交）不受影响。
        """
        current = self.db.get_epoch()
        with self._lock:
            stale = [(jid, fut) for jid, (fut, ep) in self._futures.items()
                     if ep != current]
        for job_id, fut in stale:
            fut.cancel()
            if fut.cancelled():
                # 尚未开工就被撤销的任务不会执行 _run，这里清掉登记项。
                with self._lock:
                    self._futures.pop(job_id, None)
                    self._active.pop(job_id, None)
                self.db.discard_job(job_id, current)

    def shutdown(self) -> None:
        with self._lock:
            tokens = list(self._active.values())
            futures = [fut for fut, _ep in self._futures.values()]
        for token in tokens:
            token.cancel()
        for fut in futures:
            fut.cancel()
        self.pool.shutdown(wait=False, cancel_futures=True)

    # ------------------------------------------------------------------
    def _run(self, job_id: str, token: CancelToken) -> None:
        job_epoch = -1
        try:
            with use_cancel_token(token):
                # 排队期间已被更新淘汰：直接退出，不进入求解、不写任何状态。
                token.check()
                job = self.db.get_job(job_id)
                if job is None:
                    return
                job_epoch = job["epoch"]
                if not self.db.mark_running(job_id, job_epoch):
                    return
                token.check()

                snapshot = json.loads(job["input_snapshot"])
                samples = snapshot["samples"]
                targets = snapshot["targets"]

                try:
                    # 取消令牌已绑定到工作线程，求解器内部的可中断点会读取它。
                    res = solver.solve_qp(samples, targets)
                except JobCancelled:
                    return
                except solver.SolverError as exc:
                    # 数值失败 ≠ 无解：单独标记 failed，保留输入，绝不返回冲突集
                    self._finish_if_current(
                        job_id, job_epoch, "failed", None, str(exc)
                    )
                    return
                except Exception as exc:  # pragma: no cover - 防御性
                    self._finish_if_current(
                        job_id, job_epoch, "failed", None,
                        f"{type(exc).__name__}: {exc}",
                    )
                    return

                # 不可中断的底层调用可能刚好返回：写结果前再确认一次版本。
                try:
                    token.check()
                except JobCancelled:
                    return

                if res.status == "infeasible":
                    iis_rows = [targets[i] for i in res.infeasible_targets]
                    self._finish_if_current(
                        job_id, job_epoch, "infeasible",
                        {"infeasible_targets": iis_rows}, None,
                    )
                    return

                if res.status == "failed":
                    self._finish_if_current(
                        job_id, job_epoch, "failed", None,
                        f"数值求解失败（未判定为无解）：{res.reason}",
                    )
                    return

                # 成功：组装完整结果与输入依据
                weights, totals = self._assemble(samples, targets, res.weights)
                payload = {
                    "weights": weights,
                    "objective": res.objective,
                    "residual_abs_sum": res.residual_abs_sum,
                    "residual_max_abs": res.residual_max_abs,
                    "target_totals": totals,
                }
                self._finish_if_current(
                    job_id, job_epoch, "succeeded", payload, None
                )
        finally:
            # 计算位在 finally 中释放：无论正常结束、取消还是底层调用异常返回，
            # 都立即腾出线程池容量给当前版本的作业。
            with self._lock:
                self._active.pop(job_id, None)
                self._futures.pop(job_id, None)
            self.db.discard_job(job_id, job_epoch)

    def _finish_if_current(self, job_id, job_epoch, status, result, error):
        """带 epoch 条件的终态写入；旧版本作业永远写不进去。"""
        try:
            self.db.finish_job(job_id, job_epoch, status, result, error)
        except Exception:
            # 存储层失败不得影响计算位释放；作业保持未完成状态，不会出现半份结果。
            return

    @staticmethod
    def _assemble(samples, targets, w):
        weights = []
        for s, wi in zip(samples, w):
            weights.append({
                "person_id": s["person_id"],
                "d": s["d"],
                "w": wi,
                "lower": 0.5 * s["d"],
                "upper": 2.0 * s["d"],
            })
        a, p = solver.build_matrix(samples, targets)
        import numpy as np
        wa = np.array(w, dtype=float)
        totals = []
        for j, t in enumerate(targets):
            total = float(a[j] @ wa)
            totals.append({
                "dimension": t["dimension"],
                "code": t["code"],
                "population": t["population"],
                "weighted_total": total,
                "abs_error": abs(total - t["population"]),
            })
        return weights, totals
