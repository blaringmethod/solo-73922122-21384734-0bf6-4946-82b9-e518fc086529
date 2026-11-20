"""取消语义测试：更正与求解交错时的排队、在途、不可中断调用与计算位释放。"""

from __future__ import annotations

import threading
import time

import pytest

from app import solver
from app.cancellation import JobCancelled, CancelToken, checkpoint, use_cancel_token
from app.db import Database
from app.jobs import JobManager

SAMPLES = [
    {"person_id": "P1", "region": "East", "age_group": "youth", "d": 1},
    {"person_id": "P2", "region": "East", "age_group": "mid", "d": 1},
    {"person_id": "P3", "region": "South", "age_group": "youth", "d": 2},
]
FEAS_TARGETS = [
    {"dimension": "region", "code": "East", "population": 2},
    {"dimension": "region", "code": "South", "population": 3},
]
# R1 两人上限之和为 4，与 a1=1 冲突：会走到 IIS 定位。
CONFLICT_SAMPLES = [
    {"person_id": "X", "region": "R1", "age_group": "a1", "d": 1},
    {"person_id": "Y", "region": "R1", "age_group": "a2", "d": 1},
    {"person_id": "Z", "region": "R2", "age_group": "a1", "d": 1},
]
CONFLICT_TARGETS = [
    {"dimension": "region", "code": "R1", "population": 4},
    {"dimension": "age_group", "code": "a1", "population": 1},
]


@pytest.fixture()
def db(tmp_path):
    database = Database(tmp_path / "cancel.db")
    database.replace_samples(SAMPLES)
    database.replace_targets(FEAS_TARGETS)
    yield database
    database.close()


def wait_job(database: Database, job_id: str, timeout: float = 10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        row = database.get_job(job_id)
        if row is None:
            return None
        if row["status"] not in ("queued", "running"):
            return row
        time.sleep(0.02)
    raise TimeoutError(job_id)


def wait_until(predicate, timeout: float = 5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise TimeoutError("condition not met")


# ---------------------------------------------------------------------------
# 取消令牌 / 求解器检查点
# ---------------------------------------------------------------------------

def test_checkpoint_raises_and_token_is_bound_to_thread():
    token = CancelToken()
    checkpoint(token)  # 未取消：不抛
    token.cancel()
    with pytest.raises(JobCancelled):
        checkpoint(token)

    # 工作线程绑定：求解器不显式传令牌也能感知取消
    seen = {}

    def work():
        with use_cancel_token(token):
            try:
                checkpoint()  # 无参数，读取线程绑定令牌
            except JobCancelled:
                seen["cancelled"] = True

    t = threading.Thread(target=work)
    t.start()
    t.join()
    assert seen.get("cancelled") is True


def test_solve_qp_interrupts_at_qp_iteration():
    """可中断的优化迭代在令牌置位后立刻抛 JobCancelled。"""
    started = threading.Event()
    token = CancelToken()

    def endless_slsqp(a, p, d, cancel=None):
        started.set()
        while True:
            checkpoint(cancel)
            time.sleep(0.005)

    real = solver._slsqp
    solver._slsqp = endless_slsqp
    try:
        outcome = {}

        def work():
            with use_cancel_token(token):
                try:
                    solver.solve_qp(SAMPLES, FEAS_TARGETS)
                except JobCancelled:
                    outcome["cancelled"] = True

        t = threading.Thread(target=work)
        t.start()
        assert started.wait(5)
        token.cancel()
        t.join(5)
        assert not t.is_alive(), "取消后求解器未在迭代点退出"
        assert outcome.get("cancelled") is True
    finally:
        solver._slsqp = real


# ---------------------------------------------------------------------------
# 作业层：排队 / 在途 / 不可中断调用
# ---------------------------------------------------------------------------

def test_correction_drops_queued_old_job_without_running_it(db, monkeypatch):
    """排在后面的旧作业在更正后直接退出，不占用计算位、不调用求解器。"""
    gate = threading.Event()
    started = threading.Event()
    calls = []

    def blocking_linprog(*args, **kwargs):
        # 模拟不可中断的 HiGHS 调用：先占住唯一的工作线程
        started.set()
        assert gate.wait(10)
        return real_linprog(*args, **kwargs)

    import scipy.optimize as so
    real_linprog = so.linprog
    monkeypatch.setattr(solver, "linprog", blocking_linprog)

    manager = JobManager(db, max_workers=1)
    try:
        first = manager.submit()
        assert started.wait(5)

        # 第二个作业排队；在它开始前更正输入
        second = manager.submit()
        wait_until(lambda: db.get_job(second["job_id"])["status"] == "queued")
        real_solve = solver.solve_qp
        monkeypatch.setattr(solver, "solve_qp",
                            lambda *a, **k: calls.append(1) or real_solve(*a, **k))

        db.replace_samples(SAMPLES)  # 更正：epoch+1，取消旧令牌
        manager.notify_input_changed()

        # 旧作业记录已删除（查不到）
        assert db.get_job(first["job_id"]) is None
        assert db.get_job(second["job_id"]) is None

        gate.set()
        time.sleep(0.2)
        # 排队中的旧作业从未进入求解
        assert calls == []
        # 两个旧作业都没有留下任何状态
        rows = db._conn.execute("SELECT job_id, status, result FROM jobs").fetchall()
        assert rows == []
    finally:
        gate.set()
        manager.shutdown()


def test_correction_interrupts_running_job_at_iteration(db, monkeypatch):
    """运行中的旧作业在下一个可中断迭代点退出，且不写状态/结果。"""
    started = threading.Event()

    def endless_slsqp(a, p, d, cancel=None):
        started.set()
        while True:
            checkpoint(cancel)
            time.sleep(0.005)

    monkeypatch.setattr(solver, "_slsqp", endless_slsqp)
    manager = JobManager(db, max_workers=1)
    try:
        job = manager.submit()
        assert started.wait(5)
        assert db.get_job(job["job_id"])["status"] == "running"

        db.replace_targets(FEAS_TARGETS)
        manager.notify_input_changed()

        # 作业记录立即删除，且工作线程很快退出（下一次迭代检查点）
        assert db.get_job(job["job_id"]) is None
        wait_until(lambda: not manager._active)
        assert manager._active == {}
    finally:
        manager.shutdown()


def test_uninterruptible_call_returns_then_frees_slot_without_further_work(
        db, monkeypatch):
    """底层调用返回后立即释放计算位：不再做可行性判定/IIS/权重求解。"""
    import scipy.optimize as so
    real_linprog = so.linprog
    released = threading.Event()
    token_seen = {}

    def slow_linprog(*args, **kwargs):
        # 不可中断：更正发生在调用内部
        token_seen["during"] = True
        time.sleep(0.4)
        return real_linprog(*args, **kwargs)

    monkeypatch.setattr(solver, "linprog", slow_linprog)
    iis_calls = []
    qp_calls = []
    monkeypatch.setattr(solver, "find_iis",
                        lambda *a, **k: iis_calls.append(1) or [])
    monkeypatch.setattr(solver, "_slsqp",
                        lambda *a, **k: qp_calls.append(1))
    monkeypatch.setattr(solver, "_trust_constr",
                        lambda *a, **k: qp_calls.append(1))

    # 用不可行输入，若未取消就会进入 find_iis
    db.replace_samples(CONFLICT_SAMPLES)
    db.replace_targets(CONFLICT_TARGETS)

    manager = JobManager(db, max_workers=1)
    try:
        job = manager.submit()
        wait_until(lambda: token_seen.get("during"))
        # 更正与求解交错：更正立即返回，不等待慢调用
        t0 = time.time()
        db.replace_targets(CONFLICT_TARGETS)
        manager.notify_input_changed()
        assert time.time() - t0 < 0.3, "更正被旧作业阻塞"

        wait_until(lambda: not manager._active, timeout=5)
        released.set()
        # 慢调用返回后：不继续可行性判定之外的任何阶段
        assert iis_calls == []
        assert qp_calls == []
        assert db.get_job(job["job_id"]) is None
        assert db._conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0
        assert db.latest_succeeded_job() is None
    finally:
        released.set()
        manager.shutdown()


def test_new_epoch_job_uses_freed_slot_while_old_call_finishes(
        db, monkeypatch):
    """更正后新作业可用腾出的计算位；旧作业晚返回也不得写任何东西。"""
    import scipy.optimize as so
    real_linprog = so.linprog
    in_old_call = threading.Event()
    old_done = threading.Event()
    first_call = {"n": 0}

    def slow_first_linprog(*args, **kwargs):
        first_call["n"] += 1
        if first_call["n"] == 1:
            in_old_call.set()
            old_done.wait(10)  # 不可中断的旧调用，直到测试放行
        return real_linprog(*args, **kwargs)

    monkeypatch.setattr(solver, "linprog", slow_first_linprog)
    manager = JobManager(db, max_workers=1)
    try:
        old = manager.submit()
        assert in_old_call.wait(5)

        t0 = time.time()
        db.replace_samples(SAMPLES)  # 更正
        manager.notify_input_changed()
        elapsed = time.time() - t0

        # 更正未被旧作业阻塞；旧记录立刻查不到
        assert elapsed < 0.3
        assert db.get_job(old["job_id"]) is None

        # 当前版本的新作业已提交并等待同一个计算位
        new = manager.submit()
        assert new["epoch"] > old["epoch"]

        # 放行不可中断的旧调用；它返回后不得写结果，槽位立即转给新作业
        old_done.set()
        row = wait_job(db, new["job_id"], timeout=10)
        assert row is not None and row["status"] == "succeeded"
        assert db.get_job(old["job_id"]) is None
        assert db.latest_succeeded_job()["job_id"] == new["job_id"]
    finally:
        old_done.set()
        manager.shutdown()


def test_stale_epoch_cannot_write_status_or_result(db):
    """即使绕过作业层直接调用存储层，旧 epoch 也写不进任何状态/结果。"""
    stale_epoch = db.get_epoch()
    job, token = db.create_job_current()
    epoch = job["epoch"]
    assert epoch == stale_epoch

    db.replace_samples(SAMPLES)  # 更正使旧作业失效
    assert db.get_job(job["job_id"]) is None

    # 旧作业晚到：mark_running / finish_job 都必须拒绝
    assert db.mark_running(job["job_id"], epoch) is False
    assert db.finish_job(job["job_id"], epoch, "succeeded",
                         {"weights": []}, None) is False
    assert db.get_job(job["job_id"]) is None
    assert db.latest_succeeded_job() is None
