"""
演示主进程启动多个子进程，并等待所有子进程初始化完成的机制。
模拟 SGLang Engine 启动 Scheduler 子进程的流程。
"""
import multiprocessing as mp
import time
import random
from typing import List, Dict


def worker_process(worker_id: int, init_time: int, pipe_writer):
    """
    子进程的工作函数，模拟 Scheduler 的初始化过程。

    Args:
        worker_id: 进程 ID（模拟 tp_rank）
        init_time: 初始化耗时（秒）
        pipe_writer: 管道写端，用于向主进程发送就绪信号
    """
    print(f"[Worker {worker_id}] 子进程启动，PID={mp.current_process().pid}")
    print(f"[Worker {worker_id}] 开始初始化，预计耗时 {init_time} 秒...")

    # 模拟耗时的初始化操作（加载模型、分配显存、编译 CUDA Graph 等）
    for i in range(init_time):
        time.sleep(1)
        print(f"[Worker {worker_id}] 初始化进度: {i + 1}/{init_time} 秒")

    # 初始化完成，向主进程发送就绪信号
    ready_info = {
        "status": "ready",
        "worker_id": worker_id,
        "pid": mp.current_process().pid,
        "max_batch_size": 32,  # 模拟一些配置信息
        "memory_allocated_gb": random.randint(10, 20),
    }

    print(f"[Worker {worker_id}] 初始化完成！发送就绪信号...")
    pipe_writer.send(ready_info)
    pipe_writer.close()

    # 模拟子进程继续运行（实际的 Scheduler 会进入事件循环处理请求）
    print(f"[Worker {worker_id}] 进入工作循环...")
    while True:
        time.sleep(5)
        print(f"[Worker {worker_id}] 正在运行中...")


def wait_for_workers_ready(pipe_readers: List, worker_procs: List) -> List[Dict]:
    """
    等待所有子进程初始化完成，模拟 _wait_for_scheduler_ready 函数。

    Args:
        pipe_readers: 管道读端列表
        worker_procs: 子进程列表

    Returns:
        所有子进程的配置信息列表
    """
    print("\n[主进程] 等待所有子进程初始化完成...")
    worker_infos = []

    for i in range(len(pipe_readers)):
        try:
            print(f"[主进程] 等待 Worker {i} 就绪...")
            # 阻塞等待第 i 个子进程发送就绪信号
            data = pipe_readers[i].recv()
            print(f"[主进程] ✓ Worker {i} 已就绪！收到信息: {data}")
        except EOFError:
            print(f"[主进程] ✗ Worker {i} 意外退出！")
            worker_procs[i].join()
            print(f"[主进程] Worker {i} 退出码: {worker_procs[i].exitcode}")
            raise

        if data["status"] != "ready":
            raise RuntimeError(f"Worker {i} 初始化失败！")

        worker_infos.append(data)

    return worker_infos


def main():
    """主进程入口"""
    print("=" * 60)
    print("多进程初始化 Demo - 模拟 SGLang Engine 启动流程")
    print("=" * 60)

    # 配置：启动 3 个子进程，模拟 tp_size=3
    num_workers = 3
    init_times = [3, 5, 4]  # 每个子进程的初始化耗时（秒）

    print(f"\n[主进程] 准备启动 {num_workers} 个子进程...")
    print(f"[主进程] 初始化耗时: {init_times}")

    # 启动子进程
    worker_procs = []
    pipe_readers = []

    for i in range(num_workers):
        # 创建单向管道：子进程写，主进程读
        reader, writer = mp.Pipe(duplex=False)

        # 创建子进程
        proc = mp.Process(
            target=worker_process,
            args=(i, init_times[i], writer)
        )
        proc.start()

        worker_procs.append(proc)
        pipe_readers.append(reader)

    print(f"[主进程] 已启动 {num_workers} 个子进程")

    # 等待所有子进程初始化完成
    start_time = time.time()
    worker_infos = wait_for_workers_ready(pipe_readers, worker_procs)
    elapsed_time = time.time() - start_time

    # 打印汇总信息
    print("\n" + "=" * 60)
    print(f"[主进程] 所有子进程初始化完成！总耗时: {elapsed_time:.2f} 秒")
    print("=" * 60)
    print("\n[主进程] 子进程配置信息汇总:")
    for info in worker_infos:
        print(f"  - Worker {info['worker_id']}: "
              f"PID={info['pid']}, "
              f"max_batch_size={info['max_batch_size']}, "
              f"memory={info['memory_allocated_gb']}GB")

    print("\n[主进程] 引擎已就绪，可以开始处理请求！")
    print("[主进程] 按 Ctrl+C 退出...")

    # 主进程继续运行（实际的 Engine 会在这里处理用户请求）
    try:
        while True:
            time.sleep(2)
            print("[主进程] 引擎运行中...")
    except KeyboardInterrupt:
        print("\n[主进程] 收到退出信号，正在关闭子进程...")
        for proc in worker_procs:
            proc.terminate()
            proc.join()
        print("[主进程] 所有子进程已关闭，退出。")


if __name__ == "__main__":
    # 设置多进程启动方式为 spawn（与 SGLang 一致）
    mp.set_start_method("spawn", force=True)
    main()