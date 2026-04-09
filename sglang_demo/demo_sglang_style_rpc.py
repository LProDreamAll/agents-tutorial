import multiprocessing as mp
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union
import zmq


# ===== 消息体（像 sglang io_struct）=====
@dataclass
class RpcReqInput:
    method: str
    parameters: Optional[Dict[str, Any]] = None


@dataclass
class RpcReqOutput:
    success: bool
    message: str


@dataclass
class SchedulerEvent:
    event: str
    detail: Dict[str, Any]


Message = Union[RpcReqInput, RpcReqOutput, SchedulerEvent]
ENDPOINT = "ipc:///tmp/sglang_rpc_demo.sock"


# 也可用 tcp://127.0.0.1:29788
def scheduler_proc(endpoint: str):
    ctx = zmq.Context(1)
    # 模拟 sglang: scheduler 侧 connect
    rpc_sock = ctx.socket(zmq.DEALER)
    rpc_sock.connect(endpoint)
    print("[scheduler] connected")
    running = True
    while running:
        msg: Message = rpc_sock.recv_pyobj()
        print(f"[scheduler] recv: {msg}")
        if isinstance(msg, RpcReqInput):
            if msg.method == "shutdown":
                rpc_sock.send_pyobj(RpcReqOutput(success=True, message="scheduler bye"))
                running = False
                continue
            # 1) 回 RPC 响应
            rpc_sock.send_pyobj(
                RpcReqOutput(
                    success=True,
                    message=f"executed method={msg.method}, params={msg.parameters}",
                )
            )
            # 2) 再主动推一条事件
            rpc_sock.send_pyobj(
                SchedulerEvent(
                    event="task_done",
                    detail={"method": msg.method, "ts": time.time()},
                )
            )
    rpc_sock.close(0)
    ctx.term()
    print("[scheduler] exit")


def engine_main(endpoint: str):
    ctx = zmq.Context(1)
    # 模拟 sglang: engine 侧 bind
    rpc_sock = ctx.socket(zmq.DEALER)
    rpc_sock.bind(endpoint)
    p = mp.Process(target=scheduler_proc, args=(endpoint,), daemon=True)
    p.start()
    time.sleep(0.2)  # 等 scheduler connect
    # 主进程 -> scheduler
    req = RpcReqInput(method="save_sharded_model", parameters={"path": "/tmp/demo"})
    rpc_sock.send_pyobj(req)
    print(f"[engine] sent: {req}")
    # 接收两条：RPC response + scheduler event
    for i in range(2):
        obj: Message = rpc_sock.recv_pyobj()
        print(f"[engine] recv[{i}]: {obj}")
    # 关闭 scheduler
    rpc_sock.send_pyobj(RpcReqInput(method="shutdown"))
    print("[engine] sent: shutdown")
    print(f"[engine] recv: {rpc_sock.recv_pyobj()}")
    p.join(timeout=2)
    rpc_sock.close(0)
    ctx.term()


if __name__ == "__main__":
    engine_main(ENDPOINT)
