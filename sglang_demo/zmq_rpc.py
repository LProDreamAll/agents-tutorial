import multiprocessing as mp
import time
import zmq

ENDPOINT = "tcp://127.0.0.1:29777"


def scheduler_proc(endpoint: str):
    ctx = zmq.Context()
    router = ctx.socket(zmq.ROUTER)  # scheduler 端
    router.bind(endpoint)
    print(f"[scheduler] listening at {endpoint}")
    while True:
        # DEALER -> ROUTER: [identity, payload]
        identity, payload = router.recv_multipart()
        msg = payload.decode("utf-8")
        print(f"[scheduler] recv: {msg}")
        if msg == "shutdown":
            router.send_multipart([identity, b"ack: shutdown"])
            break
        # 1) 回应主进程（request -> response）
        router.send_multipart([identity, f"ack: got '{msg}'".encode("utf-8")])
        # 2) scheduler 主动再发一条事件（server push）
        time.sleep(0.2)
        router.send_multipart([identity, b"event: scheduler finished task"])
    router.close()
    ctx.term()
    print("[scheduler] exit")


def engine_main(endpoint: str):
    ctx = zmq.Context()
    dealer = ctx.socket(zmq.DEALER)  # 主进程端
    dealer.setsockopt(zmq.IDENTITY, b"engine-1")  # 固定身份，便于 scheduler 回发
    dealer.connect(endpoint)
    # 主进程发消息给 scheduler
    dealer.send_string("hello scheduler")
    print("[engine] sent: hello scheduler")
    # 主进程接收两条：ack + event
    for _ in range(2):
        reply = dealer.recv().decode("utf-8")
        print(f"[engine] recv: {reply}")
    # 关闭 scheduler
    dealer.send_string("shutdown")
    print("[engine] sent: shutdown")
    print(f"[engine] recv: {dealer.recv().decode('utf-8')}")
    dealer.close()
    ctx.term()


if __name__ == "__main__":
    p = mp.Process(target=scheduler_proc, args=(ENDPOINT,), daemon=True)
    p.start()
    time.sleep(0.3)  # 等 scheduler bind
    engine_main(ENDPOINT)
    p.join(timeout=2)
