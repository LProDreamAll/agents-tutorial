# yield_demo.py
import asyncio
from dataclasses import dataclass


# 1) 普通 yield：生成器
def count(n: int):
    for i in range(n):
        print(f"[producer] make {i}")
        yield i


# 2) async yield：异步生成器
async def stream_words(text: str):
    for w in text.split():
        await asyncio.sleep(0.2)  # 模拟异步IO
        yield w


@dataclass
class Obj:
    is_single: bool
    items: list[str]


class Handler:
    async def _tokenize_one_request(self, obj: Obj):
        await asyncio.sleep(0.05)
        return obj.items[0].split()

    def _send_one_request(self, obj: Obj, tokenized_obj, created_time):
        return {"tokens": tokenized_obj, "created_time": created_time}

    async def _wait_one_response(self, obj: Obj, state, request):
        for t in state["tokens"]:
            await asyncio.sleep(0.1)  # 模拟服务端逐步返回
            yield {"mode": "single", "token": t}

    async def _handle_batch_request(self, obj: Obj, request, created_time):
        for i, text in enumerate(obj.items):
            for t in text.split():
                await asyncio.sleep(0.08)
                yield {"mode": "batch", "item": i, "token": t}

    async def handle(self, obj: Obj, request):
        created_time = asyncio.get_running_loop().time()
        if obj.is_single:
            tokenized_obj = await self._tokenize_one_request(obj)
            state = self._send_one_request(obj, tokenized_obj, created_time)
            async for response in self._wait_one_response(obj, state, request):
                yield response
        else:
            async for response in self._handle_batch_request(
                    obj, request, created_time
            ):
                yield response


async def main():
    # print("=== 普通生成器 ===")
    # for x in count(3):
    #     print(f"[consumer] got {x}")
    #
    # print("\n=== 异步生成器 ===")
    # async for w in stream_words("hello async yield demo"):
    #     print(f"[consumer] got {w}")

    h = Handler()

    # print("\n=== 单请求流式 ===")
    # async for r in h.handle(Obj(is_single=True, items=["a b c"]), request={}):
    #     print(r)

    print("\n=== 批请求流式 ===")
    async for r in h.handle(Obj(is_single=False, items=["x y", "m n"]), request={}):
        print(r)


if __name__ == "__main__":
    asyncio.run(main())
