import cv2 as cv
import pickle
import aiohttp
import asyncio
from asgiref import sync
from tqdm.asyncio import tqdm_asyncio

async def gather_with_concurrency(n, *coros):
    semaphore = asyncio.Semaphore(n)
    async def sem_coro(coro):
        async with semaphore:
            return await coro
    return await tqdm_asyncio.gather(*(sem_coro(c) for c in coros))

def caption_video_stream(url,cap,max_concurrency = 500):
    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    async def pipe():
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout()) as session:
            async def pipe_frame():
                ret, frame = cap.read()
                frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB) if ret else None
                frame = pickle.dumps(frame)
                async with session.post(url, data=frame) as response:
                    return await response.text()
            return await gather_with_concurrency(max_concurrency,*[pipe_frame() for _ in range(total_frames)])
    return sync.async_to_sync(pipe)()

def caption_video_batch(url,frames):
    async def send_all(frames):
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout()) as session:
            async def send(frame):
                frame = pickle.dumps(frame)
                async with session.post(url, data=frame) as response:
                    return await response.text()
            return await asyncio.gather(*[send(frame) for frame in frames])
    return sync.async_to_sync(send_all)(frames)

