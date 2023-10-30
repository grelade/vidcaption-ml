### ML Video captioning tool

A tool for creating video captions in bulk. Forms the basic element in the preprocessing pipeline used in the <a>putin-dataset</a> to extract clips with the president Putin. Uses the BLIP2 img2txt model as the captioning module.

#### single-gpu server

Running the single-GPU captioning tool works on a Flask HTTP server.
```
cd flask
pip install -r requirements.txt
./start.sh
```
(loading BLIP2 into memory takes several seconds)

Two example clients are provided: 
* client_example-dummydata.py (finds captions on a random noise data)
* client_example-putinmask.py (finds a putin mask in an example video `sample_videos/putin_test.mp4` based on the provided captions)

#### multi-gpu server

Running the captioning tool on multiple GPUs works with torchserve. 
```
cd torchserve
pip install -r requirements-torchserve.txt
./server-create.sh
./start.sh
```
(takes some time to load BLIP2 into GPU memory)

After the server is up and running, several example clients are provided:
* client_example-dummydata.py
* client_example-batch.py
* client_example-stream.py
* client_example-putinmask.py

### Technology used

* huggingface transformers - BLIP2 model
* flask - used by the single-gpu server
* torchserve - used by the multi-gpu server
* aiohttp, asyncio - concurrent runtime in bulk
