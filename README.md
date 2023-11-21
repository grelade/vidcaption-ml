### ML Video captioning tool

A tool for creating video captions in bulk. Uses the [BLIP2 + OPT](https://huggingface.co/Salesforce/blip2-opt-2.7b) img2txt model as the captioning module.

Forms the basic building block in the preprocessing pipeline used in the forthcoming <a>putin-dataset</a> comprising of Vladimir Putin clips. 

#### single-gpu server

Running the single-GPU captioning tool works on a Flask HTTP server.
```
cd flask
pip install -r requirements.txt
./start.sh
```
(loading BLIP2 into memory takes several seconds)

Two example clients are provided: 
* `client_example-dummydata.py` (finds captions on a random noise data)
* `client_example-putinmask.py` (finds a putin mask in an example video `sample_videos/putin_test.mp4` based on the provided captions)

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
* `client_example-dummydata.py` (finds captions on a random noise data)
* `client_example-batch.py` (find captions with batches *can be slower*)
* `client_example-stream.py` (find captions using asyncio stream *faster but can clog the computer*)
* `client_example-stream-v2.py` (find captions using batched asyncio streaming *fastest*)
* `client_example-putinmask.py` (finds a putin mask in an example video `sample_videos/putin_test.mp4` based on the provided captions)

##### config.properties

To **change the number of gpus** we must set the `number_of_gpu` parameter and change the `minWorkers`, `maxWorkers` parameters accordingly in the model specification. To use only a subset of gpus available, set the `CUDA_VISIBLE_DEVICES` variable to a list in the `start.sh` script.

### Technology used

* **huggingface transformers** - BLIP2 model
* **flask** - used by the single-gpu server
* **torchserve** - used by the multi-gpu server
* **aiohttp, asyncio** - concurrent runtime in bulk
