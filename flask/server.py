from flask import Flask
from flask import request
from flask import jsonify
import pickle

from transformers import AutoProcessor, Blip2ForConditionalGeneration
import torch

PORT = 5000

device = "cuda:0" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b", device=device)
print(f'loading the BLIP2 model... device={device}')
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)
_ = model.to(device)

app = Flask(__name__)

@app.route("/")
def main():
    return '''captions-model server<br>use /caption'''

@app.post("/caption")
def caption():
    f = request.files['media']
    frames = pickle.loads(f.read())
    
    inputs = processor(frames, return_tensors="pt").to(device, torch.float16)   
    generated_ids = model.generate(**inputs, max_new_tokens=100)
    generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
    del frames
    del inputs
    del generated_ids
    torch.cuda.empty_cache()
    
    return jsonify(generated_texts)

if __name__ == "__main__":
    
    app.run(debug = False, host='0.0.0.0', port = PORT)