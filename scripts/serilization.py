import time
import pickle
import joblib
import onnx
import onnxruntime as ort
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import torch

def save_model_pickle(model, file):
    start_time = time.time()
    with open(file, 'wb') as f:
        pickle.dump(model, f)
    return time.time() - start_time  

def load_model_pickle(file):
    start_time = time.time()
    with open(file, 'rb') as f:
        model = pickle.load(f)
    return time.time() - start_time, model 

def save_model_joblib(model, file) :
    start_time = time.time()
    joblib.dump(model, file)
    return time.time() - start_time

def load_model_joblib(file) :
    start_time = time.time()
    model = joblib.load(file)
    return time.time() - start_time, model

def save_model_onnx(model, file, x_sample):
    start_time = time.time()
    initial_type = [('float_input', FloatTensorType([None, x_sample.shape[1]]))]
    onnx_model = convert_sklearn(model, initial_types=initial_type)
    with open(file, 'wb') as f:
        f.write(onnx_model.SerializeToString())
    return time.time() - start_time

def load_model_onnx(file):
    start_time = time.time()
    onnx_model = onnx.load(file)
    ort_session = ort.InferenceSession(file)
    return time.time() - start_time, ort_session

def save_torch_model_full(model, file):
    start_time = time.time()
    torch.save(model, file)
    return time.time() - start_time

def load_torch_model_full(file):
    start_time = time.time()
    model = torch.load(file)
    return time.time() - start_time, model

file_size = {}
result = {}
