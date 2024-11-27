from model_train import load_data, train_model, train_torch_model
from serilization import save_model_pickle, save_model_joblib, save_model_onnx, save_torch_model_full, load_model_joblib, load_model_onnx, load_model_pickle, load_torch_model_full
from utils import get_file_size
from sklearn import preprocessing

data = load_data()
x = data.iloc[:, 0:-1]
y = data.iloc[:, -1] 

x_normalized = preprocessing.normalize(x).astype(float)
input_size = x_normalized.shape[1] 

model_bayes = train_model(x_normalized, y)
model_torch = train_torch_model(x_normalized, y, input_size,epochs=10, learning_rate=0.01)

result = {}
sizes = {}

# Pickle
pickle_save_time = save_model_pickle(model_bayes, 'models/model_bayes.pkl')
pickle_load_time, _ = load_model_pickle('models/model_bayes.pkl')
sizes['Pickle'] = get_file_size('models/model_bayes.pkl')
result['Pickle'] = (save_model_pickle, load_model_pickle)

# Joblib
joblib_save_time = save_model_joblib(model_bayes, 'models/model_bayes.joblib')
joblib_load_time, _ = load_model_joblib('models/model_bayes.joblib')
sizes['Joblib'] = get_file_size('models/model_bayes.joblib')
result['Joblib'] = (save_model_joblib, load_model_joblib)

# ONNX
x_sample = x_normalized[:1]  
onnx_save_time = save_model_onnx(model_bayes, 'models/model_bayes.onnx', x_sample)
onnx_load_time, _ = load_model_onnx('models/model_bayes.onnx') 
sizes['ONNX'] = get_file_size('models/model_bayes.onnx')
result['ONNX'] = (onnx_save_time, onnx_load_time)

# PyTorch
torch_model = train_torch_model(x_normalized, y, input_size)
torch_save_time = save_torch_model_full(torch_model, "models/torch_model.pth")
torch_load_time, _ = load_torch_model_full("models/torch_model.pth")
sizes["PyTorch"] = get_file_size("models/torch_model.pth")
result["PyTorch"] = (torch_save_time, torch_load_time)

# Results Dictionary
result['Pickle'] = (pickle_save_time, pickle_load_time)  # Use the first element of the tuple
result['Joblib'] = (joblib_save_time, joblib_load_time)
result['ONNX'] = (onnx_save_time, onnx_load_time)
result['PyTorch'] = (torch_save_time, torch_load_time)

# Display Results
with open('result.txt', 'w') as f :
    f.write("Results (Save Time, Load Time):\n")
    for key, (save_time, load_time) in result.items():
        f.write(f"- {key}: Save Time = {save_time:.4f}s, Load Time = {load_time:.4f}s\n")

    f.write("\nFile Sizes (Bytes):\n")
    for key, size in sizes.items():
        f.write(f"- {key}: {size} bytes\n")

print('Results Saved in "result.txt"')

