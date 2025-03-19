from nn import FalloverPredictor
import torch
from torch.autograd import profiler
import numpy as np
import yaml
import time

lstm_weights_path = "LSTM_weights/experiment1_weights.pth"
config_file = "g1.yaml"
num_features = 55

with open(config_file, "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    window_size = config["window_size"]

model = FalloverPredictor(input_size=num_features)
state_dict = torch.load(lstm_weights_path)
model.load_state_dict(state_dict)

num_param = 0
for name, parameter in model.named_parameters():
    num_param += np.prod(parameter.size())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processing_unit = "GPU" if device == "cuda" else "CPU"
model = model.to(device)
input_data = torch.randn(1, window_size, num_features).to(device)

# Warm-up runs
for _ in range(10):
    _ = model(input_data)

if device == "cuda":
    torch.cuda.synchronize()
start_time = time.time()
output = model(input_data)
if device == "cuda":
    torch.cuda.synchronize()
end_time = time.time()
inference_time = end_time - start_time

print(f"Number of parameters: {num_param}")
print(f"Average inference time with {processing_unit} synchronization: {inference_time:.4f} seconds")

# with profiler.profile(record_shapes=True) as prof:
#     with profiler.record_function("model_inference"):
#         output = model(input_data)
# print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))