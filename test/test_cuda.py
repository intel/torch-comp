import torch_comp;torch_comp.compatible_mode()
import torch
import torch.nn as nn
import time

# 1. 检查CUDA可用性
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}\n")

# 2. 定义简单线性网络
class LinearNet(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        return self.linear(x)

# 3. 参数配置
input_size = 1000
output_size = 500
batch_size = 1024
num_tests = 100

# 4. 初始化模型和数据
model = LinearNet(input_size, output_size).to(device)
input_data = torch.randn(batch_size, input_size).to(device)

# 5. GPU预热 (避免首次运行的初始化时间影响测试)
with torch.no_grad():
    for _ in range(10):
        _ = model(input_data)
torch.cuda.synchronize()  # 等待CUDA操作完成

# 6. 性能测试函数
def run_performance_test(device_name):
    """测试指定设备的计算性能"""
    local_model = LinearNet(input_size, output_size)
    local_data = torch.randn(batch_size, input_size)
    
    if device_name == "cuda":
        local_model = local_model.cuda()
        local_data = local_data.cuda()
    
    # 预热
    with torch.no_grad():
        for i in range(10):
            res = local_model(local_data)
            print(f"第{i}次，结果的device是: {res.device}")

    if device_type == "cuda":
        print("cuda path")
    elif device_type == "cpu":
        print("cpu path")
    else:
        raise RuntimeError(f"Unsupported device: {res.device}")


run_performance_test("cuda")