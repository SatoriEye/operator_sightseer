import subprocess
import torch
import time
import os
import torchvision.models as models
from testnn import HybridModel


def get_gpu_memory_info():
    # 使用 nvidia-smi 获取 GPU 信息
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.total,memory.used', '--format=csv,noheader,nounits'],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if result.returncode != 0:
        raise Exception("Failed to execute nvidia-smi: " + result.stderr.decode())

    # 解码输出
    output = result.stdout.decode().strip()

    # 解析输出
    memory_total, memory_used = map(int, output.split(', '))

    return memory_total, memory_used

# 示例使用：get_gpu_memory_info：测试GPU显存

# try:
#     total_memory, used_memory = get_gpu_memory_info()
#     print(f"Total GPU Memory: {total_memory} MiB")
#     print(f"Used GPU Memory: {used_memory} MiB")
# except Exception as e:
#     print(e)


def measure_tensor_transfer_times(sizes, num_trials=50):
    # 检查是否有可用的GPU
    if not torch.cuda.is_available():
        raise Exception(
            "No CUDA-enabled GPU available. Please ensure that you have a GPU and the correct drivers installed.")

    # 获取所有可用的GPU设备
    devices = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]

    # 初始化结果字典
    results = {f"GPU_{i}": {} for i in range(len(devices))}

    # 在CPU上初始化多个不同尺寸的张量
    tensors_cpu = {size: torch.rand(size) for size in sizes}

    # 对每个张量进行测量
    for size, tensor_cpu in tensors_cpu.items():
        for device_idx, device in enumerate(devices):
            total_time = 0.0
            for _ in range(num_trials):
                start_time = time.time()
                tensor_gpu = tensor_cpu.to(device)
                del tensor_gpu  # 删除以释放GPU内存
                end_time = time.time()
                total_time += (end_time - start_time)

            # 计算平均时间
            avg_time = total_time / num_trials
            results[f"GPU_{device_idx}"][str(size)] = avg_time

    return results

# 示例使用：measure_tensor_transfer_times：从torch角度测试移动时间

# sizes = [(4, 4), (8, 8), (16, 16), (32, 32), (64, 64), (128, 128),
#          (256, 256), (512, 512), (1024, 1024), (2048, 2048), (4096, 4096), (8192, 8192)]
#
# try:
#     results = measure_tensor_transfer_times(sizes)
#     for gpu, data in results.items():
#         print(f"{gpu}:")
#         for size, avg_time in data.items():
#             print(f"  Size: {size}, Avg Transfer Time: {avg_time:.6f} seconds")
# except Exception as e:
#     print(e)


def model_moving_speed_test(epochs=50):
    # 模型移动速度测试，使用resnet18做重复测试
    # 检查 CUDA 是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    weight_path = os.path.join(script_dir, 'test_weight/resnet18-f37072fd.pth')

    total_time = 0.0
    for i in range(epochs):
        start_time = time.time()

        # 加载模型权重
        model = models.resnet18(pretrained=False)
        state_dict = torch.load(weight_path)
        # 将加载的权重加载到模型中
        model.load_state_dict(state_dict)
        model.to(device)

        end_time = time.time()
        total_time += (end_time - start_time)

    return total_time / epochs


def model_forward_speed_test(epochs=50):
    # 参数设置
    input_size = 100  # Input size for RNN
    hidden_size = 128  # Hidden size for RNN
    num_classes = 10  # Number of classes for classification
    graph_input_dim = 1433  # Input dimension for GCN (e.g., number of node features)
    gcn_output_dim = 16  # Output dimension for GCN
    d_model = 512  # Dimension of the model
    nhead = 8  # Number of attention heads
    num_encoder_layers = 6  # Number of encoder layers
    dim_feedforward = 2048  # Dimension of the feedforward network

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HybridModel(input_size, hidden_size, num_classes, graph_input_dim,
                        gcn_output_dim, d_model, nhead, num_encoder_layers, dim_feedforward)

    # 示例输入数据
    batch_size = 32
    seq_length = 50
    rnn_input = torch.randn(batch_size, seq_length, input_size)
    cnn_input = torch.randn(batch_size, 1, 28, 28)  # Example input for a 28x28 image
    transformer_input = torch.randn(batch_size, seq_length, d_model)  # Example input for transformer
    graph_input = torch.randn(batch_size, graph_input_dim)  # Example graph node features
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)  # Example graph edges

    rnn_input = rnn_input.to(device)
    cnn_input = cnn_input.to(device)
    transformer_input = transformer_input.to(device)
    graph_input = graph_input.to(device)
    edge_index = edge_index.to(device)
    model.to(device)

    model.eval()

    # 不需要计算梯度
    start_time = time.time()
    with torch.no_grad():
        # 前向传播
        for i in range(epochs):
            output = model(rnn_input, cnn_input, transformer_input, graph_input, edge_index)

    end_time = time.time()

    return end_time-start_time, model.time_calculator
