import subprocess
import torch
import time
import os
import torchvision.models as models


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

    # 检查 CUDA 是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    weight_path = os.path.join(script_dir, 'test_weight')

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
