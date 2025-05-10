import torch

def generate_noise(distribution_type='uniform', size=(25, 32), **kwargs):
    """
    生成随机噪声

    参数:
        distribution_type (str): 噪声的分布类型，支持 'uniform', 'normal', 'custom_normal', 'bernoulli', 'integer', 'binary'
        size (tuple): 生成噪声的张量形状
        **kwargs: 其他可选参数，如 mean, std, p 等

    返回:
        torch.Tensor: 生成的噪声张量。
    """
    # 均匀分布 [0, 1)
    if distribution_type == 'uniform':
        noise = torch.rand(size)
    
    # 标准正态分布
    elif distribution_type == 'normal':
        noise = torch.randn(size)
    
    # 自定义正态分布
    elif distribution_type == 'custom_normal':
        mean = kwargs.get('mean', 0)
        std = kwargs.get('std', 1)
        noise = torch.normal(mean=mean, std=std, size=size)
    
    # 伯努利分布 (0 或 1)
    elif distribution_type == 'bernoulli':
        p = kwargs.get('p', 0.5)
        noise = torch.bernoulli(torch.full(size, p))
    
    # 离散均匀分布，生成整数
    elif distribution_type == 'integer':
        low = kwargs.get('low', 0)
        high = kwargs.get('high', 10)
        noise = torch.randint(low, high, size)
    
    # 二值化噪声
    elif distribution_type == 'binary':
        threshold = kwargs.get('threshold', 0.5)
        noise = (torch.rand(size) > threshold).float()
    
    else:
        raise ValueError(f"Unsupported distribution type: { distribution_type }")
    
    return noise
