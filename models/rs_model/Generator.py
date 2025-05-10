import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.generateNoise import generate_noise

class Generator(nn.Module):

    def __init__(self, gen_batch_size, gen_dim_user, gen_dim_item, device):
        super().__init__()

        self.gen_batch_size = gen_batch_size
        self.gen_dim_user = gen_dim_user
        self.gen_dim_item = gen_dim_item
        self.device = device

        self.user_data = nn.Parameter((torch.rand(self.gen_batch_size, self.gen_dim_user)).to(self.device))
        self.item_data = nn.Parameter((torch.rand(self.gen_batch_size, self.gen_dim_item)).to(self.device))
    
    def reset(self):
        self.user_data.data = torch.rand(self.gen_batch_size, self.gen_dim_user).to(self.device)
        self.item_data.data = torch.rand(self.gen_batch_size, self.gen_dim_item).to(self.device)
    
    def set_grad(self, need_grad=True):
        self.user_data.requires_grad_(need_grad)
        self.item_data.requires_grad_(need_grad)
    
    def forward(self, difficulty_balance=1):
        return F.softmax(self.user_data / difficulty_balance, dim=1), F.softmax(self.item_data / difficulty_balance, dim=1)
    
    def pretrain_target_labels(self, teacher_list, optim_g, target_labels, pretrain_epoch, difficulty_balance=1):
        self.set_grad(True)
        loss_fn = nn.L1Loss()
        features = {
            'user_feature': None,
            'item_feature': None
        }

        for _ in range(pretrain_epoch):
            optim_g.zero_grad()
            features['user_feature'], features['item_feature'] = self.forward(difficulty_balance=difficulty_balance)
            t_logit_list = torch.stack([torch.sigmoid(teacher(features, discrete=False)) for teacher in teacher_list], dim=0)
            t_logit_overall = t_logit_list.mean(dim=0)
            t_logit_sorted, _ = torch.sort(t_logit_overall)
            print(t_logit_sorted)
            loss = loss_fn(t_logit_sorted, target_labels)
            loss.backward()
            optim_g.step()
        
        if pretrain_epoch > 0:
            del t_logit_list, t_logit_overall, t_logit_sorted
            torch.cuda.empty_cache()


    def pretrain_distribution(self, teacher_list, optim_g, mean, std, pretrain_epoch):
        # [0, 1]均匀分布下 均值为0.5，标准差为sqrt(1/12)≈0.289
        self.set_grad(True)
        loss_fn = nn.L1Loss()
        mean, std = torch.tensor(mean), torch.tensor(std)
        features = {
            'user_feature': None,
            'item_feature': None
        }
        for _ in range(pretrain_epoch):
            optim_g.zero_grad()
            features['user_feature'], features['item_feature'] = self.forward()
            t_logit_list = torch.stack([torch.sigmoid(teacher(features, discrete=False)) for teacher in teacher_list], dim=0)
            t_logit_overall = t_logit_list.mean(dim=0)
            _mean, _std = t_logit_overall.mean(), t_logit_overall.std()
            loss = loss_fn(_mean, mean) + loss_fn(_std, std)
            loss.backward()
            optim_g.step()


class Generator_f(nn.Module):
    '''
        distribution_type: 构建生成样本的随机噪声分布
    '''
    def __init__(self, distribution_type, gen_batch_size, gen_dim_feature, device, **kwargs):
        super().__init__()
        
        self.distribution_type = distribution_type
        self.distribution_params = kwargs
        self.device = device

        self.gen_batch_size = gen_batch_size
        self.feature_name_record = []
        for key in gen_dim_feature:
            self.feature_name_record.append(key)
            setattr(self, f'gen_dim_{ key }', gen_dim_feature[key])
            setattr(self, f'{ key }_data', nn.Parameter(generate_noise(distribution_type, 
                                                                       size=(gen_batch_size, gen_dim_feature[key]), kwargs=kwargs).to(device)))
        
        self.batch_data = {}
        for data in self.feature_name_record:
            print(data)
            self.batch_data[data] = None

    def reset(self):
        for data in self.feature_name_record:
            gen_dim = getattr(self, f'gen_dim_{ data }')
            noise = generate_noise(self.distribution_type, 
                                   size=(self.gen_batch_size, gen_dim), kwargs=self.distribution_params).to(self.device)
            getattr(self, f'{ data }_data').data = noise
    
    def set_grad(self, need_grad=True):
        for data in self.feature_name_record:
            getattr(self, f'{ data }_data').requires_grad_(need_grad)

    def forward(self, difficulty_balance):
        for data in self.feature_name_record:
            self.batch_data[data] = F.softmax(getattr(self, f'{ data }_data') / difficulty_balance[data], dim=1)
        return self.batch_data
    
    def pretrain_target_labels(self, teacher_list, optim_g, target_labels, pretrain_epoch, difficulty_balance):
        self.set_grad(True)
        loss_fn = nn.L1Loss()
        for _ in range(pretrain_epoch):
            optim_g.zero_grad()
            features = self.forward(difficulty_balance)
            t_logit_list = torch.stack([torch.sigmoid(teacher(features, discrete=False)) for teacher in teacher_list], dim=0)
            t_logit_overall = t_logit_list.mean(dim=0)
            t_logit_sorted, _ = torch.sort(t_logit_overall)
            loss = loss_fn(t_logit_sorted, target_labels)
            loss.backward()
            optim_g.step()

    def pretrain_distribution(self, teacher_list, optim_g, mean, std, pretrain_epoch, difficulty_balance):
        # [0, 1]均匀分布下 均值为0.5，标准差为sqrt(1/12)≈0.289
        self.set_grad(True)
        loss_fn = nn.L1Loss()
        mean, std = torch.tensor(mean), torch.tensor(std)
        for _ in range(pretrain_epoch):
            optim_g.zero_grad()
            features = self.forward(difficulty_balance)
            t_logit_list = torch.stack([torch.sigmoid(teacher(features, discrete=False)) for teacher in teacher_list], dim=0)
            t_logit_overall = t_logit_list.mean(dim=0)
            _mean, _std = t_logit_overall.mean(), t_logit_overall.std()
            loss = loss_fn(_mean, mean) + loss_fn(_std, std)
            loss.backward()
            optim_g.step()

    

class Feature_generation_block(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()
        dim_diff = output_dim // input_dim

        # '''
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim * dim_diff // 3),
            nn.BatchNorm1d(input_dim * dim_diff // 3),

            nn.Linear(input_dim * dim_diff // 3, input_dim * dim_diff // 3 * 2),
            nn.BatchNorm1d(input_dim * dim_diff // 3 * 2),

            nn.Linear(input_dim * dim_diff // 3 * 2, output_dim),
            nn.Softmax(dim=1)
        )

    
    def forward(self, noise):
        return self.net(noise)


class Generator_v(nn.Module):

    def __init__(self, gen_dim_user, gen_dim_item, input_dim):
        super().__init__()
        self.user_data = Feature_generation_block(input_dim, gen_dim_user)
        self.item_data = Feature_generation_block(input_dim, gen_dim_item)

    def set_grad(self, need_grad=True):

        for param in self.user_data.parameters():
            param.requires_grad = need_grad
        
        for param in self.item_data.parameters():
            param.requires_grad = need_grad

    def forward(self, noise):
        return self.user_data(noise), self.item_data(noise)
    
    def pretrain_target_labels(self, teacher_list, optim_g, target_labels, pretrain_epoch, noise):
        self.set_grad(True)
        loss_fn = nn.L1Loss()
        features = {
            'user_feature': None,
            'item_feature': None
        }
        
        for _ in range(pretrain_epoch):
            optim_g.zero_grad()
            noise = torch.rand(25, 100).to('cuda')
            features['user_feature'], features['item_feature'] = self.forward(noise)
            t_logit_list = torch.stack([torch.sigmoid(teacher(features, discrete=False)) for teacher in teacher_list], dim=0)
            t_logit_overall = t_logit_list.mean(dim=0)
            t_logit_sorted, _ = torch.sort(t_logit_overall)
            loss = loss_fn(t_logit_sorted, target_labels)
            loss.backward()
            optim_g.step()
    

class Generator_v_c(nn.Module):

    def __init__(self, gen_dim_user, gen_dim_item, input_dim):
        super().__init__()
        self.user_data = conditionFeatureGenerationBlock(input_dim, gen_dim_user)
        self.item_data = conditionFeatureGenerationBlock(input_dim, gen_dim_item)

    def set_grad(self, need_grad=True):

        for param in self.user_data.parameters():
            param.requires_grad = need_grad
        
        for param in self.item_data.parameters():
            param.requires_grad = need_grad

    def forward(self, noise, c):
        return self.user_data(noise, c), self.item_data(noise, c)
    
    def pretrain_target_labels(self, teacher_list, optim_g, target_labels, pretrain_epoch, noise):
        self.set_grad(True)
        loss_fn = nn.L1Loss()
        features = {
            'user_feature': None,
            'item_feature': None
        }
        
        for _ in range(pretrain_epoch):
            optim_g.zero_grad()
            noise = torch.rand(25, 100).to('cuda')
            features['user_feature'], features['item_feature'] = self.forward(noise)
            t_logit_list = torch.stack([torch.sigmoid(teacher(features, discrete=False)) for teacher in teacher_list], dim=0)
            t_logit_overall = t_logit_list.mean(dim=0)
            t_logit_sorted, _ = torch.sort(t_logit_overall)
            loss = loss_fn(t_logit_sorted, target_labels)
            loss.backward()
            optim_g.step()



class LoRALinear(nn.Module):
    def __init__(self, input_dim, output_dim, r=8):
        super(LoRALinear, self).__init__()
        # 原始矩阵的低秩分解
        self.A = nn.Linear(input_dim, r, bias=False)  # 第一部分低秩矩阵
        self.B = nn.Linear(r, output_dim, bias=False)  # 第二部分低秩矩阵

    def forward(self, x):
        # 用低秩分解的矩阵计算近似的线性变换
        return self.B(self.A(x))
    
class LoRALayer(nn.Module):
    def __init__(self, original_layer, rank):
        super(LoRALayer, self).__init__()
        self.original_layer = original_layer
        self.rank = rank
        # 低秩分解的 A 和 B 矩阵
        self.A = nn.Parameter(torch.randn(original_layer.weight.size(0), rank))
        self.B = nn.Parameter(torch.randn(rank, original_layer.weight.size(1)))
    
    def forward(self, x):
        # 残差相加：W + A @ B
        delta_w = self.A @ self.B
        return (self.original_layer.weight + delta_w) @ x


class Feature_generation_block_2(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, repeat_factor=5):
        super().__init__()
        self.hidden_count = len(hidden_dim)
        last_dim = input_dim
        for idx, dim in enumerate(hidden_dim):
            setattr(self, f'fc_{ idx }', nn.Linear(last_dim, dim))
            setattr(self, f'bn_{ idx }', nn.BatchNorm1d(dim))
            last_dim = dim
        group_count = math.ceil(output_dim / repeat_factor)
        self.repeat_factor = repeat_factor
        self.last_repeat = output_dim % repeat_factor
        setattr(self, f'fc_{ len(hidden_dim) }', nn.Linear(last_dim, group_count))
        
    def forward(self, x):
        for idx in range(self.hidden_count):
            x = getattr(self, f'fc_{ idx }')(x)
            x = getattr(self, f'bn_{ idx }')(x)
        self.outputs = getattr(self, f'fc_{ self.hidden_count }')(x)
        if self.last_repeat != 0:
            out_main = self.outputs[:, :-1].repeat_interleave(self.repeat_factor, dim=1)
            out_last = self.outputs[:, -1].unsqueeze(1).repeat_interleave(self.last_repeat, dim=1)
            final_output = torch.cat([out_main, out_last], dim=1)
        else:
            final_output = self.outputs.repeat_interleave(self.repeat_factor, dim=1)
        return F.softmax(final_output, dim=1)

    def get_original_outputs(self):
        return self.outputs
    

class Feature_generation_block_3(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),  # 减少中间层大小
            nn.BatchNorm1d(1024),
            nn.Linear(1024, output_dim),  # 依然输出为162542
        )
        
    def forward(self, x):
        return torch.softmax(self.net(x), dim=1)
    

class Generator_ml_25m(nn.Module):

    def __init__(self, gen_dim_user, gen_dim_item, input_dim):
        super().__init__()
        self.user_data = Feature_generation_block_3(input_dim, gen_dim_user)
        self.item_data = Feature_generation_block_3(input_dim, gen_dim_item)

    def forward(self, noise):
        return self.user_data(noise), self.item_data(noise)
    
    def pretrain_target_labels(self, teacher_list, optim_g, target_labels, pretrain_epoch, noise):
        loss_fn = nn.L1Loss()
        features = {
            'user_feature': None,
            'item_feature': None
        }
        for _ in range(pretrain_epoch):
            optim_g.zero_grad()
            noise = torch.rand(25, 100).to('cuda')
            features['user_feature'], features['item_feature'] = self.forward(noise)
            t_logit_list = torch.stack([torch.sigmoid(teacher(features, discrete=False)) for teacher in teacher_list], dim=0)
            t_logit_overall = t_logit_list.mean(dim=0)
            t_logit_sorted, _ = torch.sort(t_logit_overall)
            loss = loss_fn(t_logit_sorted, target_labels)
            print(t_logit_sorted)
            loss.backward()
            optim_g.step()
        
        if pretrain_epoch > 0:
            del t_logit_list, t_logit_overall, t_logit_sorted
            torch.cuda.empty_cache()


class SharedOutputMLP(nn.Module):
    def __init__(self, input_dim, shared_output_dim, repeat_factor):
        super(SharedOutputMLP, self).__init__()
        # 定义一个较小的全连接层，输出维度为 shared_output_dim
        self.fc = nn.Linear(input_dim, shared_output_dim)
        # Repeat factor 表示每个区域的值需要重复的次数
        self.repeat_factor = repeat_factor

    def forward(self, x):
        # 计算小的输出
        out = self.fc(x)  # 形状为 [batch_size, shared_output_dim]
        # 重复每个值 repeat_factor 次，得到最终输出
        out = out.repeat_interleave(self.repeat_factor, dim=1)  # 形状为 [batch_size, shared_output_dim * repeat_factor]
        return out


class DeconvGenerator(nn.Module):
    def __init__(self, input_dim, output_size):
        super(DeconvGenerator, self).__init__()
        self.input_dim = input_dim
        self.output_size = output_size  # 最终输出的大小（200,000）

        # print(self.output_size)
        
        self.fc = nn.Linear(input_dim, 512 * 4 * 4)  # 将噪声扩展到较高维度
        
        # 反卷积层用于上采样
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1),  # 4x4 -> 8x8
            nn.BatchNorm2d(512),
            # nn.ReLU(True),
            
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # 8x8 -> 16x16
            nn.BatchNorm2d(256),
            # nn.ReLU(True),
            
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 16x16 -> 32x32
            nn.BatchNorm2d(128),
            # nn.ReLU(True),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # 32x32 -> 64x64
            nn.BatchNorm2d(64),
            # nn.ReLU(True),
            
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),    # 64x64 -> 128x128
            nn.BatchNorm2d(32),
            # nn.ReLU(True),
            
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),    # 128x128 -> 256x256
            nn.BatchNorm2d(16),
            # nn.ReLU(True),
            
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),     # 256x256 -> 512x512
            nn.BatchNorm2d(8),
            # nn.ReLU(True),
            
            nn.ConvTranspose2d(8, 1, kernel_size=3, stride=1, padding=1),      # 保持 512x512
            nn.Tanh()  # 假设图像范围在 [-1, 1] 之间
        )

    def forward(self, z):
        # 将噪声 z 经过全连接层扩展并reshape
        x = self.fc(z).view(-1, 512, 4, 4)
        
        # 经过反卷积网络生成 512x512 图像
        x = self.deconv(x)
        
        # 展平图像并取前 200,000 个像素
        x = x.view(-1, 512 * 512)[:, :self.output_size]
        
        # 对输出做 Softmax
        x = torch.softmax(x / 0.4, dim=1)
        
        return x


class conditionFeatureGenerationBlock(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()

        dim_diff = output_dim // input_dim

        self.label_embedding = nn.Embedding(2, 8)

        self.net = nn.Sequential(
            nn.Linear(input_dim + 8, input_dim * dim_diff // 3),
            nn.BatchNorm1d(input_dim * dim_diff // 3),

            nn.Linear(input_dim * dim_diff // 3, input_dim * dim_diff // 3 * 2),
            nn.BatchNorm1d(input_dim * dim_diff // 3 * 2),

            nn.Linear(input_dim * dim_diff // 3 * 2, output_dim),
            nn.Softmax(dim=1)
        )
    
    def get_label_embedding(self, c):
        emb_0 = self.label_embedding(torch.tensor(0).to('cuda'))
        emb_1 = self.label_embedding(torch.tensor(1).to('cuda'))
        weighted_embedding = c * emb_0 + (1 - c) * emb_1
        return weighted_embedding

    def forward(self, z, c):
        c = c.view(-1, 1)
        weighted_embedding = self.get_label_embedding(c)
        x = torch.cat([z, weighted_embedding], dim=1)
        return self.net(x)
