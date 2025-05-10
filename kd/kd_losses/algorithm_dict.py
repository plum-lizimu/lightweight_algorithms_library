from kd_losses.soft_target import soft_target
from kd_losses.NKD import NKD
from kd_losses.NormKD import NormKD
from kd_losses.soft_target_binary import soft_target_binary

algorithm_dict = {
    'soft_target': soft_target,
    'NKD': NKD,
    'NormKD': NormKD,

    'soft_target_binary': soft_target_binary
}