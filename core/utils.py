import cv2
import math
import torch
import numpy as np

from torch.optim.lr_scheduler import LambdaLR

def get_one_hot_vector(label, classes):
    vector = np.zeros((classes), dtype = np.float32)
    if len(label) > 0:
        vector[label] = 1.
    return vector

def get_learning_rate_from_optimizer(optimizer):
    return optimizer.param_groups[0]['lr']

def get_numpy_from_tensor(tensor):
    return tensor.cpu().detach().numpy()

def load_model(model, model_path):
    if torch.cuda.device_count() > 1:
        model.module.load_state_dict(torch.load(model_path))
    else:
        model.load_state_dict(torch.load(model_path))

def save_model(model, model_path):
    if torch.cuda.device_count() > 1:
        torch.save(model.module.state_dict(), model_path)
    else:
        torch.save(model.state_dict(), model_path)

def get_cosine_schedule_with_warmup(optimizer,
                                    warmup_iteration,
                                    max_iteration,
                                    cycles=7./16.
                                    ):
    def _lr_lambda(current_iteration):
        if current_iteration < warmup_iteration:
            return float(current_iteration) / float(max(1, warmup_iteration))

        no_progress = float(current_iteration - warmup_iteration) / float(max(1, max_iteration - warmup_iteration))
        return max(0., math.cos(math.pi * cycles * no_progress))
    
    return LambdaLR(optimizer, _lr_lambda, -1)