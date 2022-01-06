import numpy as np
import torch


beta1, beta2 = 0.9, 0.999


#  sgd
def optimize_w_sgd(model, optimizer, epsilon, loss, update_linear_lr, update_embedding_lr):
    optimizer.zero_grad()
    loss.backward(retain_graph=False)

    modules_list = ['classifier2', 'fusion']
    for name, params in model.named_parameters():
        if any(module in name for module in modules_list) is False:
            # print(name)
            if params is not None and params.grad is not None:
                lr = update_linear_lr
                if 'sentence_embedding' in name or 'image_embedding' in name:
                    lr = update_embedding_lr
                params.data.copy_(params - epsilon * lr * params.grad)
    return model
