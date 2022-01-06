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


#  sgd text
def optimize_text_sgd(model, optimizer, epsilon, loss, update_lr):
    optimizer.zero_grad()
    loss.backward(retain_graph=False)

    modules_list = ['image_embedding', 'classifier2', 'fusion']
    for name, params in model.named_parameters():
        if any(module in name for module in modules_list) is False:
            # print(name)
            if params is not None and params.grad is not None:
                lr = update_lr
                if 'sentence_embedding' in name:
                    lr = update_lr / 100
                params.data.copy_(params - epsilon * lr * params.grad)
    return model


#  image
def optimize_image_sgd(model, optimizer, epsilon, loss, update_lr):
    optimizer.zero_grad()
    loss.backward(retain_graph=False)

    modules_list = ['sentence_embedding', 'classifier2', 'fusion']
    for name, params in model.named_parameters():
        if any(module in name for module in modules_list) is False:
            # print(name)
            if params is not None and params.grad is not None:
                lr = update_lr
                if 'image_embedding' in name:
                    lr = update_lr / 100
                params.data.copy_(params - epsilon * lr * params.grad)
    return model


# adam
def optimize_w_adam(model, optimizer, epsilon, loss, iteration, vs, ss):
    step_iteration = iteration + 1
    eps = 1e-8
    step_vs, step_ss = vs, ss
    optimizer.zero_grad()
    loss.backward(retain_graph=False)

    for named_parameters, v, s in zip(model.named_parameters(),  step_vs, step_ss):
        name, params = named_parameters
        if 'classifier2' not in name:
            if params is not None and params.grad is not None:
                lr = optimizer.param_groups[0]['lr']
                if 'sentence_embedding' in name:
                    lr = optimizer.param_groups[1]['lr']
                if 'image_embedding' in name:
                    lr = optimizer.param_groups[2]['lr']
                v[:] = beta1 * v + (1 - beta1) * params.grad
                s[:] = beta2 * s + (1 - beta2) * params.grad ** 2
                v_hat = v / (1 - beta1 ** step_iteration)
                s_hat = s / (1 - beta2 ** step_iteration)
                params.data.copy_(params - epsilon * lr * v_hat / torch.sqrt(s_hat + eps))
    return model


# def copy_model_params(model, copy_model):
#     update_modules = ['sentence_embedding', 'linear1', 'image_embedding', 'linear3']
#     for named_model_param, copy_model_param in zip(model.named_parameters(), copy_model.parameters()):
#         name, model_param = named_model_param
#         for update_module in update_modules:
#             if update_module in name:
#                 model_param.data.copy_(copy_model_param.data)
#     return model


def copy_model_params(model, copy_model):
    for model_param, copy_model_param in zip(model.parameters(), copy_model.parameters()):
        model_param.data.copy_(copy_model_param.data)
    return model


def update_vs_ss(model, vs, ss):
    for named_parameters, v, s in zip(model.named_parameters(), vs, ss):
        name, params = named_parameters
        if params.grad is None:
            continue
        v[:] = beta1 * v + (1 - beta1) * params.grad
        s[:] = beta2 * s + (1 - beta2) * params.grad ** 2
    return vs, ss


def my_optimizer(model, optimizer, epsilon, loss, opt_names: []):
    # opt_names = ['sentence_embedding', 'linear1', 'image_embedding', 'linear3']
    optimizer.zero_grad()
    loss.backward(retain_graph=False)

    for name, params in model.named_parameters():
        for opt_name in opt_names:
            if opt_name in name:
                if params is not None and params.grad is not None:
                    lr = optimizer.param_groups[0]['lr']
                    if opt_name == 'sentence_embedding':
                        lr = optimizer.param_groups[1]['lr']
                    if opt_name == 'image_embedding':
                        lr = optimizer.param_groups[2]['lr']
                    params.data.copy_(params - epsilon * lr * params.grad)
    return model
