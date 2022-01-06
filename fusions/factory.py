import copy
# from bootstrap.lib.logger import Logger
from fusions.fusions import ConcatMLP


def factory(opt):
    ftype = opt.pop('type', None)  # rm type from dict
    
    if ftype == 'cat_mlp':
        fusion = ConcatMLP(**opt)
    else:
        raise ValueError()

    return fusion
