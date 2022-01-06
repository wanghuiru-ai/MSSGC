import copy
# from bootstrap.lib.logger import Logger
from fusions.fusions import Block
from fusions.fusions import BlockTucker
from fusions.fusions import MLB
from fusions.fusions import MFB
from fusions.fusions import MFH
from fusions.fusions import MCB
from fusions.fusions import Mutan
from fusions.fusions import Tucker
from fusions.fusions import LinearSum
from fusions.fusions import ConcatMLP


def factory(opt):
    ftype = opt.pop('type', None)  # rm type from dict

    if ftype == 'block':
        fusion = Block(**opt)
    elif ftype == 'block_tucker':
        fusion = BlockTucker(**opt)
    elif ftype == 'mlb':
        fusion = MLB(**opt)
    elif ftype == 'mfb':
        fusion = MFB(**opt)
    elif ftype == 'mfh':
        fusion = MFH(**opt)
    elif ftype == 'mcb':
        fusion = MCB(**opt)
    elif ftype == 'mutan':
        fusion = Mutan(**opt)
    elif ftype == 'tucker':
        fusion = Tucker(**opt, mm_dim=256)
    elif ftype == 'linear_sum':
        fusion = LinearSum(**opt, mm_dim=2816)
    elif ftype == 'cat_mlp':
        fusion = ConcatMLP(**opt)
    # elif ftype == 'concat':
    #     fusion = ConcatMLP(**opt)
    else:
        raise ValueError()

    return fusion
