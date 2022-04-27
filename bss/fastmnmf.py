import torch
from .base import tf_bss_model_base
from pyroomacoustics.bss import fastmnmf as pra_mnmf


class fastmnmf(tf_bss_model_base):
    def __init__(self, **stft_args):
        super().__init__(**stft_args)

    def separate(self, X, **separate_args):
        # とりあえず動かせるようにするために pyroomacoustics を利用
        # 普段は pytorch 用に書き換えたものを利用しているが，ここでは省略

        X_np = X.numpy()
        Y = torch.from_numpy(pra_mnmf(X, **separate_args))

        return Y
