"""
pyroomacoustics から pytorch で実行できるように移植したもの
実装は省略
"""

import torch
from .base import tf_bss_model_base


class fastmnmf(tf_bss_model_base):
    def __init__(self, **stft_args):
        super().__init__(**stft_args)

    def separate(self, X):
        # 実装は省略
        # 簡単に実行するなら，
        # (1): torch tensor の X を numpy ndarray に変換
        # (2): pyroomacoustics の bss モジュールに渡して 結果を torch tensor に変換
        # (3): return
        raise NotImplementedError
