import torch
import torchaudio


class tf_bss_model_base(torch.nn.Module):
    def __init__(self, **stft_args):
        super(tf_bss_model_base, self).__init__()

        spec_fn = torchaudio.transforms.Spectrogram
        spec_args = {
            k: v for k, v in stft_args.items()
            if k in spec_fn.__init__.__annotations__.keys()
        }
        spec_args['power'] = None
        self.stft = spec_fn(**spec_args)

        ispec_fn = torchaudio.transforms.InverseSpectrogram
        ispec_args = {
            k: v for k, v in stft_args.items()
            if k in ispec_fn.__init__.__annotations__.keys()
        }
        self.istft = ispec_fn(**ispec_args)

        self.stft_args = stft_args

    def forward(self, xnt, **separate_args):
        xlkn = self.stft(xnt).permute(2, 1, 0)
        ylkn = self.separate(xlkn, **separate_args)
        ynt = self.istft(ylkn.permute(2, 1, 0), xnt.shape[-1])

        return ynt

    def separate(self, X):
        raise NotImplementedError
