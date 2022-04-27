"""
Gaussian modeling-based multichannel audio sourceseparation
exploiting generic source spectral model
の実装 (途中)
    - init は MNMF の分離結果から引っ張ってくるように
    - とりあえず分離できている雰囲気
    - 初期値はかなり重要，sigma^2 のパラメータには気を遣った方がよい
        - sigma^2: 小 => 定常ノイズ: 小，sigma^2: 大 => 定常ノイズ: 大
"""

import torch
from torchnmf.nmf import NMF
from .base import tf_bss_model_base
from .fastmnmf import fastmnmf

from tqdm.auto import tqdm


class lgm(tf_bss_model_base):
    def __init__(self, **stft_args):
        super().__init__(**stft_args)

    def forward(self, xnt, **separate_args):
        xlkn = self.stft(xnt).permute(2, 1, 0)
        ret_val = self.separate(xlkn, **separate_args)
        ymlkn, losses = ret_val if type(ret_val) == tuple else (ret_val, None)
        ymnt = self.istft(ymlkn.permute(0, 3, 2, 1), xnt.shape[-1])

        return ymnt if losses is None else (ymnt, losses)

    def separate(self, X):
        # ここでは実装を省略 (自分が書いたわけではないので)
        raise NotImplementedError


class nfm_lgm(lgm):
    def __init__(self, **stft_args):
        super().__init__(**stft_args)

    def separate(
        self,
        X,
        n_src=None, n_iter=10, n_components=4, sigma2=1e-2, eps=1e-10,
        return_losses=False
    ):
        # X[t, f, c] -> Y[n, t, f, c]
        n_frames, n_freq, n_chan = X.shape
        dtype, device = X.dtype, X.device

        if n_src is None:
            n_src = n_chan

        X_ftc = X.permute(1, 0, 2)
        I = torch.eye(n_chan, device=X.device)
        I_eps = I*eps

        def normalize_RV(P):
            # normalize R[..., c, c] (complex tensor)
            if P.is_complex():
                Pnorm = (P*P.conj()).real.sum((-2, -1), keepdim=True)
            # normalize V[..., f, t] (real tensor)
            else:
                Pnorm = (P**2).sum((-2, -1), keepdim=True)

            return P / Pnorm.clip(eps)

        def normalize_W(W):
            Wnorm = W.sum(1, keepdim=True)
            return W / Wnorm.clip(eps)

        def normalize_H(H):
            Hnorm = H.sum(2, keepdim=True)
            return H / Hnorm.clip(eps)

        def trace(A):
            tr = lambda x: torch.diagonal(x, offset=0, dim1=-1, dim2=-2).sum(-1)
            trA = tr(A)
            return trA

        def determinant(A):
            det = lambda x: torch.det(x)
            detA = det(A)
            return detA

        # init parameters using MNMF
        mnmf = fastmnmf(**self.stft_args)
        V_nft = mnmf.separate(
            X, n_src=n_src, n_iter=30, n_components=2
        ).abs().permute(2, 1, 0)
        V_nft /= V_nft.max()

        Rx_ftcc = torch.einsum('ftc,ftd->ftcd', X_ftc, X_ftc.conj())
        R_nfcc = torch.einsum('nft,ftcd->nftcd', V_nft, Rx_ftcc).mean(2)

        nmf = NMF(V_nft.shape[1:], n_components)
        W_nfk, H_nkt = [], []
        for V_ft in V_nft:
            nmf.fit(V_ft)
            W_nfk.append(nmf.H.detach())
            H_nkt.append(nmf.W.detach().mT)

        W_nfk = normalize_W(torch.stack(W_nfk))
        H_nkt = normalize_H(torch.stack(H_nkt))
        V_nft = normalize_RV(torch.bmm(W_nfk, H_nkt)).clip(0)

        Rb_fcc = sigma2*torch.eye(n_chan, dtype=dtype, device=device).tile(n_freq, 1, 1)
        R_nfcc = normalize_RV(R_nfcc)

        loss_list = []
        for epoch in tqdm(range(n_iter)):
            # E-step
            V_nft = V_nft if epoch == 0 else torch.bmm(W_nfk, H_nkt).clip(0)
            R_nftcc = torch.einsum('nft,nfcd->nftcd', V_nft, R_nfcc)
            Rx_ftcc = R_nftcc.sum(0) + Rb_fcc[:, None]
            Rx_inv = (Rx_ftcc+I_eps).pinverse()
            R_nkftcc = torch.einsum('nfk,nkt,nfcd->nkftcd', W_nfk, H_nkt, R_nfcc)

            G_nftcc = torch.matmul(R_nftcc, Rx_inv)
            Yh_nftc = torch.einsum('nftcd,ftd->nftc', G_nftcc, X_ftc)
            Rh_nftcc = \
                torch.einsum('nftc,nftd->nftcd', Yh_nftc, Yh_nftc.conj()) + \
                torch.matmul(I-G_nftcc, R_nftcc)

            G_nkftcc = torch.matmul(R_nkftcc, Rx_inv)
            Yh_nkftc = torch.einsum('nkftcd,ftd->nkftc', G_nkftcc, X_ftc)
            Rh_nkftcc = \
                torch.einsum('nkftc,nkftd->nkftcd', Yh_nkftc, Yh_nkftc.conj()) + \
                torch.matmul(I-G_nkftcc, R_nkftcc)

            Gb_ftcc = torch.matmul(Rb_fcc[:, None], Rx_inv)
            Nh_ftc = torch.einsum('ftcd,ftd->ftc', Gb_ftcc, X_ftc)
            Rbh_ftcc = \
                torch.einsum('ftc,ftd->ftcd', Nh_ftc, Nh_ftc.conj()) + \
                torch.matmul(I-Gb_ftcc, Rb_fcc[:, None])

            # M-step
            R_nfcc = (Rh_nftcc/V_nft.clip(eps)[..., None, None]).mean(2)
            R_inv = (R_nfcc+I_eps).pinverse()

            Vh_nkft = trace(
                torch.einsum('nfcd,nkftde->nkftce', R_inv, Rh_nkftcc)
            ).real.clip(0) / n_chan
            W_nfk, H_nkt = (
                normalize_W((Vh_nkft/H_nkt.clip(eps)[..., None, :]).mean(3).mT),
                normalize_H((Vh_nkft/W_nfk.clip(eps).mT[..., None]).mean(2))
            )

            Rb_fcc = Rbh_ftcc.mean(1)*I

            # loss calculation
            if return_losses:
                loss = (
                    torch.einsum(
                        'ftc,ftcd,ftd->ft', X_ftc.conj(), Rx_inv, X_ftc
                    ).real.abs() + \
                    torch.log(determinant(Rx_ftcc).real.abs()+eps)
                ).sum()
                loss_list.append(loss.item())

        # output of separated signal using last parameters
        V_nft = torch.bmm(W_nfk, H_nkt).clip(0)
        R_nftcc = torch.einsum('nft,nfcd->nftcd', V_nft, R_nfcc)
        Rx_ftcc = R_nftcc.sum(0) + Rb_fcc[:, None]
        Rx_inv = (Rx_ftcc+I_eps).pinverse()
        G_nftcc = torch.matmul(R_nftcc, Rx_inv)
        Yh_nftc = torch.einsum('nftcd,ftd->nftc', G_nftcc, X_ftc)

        # Y[n, t, f, c]
        Y = Yh_nftc.permute(0, 2, 1, 3)
        # normalize amplitude
        Y = Y*(
            X.abs().max()/Y.reshape(n_src, -1).abs().max(-1).values
        )[:, None, None, None]

        if return_losses:
            return Y, loss_list
        else:
            return Y
