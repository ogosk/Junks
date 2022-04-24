"""
Gaussian modeling-based multichannel audio sourceseparation exploiting generic source spectral model
の実装 (未完成)
- 多分 init がよくない
"""

import torch
from torchnmf.nmf import NMF
from .base import tf_bss_model_base

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

    # ここでは実装を省略 (自分が書いたわけではないので)


class nfm_lgm(lgm):
    def __init__(self, **stft_args):
        super().__init__(**stft_args)

    def separate(
        self,
        X,
        n_src=None, n_iter=10, n_components=4, sigma2=1e-4, return_losses=False
    ):
        n_frames, n_freq, n_chan = X.shape

        if n_src is None:
            n_src = n_chan

        eps = 1e-18

        X_ftc = X.permute(1, 0, 2)
        I = torch.eye(n_chan)
        I_eps = I*eps

        # parameters that require init
        R_nfcc = torch.einsum(
            'ftc,ftd->ftcd', X_ftc, X_ftc.conj()
        ).mean(1)[None].tile(n_src, 1, 1, 1)
        Rb_fcc = sigma2*torch.eye(n_chan, dtype=X.dtype)[None].tile(n_freq, 1, 1)

        V_nft = torch.ones(n_src, n_freq, n_frames)
        nmf = NMF(V_nft.shape[1:], n_components)
        nmf.fit(V_nft[0])
        W_nfk = nmf.H.detach()[None].tile(n_src, 1, 1)
        H_nkt = nmf.W.detach()[None].tile(n_src, 1, 1).mT

        def abs_trace(A):
            return A.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1).abs()

        def normalize_RV(P):
            if P.is_complex():
                Pnorm = (P*P.conj()).real.sum((-2, -1), keepdim=True)
            else:
                Pnorm = (P**2).sum((-2, -1), keepdim=True)

            return P/Pnorm.clip(eps)

        def normalize_W(W):
            Wnorm = W.sum(1, keepdim=True).clip(eps)
            return W / Wnorm

        def normalize_H(H):
            Hnorm = H.sum(2, keepdim=True).clip(eps)
            return H / Hnorm

        loss_list = []
        for epoch in tqdm(range(n_iter)):
            # E-step
            V_nft = torch.bmm(W_nfk, H_nkt)
            R_nftcc = torch.einsum('nft,nfcd->nftcd', V_nft, R_nfcc)
            Rx_ftcc = R_nftcc.sum(0) + Rb_fcc[:, None]
            Rx_inv = (Rx_ftcc + I_eps).inverse()
            R_nkftcc = torch.einsum('nfk,nkt,nfcd->nkftcd', W_nfk, H_nkt, R_nfcc)

            G_nftcc = torch.matmul(R_nftcc, Rx_inv)
            Yh_nftc = torch.einsum('nftcd,ftd->nftc', G_nftcc, X_ftc)
            Rh_nftcc = \
                torch.einsum('nftc,nftd->nftcd', Yh_nftc, Yh_nftc.conj()) + \
                torch.matmul((I-G_nftcc), R_nftcc)

            G_nkftcc = torch.matmul(R_nkftcc, Rx_inv)
            Yh_nkftc = torch.einsum('nkftcd,ftd->nkftc', G_nkftcc, X_ftc)
            Rh_nkftcc = \
                torch.einsum('nkftc,nkftd->nkftcd', Yh_nkftc, Yh_nkftc.conj()) + \
                torch.matmul((I-G_nkftcc), R_nkftcc)

            Gb_ftcc = torch.matmul(Rb_fcc[:, None], Rx_inv)
            Nh_ftc = torch.einsum('ftcd,ftd->ftc', Gb_ftcc, X_ftc)
            Rbh_ftcc = \
                torch.einsum('ftc,ftd->ftcd', Nh_ftc, Nh_ftc.conj()) + \
                torch.matmul((I-Gb_ftcc), Rb_fcc[:, None])

            # M-step
            R_inv = (R_nfcc + I_eps).pinverse()
            Vh_nkft = normalize_RV(abs_trace(
                torch.einsum('nfcd,nkftde->nkftce', R_inv, Rh_nkftcc)
            ) / n_chan)

            R_nfcc = normalize_RV((Rh_nftcc/V_nft[..., None, None]).mean(2))
            W_nfk, H_nkt = (
                normalize_W((Vh_nkft/H_nkt[..., None, :].clip(eps)).mean(3).mT),
                normalize_H((Vh_nkft/W_nfk.mT[..., None].clip(eps)).mean(2))
            )
            Rb_fcc = Rbh_ftcc.mean(1)*I

            # loss
            loss = (
                torch.einsum(
                    'ftc,ftcd,ftd->ft', X_ftc.conj(), Rx_inv, X_ftc
                ) + torch.log(Rx_ftcc.det().abs())
            ).sum()
            loss_list.append(loss.item())

        # V_nft = torch.bmm(W_nfk, H_nkt)
        # R_nftcc = torch.einsum('nft,nfcd->nftcd', V_nft, R_nfcc)
        # Rx_inv = (Rx_ftcc + I_eps).inverse()
        # G_nftcc = torch.matmul(R_nftcc, Rx_inv)
        # Yh_nftc = torch.einsum('nftcd,ftd->nftc', G_nftcc, X_ftc)

        Y = Yh_nftc.permute(0, 2, 1, 3)

        if return_losses:
            return Y, loss_list
        else:
            return Y
