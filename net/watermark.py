from typing import *
import torch
from torch import nn
from .gan import G
from .gan import Discriminator


class WaterMark(nn.Module):
    def __init__(
            self,
            img_c=3,
            g_layerX2=4,
            d_layers=4,
            dlr=0.01,
            glr=0.01,
            dloss_ratio=0.5,
            transitions: Callable[[torch.Tensor], List[torch.Tensor]] = lambda x: [x]
    ):
        """
        WaterMark network main class
        :param img_c: the channel of image
        :param g_layerX2: this parameter * 2 is the generator layer number
        :param d_layers: discriminator layer number
        :param dlr: discriminator learning rate
        :param glr: generator learning rate
        :param transitions: transitions function to support the watermark robustness
        """
        super(WaterMark, self).__init__()
        self.G = G(img_c, g_layerX2)
        self.D = Discriminator(img_c, d_layers)
        self.raw_img: Optional[torch.Tensor] = None
        self.add_watermark_img: Optional[torch.Tensor] = None
        self.add_watermark_img_trans: Optional[torch.Tensor] = None
        self.raw_watermark: Optional[torch.Tensor] = None
        self.watermarked_d_res: Optional[List[torch.Tensor]] = None
        self.raw_img_d_res: Optional[List[torch.Tensor]] = None
        self.d_gloss: Optional[torch.Tensor] = None
        self.dloss_func = nn.MSELoss()
        self.d_opt = torch.optim.Adam(self.D.parameters(), dlr)
        self.g_opt = torch.optim.Adam(self.G.parameters(), glr)
        self.transitions = transitions
        self.dloss_ratio = dloss_ratio

    def reset_grad(self):
        self.d_opt.zero_grad()
        self.g_opt.zero_grad()

    def g_forward(self, img: torch.Tensor):
        self.raw_img = img
        watermark = self.G(img)
        self.raw_watermark = watermark
        self.add_watermark_img = watermark + img
        return watermark

    def d_forward(self, img1: Optional[torch.Tensor] = None, img2: Optional[torch.Tensor] = None, train=True):
        if not train:
            return self.D(img1, img2)
        else:
            trans_raw_imgs: List[torch.Tensor] = self.transitions(self.raw_img)
            trans_marked_imgs: List[torch.Tensor] = self.transitions(self.add_watermark_img)
            self.watermarked_d_res = []
            self.raw_img_d_res = []
            self.add_watermark_img_trans = trans_marked_imgs
            for watermarked_img in trans_marked_imgs:
                self.watermarked_d_res.append(
                    self.D(watermarked_img)
                )
            for raw_img in trans_raw_imgs:
                self.raw_img_d_res.append(
                    self.D(raw_img)
                )

    def calc_d_loss(self):
        same_loss = self.dloss_func(
            torch.stack([*self.raw_img_d_res]),
            torch.tensor([1]).float()
        )
        marked_loss = self.dloss_func(
            torch.stack([*self.watermarked_d_res]),
            torch.tensor([0]).float()
        )
        self.d_gloss = (same_loss + marked_loss) / 2
        return self.d_gloss

    def calc_g_loss(self):
        mark_loss = self.raw_watermark.mean()
        return (self.d_gloss * self.dloss_ratio + mark_loss * (1 - self.dloss_ratio)) / 2
