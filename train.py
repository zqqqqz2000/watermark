import torch
from torch.utils.data import DataLoader
import visdom

from net.watermark import WaterMark
from utils.dataset import Mp4DataSet
from utils.utils import denormalize


class Trainer:
    def __init__(self, net: WaterMark, data_loader: DataLoader, out_inv=100):
        self.net = net
        self.data_loader = data_loader
        self.vis = visdom.Visdom()
        self.out_inv = out_inv

    def train(self):
        # device = 'cuda' if torch.cuda.is_available() else 'cpu'
        device = 'cpu'
        self.net.to(device)
        opts = {
            "title": 'train',
            "xlabel": 'times',
            "ylabel": 'loss',
            "legend": ['dloss', 'gloss']
        }
        count = 0
        for step in range(100):
            for img in self.data_loader:
                img = img.to(device)
                count += 1
                self.net.reset_grad()

                self.net.g_forward(img)
                self.net.d_forward()

                dloss = self.net.calc_d_loss()
                dloss.backward(retain_graph=True)

                gloss = self.net.calc_g_loss()
                gloss.backward()
                # self.net.G.frozen(False)
                self.net.d_opt.step()
                # self.net.G.frozen(True)

                self.net.D.frozen(False)
                self.net.g_opt.step()
                self.net.D.frozen(True)

                if not count % self.out_inv:
                    self.vis.line(
                        X=[self.out_inv * count],
                        Y=[[dloss.detach().cpu(), gloss.detach().cpu()]],
                        update='append',
                        opts=opts,
                        win='training loss'
                    )
                    self.vis.images(
                        [denormalize(self.net.raw_img[0]), denormalize(self.net.add_watermark_img[0])],
                        win='img'
                    )


if __name__ == '__main__':
    img_size = 256
    dataset = Mp4DataSet('data/data.mp4', img_resize=img_size)
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)
    net = WaterMark(image_size=img_size, d_layers=4, g_layerX2=4)
    trainer = Trainer(net, loader, out_inv=1)
    trainer.train()
