import argparse

from torch import nn
from torchvision import models
from torch.autograd import Variable
import torch
from datasets import CLEVRDataset, transform, collate_fn
from utils import to_var, weights_init, mask_tensor
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence


class InputUnit(nn.Module):
    def __init__(self, args=None, fixed_extractor=True):
        super(InputUnit, self).__init__()
        extractor = nn.Sequential(
            *list(models.resnet101(pretrained=True).children())[:-3],
        )

        d, p = args.d, args.p
        self.d = d
        self.p = p

        if fixed_extractor:
            for param in extractor.modules():
                param.requires_grad = False

        self.image_extractor = nn.Sequential(
            extractor,
            nn.Conv2d(1024, d, 1, 1),
            nn.BatchNorm2d(d),
            nn.ELU(),
            nn.Conv2d(d, d, 1, 1),
            nn.BatchNorm2d(d),
            nn.ELU()
        )

        self.embed = nn.Embedding(args.vocab_size, args.embed_size)
        self.lstm = nn.LSTM(args.embed_size, d, 1, batch_first=True, dropout=0.85, bidirectional=True)
        self.linear_q = nn.ModuleList([nn.Linear(2*d, d) for _ in range(p)])
        self.linear_cw = nn.Linear(2*d, d)

        self.apply(weights_init)

    def forward(self, x, q, lengths, mask):
        k = self.image_extractor(x)

        embedding = self.embed(q)

        cw = self.lstm(embedding)[0]

        indices = lengths[:, None, None].repeat(1, 1, self.d)
        cw_1 = cw[:, 0, self.d:]
        cw_s = cw[:, :, :self.d].gather(1, indices).squeeze()
        q_ = torch.cat([cw_1, cw_s], -1)
        q = torch.stack([self.linear_q[i](q_) for i in range(self.p)])

        cw = self.linear_cw(cw)
        cw = mask_tensor(cw, mask, 0.)

        return k, q_, q, cw


class ControlUnit(nn.Module):
    def __init__(self, args=None):
        super(ControlUnit, self).__init__()

        d = args.d
        self.linear_cq = nn.Linear(2*d, d)
        self.linear_ca = nn.Linear(d, 1)

        self.apply(weights_init)

    def forward(self, c, q, cw, mask):
        c = c[:q.size(0)]

        cq = self.linear_cq(torch.cat([c, q], dim=1))

        ca = self.linear_ca(cq[:, None, :] * cw)
        ca = mask_tensor(ca, mask, float('-inf'))
        cv = F.softmax(ca, 1)
        c = (cv * cw).sum(1)
        return c, cv


class ReadUnit(nn.Module):
    def __init__(self, args=None):
        super(ReadUnit, self).__init__()

        d = args.d

        self.linear_m = nn.Linear(d, d)
        self.conv_k = nn.Conv2d(d, d, 1, 1)
        self.conv_I = nn.Conv2d(2*d, d, 1, 1)
        self.conv_ra = nn.Conv2d(d, d, 1, 1)
        self.softmax = nn.Softmax2d()

        self.apply(weights_init)

    def forward(self, m, k, c):
        m = m[:k.size(0)]
        I = self.linear_m(m)[:, :, None, None] * self.conv_k(k)
        I_ = self.conv_I(torch.cat([I, k], 1))
        ra = self.conv_ra(I_ * c[:, :, None, None])
        rv = self.softmax(ra)
        r = (rv * k).sum(-1).sum(-1)

        return r, rv


class WriteUnit(nn.Module):
    def __init__(self, args=None):
        super(WriteUnit, self).__init__()

        d = args.d

        self.linear_m = nn.Linear(2*d, d)
        self.linear_sa = nn.Linear(d, 1)
        self.linear_m_ = nn.Linear(d, d)
        self.linear_c = nn.Linear(d, 1)
        self.linear_s = nn.Linear(d, d, bias=False)

        self.apply(weights_init)
        self.linear_c.bias.data.fill_(1)

    def forward(self, r, m, c, cs, ms):
        m, c = m[:r.size(0)], c[:r.size(0)]
        m_prev = self.linear_m(torch.cat([r, m], 1))

        m_ = self.linear_m_(m_prev)
        c = F.sigmoid(self.linear_c(c))

        if cs and ms:
            cs, ms = torch.stack(cs), torch.stack(ms)
            sa = F.softmax(self.linear_sa(cs * c[None, :, :]), 0)
            m_sa = (sa * ms).sum(0)
            m_ += self.linear_s(m_sa)

        m = c * m + (1 - c) * m_
        return m


class MACCell(nn.Module):
    def __init__(self, args=None):
        super(MACCell, self).__init__()

        self.control_unit = ControlUnit(args)
        self.read_unit = ReadUnit(args)
        self.write_unit = WriteUnit(args)

        self.apply(weights_init)

    def forward(self, c, m, k, q, cw, mask, cs, ms):
        c, cv = self.control_unit(c, q, cw, mask)
        r, rv = self.read_unit(m, k, c)
        m = self.write_unit(r, m, c, cs, ms)

        return c, m, cv, rv


class OutputUnit(nn.Module):
    def __init__(self, args=None):
        super(OutputUnit, self).__init__()
        d = args.d

        self.linear = nn.Sequential(
            nn.Linear(3*d, d),
            nn.ReLU(),
            nn.Linear(d, args.num_classes)
        )

        self.apply(weights_init)

    def forward(self, q, m):
        x = self.linear(torch.cat([q, m], 1))
        return x


class MACNetwork(nn.Module):
    def __init__(self, args=None, d=512, p=12):
        super(MACNetwork, self).__init__()

        self.input_unit = InputUnit(args)
        self.mac_cell = MACCell(args)
        self.output_unit = OutputUnit(args)
        self.p = p
        self.m0 = nn.Parameter(torch.randn(args.batch_size, d))
        self.c0 = nn.Parameter(torch.randn(args.batch_size, d))

    def forward(self, x, question, lengths):
        mask = (question == 0).data
        k, q_, q, cw = self.input_unit(x, question, lengths, mask)

        c, m = self.c0, self.m0
        cs, ms, cvs, rvs = [], [], [], []

        for i in range(self.p):
            c, m, cv, rv = self.mac_cell(c, m, k, q[i], cw, mask, cs, ms)

            cs.append(c)
            ms.append(m)
            cvs.append(cv)
            rvs.append(rv)

        out = self.output_unit(q_, m)
        return out, cvs, rvs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CapsNet')

    parser.add_argument('-batch_size', type=int, default=16)
    parser.add_argument('-d', type=int, default=512, help='Hidden dim')
    parser.add_argument('-p', type=int, default=12, help='Num of MAC Cell')

    args = parser.parse_args()

    model = MACNetwork(args)

    if torch.cuda.is_available():
        model.cuda()

    clevr = CLEVRDataset('data/CLEVR_v1.0/', download=False, mode='val', transform=transform)

    loader = torch.utils.data.DataLoader(dataset=clevr,
                                         batch_size=args.batch_size,
                                         num_workers=4,
                                         collate_fn=collate_fn,
                                         shuffle=False)

    for data in loader:
        images, questions, lengths, answers = [to_var(x) for x in data]
        # images, questions, lengths, answers = data
        # images, questions, answers = to_var(images), to_var(questions), to_var(answers)

        model(images, questions, lengths)
