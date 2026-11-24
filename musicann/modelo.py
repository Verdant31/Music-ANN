"""U-Net que transforma o log-Mel da original no log-Mel da lo-fi.

É toda convolucional, então aceita qualquer duração: treina com janelas de
~3 s e depois roda na música inteira. O comprimento só precisa ser múltiplo
de 16 (quatro reduções pela metade); `completar` cuida disso.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def bloco(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1), nn.GroupNorm(8, c_out), nn.SiLU(),
        nn.Conv2d(c_out, c_out, 3, padding=1), nn.GroupNorm(8, c_out), nn.SiLU(),
    )


class UNet(nn.Module):
    def __init__(self, base=32, niveis=4):
        super().__init__()
        canais = [base * 2**i for i in range(niveis + 1)]
        self.entrada = bloco(1, canais[0])
        self.desce = nn.ModuleList(bloco(canais[i], canais[i + 1]) for i in range(niveis))
        self.sobe = nn.ModuleList(
            nn.ConvTranspose2d(canais[i + 1], canais[i], 2, stride=2) for i in reversed(range(niveis))
        )
        self.junta = nn.ModuleList(bloco(2 * canais[i], canais[i]) for i in reversed(range(niveis)))
        self.saida = nn.Conv2d(canais[0], 1, 1)
        self.multiplo = 2**niveis

    def forward(self, x):
        # x: (lote, 128 bandas, quadros). A rede prevê a diferença em relação
        # à entrada, então partir da identidade já é um bom começo.
        h = self.entrada(x.unsqueeze(1))
        atalhos = [h]
        for d in self.desce:
            h = d(F.max_pool2d(h, 2))
            atalhos.append(h)
        atalhos.pop()
        for s, j in zip(self.sobe, self.junta):
            h = j(torch.cat([s(h), atalhos.pop()], dim=1))
        return x + self.saida(h).squeeze(1)


class Discriminador(nn.Module):
    """PatchGAN: olha (entrada, saída) e dá uma nota de "parece real" para
    cada pedaço do espectrograma. Usado no treino adversarial (pix2pix)."""

    def __init__(self, base=32):
        super().__init__()
        camadas, c = [], 2
        for i, s in enumerate([2, 2, 2, 1]):
            co = base * 2**i
            camadas += [nn.utils.spectral_norm(nn.Conv2d(c, co, 4, stride=s, padding=1)), nn.LeakyReLU(0.2)]
            c = co
        camadas.append(nn.Conv2d(c, 1, 4, padding=1))
        self.rede = nn.Sequential(*camadas)

    def forward(self, entrada, saida):
        return self.rede(torch.stack([entrada, saida], dim=1))


def completar(x, multiplo):
    """Completa o eixo do tempo até um múltiplo de `multiplo` (repetindo a borda)."""
    falta = (-x.shape[-1]) % multiplo
    if falta:
        x = F.pad(x.unsqueeze(1), (0, falta, 0, 0), mode="replicate").squeeze(1)
    return x, falta
