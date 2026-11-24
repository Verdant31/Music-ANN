"""Etapa 3: treina a U-Net e compara com duas referências simples.

As músicas de teste ficam inteiras de fora do treino. No fim imprime o erro
médio (L1, na escala normalizada) de três jeitos de prever a lo-fi:

  - identidade: devolver a original sem mudar nada
  - equalizador: somar a cada banda a diferença média original -> lo-fi
    aprendida no treino (um "EQ lo-fi" fixo)
  - U-Net

Se a U-Net não ganhar do equalizador, ela não aprendeu nada além de um EQ.

    python treinar.py
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch

from musicann import dados
from musicann.dados import carregar_pares, montar
from musicann.modelo import Discriminador, UNet


def dividir(nomes, n_teste, semente=31):
    rng = np.random.default_rng(semente)
    teste = sorted(rng.choice(nomes, n_teste, replace=False).tolist())
    return [n for n in nomes if n not in teste], teste


def variacao(z):
    """Quanto o espectrograma muda de um quadro para o outro (média de |Δt|)."""
    return float(np.abs(np.diff(z, axis=-1)).mean())


@torch.no_grad()
def avaliar(modelo, X, Y, dev, lote=64, nitidez=False):
    """Erro L1 médio. Com nitidez=True devolve também a razão entre a
    variação no tempo da previsão e a da lo-fi real: perto de 1 = tão
    detalhada quanto a real; perto de 0 = borrada."""
    modelo.eval()
    erro, var_p = 0.0, 0.0
    for i in range(0, len(X), lote):
        x = torch.from_numpy(X[i : i + lote]).to(dev)
        y = torch.from_numpy(Y[i : i + lote]).to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = modelo(x).float()
        erro += (p - y).abs().mean().item() * len(x)
        var_p += variacao(p.cpu().numpy()) * len(x)
    if nitidez:
        return erro / len(X), var_p / len(X) / variacao(Y)
    return erro / len(X)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pares", default="dados/pares")
    ap.add_argument("--custo-max", type=float, default=0.25)
    ap.add_argument("--custo-fino-max", type=float, default=None,
                    help="filtro do alinhamento refinado (custo cosseno por quadro), ex.: 0.45")
    ap.add_argument("--n-teste", type=int, default=5)
    ap.add_argument("--epocas", type=int, default=40)
    ap.add_argument("--lote", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--saida", default="modelos")
    ap.add_argument("--gan", type=float, default=0.0,
                    help="peso da perda adversarial (pix2pix); 0 = só L1. Ex.: 0.01")
    ap.add_argument("--volume-relativo", action="store_true",
                    help="tira o volume médio de cada janela (entrada e alvo): a rede aprende só a forma do "
                         "espectro, não a diferença de volume entre as faixas do YouTube")
    ap.add_argument("--lista", nargs="*", default=None,
                    help="usar só as músicas destes CSVs (ex.: pares.csv); sem isso usa todas da pasta")
    ap.add_argument("--teste", nargs="*", default=None, help="nomes fixos das músicas de teste")
    ap.add_argument("--n-validacao", type=int, default=0,
                    help="músicas do treino separadas para escolher a melhor época (o teste fica intocado)")
    args = ap.parse_args()

    torch.manual_seed(0)
    dev = "cuda"
    dados.CUSTO_FINO_MAX = args.custo_fino_max
    pares = carregar_pares(args.pares)
    if args.lista:
        nomes = set()
        for csv in args.lista:
            with open(csv, encoding="utf-8") as f:
                nomes |= {linha.split(";")[0] for linha in f.read().splitlines()[1:] if linha.strip()}
        pares = {k: v for k, v in pares.items() if k in nomes}
    if args.teste:
        teste = sorted(n for n in args.teste if n in pares)
        treino = [n for n in sorted(pares) if n not in teste]
    else:
        treino, teste = dividir(sorted(pares), args.n_teste)
    validacao = []
    if args.n_validacao:
        treino, validacao = dividir(treino, args.n_validacao, semente=7)
    Xtr, Ytr = montar(pares, treino, args.custo_max)
    Xte, Yte = montar(pares, teste, args.custo_max, passo=64)
    Xva, Yva = montar(pares, validacao, args.custo_max, passo=64) if validacao else (Xte, Yte)
    if args.volume_relativo:
        Xtr, Ytr, Xte, Yte, Xva, Yva = (z - z.mean(axis=(1, 2), keepdims=True)
                                        for z in (Xtr, Ytr, Xte, Yte, Xva, Yva))
    print(f"treino: {len(treino)} músicas, {len(Xtr)} janelas | validação: {validacao} | "
          f"teste: {teste}, {len(Xte)} janelas")

    # Referências
    eq = (Ytr - Xtr).mean(axis=(0, 2), keepdims=True)
    ref = {
        "identidade": float(np.abs(Xte - Yte).mean()),
        "equalizador": float(np.abs(Xte + eq - Yte).mean()),
    }
    print(f"referências no teste: {ref}")

    modelo = UNet(base=args.base).to(dev)
    print(f"U-Net: {sum(p.numel() for p in modelo.parameters()) / 1e6:.1f} M parâmetros")
    opt = torch.optim.AdamW(modelo.parameters(), lr=args.lr, weight_decay=1e-4)
    passos = args.epocas * math.ceil(len(Xtr) / args.lote)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=passos, pct_start=0.05)

    if args.gan > 0:
        disc = Discriminador().to(dev)
        opt_d = torch.optim.AdamW(disc.parameters(), lr=2e-4, betas=(0.5, 0.999))

    Xtr_t, Ytr_t = torch.from_numpy(Xtr), torch.from_numpy(Ytr)
    os.makedirs(args.saida, exist_ok=True)
    melhor_val, historico = float("inf"), []
    for ep in range(1, args.epocas + 1):
        modelo.train()
        t0, soma = time.time(), 0.0
        perm = torch.randperm(len(Xtr_t))
        for i in range(0, len(perm), args.lote):
            idx = perm[i : i + args.lote]
            x, y = Xtr_t[idx].to(dev), Ytr_t[idx].to(dev)
            # Aumento de dados: mesmo ganho (±6 dB) na entrada e no alvo.
            g = (torch.rand(len(x), 1, 1, device=dev) - 0.5) * 0.2
            x, y = x + g, y + g
            with torch.autocast("cuda", dtype=torch.bfloat16):
                p = modelo(x).float()
                perda = (p - y).abs().mean()
                if args.gan > 0:
                    # Discriminador: real -> 1, gerado -> 0 (LSGAN).
                    d_real, d_falso = disc(x, y).float(), disc(x, p.detach()).float()
                    perda_d = ((d_real - 1) ** 2).mean() + (d_falso**2).mean()
                    opt_d.zero_grad(set_to_none=True)
                    perda_d.backward()
                    opt_d.step()
                    # Gerador: quer que o discriminador ache a saída real.
                    perda = perda + args.gan * ((disc(x, p).float() - 1) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            perda.backward()
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
            opt.step()
            sched.step()
            soma += perda.item() * len(x)
        erro_val = avaliar(modelo, Xva, Yva, dev)
        historico.append(dict(epoca=ep, treino=soma / len(Xtr_t), validacao=erro_val))
        marca = ""
        if erro_val < melhor_val:
            melhor_val = erro_val
            torch.save(dict(estado=modelo.state_dict(), base=args.base, eq=torch.from_numpy(eq),
                            volume_relativo=args.volume_relativo), os.path.join(args.saida, "unet.pt"))
            marca = " *"
        nome_val = "validação" if validacao else "teste"
        print(f"época {ep:3d}  treino {soma / len(Xtr_t):.4f}  {nome_val} {erro_val:.4f}  ({time.time() - t0:.0f}s){marca}")

    # O melhor pelo L1 favorece saídas "médias"; com GAN vale ouvir também o final.
    torch.save(dict(estado=modelo.state_dict(), base=args.base, eq=torch.from_numpy(eq),
                    volume_relativo=args.volume_relativo), os.path.join(args.saida, "unet_final.pt"))
    erro_final, nit_final = avaliar(modelo, Xte, Yte, dev, nitidez=True)
    ck = torch.load(os.path.join(args.saida, "unet.pt"), map_location=dev)
    modelo.load_state_dict(ck["estado"])
    melhor, nit_melhor = avaliar(modelo, Xte, Yte, dev, nitidez=True)
    nit_ref = {"identidade": variacao(Xte) / variacao(Yte), "equalizador": variacao(Xte + eq) / variacao(Yte)}
    print(f"nitidez (1 = igual à lo-fi real): identidade {nit_ref['identidade']:.2f} | "
          f"U-Net melhor {nit_melhor:.2f} | U-Net final {nit_final:.2f} (erro final {erro_final:.4f})")
    resultado = dict(referencias=ref, unet=melhor, nitidez_melhor=nit_melhor, nitidez_final=nit_final,
                     erro_final=erro_final, nitidez_referencias=nit_ref, treino=treino, teste=teste,
                     validacao=validacao, volume_relativo=args.volume_relativo,
                     janelas_treino=len(Xtr), janelas_teste=len(Xte), custo_max=args.custo_max,
                     historico=historico)
    with open(os.path.join(args.saida, "resultado.json"), "w", encoding="utf-8") as f:
        json.dump(resultado, f, ensure_ascii=False, indent=1)
    print(f"\nerro L1 no teste | identidade {ref['identidade']:.4f} | equalizador {ref['equalizador']:.4f} | U-Net {melhor:.4f}")


if __name__ == "__main__":
    main()
