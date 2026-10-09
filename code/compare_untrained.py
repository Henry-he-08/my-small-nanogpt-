"""对照实验：未训练模型（随机初始化）生成的文本 —— 任务【进阶】要求

同一个 prompt、同样的采样参数、同样的模型结构，
唯一区别是【有没有加载训练好的权重】。

  · 未训练：随机初始化，等价于"没学过的模型"
  · 训练后：加载 out-shakespeare-char/ckpt.pt

用法：
    python compare_untrained.py
    python compare_untrained.py --ckpt D:\\ML\\nanogpt_orig\\out-shakespeare-char\\ckpt.pt
"""
import os
import sys
import glob
import pickle
import argparse
from contextlib import nullcontext

import torch

from model import GPTConfig, GPT

PROMPT = "ROMEO: "
N_TOKENS = 500
TEMPERATURE = 0.8
TOP_K = 200


# ═════════════════════════════════════════════════════════════════
# 找文件（与 demo_generate.py 同一套逻辑，保持独立不互相 import）
# ═════════════════════════════════════════════════════════════════

def _candidate_roots():
    here = os.path.dirname(os.path.abspath(__file__))
    roots = [here, os.getcwd()]
    for base in (here, os.getcwd()):
        p = base
        for _ in range(2):
            p = os.path.dirname(p)
            if p and p not in roots:
                roots.append(p)
    out = list(roots)
    for r in roots:
        try:
            for name in sorted(os.listdir(r)):
                d = os.path.join(r, name)
                if os.path.isdir(d) and d not in out:
                    out.append(d)
        except OSError:
            pass
    return out


def locate(relpath, explicit=None):
    if explicit and os.path.exists(explicit):
        return os.path.abspath(explicit)
    for root in _candidate_roots():
        cand = os.path.join(root, relpath)
        if os.path.exists(cand):
            return os.path.abspath(cand)
    if relpath.endswith('ckpt.pt'):
        for root in _candidate_roots():
            for hit in glob.glob(os.path.join(root, 'out-*', 'ckpt.pt')):
                return os.path.abspath(hit)
    return None


def load_meta(data_dir):
    """读词表；meta.pkl 不在就从 input.txt 现算（结果完全一致）。"""
    pkl = os.path.join(data_dir, 'meta.pkl')
    if os.path.exists(pkl):
        with open(pkl, 'rb') as f:
            return pickle.load(f)
    txt = os.path.join(data_dir, 'input.txt')
    if not os.path.exists(txt):
        return None
    with open(txt, encoding='utf-8') as f:
        text = f.read()
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    return {'vocab_size': len(chars), 'stoi': stoi,
            'itos': {i: ch for ch, i in stoi.items()}}


# ═════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="未训练 vs 训练后 生成结果对照")
    ap.add_argument("--ckpt", default=None, help="训练好的 checkpoint 路径（不给就自动搜索）")
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--n", type=int, default=N_TOKENS)
    ap.add_argument("-t", "--temperature", type=float, default=TEMPERATURE)
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ctx = nullcontext() if device == 'cpu' else torch.amp.autocast('cuda', dtype=torch.bfloat16)
    here = os.path.dirname(os.path.abspath(__file__))
    os.chdir(here)

    # ---- 词表 ----
    meta = None
    for d in (os.path.join(here, 'data', 'shakespeare_char'),
              os.path.join(os.getcwd(), 'data', 'shakespeare_char')):
        meta = load_meta(d)
        if meta:
            break
    if meta is None:
        hit = locate('data/shakespeare_char/input.txt')
        meta = load_meta(os.path.dirname(hit)) if hit else None
    if meta is None:
        print("[错误] 找不到词表。先跑：python data/shakespeare_char/prepare.py")
        return 1

    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join(itos[i] for i in l)

    x = torch.tensor(encode(args.prompt), dtype=torch.long, device=device)[None, ...]

    def gen(model):
        with torch.no_grad(), ctx:
            y = model.generate(x.clone(), args.n, temperature=args.temperature, top_k=TOP_K)
        return decode(y[0].tolist())

    # ══ 未训练模型（随机初始化） ══
    torch.manual_seed(42)
    cfg = GPTConfig(block_size=256, vocab_size=meta['vocab_size'],
                    n_layer=6, n_head=6, n_embd=384, dropout=0.0)
    fresh = GPT(cfg).to(device)
    fresh.eval()

    print("=" * 68)
    print("  ① 未训练模型（权重完全随机，不加载任何 checkpoint）")
    print(f"     参数量 {sum(p.numel() for p in fresh.parameters())/1e6:.2f}M"
          f" | prompt={args.prompt!r} | temperature={args.temperature}")
    print("=" * 68)
    print(gen(fresh))

    # ══ 训练后的模型 ══
    ckpt_path = locate('out-shakespeare-char/ckpt.pt', explicit=args.ckpt)
    if ckpt_path is None:
        print("\n" + "=" * 68)
        print("  ② 训练后的模型 —— 跳过（没找到 checkpoint）")
        print("=" * 68)
        print("  想比较的话，二选一：")
        print("    python compare_untrained.py --ckpt D:\\ML\\nanogpt_orig\\out-shakespeare-char\\ckpt.pt")
        print("    或先训练：python train.py config/train_shakespeare_char.py --compile=False")
        return 0

    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    trained = GPT(GPTConfig(**ck['model_args'])).to(device)
    trained.load_state_dict(ck['model'])
    trained.eval()

    print("\n" + "=" * 68)
    print("  ② 训练后的模型")
    print(f"     checkpoint: {ckpt_path}")
    print(f"     第 {ck['iter_num']} 步 | val loss {float(ck['best_val_loss']):.4f}"
          f" | prompt={args.prompt!r} | temperature={args.temperature}")
    print("=" * 68)
    print(gen(trained))

    print("\n" + "=" * 68)
    print("  结论")
    print("=" * 68)
    print("  同一个 prompt、同样的采样参数，唯一区别是有没有训练：")
    print("    ① 随机权重 → 全是乱码符号，没有任何语言结构")
    print("    ② 训练后的权重 → 出现真实单词、剧本格式、角色名")
    print("  → 说明模型学到的不是'字符频率'，而是【语言的结构】。")

    print("\n" + "=" * 68)
    print("  两个数字的差：参数量 10.65M vs 10.75M")
    print("=" * 68)
    print("  本脚本打印的是【全部参数】（含位置嵌入）。")
    print("  nanoGPT 自己打印时会减掉位置嵌入（block_size*n_embd = 256*384 = 98,304），")
    print("  所以显示 10.65M。两者都对，只是口径不同。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
