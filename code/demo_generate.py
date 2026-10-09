"""
demo_generate.py —— 点一下就能生成文本（VS Code 里直接按 ▶ 运行）

════════════════════════════════════════════════════════════════════════
【这个文件是干什么的】
════════════════════════════════════════════════════════════════════════

  官方 sample.py 要传一堆命令行参数，直接点运行会用默认值（out_dir='out'）
  然后报错说找不到 checkpoint。所以这里写一个【参数都填好】的版本，
  你打开这个文件、点运行，就能看到文本。

  同时它把生成的完整流程写清楚了（只有 5 步），比 sample.py 好读。

════════════════════════════════════════════════════════════════════════
【它需要什么？找不到会怎样？】
════════════════════════════════════════════════════════════════════════

  生成文本需要两样东西：

    ① checkpoint（ckpt.pt）  ── 训练好的【权重】，必需
    ② meta.pkl（词表）       ── 可缺，缺了就【从 input.txt 现算】

  ① 是本脚本自动搜索的。搜索顺序：
       命令行 --ckpt 参数  >  环境变量 NANOGPT_CKPT  >  以下候选位置
          <本文件所在目录>/out-shakespeare-char/ckpt.pt
          当前目录/out-shakespeare-char/ckpt.pt
          向上 2 层各级目录、以及它们的【同级子目录】下的同名路径
     （所以就算你在提交包里运行，它也能找到隔壁 nanogpt_orig/ 里的 ckpt）

  ② 词表只有 65 个字符，是从 input.txt 里 sorted(set(text)) 得来的，
     完全可以现算 —— 所以【不需要】先跑 prepare.py。

  ⚠️ 如果搜索完还是找不到 ckpt，脚本会打印具体该怎么做（两种办法）。

════════════════════════════════════════════════════════════════════════
【文本到底在哪里生成的】
════════════════════════════════════════════════════════════════════════

    这个文件（流程编排）
        ↓ 调用
    model.py 第 306 行  GPT.generate()
        ↓ 里面是一个 7 步循环
        for _ in range(max_new_tokens):
            idx_cond = idx 截断到 block_size      ← 步骤1
            logits = self(idx_cond)               ← 步骤2 前向
            logits = logits[:, -1, :] / temp      ← 步骤3 只取最后一个位置
            logits[低于top_k的] = -inf            ← 步骤4 截断
            probs = softmax(logits)               ← 步骤5 变概率
            idx_next = multinomial(probs)         ← 步骤6 采样一个字符
            idx = cat([idx, idx_next])            ← 步骤7 拼回去

  ★ 核心：每次只生成【一个】字符，然后把它拼回输入，再预测下一个。
    所以生成 500 个字符 = 前向计算 500 次。这就是"自回归"。

    对比训练：训练时一次前向能同时预测 256 个位置（并行）。
    生成时只能一次一个（串行）—— 这是两者最大的区别。

════════════════════════════════════════════════════════════════════════
【可以改的地方】
════════════════════════════════════════════════════════════════════════

  下面有 5 个参数，改完重新运行就行。建议试：
    PROMPT      = "KING HENRY: "   换个开头，看模型怎么接
    TEMPERATURE = 0.3 / 1.5        越小越保守，越大越胡言乱语
    TOP_K       = 10 / None        只在概率最高的 K 个里选
"""

import os
import sys
import glob
import pickle
import argparse
from contextlib import nullcontext

import torch

from model import GPTConfig, GPT

# ═════════════════════════════════════════════════════════════════════
# 参数（改这里）
# ═════════════════════════════════════════════════════════════════════

CKPT        = "out-shakespeare-char/ckpt.pt"   # 默认相对路径（会被自动搜索覆盖）
PROMPT      = "TO be,or not to be "            # 起始文本
N_TOKENS    = 500                              # 生成多少个字符
TEMPERATURE = 0.75                             # 温度
TOP_K       = 200                              # 只在前 200 个候选里采样


# ═════════════════════════════════════════════════════════════════════
# 定位文件（解决"找不到 ckpt.pt"的问题）
# ═════════════════════════════════════════════════════════════════════

def _candidate_roots():
    """列出可能放着模型/数据的目录（有界，不会全盘扫）。

    构成：本文件目录 + 当前目录 + 各自向上 2 层 + 这些层的同级子目录。
    举例（本文件在 D:\\ML\\nanogpt_submit\\code\\）：
        D:\\ML\\nanogpt_submit\\code
        D:\\ML\\nanogpt_submit            ← 向上 1
        D:\\ML                            ← 向上 2 ← 扫描它的子目录就能找到 nanogpt_orig
        以及上面这些目录各自的子目录
    """
    here = os.path.dirname(os.path.abspath(__file__))
    roots = [here, os.getcwd()]

    for base in (here, os.getcwd()):
        p = base
        for _ in range(2):                      # 向上 2 层
            p = os.path.dirname(p)
            if p and p not in roots:
                roots.append(p)

    # 对上面每个根，再把它的一级子目录也加进来（找同级项目，如 nanogpt_orig）
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


def locate(relpath, explicit=None, env_var=None):
    """在候选目录里找 relpath，返回第一个存在的绝对路径；找不到返回 None。"""
    # ① 命令行 / 环境变量优先
    if explicit and os.path.exists(explicit):
        return os.path.abspath(explicit)
    if env_var:
        v = os.environ.get(env_var)
        # 只要含 ckpt.pt 的路径才认，避免用户把普通变量当成路径用
        if v and 'ckpt' in v and os.path.exists(v):
            return os.path.abspath(v)

    # ② 候选根目录
    for root in _candidate_roots():
        cand = os.path.join(root, relpath)
        if os.path.exists(cand):
            return os.path.abspath(cand)

    # ③ 兜底：在候选根目录下用通配符找 out-*/ckpt.pt（训练目录名可能不同）
    if relpath.endswith('ckpt.pt'):
        for root in _candidate_roots():
            for hit in glob.glob(os.path.join(root, 'out-*', 'ckpt.pt')):
                return os.path.abspath(hit)
    return None


def load_meta(data_dir):
    """读词表。meta.pkl 在就直接读；不在就从 input.txt 现算（结果完全一致）。"""
    pkl = os.path.join(data_dir, 'meta.pkl')
    if os.path.exists(pkl):
        with open(pkl, 'rb') as f:
            meta = pickle.load(f)
        meta.setdefault('_source', pkl)
        return meta

    txt_path = os.path.join(data_dir, 'input.txt')
    if not os.path.exists(txt_path):
        return None

    # 与官方 prepare.py 的算法完全一致：sorted(list(set(text)))
    with open(txt_path, encoding='utf-8') as f:
        text = f.read()
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    return {'vocab_size': len(chars), 'stoi': stoi, 'itos': itos,
            '_source': txt_path + '  (现算，未用 meta.pkl)'}


# ═════════════════════════════════════════════════════════════════════
# 主流程
# ═════════════════════════════════════════════════════════════════════

def parse_args():
    ap = argparse.ArgumentParser(
        description="生成莎士比亚风格文本（参数都填好了，直接跑即可）")
    ap.add_argument("--ckpt", default=None,
                    help="checkpoint 路径。不给就自动搜索。")
    ap.add_argument("--prompt", default=None, help=f"起始文本（默认 {PROMPT!r}）")
    ap.add_argument("--n", type=int, default=None, help=f"生成字符数（默认 {N_TOKENS}）")
    ap.add_argument("-t", "--temperature", type=float, default=None,
                    help=f"温度（默认 {TEMPERATURE}）")
    ap.add_argument("--top_k", type=int, default=None, help=f"top-k（默认 {TOP_K}）")
    return ap.parse_args()


def main():
    args = parse_args()
    prompt = args.prompt if args.prompt is not None else PROMPT
    n_tok = args.n if args.n is not None else N_TOKENS
    temp = args.temperature if args.temperature is not None else TEMPERATURE
    top_k = args.top_k if args.top_k is not None else TOP_K

    # ── 环境 ────────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ctx = nullcontext() if device == "cpu" else torch.amp.autocast("cuda", dtype=torch.bfloat16)

    here = os.path.dirname(os.path.abspath(__file__))
    os.chdir(here)          # 让 data/ 之类的相对路径稳定

    # ── 定位 checkpoint ─────────────────────────────────────────────
    ckpt_path = locate(CKPT, explicit=args.ckpt, env_var="NANOGPT_CKPT")
    if ckpt_path is None:
        print("=" * 68)
        print("[错误] 找不到 checkpoint（ckpt.pt）")
        print("=" * 68)
        print()
        print("  已搜索的目录（最多列出 8 个）：")
        for r in list(dict.fromkeys(_candidate_roots()))[:8]:
            print(f"    {r}")
        print("    ...（同级子目录也搜过了）")
        print()
        print("  两个办法，任选一个：")
        print()
        print("  【办法 1】直接指定路径（最快，立即能用）")
        print("      python demo_generate.py --ckpt "
              "D:\\ML\\nanogpt_orig\\out-shakespeare-char\\ckpt.pt")
        print()
        print("  【办法 2】先在这个目录训练一次（约 10 分钟，产出自己的 ckpt）")
        print("      python data/shakespeare_char/prepare.py")
        print("      python train.py config/train_shakespeare_char.py --compile=False")
        print("      然后重新运行本脚本")
        print()
        print("  💡 为什么找不到：checkpoint 有 123 MB，提交包里【故意没放】")
        print("     （GitHub 单文件上限 100 MB 会拒收）。它在训练时产出，")
        print("     跑一次 train.py 就有了。")
        print("=" * 68)
        return 1

    # ── 定位词表（找不到就现算） ────────────────────────────────────
    data_dir, meta = None, None
    for cand_dir in [os.path.join(here, 'data', 'shakespeare_char'),
                     os.path.join(os.getcwd(), 'data', 'shakespeare_char')]:
        meta = load_meta(cand_dir)
        if meta is not None:
            data_dir = cand_dir
            break
    if meta is None:
        # 全都没找到，就在候选根目录里找 input.txt
        hit = locate('data/shakespeare_char/input.txt')
        if hit:
            data_dir = os.path.dirname(hit)
            meta = load_meta(data_dir)
    if meta is None:
        print("[错误] 找不到词表：既没有 meta.pkl，也没有 data/shakespeare_char/input.txt")
        print("       跑一次：python data/shakespeare_char/prepare.py")
        return 1

    stoi, itos = meta["stoi"], meta["itos"]
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda ids: "".join(itos[i] for i in ids)

    # ── 第 1 步：词表 ───────────────────────────────────────────────
    print(f"[1/5] 词表读好了，vocab_size = {meta['vocab_size']}")
    print(f"      来源: {meta.get('_source', '(meta.pkl)')}")

    # ── 第 2 步：读 checkpoint，重建模型结构，装权重 ────────────────
    #    ★ 关键：checkpoint 里【只有数字，没有代码】。
    #      所以必须先按 config 把模型搭出来，再把权重装进去。
    print(f"      checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = GPTConfig(**checkpoint["model_args"])
    model = GPT(cfg).to(device)
    model.load_state_dict(checkpoint["model"])   # 键名是 "model"，不是 "state_dict"
    model.eval()          # ★ 生成时必须 eval（关掉 dropout）
    print(f"[2/5] 模型重建完成，参数量 "
          f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M"
          f"（nanoGPT 自己打印的 10.65M 是减掉了位置嵌入，口径不同）")
    print(f"      这是第 {checkpoint['iter_num']} 步存的 checkpoint，"
          f"当时 val loss = {float(checkpoint['best_val_loss']):.4f}")

    # ── 第 3 步：把 prompt 编码成整数 ──────────────────────────────
    bad = sorted({c for c in prompt if c not in stoi})
    if bad:
        print(f"[错误] prompt 里有词表外的字符（词表只有 {meta['vocab_size']} 个字符）: {bad}")
        return 1
    idx = torch.tensor(encode(prompt), dtype=torch.long, device=device)[None, ...]
    print(f"[3/5] prompt {prompt!r} 编码成 {idx.shape[1]} 个整数")

    # ── 第 4 步：生成（真正出文本的地方） ──────────────────────────
    print(f"[4/5] 生成中…… 参数: temperature={temp}, top_k={top_k}, 长度={n_tok}")
    with torch.no_grad():
        with ctx:
            out = model.generate(idx, n_tok, temperature=temp, top_k=top_k)

    # ── 第 5 步：解码回文本并打印 ──────────────────────────────────
    text = decode(out[0].tolist())
    print(f"[5/5] 完成\n")

    print("=" * 68)
    print(f" 生成结果（prompt: {prompt!r}，temperature={temp}）")
    print("=" * 68)
    print(text)
    print("=" * 68)

    print("""
【观察一下这些】
  1. 角色名的格式对不对？（应该有 "XXX:" 然后换行然后台词）
  2. 出现的词是真英文单词，还是乱码？
  3. 语义连贯吗？读两句就知道 —— 局部通顺，整体不连贯，这是小模型的正常水平
  4. 把 TEMPERATURE 改成 0.3 再跑，看有什么变化（会更通顺但更重复）
  5. 改成 1.5 再跑（会更"有创意"但更容易胡说）

【命令行也能调（不用改文件）】
  python demo_generate.py -t 0.3
  python demo_generate.py --prompt "KING HENRY: " --n 800
""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
