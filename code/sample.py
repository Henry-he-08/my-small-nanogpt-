"""
Sample from a trained model
"""
import os
import pickle
from contextlib import nullcontext
import torch
try:
    import tiktoken
except ImportError:
    tiktoken = None  # 只有加载 GPT-2 权重时才需要；char 级用 meta.pkl，不需要
from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out' # ignored if init_from is not 'resume'
start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 10 # number of samples to draw
max_new_tokens = 500 # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# model
if init_from == 'resume':
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    if not os.path.exists(ckpt_path):
        # 原本这里直接 torch.load，报错是 FileNotFoundError，看不出该怎么做。
        # 加一段自动搜索 + 明确指引（不改变原有逻辑，只是提前给出更好的提示）。
        import glob as _glob
        _here = os.path.dirname(os.path.abspath(__file__))
        _cands = [_ckpt for _d in (out_dir, 'out', 'out-shakespeare-char')
                  for _ckpt in [os.path.join(_d, 'ckpt.pt')]]
        _cands += _glob.glob(os.path.join(_here, 'out-*', 'ckpt.pt'))
        _cands += _glob.glob(os.path.join(os.path.dirname(_here), '*', 'out-*', 'ckpt.pt'))
        _cands += _glob.glob(os.path.join(os.path.dirname(os.path.dirname(_here)),
                                          '*', 'out-*', 'ckpt.pt'))
        _cands += _glob.glob(os.path.join(os.path.dirname(os.path.dirname(_here)),
                                          '*', '*', 'out-*', 'ckpt.pt'))
        _found = next((c for c in _cands if os.path.exists(c)), None)
        if _found is None:
            raise SystemExit(
                f"[错误] 找不到 checkpoint: {ckpt_path}\n"
                f"\n  两种办法：\n"
                f"    1) 先训练一次（约 10 分钟）\n"
                f"       python train.py config/train_shakespeare_char.py --compile=False\n"
                f"    2) 指定已有 checkpoint 所在目录\n"
                f"       python sample.py --out_dir=out-shakespeare-char\n"
                f"\n  💡 更省事：直接用 demo_generate.py —— 它会自动搜索 checkpoint。")
        print(f"[提示] {ckpt_path} 不存在，自动改用 {_found}")
        ckpt_path = _found
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    # init from a given GPT-2 model
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

# look for the meta pickle in case it is available in the dataset folder
load_meta = False
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']: # older checkpoints might not have these...
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)
if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    # TODO want to make this more general to arbitrary encoder/decoder schemes
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
elif init_from == 'resume' and os.path.exists(os.path.join('data', 'shakespeare_char', 'input.txt')):
    # 补丁：char 级数据集的 meta.pkl 只是 sorted(set(text)) 的结果，
    # 完全可以从 input.txt 现算 —— 这样就不必先跑 prepare.py，
    # 也避免下面错误地退回 GPT-2 词表（50304 vs 65，会对不上）。
    _txt = os.path.join('data', 'shakespeare_char', 'input.txt')
    print(f"No meta.pkl found; building char vocab from {_txt} ...")
    with open(_txt, encoding='utf-8') as f:
        _chars = sorted(list(set(f.read())))
    stoi = {ch: i for i, ch in enumerate(_chars)}
    itos = {i: ch for ch, i in stoi.items()}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
    print(f"Built char vocab: vocab_size={len(_chars)}")
else:
    # ok let's assume gpt-2 encodings by default
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

# encode the beginning of the prompt
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()
start_ids = encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

# run generation
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            print(decode(y[0].tolist()))
            print('---------------')
