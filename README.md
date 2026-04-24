可以，下面我直接给你一套**能交作业的实现思路 + 可跑的 Python 脚本骨架**。这题的关键点我先帮你定下来：

Spritz 2b 的数据说明里明确写了：这是 **second-generation Michelson TDI X/Y/Z** 数据，采样间隔是 **5 秒**；官方 WDM 仓库也明确提供了 `transform_wavelet_time` 和 `transform_wavelet_freq_time` 两个 Python 接口；FRFT 方面，可以直接用开源仓库 `siddharth-maddali/frft` 这类实现。([lisa-ldc.in2p3.fr][1])

另外，这个数据集里的 MBHB 信号本来就是并合前后持续数天到数周的啁啾型信号，所以你在时频图里应该能看到**频率随时间上升**的结构；Spritz 2b 还带有 gaps / glitches / 非平稳噪声，这也是图上会有杂散结构的原因。([lisa-ldc.in2p3.fr][1])

## 你最终要做的事

1. 下载 `LDC2_spritz_mbhb1_training_v1.h5`
2. 运行下面脚本，生成：

   * `wdm_timeseries.png`
   * `frft_scan.png`
3. 把脚本和结果图推到你自己的 GitHub repo
4. 报名回复 repo 链接

## 依赖安装

```bash
pip install numpy scipy matplotlib h5py
pip install WDMWaveletTransforms
```

FRFT 你可以二选一：

### 方案 A：直接用现成库

如果你本地能装到带 `frft` 的包，就直接用。

### 方案 B：把 GitHub 里的 `frft.py` 放到项目目录

参考这个仓库：`siddharth-maddali/frft`。它说明了 FRFT 的参数约定：`alpha=1` 对应普通傅里叶变换的四分之一周期旋转。([GitHub][2])

---

## 推荐脚本：`lisa_wdm_frft.py`

这个版本特意做成了**尽量不依赖 HDF5 内部固定键名**。如果你的 h5 文件字段稍有不同，它会先自动搜索常见数据集。

```python
import os
import sys
import numpy as np
import h5py
import matplotlib.pyplot as plt

# -----------------------------
# 可选：WDM
# -----------------------------
try:
    from WDMWaveletTransforms.wavelet_transforms import transform_wavelet_time
except Exception:
    try:
        from wavelet_transforms import transform_wavelet_time
    except Exception:
        transform_wavelet_time = None

# -----------------------------
# 可选：FRFT
# 你可以把第三方 frft.py 放在当前目录，提供 frft(x, alpha)
# -----------------------------
try:
    from frft import frft
except Exception:
    frft = None


def list_h5_datasets(h5obj):
    """递归列出所有 dataset 路径"""
    ds_paths = []

    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset):
            ds_paths.append(name)

    h5obj.visititems(visitor)
    return ds_paths


def find_best_timeseries_dataset(h5obj):
    """
    自动寻找最像 TDI 时间序列的 dataset。
    优先匹配常见名字：X/Y/Z、tdi、obs 等。
    """
    ds_paths = list_h5_datasets(h5obj)

    # 先按常见命名偏好排序
    preferred_keywords = [
        "tdi/x", "obs/tdi/x", "x", "t/x",
        "tdi", "obs", "data", "strain"
    ]

    scored = []
    for path in ds_paths:
        low = path.lower()
        score = 0
        for i, kw in enumerate(preferred_keywords):
            if kw in low:
                score += 100 - i
        try:
            shape = h5obj[path].shape
            ndim = len(shape)
            size = np.prod(shape)
            # 偏好一维长序列，或二维中一维明显很长
            if ndim == 1 and size > 1000:
                score += 50
            elif ndim == 2 and max(shape) > 1000:
                score += 30
        except Exception:
            pass
        scored.append((score, path))

    scored.sort(reverse=True)

    # 先尝试前几个
    for _, path in scored[:10]:
        data = np.array(h5obj[path])
        if data.ndim == 1 and data.size > 1000:
            return path, data
        if data.ndim == 2 and max(data.shape) > 1000:
            # 若是二维，选更长的那一维作为时间轴
            if data.shape[0] > data.shape[1]:
                return path, data[:, 0]
            else:
                return path, data[0, :]

    # 再兜底：任选一个最大的 1D dataset
    best_path = None
    best_size = -1
    best_data = None
    for path in ds_paths:
        data = np.array(h5obj[path])
        if data.ndim == 1 and data.size > best_size:
            best_path = path
            best_size = data.size
            best_data = data

    if best_path is None:
        raise RuntimeError("没有找到合适的一维时间序列数据集，请先打印 HDF5 结构手动确认。")

    return best_path, best_data


def read_time_series_from_hdf5(h5_path):
    """
    读取 HDF5 中最可能的 TDI-X 时间序列。
    NaN -> 0
    """
    with h5py.File(h5_path, "r") as f:
        ds_path, x = find_best_timeseries_dataset(f)

        # 尝试找时间步长 dt
        dt = 5.0  # Spritz 2b 文档给出 5 秒采样；这里也做默认值
        for key in ["dt", "cadence", "delta_t", "sample_dt"]:
            if key in f.attrs:
                try:
                    dt = float(f.attrs[key])
                    break
                except Exception:
                    pass

        print(f"[INFO] selected dataset: {ds_path}")
        print(f"[INFO] raw shape: {x.shape}")
        print(f"[INFO] dt = {dt}")

    x = np.asarray(x, dtype=np.float64)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    t = np.arange(len(x)) * dt
    return t, x, dt


def choose_wdm_shape(n):
    """
    给 WDM 选 Nt, Nf。
    官方示例常见的是 Nt 为偶数、Nf 为 2 的幂量级。
    这里取一个兼顾速度和图像效果的近似。
    """
    # Nt 取不超过数据长度的 2 的幂，且为偶数
    Nt = 1
    while Nt * 2 <= n:
        Nt *= 2
    Nt = max(512, Nt // 4)  # 不直接吃满，防止太慢
    Nt = (Nt // 2) * 2

    # Nf 经验取比 Nt 小一些
    Nf = min(4096, max(256, Nt // 2))
    # 向最近 2 的幂靠拢
    p = 1
    while p < Nf:
        p *= 2
    Nf = p if abs(p - Nf) < abs(p // 2 - Nf) else p // 2
    Nf = max(256, Nf)

    return Nt, Nf


def run_wdm(x, dt, mult=16):
    if transform_wavelet_time is None:
        raise ImportError(
            "没有导入到 WDMWaveletTransforms。请先 pip install WDMWaveletTransforms"
        )

    Nt, Nf = choose_wdm_shape(len(x))
    x_use = x[:Nt].copy()

    print(f"[INFO] WDM input length={len(x_use)}, Nt={Nt}, Nf={Nf}, mult={mult}")

    w = transform_wavelet_time(x_use, Nf, Nt, dt, mult=mult)
    w = np.asarray(w)

    # 常见情况下输出可 reshape 为 (Nt, Nf) 或 (Nf, Nt)
    if w.ndim == 1:
        if w.size == Nt * Nf:
            w = w.reshape(Nt, Nf)
        else:
            raise RuntimeError(f"WDM 输出长度 {w.size} 不能 reshape 成 ({Nt}, {Nf})")

    # 保证 shape 为 (Nf, Nt) 便于画图
    if w.shape == (Nt, Nf):
        w_plot = np.abs(w.T)
    elif w.shape == (Nf, Nt):
        w_plot = np.abs(w)
    else:
        # 自动猜一下
        if w.shape[0] < w.shape[1]:
            w_plot = np.abs(w)
        else:
            w_plot = np.abs(w.T)

    # 压缩动态范围，图更清晰
    w_plot = np.log10(w_plot + 1e-12)

    t_axis = np.arange(w_plot.shape[1]) * dt
    # 粗略频率轴：0 ~ Nyquist
    f_axis = np.linspace(0, 1.0 / (2.0 * dt), w_plot.shape[0])

    return t_axis, f_axis, w_plot


def frft_single_signal(x, alpha):
    """
    调用外部 frft(x, alpha)。
    如果你放的是别家实现，可能参数名不同，可在这里改。
    """
    if frft is None:
        raise ImportError(
            "没有导入到 frft。请把 frft.py 放在当前目录，或安装可用 FRFT 实现。"
        )
    y = frft(x, alpha)
    return np.asarray(y)


def choose_frft_segment(x, dt, duration_days=8, downsample=4):
    """
    对 MBHB 啁啾，通常看末段更明显。
    取末尾若干天，并可适度降采样加快 FRFT 扫描。
    """
    n_tail = int(duration_days * 24 * 3600 / dt)
    n_tail = min(n_tail, len(x))
    seg = x[-n_tail:].copy()

    if downsample > 1:
        seg = seg[::downsample]
        dt_eff = dt * downsample
    else:
        dt_eff = dt

    # 去均值并标准化
    seg = seg - np.mean(seg)
    std = np.std(seg)
    if std > 0:
        seg = seg / std

    return seg, dt_eff


def run_frft_scan(x, dt, alpha_min=0.0, alpha_max=1.0, n_alpha=120):
    """
    对不同 alpha 做 FRFT 扫描，形成 alpha-样本 index 图。
    这不是传统的 time-frequency 图，但很适合把啁啾信号“聚焦”出来。
    """
    seg, dt_eff = choose_frft_segment(x, dt, duration_days=8, downsample=4)
    alphas = np.linspace(alpha_min, alpha_max, n_alpha)

    spec = []
    for a in alphas:
        y = frft_single_signal(seg, a)
        spec.append(np.log10(np.abs(y) + 1e-10))

    spec = np.array(spec)  # shape: (n_alpha, n_samples)
    u_axis = np.arange(spec.shape[1])

    return alphas, u_axis, spec, dt_eff


def plot_wdm(t_axis, f_axis, w_plot, out_png):
    plt.figure(figsize=(12, 6))
    pcm = plt.pcolormesh(t_axis / 86400.0, f_axis * 1e3, w_plot, shading="auto")
    plt.colorbar(pcm, label="log10 |WDM coefficient|")
    plt.xlabel("Time [days]")
    plt.ylabel("Frequency [mHz]")
    plt.title("WDM time-frequency map")
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def plot_frft(alphas, u_axis, spec, out_png):
    plt.figure(figsize=(12, 6))
    pcm = plt.pcolormesh(u_axis, alphas, spec, shading="auto")
    plt.colorbar(pcm, label="log10 |FRFT amplitude|")
    plt.xlabel("Fractional-domain sample index")
    plt.ylabel("FRFT order α")
    plt.title("FRFT scan")
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def main():
    if len(sys.argv) < 2:
        print("用法: python lisa_wdm_frft.py LDC2_spritz_mbhb1_training_v1.h5")
        sys.exit(1)

    h5_path = sys.argv[1]
    if not os.path.exists(h5_path):
        raise FileNotFoundError(h5_path)

    t, x, dt = read_time_series_from_hdf5(h5_path)

    # 先画原始时间序列概览
    plt.figure(figsize=(12, 3.5))
    plt.plot(t / 86400.0, x, lw=0.5)
    plt.xlabel("Time [days]")
    plt.ylabel("Amplitude")
    plt.title("Selected LISA time series")
    plt.tight_layout()
    plt.savefig("timeseries_overview.png", dpi=180)
    plt.close()

    # WDM
    try:
        t_w, f_w, w_plot = run_wdm(x, dt, mult=16)
        plot_wdm(t_w, f_w, w_plot, "wdm_timeseries.png")
        print("[OK] saved wdm_timeseries.png")
    except Exception as e:
        print(f"[WARN] WDM failed: {e}")

    # FRFT
    try:
        alphas, u_axis, spec, dt_eff = run_frft_scan(
            x, dt, alpha_min=0.0, alpha_max=1.0, n_alpha=120
        )
        plot_frft(alphas, u_axis, spec, "frft_scan.png")
        print("[OK] saved frft_scan.png")
    except Exception as e:
        print(f"[WARN] FRFT failed: {e}")

    print("[DONE]")


if __name__ == "__main__":
    main()
```

---

## 怎么调，图会更像“作业图”

### 1) WDM 图不清晰时

重点调这几个参数：

```python
run_wdm(x, dt, mult=16)
```

你可以试：

* `mult=8`
* `mult=16`
* `mult=32`

再改：

* `Nt` 再大一点：让时间分辨率更细
* `Nf` 再大一点：让频率分辨率更细

你还可以把颜色范围卡一下：

```python
vmin = np.percentile(w_plot, 20)
vmax = np.percentile(w_plot, 99.5)
plt.pcolormesh(..., vmin=vmin, vmax=vmax)
```

这通常会比默认颜色条更容易看到主信号脊线。

### 2) FRFT 图不明显时

FRFT 最容易踩坑的是：**直接对全年数据做变换，信号太弱太长，图会糊**。
对 MBHB，建议：

* 只取**最后几天到一周**
* 适度下采样
* 对不同 `α` 扫描，而不是只画一个 α

也就是我上面脚本里的思路：**取尾段 + α 扫描图**。
这种图虽然不是标准 spectrogram，但很适合展示“哪个分数阶最能把啁啾聚焦”。

如果你想更像“时频图”，可以进一步做**滑动窗 FRFT**：

* 每个窗口做一次 FRFT
* 对每个窗口取最优 α 或最大峰值
* 再拼成二维图

不过作为选拔题，先把上面的 `frft_scan.png` 做清楚，通常已经够用了。

---

## 如果 HDF5 键名不对，怎么快速修

先打印结构：

```python
import h5py

with h5py.File("LDC2_spritz_mbhb1_training_v1.h5", "r") as f:
    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(name, obj.shape, obj.dtype)
    f.visititems(visitor)
```

看输出里有没有类似：

* `X`
* `Y`
* `Z`
* `obs/tdi/X`
* `H5LISA/PreProcess/TDI/X`
* `obs/X`

然后把读取部分直接改成：

```python
x = np.array(f["你的真实路径"])
x = np.nan_to_num(x, nan=0.0)
```

---

## GitHub 提交最简流程

假设你已经建好了 repo：

```bash
git init
git add lisa_wdm_frft.py
git add timeseries_overview.png wdm_timeseries.png frft_scan.png
git commit -m "Add LISA LDC WDM and FRFT analysis"
git branch -M main
git remote add origin 你的仓库地址
git push -u origin main
```

报名时回复 repo 链接即可。

---

## 你在作业说明里可以这样写

你可以把下面这段稍微改改放进 `README.md`：

```markdown
本项目完成了 LISA Data Challenge 2b Spritz MBHB 训练数据的时频分析与可视化：

1. 使用 h5py 读取 HDF5 数据，并将 NaN 值置 0；
2. 使用 WDMWaveletTransforms 实现 WDM 小波时频变换；
3. 使用 FRFT 实现分数阶傅里叶变换，并通过 alpha 扫描展示啁啾信号特征；
4. 使用 matplotlib 的 pcolormesh 绘制时频图和 FRFT 扫描图。

说明：
- 数据集为 LISA Data Challenge 2b: Spritz 的 MBHB 训练数据；
- 数据为 second-generation Michelson TDI X/Y/Z，采样间隔 5 s；
- WDM 使用官方仓库提供的 transform_wavelet_time 接口；
- FRFT 使用开源 Python 实现。
```

这些描述和方法选择都与官方 Challenge 页面、Spritz 文档和 WDM/FRFT 仓库说明是一致的。([lisa-ldc.in2p3.fr][1])

---

## 我建议你实际提交时这样组织仓库

```text
repo/
├── README.md
├── lisa_wdm_frft.py
├── requirements.txt
├── timeseries_overview.png
├── wdm_timeseries.png
└── frft_scan.png
```

`requirements.txt` 可以写：

```txt
numpy
scipy
matplotlib
h5py
WDMWaveletTransforms
```

---

## 两个现实提醒

第一，我这里不能替你直接登录百度网盘下载文件，也不能替你登录 GitHub 账号上传仓库，所以“下载数据”和“推送到你自己的 repo”这两步你需要在本地完成。

第二，**真正最可能卡你的不是算法，而是 HDF5 里具体的 dataset 路径**。所以最稳的做法是：
先跑一遍“打印 HDF5 结构”的小段代码，确认真实字段名，再把读取路径固定下来。

你要是愿意，我下一条可以直接继续给你一份**更偏“最终提交版”**的 `README.md` 模板和一个**支持滑动窗 FRFT 的增强版脚本**。

[1]: https://lisa-ldc.in2p3.fr/challenge2b "LISA Data Challenge"
[2]: https://github.com/siddharth-maddali/frft "GitHub - siddharth-maddali/frft: Fast implementation of the multi-dimensional fractional Fourier transform in Python · GitHub"
