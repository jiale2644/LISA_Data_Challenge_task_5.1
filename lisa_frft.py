import os
import sys
import numpy as np
import h5py
import matplotlib.pyplot as plt


# WDM

try:
    from WDMWaveletTransforms.wavelet_transforms import (
        transform_wavelet_time,
        transform_wavelet_freq_time,
    )
    print("[INFO] WDM imported successfully from package")
except Exception as e1:
    try:
        from wavelet_transforms import (
            transform_wavelet_time,
            transform_wavelet_freq_time,
        )
        print("[INFO] WDM imported successfully from local module")
    except Exception as e2:
        print(f"[WARN] WDM import failed from package: {e1}")
        print(f"[WARN] WDM import failed from local module: {e2}")
        transform_wavelet_time = None
        transform_wavelet_freq_time = None


# FRFT

try:
    from frft import frft
    print("[INFO] FRFT imported successfully")
except Exception as e:
    print(f"[WARN] FRFT import failed: {e}")
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


def read_time_series_from_hdf5(h5_path, channel="X"):
    """
    读取 HDF5 中的 TDI 时间序列。
    支持普通 1D 数组，也支持 structured array，如 [('t','<f8'),('X','<f8'),...]
    NaN -> 0
    """
    with h5py.File(h5_path, "r") as f:
        ds_path, data = find_best_timeseries_dataset(f)

        dt = 5.0
        for key in ["dt", "cadence", "delta_t", "sample_dt"]:
            if key in f.attrs:
                try:
                    dt = float(f.attrs[key])
                    break
                except Exception:
                    pass

        print(f"[INFO] selected dataset: {ds_path}")
        print(f"[INFO] raw shape: {data.shape}")
        print(f"[INFO] dtype: {data.dtype}")
        print(f"[INFO] dt = {dt}")

    # ---------- 关键修复 ----------
    # 如果是 structured array，例如含 t/X/Y/Z 字段
    if getattr(data.dtype, "names", None) is not None:
        names = data.dtype.names
        print(f"[INFO] structured fields: {names}")

        if channel in names:
            x = np.asarray(data[channel], dtype=np.float64)
        elif "X" in names:
            x = np.asarray(data["X"], dtype=np.float64)
        else:
            # 兜底：找第一个非时间字段
            non_time = [n for n in names if n.lower() not in ("t", "time")]
            if not non_time:
                raise RuntimeError(f"结构化数组里没有可用信号字段：{names}")
            x = np.asarray(data[non_time[0]], dtype=np.float64)

        if "t" in names:
            t = np.asarray(data["t"], dtype=np.float64)
            if len(t) > 1:
                dt = float(np.median(np.diff(t)))
        else:
            t = np.arange(len(x)) * dt

    else:
        # 普通数组
        data = np.asarray(data)
        if data.ndim == 1:
            x = np.asarray(data, dtype=np.float64)
        elif data.ndim == 2:
            # 二维时默认取第一列
            if data.shape[0] > data.shape[1]:
                x = np.asarray(data[:, 0], dtype=np.float64)
            else:
                x = np.asarray(data[0, :], dtype=np.float64)
        else:
            raise RuntimeError(f"不支持的数据形状: {data.shape}")

        t = np.arange(len(x)) * dt

    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return t, x, dt


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


def choose_frft_segment(x, dt, duration_days=4, downsample=2):
    """
    只取末段，更容易看到 MBHB chirp。
    duration_days 缩短到 4 天，信号会更集中。
    """
    n_tail = int(duration_days * 24 * 3600 / dt)
    n_tail = min(n_tail, len(x))

    seg = x[-n_tail:].copy()
    seg = np.nan_to_num(seg, nan=0.0, posinf=0.0, neginf=0.0)

    # 去均值
    seg = seg - np.mean(seg)

    # 限幅，避免个别尖峰破坏动态范围
    low = np.percentile(seg, 1)
    high = np.percentile(seg, 99)
    seg = np.clip(seg, low, high)

    # 下采样，减少计算量，同时保留结构
    if downsample > 1:
        seg = seg[::downsample]
        dt_eff = dt * downsample
    else:
        dt_eff = dt

    # 标准化
    std = np.std(seg)
    if std > 0:
        seg = seg / std

    return seg, dt_eff


def run_frft_scan(x, dt, alpha_min=0.0, alpha_max=0.25, n_alpha=180):
    """
    FRFT 扫描。
    只扫 0~0.25 附近，通常更容易把 chirp 聚焦出来。
    """
    seg, dt_eff = choose_frft_segment(x, dt, duration_days=3, downsample=2)
    alphas = np.linspace(alpha_min, alpha_max, n_alpha)

    spec = []
    for a in alphas:
        y = frft_single_signal(seg, a)
        amp = np.abs(y)

        # 转 log 振幅
        row = np.log10(amp + 1e-10)
        spec.append(row)

    spec = np.array(spec)   # shape: (n_alpha, n_samples)

    # ===== 关键：按 alpha 行减去背景 =====
    bg = np.median(spec, axis=1, keepdims=True)
    spec = spec - bg

    # ===== 再做轻微平滑，增强 V 形/聚焦结构 =====
    from scipy.ndimage import gaussian_filter
    spec = gaussian_filter(spec, sigma=(1.0, 0.8))

    u_axis = np.arange(spec.shape[1])

    return alphas, u_axis, spec, dt_eff


def plot_frft(alphas, u_axis, spec, out_png):
    plt.figure(figsize=(12, 6))

    vals = spec[np.isfinite(spec)]
    vmin = np.percentile(vals, 20)
    vmax = np.percentile(vals, 99)

    pcm = plt.pcolormesh(
        u_axis,
        alphas,
        spec,
        shading="auto",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax
    )

    plt.colorbar(pcm, label="Relative log10 |FRFT amplitude|")
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