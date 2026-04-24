import os
import sys
import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from scipy.ndimage import gaussian_filter


def list_h5_datasets(h5obj):
    ds_paths = []

    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset):
            ds_paths.append(name)

    h5obj.visititems(visitor)
    return ds_paths


def find_best_timeseries_dataset(h5obj):
    ds_paths = list_h5_datasets(h5obj)

    preferred_keywords = [
        "tdi/x",
        "obs/tdi/x",
        "x",
        "tdi",
        "obs",
        "data",
        "strain",
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
            if ndim == 1 and size > 1000:
                score += 50
            elif ndim == 2 and max(shape) > 1000:
                score += 30
        except Exception:
            pass

        scored.append((score, path))

    scored.sort(reverse=True)

    for _, path in scored[:10]:
        data = np.array(h5obj[path])
        if data.ndim == 1 and data.size > 1000:
            return path, data
        if data.ndim == 2 and max(data.shape) > 1000:
            if data.shape[0] > data.shape[1]:
                return path, data[:, 0]
            return path, data[0, :]

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
        raise RuntimeError("没有找到合适的一维时间序列数据集。")

    return best_path, best_data


def read_time_series_from_hdf5(h5_path, channel="X"):
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

    if getattr(data.dtype, "names", None) is not None:
        names = data.dtype.names
        print(f"[INFO] structured fields: {names}")

        if channel in names:
            x = np.asarray(data[channel], dtype=np.float64)
        elif "X" in names:
            x = np.asarray(data["X"], dtype=np.float64)
        else:
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
        data = np.asarray(data)
        if data.ndim == 1:
            x = np.asarray(data, dtype=np.float64)
        elif data.ndim == 2:
            if data.shape[0] > data.shape[1]:
                x = np.asarray(data[:, 0], dtype=np.float64)
            else:
                x = np.asarray(data[0, :], dtype=np.float64)
        else:
            raise RuntimeError(f"不支持的数据形状: {data.shape}")

        t = np.arange(len(x)) * dt

    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return t, x, dt


def plot_timeseries(t, x, out_png):
    plt.figure(figsize=(12, 3.5))
    plt.plot(t / 86400.0, x, lw=0.5)
    plt.xlabel("Time [days]")
    plt.ylabel("Amplitude")
    plt.title("Selected LISA time series")
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def run_wdm_like_spectrogram(x, dt):
    """
    最终版：直接使用 spectrogram 生成接近老师风格的整段时频图
    """
    x_use = np.asarray(x, dtype=np.float64).copy()
    x_use = np.nan_to_num(x_use, nan=0.0, posinf=0.0, neginf=0.0)

    # 预处理
    x_use = x_use - np.mean(x_use)

    low = np.percentile(x_use, 1)
    high = np.percentile(x_use, 99)
    x_use = np.clip(x_use, low, high)

    std = np.std(x_use)
    if std > 0:
        x_use = x_use / std

    fs = 1.0 / dt  # Hz

    f, t_spec, Sxx = spectrogram(
        x_use,
        fs=fs,
        window="hann",
        nperseg=8192,
        noverlap=7168,
        detrend="constant",
        scaling="density",
        mode="psd"
    )

    Sxx_db = 10.0 * np.log10(Sxx + 1e-30)

    # 只保留低频段
    mask = (f >= 1e-4) & (f <= 3e-2)
    f = f[mask]
    Sxx_db = Sxx_db[mask, :]

    # 每个频率 bin 去背景
    Sxx_db = Sxx_db - np.median(Sxx_db, axis=1, keepdims=True)

    # 平滑
    Sxx_db = gaussian_filter(Sxx_db, sigma=(1.5, 1.0))

    # 低频抑制，让图更接近老师风格
    weight = np.linspace(0.3, 1.0, Sxx_db.shape[0])[:, None]
    Sxx_db = Sxx_db * weight

    # 固定颜色范围
    vmin = -5
    vmax = 6
    Sxx_db = np.clip(Sxx_db, vmin, vmax)

    t_axis = t_spec / 86400.0
    f_axis = np.log10(f)

    print("[DEBUG] spectrogram shape:", Sxx_db.shape)
    print("[DEBUG] len(f_axis):", len(f_axis))
    print("[DEBUG] len(t_axis):", len(t_axis))

    return t_axis, f_axis, Sxx_db


def plot_wdm_like_spectrogram(t_axis, f_axis, w_plot, out_png):
    plt.figure(figsize=(12, 6))

    pcm = plt.pcolormesh(
        t_axis,
        f_axis,
        w_plot,
        shading="auto",
        cmap="viridis",
        vmin=-5,
        vmax=6
    )

    plt.colorbar(pcm, label="Relative PSD [dB]")
    plt.xlabel("Time [days]")
    plt.ylabel("log-frequency [Hz]")
    plt.xlim(0, 31)
    plt.ylim(-3.4, -1.7)
    plt.yticks([-3.0, -2.5, -2.0])
    plt.title("TDI X: spectrogram")

    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def main():
    if len(sys.argv) < 2:
        print("用法: python wdm_only.py LDC2_spritz_mbhb1_training_v1.h5")
        sys.exit(1)

    h5_path = sys.argv[1]
    if not os.path.exists(h5_path):
        raise FileNotFoundError(h5_path)

    t, x, dt = read_time_series_from_hdf5(h5_path, channel="X")

    plot_timeseries(t, x, "timeseries_overview.png")
    print("[OK] saved timeseries_overview.png")

    t_w, f_w, w_plot = run_wdm_like_spectrogram(x, dt)
    plot_wdm_like_spectrogram(t_w, f_w, w_plot, "wdm_timeseries.png")
    print("[OK] saved wdm_timeseries.png")

    print("[DONE]")


if __name__ == "__main__":
    main()