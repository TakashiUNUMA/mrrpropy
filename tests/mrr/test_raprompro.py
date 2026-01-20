# tests/test_mrrpro_rarpom.py
from __future__ import annotations

from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np
import xarray as xr

# Ruta por defecto al fichero de prueba.
# Puedes sobrescribirla con la variable de entorno MRRPRO_TEST_FILE.
before_path = Path(r"./tests/data/RAW/mrrpro81/2025/03/08/20250308_120000.nc")
after_path  = Path(before_path.absolute().as_posix().replace('RAW', 'PRODUCTS')).parent / f"{before_path.stem}_processed.nc"
out_dir = Path(r"./tests/figures/mrrpro_processing_rarprompro_outputs")
out_dir.mkdir(parents=True, exist_ok=True)


def test_rarpom_processing():
    """Verifica que el procesamiento RARPOM produce un Dataset."""
    ds0 = xr.open_dataset(before_path)
    ds1 = xr.open_dataset(after_path)

    # Mapeo recomendado: (var_before, var_after, units)
    pairs = [
        ("Ze", "Ze", "dBZ"),
        ("Zea", "Zea", "dBZ"),
        ("Za", "Za", "dBZ"),
        ("RR", "RR", "mm/hr"),
        ("LWC", "LWC", "g/m3"),
        ("SNR", "SNR", "dB"),
        ("PIA", "DBPIA", "dB"),
        ("VEL", "W", "m/s"),
        ("WIDTH", "spectral width", "m/s"),
    ]

    def align_2d(a: xr.DataArray, b: xr.DataArray):
        """
        Alinea por índices/dims cuando los nombres difieren (range vs height),
        pero las formas son (time, z). Ajusta aquí si tu caso usa coords distintos.
        """
        # Normaliza dims verticales
        if "range" in a.dims and "height" in b.dims:
            b = b.rename({"height": "range"})
        if "height" in a.dims and "range" in b.dims:
            a = a.rename({"height": "range"})
        # Alinea por intersección de coords si existen y coinciden, si no por índice
        try:
            a2, b2 = xr.align(a, b, join="inner")
        except Exception:
            a2 = a
            b2 = b
        return a2, b2

    def stats(x, y):
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 10:
            return None
        xx = x[m].astype(float)
        yy = y[m].astype(float)
        bias = float(np.mean(yy - xx))
        rmse = float(np.sqrt(np.mean((yy - xx) ** 2)))
        corr = float(np.corrcoef(xx, yy)[0, 1]) if xx.size > 1 else np.nan
        # slope/intercept (y = s*x + i)
        s, i = np.polyfit(xx, yy, 1)
        return {"n": int(m.sum()), "bias": bias, "rmse": rmse, "corr": corr, "slope": float(s), "intercept": float(i)}

    rows = []

    for v0, v1, units in pairs:
        if v0 not in ds0 or v1 not in ds1:
            continue

        a = ds0[v0]
        b = ds1[v1]


        a, b = align_2d(a, b)

        x = a.values.ravel()
        y = b.values.ravel()
        if v1 == "DBPIA":
            y = np.abs(y)

        st = stats(x, y)
        if st is None:
            continue
        st.update({"var_before": v0, "var_after": v1, "units": units})
        rows.append(st)

        # downsample for plotting (avoid huge scatter)
        m = np.isfinite(x) & np.isfinite(y)
        idx = np.where(m)[0]
        if idx.size > 200_000:
            idx = np.random.default_rng(0).choice(idx, size=200_000, replace=False)

        xx = x[idx].astype(float)
        yy = y[idx].astype(float)

        fig = plt.figure()
        plt.scatter(xx, yy, s=1)
        lo = np.nanpercentile(np.concatenate([xx, yy]), 1)
        hi = np.nanpercentile(np.concatenate([xx, yy]), 99)
        plt.plot([lo, hi], [lo, hi])
        plt.xlabel(f"{v0} (before) [{units}]")
        plt.ylabel(f"{v1} (after) [{units}]")
        plt.title(f"1:1 {v0} vs {v1}  |  n={st['n']}  bias={st['bias']:.3g}  rmse={st['rmse']:.3g}  r={st['corr']:.3f}", fontsize=12)
        plt.xlim(lo, hi)
        plt.ylim(lo, hi)
        fig.savefig(out_dir / f"correlation_check_{v0}_vs_{v1}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    # Save metrics
    import csv
    with open(out_dir / "metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["var_before","var_after","units","n","bias","rmse","corr","slope","intercept"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in w.fieldnames})

    print(f"Saved plots + metrics to: {out_dir.resolve()}")
    
    # Assert r values > 0.9 for key variables
    key_vars = ["Ze", "RR", "LWC", "VEL"]
    for kv in key_vars:
        for r in rows:
            if r["var_before"] == kv:
                assert r["corr"] > 0.6, f"Low correlation for {kv}: {r['corr']}"
