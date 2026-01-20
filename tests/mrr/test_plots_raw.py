import pytest
import datetime
import numpy as np
from pathlib import Path
from matplotlib.collections import QuadMesh

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
matplotlib.use("Agg")  # imprescindible en CI/headless

from mrrpropy.raw_class import MRRProData  # cambia 'mrrpro' por el nombre real de tu módulo

# Ruta por defecto al fichero de prueba.
# Puedes sobrescribirla con la variable de entorno MRRPRO_TEST_FILE.
MRR_PATH = Path(r"./tests/data/RAW/mrrpro81/2025/03/08/20250308_120000.nc")
OUTPUT_DIR = Path(r"./tests/figures/mrr_plots_raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

@pytest.fixture(scope="session")
def mrr():
    """Carga una instancia de MRRProData para todos los tests."""
    if not MRR_PATH.exists():
        pytest.skip(f"No se encuentra el archivo de data: {MRR_PATH}")
    mrr = MRRProData.from_file(MRR_PATH)
    yield mrr
    mrr.close()


def test_quickplot_reflectivity_runs(mrr):
    """quickplot_reflectivity debe ejecutarse sin errores.

    No verificamos el contenido de la figura, solo que no haya excepciones.
    Este test se puede marcar como 'slow' si lo deseas.
    """
    pytest.importorskip("matplotlib")

    fig, ax = mrr.quickplot_reflectivity(field="Ze")
    fig.savefig('test_quickplot_reflectivity.png')  # Guardar la figura para inspección manual si se desea
    # Comprobación mínima de que devuelve objetos figura y ejes

    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)


def _pick_spectrum_var(ds) -> str:
    """Devuelve el nombre de la variable espectral a usar en tests."""
    if "spectrum_reflectivity" in ds:
        return "spectrum_reflectivity"
    if "spectrum_raw" in ds:
        return "spectrum_raw"
    pytest.skip("El Dataset no contiene 'spectrum_reflectivity' ni 'spectrum_raw'.")


def _pick_non_edge_time(ds) -> np.datetime64:
    """Elige un time no extremo para evitar problemas de borde."""
    t = ds["time"].values
    if t.size < 1:
        pytest.skip("Dataset sin dimensión time")
    # Preferimos un punto interior si hay muchos
    return t[t.size // 2]


def _pick_non_edge_range(ds) -> float:
    """Elige un range no extremo."""
    r = ds["range"].values.astype(float)
    if r.size < 1:
        pytest.skip("Dataset sin dimensión range")
    return float(r[r.size // 2])


# ============================================================
# NIVEL 1 — Smoke tests: devuelven Figure y no petan
# ============================================================
def test_plot_spectrum_returns_figure(mrr):
    ds = mrr.ds
    spectrum_var = _pick_spectrum_var(ds)

    target_time = _pick_non_edge_time(ds)
    target_range = _pick_non_edge_range(ds)

    fig, filepath = mrr.plot_spectrum(
        target_time,
        target_range,
        spectrum_var=spectrum_var,
        savefig=False,
    )

    assert isinstance(fig, Figure)
    assert filepath is None
    plt.close(fig)


def test_plot_spectrogram_returns_figure(mrr):
    ds = mrr.ds
    spectrum_var = _pick_spectrum_var(ds)
    target_time = _pick_non_edge_time(ds)

    fig, filepath = mrr.plot_spectrogram(
        target_time,
        spectrum_var=spectrum_var,
        savefig=False,
    )

    assert isinstance(fig, Figure)
    assert filepath is None
    plt.close(fig)


# ============================================================
# NIVEL 2 — Guardado a disco: crea PNG y devuelve Path
# ============================================================
def test_plot_spectrum_saves_png(tmp_path: Path, mrr):
    ds = mrr.ds
    spectrum_var = _pick_spectrum_var(ds)

    target_time = _pick_non_edge_time(ds)
    target_range = _pick_non_edge_range(ds)

    fig, filepath = mrr.plot_spectrum(
        target_time,
        target_range,
        spectrum_var=spectrum_var,
        savefig=True,
        output_dir=tmp_path,
        dpi=120,
    )

    assert isinstance(fig, Figure)
    assert filepath is not None
    assert filepath.exists()
    assert filepath.suffix.lower() == ".png"
    assert filepath.stat().st_size > 0
    plt.close(fig)


def test_plot_spectrogram_saves_png(tmp_path: Path, mrr):
    ds = mrr.ds
    spectrum_var = _pick_spectrum_var(ds)
    target_time = _pick_non_edge_time(ds)

    fig, filepath = mrr.plot_spectrogram(
        target_time,
        spectrum_var=spectrum_var,
        savefig=True,
        output_dir=tmp_path,
        dpi=120,
    )

    assert isinstance(fig, Figure)
    assert filepath is not None
    assert filepath.exists()
    assert filepath.suffix.lower() == ".png"
    assert filepath.stat().st_size > 0
    plt.close(fig)


# ============================================================
# NIVEL 3 — Validación de argumentos: savefig=True exige output_dir
# ============================================================
def test_plot_spectrum_savefig_requires_output_dir(mrr):
    ds = mrr.ds
    spectrum_var = _pick_spectrum_var(ds)

    target_time = _pick_non_edge_time(ds)
    target_range = _pick_non_edge_range(ds)

    with pytest.raises(ValueError):
        mrr.plot_spectrum(
            target_time,
            target_range,
            spectrum_var=spectrum_var,
            savefig=True,
            output_dir=None,
        )

def test_plot_ND_by_range_creates_expected_artists(mrr):
    """
    Test estándar (no visual) para plot_ND_by_range:
    - smoke test
    - devuelve Figure y no guarda si savefig=False
    - genera al menos una línea
    - añade leyenda
    """
    if not hasattr(mrr, "plot_ND_by_range"):
        pytest.skip("MRRProData.plot_ND_by_range() no existe todavía.")

    ds = mrr.ds
    if "time" not in ds or ds.sizes.get("time", 0) == 0:
        pytest.skip("Dataset sin 'time'.")
    if "range" not in ds or ds.sizes.get("range", 0) < 3:
        pytest.skip("Dataset sin suficientes gates en 'range'.")

    # tiempo representativo
    t = ds["time"].values[ds.sizes["time"] // 2]
    t = datetime.datetime(2025, 3, 8, 12, 50, 0)
    # tres rangos repartidos (usar valores existentes para evitar fragilidad)
    rvals = ds["range"].values.astype(float)
    # r_list = [
    #     float(rvals[len(rvals) // 6]),
    #     float(rvals[len(rvals) // 2]),
    #     float(rvals[5 * len(rvals) // 6]),
    # ]
    r_list = np.arange(500,4000, 200)

    fig, filepath = mrr.plot_ND_by_range(
        t,
        r_list,
        use_log10=False,
        savefig=True,
        cmap='jet',
        output_dir=OUTPUT_DIR,
        **{"xlimits": (0,12), 'N_minimum_threshold': 1e-6},
    )

    assert isinstance(fig, Figure)
    assert filepath
    assert len(fig.axes) >= 1

    ax = fig.axes[0]

    # Debe haber al menos una línea trazada (idealmente 3, pero no lo exigimos)
    assert len(ax.lines) >= 1, "No se trazaron curvas N(D)."

    # Debe existir leyenda (si hay líneas con label)
    legend = ax.get_legend()
    assert legend is not None, "Se esperaba una leyenda con los rangos."

    # Etiquetas mínimas
    assert ax.get_xlabel() != ""
    assert ax.get_ylabel() != ""

    plt.close(fig)


def test_plot_spectrogram_savefig_requires_output_dir(mrr):
    ds = mrr.ds
    spectrum_var = _pick_spectrum_var(ds)
    target_time = _pick_non_edge_time(ds)

    with pytest.raises(ValueError):
        mrr.plot_spectrogram(
            target_time,
            spectrum_var=spectrum_var,
            savefig=True,
            output_dir=None,
        )


# ============================================================
# NIVEL 4 — Nearest selection: el método debe aceptar time/range no exactos
# ============================================================
def test_plot_spectrum_accepts_non_exact_time_and_range(mrr):
    ds = mrr.ds
    spectrum_var = _pick_spectrum_var(ds)

    # Cogemos un time/range reales y los perturbamos para forzar method="nearest"
    t0 = _pick_non_edge_time(ds)
    r0 = _pick_non_edge_range(ds)

    # Perturbación pequeña: +1s y +0.49*dr
    # dr estimado de coord range
    r = ds["range"].values.astype(float)
    dr = float(np.nanmedian(np.diff(r))) if r.size > 1 else 1.0

    t_req = t0 + np.timedelta64(1, "s")
    r_req = r0 + 0.49 * dr

    fig, filepath = mrr.plot_spectrum(
        t_req,
        r_req,
        spectrum_var=spectrum_var,
        savefig=True,
        output_dir=OUTPUT_DIR,
    )

    assert isinstance(fig, Figure)
    assert filepath
    plt.close(fig)


def test_plot_spectrogram_accepts_non_exact_time(mrr):
    ds = mrr.ds
    spectrum_var = _pick_spectrum_var(ds)

    t0 = _pick_non_edge_time(ds)
    t_req = t0 + np.timedelta64(1, "s")

    fig, filepath = mrr.plot_spectrogram(
        t_req,
        spectrum_var=spectrum_var,
        savefig=True,
        output_dir=OUTPUT_DIR,
    )

    assert isinstance(fig, Figure)
    assert filepath
    plt.close(fig)


# ============================================================
# NIVEL 5 — Range limits: respeta slice y no falla
# ============================================================
def test_plot_spectrogram_with_range_limits(mrr):
    ds = mrr.ds
    spectrum_var = _pick_spectrum_var(ds)
    t0 = _pick_non_edge_time(ds)

    r = ds["range"].values.astype(float)
    if r.size < 4:
        pytest.skip("No hay suficientes gates para probar range_limits")

    # Tomamos un subrango interno
    r0 = float(r[r.size // 4])
    r1 = float(r[3 * r.size // 4])

    fig, filepath = mrr.plot_spectrogram(
        t0,
        spectrum_var=spectrum_var,
        range_limits=(r0, r1),
        savefig=True,
        output_dir=OUTPUT_DIR,
    )

    assert isinstance(fig, Figure)
    assert filepath
    plt.close(fig)


def test_plot_ND_gram_creates_expected_artists(mrr):
    """
    Test estándar de plot_ND_gram:
    - smoke + comprobaciones estructurales del gráfico (sin inspección visual).
    """
    if not hasattr(mrr, "plot_ND_gram"):
        pytest.skip("MRRProData.plot_ND_gram() no existe todavía.")

    ds = mrr.ds
    if "time" not in ds or ds.sizes.get("time", 0) == 0:
        pytest.skip("Dataset sin dimensión 'time'.")
    if "range" not in ds or ds.sizes.get("range", 0) == 0:
        pytest.skip("Dataset sin dimensión 'range'.")

    # tiempo representativo
    t = ds["time"].values[ds.sizes["time"] // 2]

    # subrango interno (evita bordes)
    r = ds["range"].values.astype(float)
    if r.size < 8:
        pytest.skip("No hay suficientes gates para probar range_limits.")
    r0 = float(r[r.size // 4])
    r1 = float(r[3 * r.size // 4])

    fig, filepath = mrr.plot_ND_gram(
        t,
        range_limits=(r0, r1),
        use_log10=True,
        savefig=True,
        output_dir=OUTPUT_DIR,
    )

    # 1) Retornos
    assert isinstance(fig, Figure)
    assert filepath

    # 2) Estructura: al menos eje principal
    assert len(fig.axes) >= 1
    ax = fig.axes[0]

    # 3) El plot debe haber creado al menos un QuadMesh (pcolormesh)
    assert any(isinstance(coll, QuadMesh) for coll in ax.collections), (
        "No se encontró QuadMesh en ax.collections. "
        "plot_ND_gram debería usar ax.pcolormesh()."
    )

    # 4) Debe existir una colorbar: normalmente añade un segundo axes
    # (esto es robusto si la colorbar está en axes separados)
    assert len(fig.axes) >= 2, "Se esperaba un axes adicional para la colorbar."

    # 5) Etiquetas mínimas (no frágiles)
    assert ax.get_xlabel() != ""
    assert ax.get_ylabel() != ""

    plt.close(fig)