import pytest
import numpy as np
import xarray as xr
from pathlib import Path


from mrrpropy.raw_class import MRRProData  # cambia 'mrrpro' por el nombre real de tu módulo

# Ruta por defecto al fichero de prueba.
# Puedes sobrescribirla con la variable de entorno MRRPRO_TEST_FILE.
MRR_PATH = Path(r"./tests/data/RAW/mrrpro81/2025/03/08/20250308_120000.nc")
OUTPUT_DIR = Path(MRR_PATH.parent.absolute().as_posix().replace('RAW', 'PRODUCTS'))  
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

@pytest.fixture(scope="session")
def mrr():
    """Carga una instancia de MRRProData para todos los tests."""
    if not MRR_PATH.exists():
        pytest.skip(f"No se encuentra el archivo de data: {MRR_PATH}")
    mrr = MRRProData.from_file(MRR_PATH)
    yield mrr
    mrr.close()


def test_dataset_loaded(mrr):
    """El dataset se carga correctamente y tiene dimensiones no vacías."""
    assert mrr.ds is not None
    assert mrr.n_time > 0
    assert mrr.n_range > 0
    assert "time" in mrr.ds.dims
    assert "range" in mrr.ds.dims


def test_basic_properties(mrr):
    """Comprobación de propiedades básicas: time, range, variables."""
    # time como índice de pandas
    t_index = mrr.time
    assert len(t_index) == mrr.n_time

    # rango como array numpy
    r = mrr.range
    assert isinstance(r, np.ndarray)
    assert r.shape == (mrr.n_range,)

    # lista de variables no vacía y contiene algunas esperadas
    vars_ = mrr.variables
    assert len(vars_) > 0
    # En tu archivo real existen estas variables:
    for expected in ["Ze", "RR", "VEL", "N", "D", "index_spectra"]:
        assert expected in vars_


def test_get_field(mrr):
    """get_field debe devolver un DataArray con las dimensiones esperadas."""
    ze = mrr.get_field("Ze")
    assert ze.dims == ("time", "range")
    assert ze.shape == (mrr.n_time, mrr.n_range)

    # Comprobar que lanza error con una variable inexistente
    with pytest.raises(KeyError):
        mrr.get_field("variable_que_no_existe")


def test_subset_time_and_range(mrr):
    """subset debe devolver un nuevo MRRProData con dimensiones reducidas."""
    # Subconjunto en tiempo: primeras 10 muestras
    mrr_sub = mrr.subset(
        time_slice=slice(mrr.time[0], mrr.time[9])
    )
    assert mrr_sub.n_time == 10
    assert mrr_sub.n_range == mrr.n_range

    # Subconjunto en rango: primeros 20 bins
    mrr_sub2 = mrr.subset(range_slice=slice(0, 2000))
    
    assert mrr_sub2.n_range == 39
    assert mrr_sub2.n_time == mrr.n_time

    # Subconjunto combinado
    mrr_sub3 = mrr.subset(
        time_slice=slice(mrr.time[0], mrr.time[9]),
        range_slice=slice(0, 2000),
    )
    assert mrr_sub3.n_time == 10

    assert mrr_sub3.n_range == 39


def test_nearest_time_index_and_profile(mrr):
    """nearest_time_index y profile_at deben ser coherentes."""
    # Usamos el primer instante del dataset
    t0 = mrr.time[0]
    idx0 = mrr.nearest_time_index(t0)
    assert idx0 == 0

    # Perfil de Ze en ese instante
    profile = mrr.profile_at(t0, field="Ze")
    assert profile.dims == ("range",)
    assert profile.shape == (mrr.n_range,)

    # Si paso una cadena equivalente debería dar un índice cercano
    t0_str = str(t0)
    idx0b = mrr.nearest_time_index(t0_str)
    assert abs(idx0b - idx0) <= 1  # tolerancia por si hay ligeras diferencias


def test_gate_spectrum(mrr):
    """gate_spectrum debe devolver espectro y eje de velocidad coherentes."""
    # Elegimos un gate "típico": tiempo 0, rango 0 (ajusta si lo necesitas)
    time_idx = 0
    range_idx = 0

    vel, spec = mrr.gate_spectrum(time_idx=time_idx, range_idx=range_idx)

    # Comprobación básica de dimensiones: 1D
    assert len(vel.shape) == 1
    assert len(spec.shape) == 1
    assert vel.shape == spec.shape

    # Deben coincidir con la dimensión spectrum_n_samples
    n_samp = mrr.ds.sizes["spectrum_n_samples"]
    assert vel.shape[0] == n_samp
    assert spec.shape[0] == n_samp

    # Si se pide use_raw=True, debe funcionar igual pero con spectrum_raw
    vel_raw, spec_raw = mrr.gate_spectrum(
        time_idx=time_idx, range_idx=range_idx, use_raw=True
    )
    assert vel_raw.shape == vel.shape
    assert spec_raw.shape == spec.shape

def test_processing_raprompro(mrr) -> xr.Dataset:
    """
    Ejecuta una vez el procesado RaProM-Pro y reutiliza el resultado en todos los tests.
    """

    out = mrr.process_raprompro()
    breakpoint()
    out.to_netcdf(OUTPUT_DIR / f"{MRR_PATH.stem}_processed.nc")
    if not isinstance(out, xr.Dataset):
        pytest.fail("process_raprompro() no devolvió un xr.Dataset.")
    return out

