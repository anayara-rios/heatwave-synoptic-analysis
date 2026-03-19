#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path

import numpy as np
import numpy.ma as ma
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import cartopy.crs as ccrs
import cartopy.feature as cfeature

from metpy.units import units
import metpy.calc as mpcalc


# =========================================================
# Paths
# =========================================================
DATA_DIR = Path("data")
OUTPUT_DIR = Path("figures/complementary")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SINGLE_LEVELS_FILE = DATA_DIR / "single_levels.nc"
PRESSURE_LEVELS_FILE = DATA_DIR / "pressure_levels.nc"


# =========================================================
# Load datasets
# =========================================================
dataset_sl = xr.open_dataset(SINGLE_LEVELS_FILE)
dataset_pl = xr.open_dataset(PRESSURE_LEVELS_FILE)


# =========================================================
# Single-level variables
# =========================================================
mslp = dataset_sl["msl"].values / 100.0  # hPa
lats = dataset_sl["latitude"].values
lons = dataset_sl["longitude"].values
u_sl = dataset_sl["u10"].values
v_sl = dataset_sl["v10"].values
t2m = dataset_sl["t2m"].values - 273.15  # °C
d2m = dataset_sl["d2m"].values - 273.15  # °C
sst = dataset_sl["sst"].values - 273.15  # °C

time_data = pd.to_datetime(dataset_sl["valid_time"].values)
datas_18h = [t for t in time_data if t.hour == 18]


# =========================================================
# Pressure-level variables
# =========================================================
lats_pl = dataset_pl["latitude"].values
lons_pl = dataset_pl["longitude"].values

z_1000 = dataset_pl.sel(pressure_level=1000)["z"].values / 100.0
z_250 = dataset_pl.sel(pressure_level=250)["z"].values
u_250 = dataset_pl.sel(pressure_level=250)["u"].values * units("m/s")
v_250 = dataset_pl.sel(pressure_level=250)["v"].values * units("m/s")

u_1000 = dataset_pl.sel(pressure_level=1000)["u"].values
v_1000 = dataset_pl.sel(pressure_level=1000)["v"].values

speed_jet = mpcalc.wind_speed(u_250, v_250).magnitude
mask_jet = ma.masked_less_equal(speed_jet, 30).mask
speed_jet[mask_jet] = np.nan


# =========================================================
# Helpers
# =========================================================
if lons.ndim == 1 and lats.ndim == 1:
    lons_2d, lats_2d = np.meshgrid(lons, lats)
else:
    lons_2d, lats_2d = lons, lats

if lons_pl.ndim == 1 and lats_pl.ndim == 1:
    lons_pl_2d, lats_pl_2d = np.meshgrid(lons_pl, lats_pl)
else:
    lons_pl_2d, lats_pl_2d = lons_pl, lats_pl


def setup_map(ax, borders_color="black", coast_color="black"):
    ax.add_feature(cfeature.COASTLINE, edgecolor=coast_color)
    ax.add_feature(cfeature.BORDERS, linestyle=":", edgecolor=borders_color)
    ax.set_extent([-180, 15, -70, 10], crs=ccrs.PlateCarree())

    gridlines = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    gridlines.xlabel_style = {"size": 10, "color": "gray"}
    gridlines.ylabel_style = {"size": 10, "color": "gray"}
    gridlines.xlocator = mticker.FixedLocator(np.arange(-180, 181, 20))
    gridlines.ylocator = mticker.FixedLocator(np.arange(-90, 91, 10))


def get_flip_flag(field_shape, lat_array):
    flip_flag = np.zeros(field_shape)
    if lat_array.ndim == 1:
        flip_flag[lat_array < 0] = 1
    else:
        flip_flag[lat_array < 0] = 1
    return flip_flag


# =========================================================
# Complementary maps
# =========================================================
def plot_dewpoint_map(data_index, timestamp):
    d2m_slice = d2m[data_index, :, :]
    mslp_slice = mslp[data_index, :, :]
    u_sl_slice = u_sl[data_index, :, :]
    v_sl_slice = v_sl[data_index, :, :]

    fig, ax = plt.subplots(
        figsize=(15, 10),
        subplot_kw={"projection": ccrs.PlateCarree()}
    )
    setup_map(ax)

    contour = ax.contour(
        lons_2d, lats_2d, mslp_slice,
        levels=20, colors="black", linewidths=1,
        transform=ccrs.PlateCarree()
    )
    ax.clabel(contour, inline=True, fontsize=10, fmt="%1.0f")

    levels = np.arange(-40, 45, 5)
    filled = ax.contourf(
        lons_2d, lats_2d, d2m_slice,
        levels=levels, cmap="coolwarm", alpha=1.0,
        transform=ccrs.PlateCarree()
    )

    cbar = plt.colorbar(filled, ax=ax, orientation="horizontal", pad=0.05, aspect=50)
    cbar.set_label("Temperatura do ponto de orvalho a 2 m (°C)")
    cbar.set_ticks(levels)

    flip_flag = get_flip_flag(u_sl_slice.shape, lats_2d)
    skip = 30
    ax.barbs(
        lons_2d[::skip, ::skip],
        lats_2d[::skip, ::skip],
        u_sl_slice[::skip, ::skip],
        v_sl_slice[::skip, ::skip],
        length=6.0,
        sizes=dict(emptybarb=0.0, spacing=0.2, height=0.5),
        linewidth=0.7,
        pivot="middle",
        barbcolor="black",
        flip_barb=flip_flag[::skip, ::skip]
    )

    ax.set_title(
        f"PRNMM (hPa), barbela de vento (m/s) e temperatura do ponto de orvalho a 2 m (°C) - "
        f"{timestamp.strftime('%d/%m/%Y %H:%M UTC')}"
    )

    filename = OUTPUT_DIR / f"{timestamp.strftime('%Y.%m.%d')}_dewpoint_2m.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()


def plot_sst_map(data_index, timestamp):
    mslp_slice = mslp[data_index, :, :]
    u_sl_slice = u_sl[data_index, :, :]
    v_sl_slice = v_sl[data_index, :, :]
    sst_slice = sst[data_index, :, :]

    fig, ax = plt.subplots(
        figsize=(15, 10),
        subplot_kw={"projection": ccrs.PlateCarree()}
    )
    setup_map(ax)

    contour = ax.contour(
        lons_2d, lats_2d, mslp_slice,
        levels=20, colors="black", linewidths=1,
        transform=ccrs.PlateCarree()
    )
    ax.clabel(contour, inline=True, fontsize=10, fmt="%1.0f")

    levels = np.arange(-40, 45, 5)
    filled = ax.contourf(
        lons_2d, lats_2d, sst_slice,
        levels=levels, cmap="coolwarm", alpha=1.0,
        transform=ccrs.PlateCarree()
    )

    cbar = plt.colorbar(filled, ax=ax, orientation="horizontal", pad=0.05, aspect=50)
    cbar.set_label("Temperatura da superfície do mar (°C)")
    cbar.set_ticks(levels)

    flip_flag = get_flip_flag(u_sl_slice.shape, lats_2d)
    skip = 30
    ax.barbs(
        lons_2d[::skip, ::skip],
        lats_2d[::skip, ::skip],
        u_sl_slice[::skip, ::skip],
        v_sl_slice[::skip, ::skip],
        length=6.0,
        sizes=dict(emptybarb=0.0, spacing=0.2, height=0.5),
        linewidth=0.7,
        pivot="middle",
        barbcolor="black",
        flip_barb=flip_flag[::skip, ::skip]
    )

    ax.set_title(
        f"PRNMM (hPa), barbela de vento (m/s) e temperatura da superfície do mar (°C) - "
        f"{timestamp.strftime('%d/%m/%Y %H:%M UTC')}"
    )

    filename = OUTPUT_DIR / f"{timestamp.strftime('%Y.%m.%d')}_sst.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()


def plot_sst_t2m_land_map(data_index, timestamp):
    t2m_slice = t2m[data_index, :, :]
    mslp_slice = mslp[data_index, :, :]
    u_sl_slice = u_sl[data_index, :, :]
    v_sl_slice = v_sl[data_index, :, :]
    sst_slice = sst[data_index, :, :]

    # Temperatura a 2 m apenas sobre o continente
    t2m_masked = np.where(np.isnan(sst_slice), t2m_slice, np.nan)

    common_temp_levels = np.arange(-40, 45, 5)

    fig, ax = plt.subplots(
        figsize=(15, 10),
        subplot_kw={"projection": ccrs.PlateCarree()}
    )
    setup_map(ax)

    contour = ax.contour(
        lons_2d, lats_2d, mslp_slice,
        levels=20, colors="black", linewidths=1,
        transform=ccrs.PlateCarree()
    )
    ax.clabel(contour, inline=True, fontsize=10, fmt="%1.0f")

    ax.contourf(
        lons_2d, lats_2d, sst_slice,
        levels=common_temp_levels, cmap="coolwarm", alpha=1.0,
        transform=ccrs.PlateCarree()
    )

    t2m_contour = ax.contourf(
        lons_2d, lats_2d, t2m_masked,
        levels=common_temp_levels, cmap="coolwarm", alpha=1.0,
        transform=ccrs.PlateCarree()
    )

    cbar = plt.colorbar(t2m_contour, ax=ax, orientation="horizontal", pad=0.05, aspect=50)
    cbar.set_label("Temperatura (°C)")
    cbar.set_ticks(common_temp_levels)

    flip_flag = get_flip_flag(u_sl_slice.shape, lats_2d)
    skip = 30
    ax.barbs(
        lons_2d[::skip, ::skip],
        lats_2d[::skip, ::skip],
        u_sl_slice[::skip, ::skip],
        v_sl_slice[::skip, ::skip],
        length=6.0,
        sizes=dict(emptybarb=0.0, spacing=0.2, height=0.5),
        linewidth=0.7,
        pivot="middle",
        barbcolor="black",
        flip_barb=flip_flag[::skip, ::skip]
    )

    ax.set_title(
        f"PRNMM (hPa), barbela de vento (m/s), temperatura da superfície do mar (°C) "
        f"e temperatura a 2 m no continente (°C) - {timestamp.strftime('%d/%m/%Y %H:%M UTC')}"
    )

    filename = OUTPUT_DIR / f"{timestamp.strftime('%Y.%m.%d')}_sst_t2m_land.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()


def plot_250hpa_jet(data_index, timestamp):
    z_250_slice = z_250[data_index, :, :]

    fig, ax = plt.subplots(
        figsize=(15, 10),
        subplot_kw={"projection": ccrs.PlateCarree()}
    )
    setup_map(ax)

    cs = ax.contour(
        lons_2d, lats_2d, z_250_slice / 10.0,
        np.arange(0, 2000, 12),
        colors="black",
        transform=ccrs.PlateCarree()
    )
    plt.clabel(cs, fmt="%d")

    jet250 = ax.contourf(
        lons_pl, lats_pl, speed_jet[data_index, :, :],
        np.arange(35, 106, 10),
        cmap="Blues",
        alpha=0.5,
        transform=ccrs.PlateCarree()
    )
    bar = plt.colorbar(jet250, orientation="horizontal", pad=0.05, aspect=50, extendrect=True)
    bar.set_label("Jato (m/s)")

    ax.set_title(
        f"Altura geopotencial (dam) e jato em 250 hPa - "
        f"{timestamp.strftime('%d/%m/%Y %H:%M UTC')}"
    )

    filename = OUTPUT_DIR / f"{timestamp.strftime('%Y.%m.%d')}_250hpa_geopot_jet.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    for data_especifica in datas_18h:
        index = int((time_data == data_especifica).argmax())

        plot_dewpoint_map(index, data_especifica)
        plot_sst_map(index, data_especifica)
        plot_sst_t2m_land_map(index, data_especifica)
        plot_250hpa_jet(index, data_especifica)


if __name__ == "__main__":
    main()

