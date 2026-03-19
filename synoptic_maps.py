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
from matplotlib.colors import ListedColormap

import cartopy.crs as ccrs
import cartopy.feature as cfeature

from metpy.units import units
import metpy.calc as mpcalc


# =========================================================
# Paths
# =========================================================
DATA_DIR = Path("data")
OUTPUT_DIR = Path("figures/synoptic")
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
z_850 = dataset_pl.sel(pressure_level=850)["z"].values / 100.0
z_500 = dataset_pl.sel(pressure_level=500)["z"].values / 100.0
z_250 = dataset_pl.sel(pressure_level=250)["z"].values

camd_1000_500 = z_500 - z_1000
camd_1000_850 = z_850 - z_1000

u_1000 = dataset_pl.sel(pressure_level=1000)["u"].values
v_1000 = dataset_pl.sel(pressure_level=1000)["v"].values

u_850 = dataset_pl.sel(pressure_level=850)["u"].values
v_850 = dataset_pl.sel(pressure_level=850)["v"].values
t_850 = dataset_pl.sel(pressure_level=850)["t"].values * units.K
q_850 = dataset_pl.sel(pressure_level=850)["q"].values

u_500 = dataset_pl.sel(pressure_level=500)["u"].values
v_500 = dataset_pl.sel(pressure_level=500)["v"].values
w_500 = dataset_pl.sel(pressure_level=500)["w"].values

u_250 = dataset_pl.sel(pressure_level=250)["u"].values * units("m/s")
v_250 = dataset_pl.sel(pressure_level=250)["v"].values * units("m/s")

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
# Main synoptic maps
# =========================================================
def plot_surface_t2m(data_index, timestamp):
    t2m_slice = t2m[data_index, :, :]
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
        lons_2d, lats_2d, t2m_slice,
        levels=levels, cmap="coolwarm", alpha=1.0,
        transform=ccrs.PlateCarree()
    )

    cbar = plt.colorbar(filled, ax=ax, orientation="horizontal", pad=0.05, aspect=50)
    cbar.set_label("Temperatura a 2 m (°C)")
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
        f"PRNMM (hPa), barbela de vento (m/s) e temperatura a 2 m (°C) - "
        f"{timestamp.strftime('%d/%m/%Y %H:%M UTC')}"
    )

    filename = OUTPUT_DIR / f"{timestamp.strftime('%Y.%m.%d')}_temp.T2m.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()


def plot_850hpa(data_index, timestamp):
    u_850_slice = u_850[data_index, :, :]
    v_850_slice = v_850[data_index, :, :]
    t_850_slice = t_850[data_index, :, :]
    q_850_slice = q_850[data_index, :, :]
    pressure_slice = 850 * units.hPa * np.ones_like(t_850_slice.magnitude)

    dewp = mpcalc.dewpoint_from_specific_humidity(
        pressure=pressure_slice,
        temperature=t_850_slice,
        specific_humidity=q_850_slice
    )
    thetae = mpcalc.equivalent_potential_temperature(
        pressure=pressure_slice,
        temperature=t_850_slice,
        dewpoint=dewp
    ).magnitude

    fig, ax = plt.subplots(
        figsize=(15, 10),
        subplot_kw={"projection": ccrs.PlateCarree()}
    )
    setup_map(ax, borders_color="lightgrey", coast_color="lightgrey")

    clevte = np.arange(260, 370, 10)
    te = ax.contourf(
        lons_pl_2d, lats_pl_2d, thetae,
        levels=clevte, cmap="RdBu_r", extend="both",
        transform=ccrs.PlateCarree()
    )
    cbar = plt.colorbar(te, orientation="horizontal", pad=0.05, aspect=50, extendrect=True)
    cbar.set_label("Theta-e (K)")

    camd_slice = camd_1000_850[data_index, :, :]
    cs0 = ax.contour(
        lons_pl, lats_pl, camd_slice,
        np.arange(0, 550, 3),
        colors="black", linewidths=1.0, linestyles="dashed",
        transform=ccrs.PlateCarree()
    )
    plt.clabel(cs0, fmt="%d")

    z_850_slice = z_850[data_index, :, :]
    cs = ax.contour(
        lons_pl, lats_pl, z_850_slice,
        np.arange(0, 800, 3),
        colors="black",
        transform=ccrs.PlateCarree()
    )
    plt.clabel(cs, fmt="%d")

    flip_flag = get_flip_flag(u_850_slice.shape, lats_2d)
    skip = 30
    ax.barbs(
        lons_pl[::skip],
        lats_pl[::skip],
        u_850_slice[::skip, ::skip],
        v_850_slice[::skip, ::skip],
        length=6.0,
        sizes=dict(emptybarb=0.0, spacing=0.2, height=0.5),
        linewidth=0.7,
        pivot="middle",
        barbcolor="black",
        flip_barb=flip_flag[::skip, ::skip]
    )

    ax.set_title(
        f"850 hPa: Theta-e (K), altura geopotencial (dam), espessura entre 1000–850 hPa (dam) "
        f"e barbelas de vento (m/s) - {timestamp.strftime('%d/%m/%Y %H:%M UTC')}"
    )

    filename = OUTPUT_DIR / f"{timestamp.strftime('%Y.%m.%d')}_850hpa.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()


def plot_500hpa(data_index, timestamp):
    u_500_slice = u_500[data_index, :, :]
    v_500_slice = v_500[data_index, :, :]
    w_500_slice = w_500[data_index, :, :]
    z_500_slice = z_500[data_index, :, :]

    fig, ax = plt.subplots(
        figsize=(15, 10),
        subplot_kw={"projection": ccrs.PlateCarree()}
    )
    setup_map(ax)

    cs = ax.contour(
        lons_2d, lats_2d, z_500_slice,
        np.arange(0, 800, 6),
        colors="black",
        transform=ccrs.PlateCarree()
    )
    plt.clabel(cs, fmt="%d")

    purple_cmap = ListedColormap(["purple"])
    csw = ax.contourf(
        lons_2d, lats_2d, w_500_slice,
        np.arange(-50, 0, 0.3),
        cmap=purple_cmap,
        transform=ccrs.PlateCarree(),
        alpha=0.5
    )
    cbar = plt.colorbar(csw, ax=ax, orientation="horizontal", pad=0.05, aspect=50)
    cbar.set_label("Omega negativo")
    cbar.set_ticks([-50, -0.3])

    flip_flag = get_flip_flag(u_500_slice.shape, lats_2d)
    skip = 30
    ax.barbs(
        lons_pl[::skip],
        lats_pl[::skip],
        u_500_slice[::skip, ::skip],
        v_500_slice[::skip, ::skip],
        length=6.0,
        sizes=dict(emptybarb=0.0, spacing=0.2, height=0.5),
        linewidth=0.7,
        pivot="middle",
        barbcolor="black",
        flip_barb=flip_flag[::skip, ::skip]
    )

    ax.set_title(
        f"500 hPa: altura geopotencial (dam), vento (m/s) e omega negativo - "
        f"{timestamp.strftime('%d/%m/%Y %H:%M UTC')}"
    )

    filename = OUTPUT_DIR / f"{timestamp.strftime('%Y.%m.%d')}_500hpa.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()

def plot_prnmm_thickness_jet_map(data_index, timestamp):
    u_1000_slice = u_1000[data_index, :, :]
    v_1000_slice = v_1000[data_index, :, :]
    mslp_slice = mslp[data_index, :, :]
    camd_slice = camd_1000_500[data_index, :, :]

    fig, ax = plt.subplots(
        figsize=(15, 10),
        subplot_kw={"projection": ccrs.PlateCarree()}
    )
    setup_map(ax)

    clevjet = np.arange(35, 106, 10)
    jet250 = ax.contourf(
        lons_pl, lats_pl, speed_jet[data_index, :, :],
        clevjet, cmap="Blues", alpha=0.5,
        transform=ccrs.PlateCarree()
    )
    bar = plt.colorbar(jet250, orientation="horizontal", pad=0.05, aspect=50, extendrect=True)
    bar.set_label("Jato (m/s)")

    clevs = (
        np.arange(0, 540, 6),
        np.array([540]),
        np.arange(546, 700, 6)
    )
    colors = ("tab:blue", "b", "tab:red")
    kw_clabels = {
        "fontsize": 11,
        "inline": True,
        "inline_spacing": 5,
        "fmt": "%i",
        "rightside_up": True,
        "use_clabeltext": True,
    }

    for clevthick, color in zip(clevs, colors):
        img2 = ax.contour(
            lons_pl, lats_pl, camd_slice,
            levels=clevthick,
            colors=color,
            linewidths=1.0,
            linestyles="dashed",
            transform=ccrs.PlateCarree()
        )
        plt.clabel(img2, **kw_clabels)

    levels = np.arange(900, 1050, 4)
    img3 = ax.contour(
        lons_pl, lats_pl, mslp_slice,
        colors="black",
        linewidths=0.7,
        levels=levels,
        transform=ccrs.PlateCarree()
    )
    ax.clabel(img3, inline=1, inline_spacing=0, fontsize=10, fmt="%1.0f", colors="black")

    flip_flag = get_flip_flag(u_1000_slice.shape, lats_2d)
    skip = 30
    ax.barbs(
        lons_pl[::skip],
        lats_pl[::skip],
        u_1000_slice[::skip, ::skip],
        v_1000_slice[::skip, ::skip],
        length=6.0,
        sizes=dict(emptybarb=0.0, spacing=0.2, height=0.5),
        linewidth=0.7,
        pivot="middle",
        barbcolor="black",
        flip_barb=flip_flag[::skip, ::skip]
    )

    ax.set_title(
        f"PRNMM (hPa), barbela de vento (m/s), espessura entre 1000–500 hPa (dam) "
        f"e jato em 250 hPa - {timestamp.strftime('%d/%m/%Y %H:%M UTC')}"
    )

    filename = OUTPUT_DIR / f"{timestamp.strftime('%Y.%m.%d')}_prnmm_thickness_jet.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()



def main():
    for data_especifica in datas_18h:
        index = int((time_data == data_especifica).argmax())

        plot_surface_t2m(index, data_especifica)
        plot_850hpa(index, data_especifica)
        plot_500hpa(index, data_especifica)
        plot_prnmm_thickness_jet_map(index, data_especifica)


if __name__ == "__main__":
    main()

