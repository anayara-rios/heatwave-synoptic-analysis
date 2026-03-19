#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.lines import Line2D


# =========================================================
# Paths
# =========================================================
DATA_DIR = Path("data")
OUTPUT_DIR = Path("figures/anomalies")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLIM_SINGLE_FILE = DATA_DIR / "climatology_single_levels.nc"
CLIM_PRESSURE_FILE = DATA_DIR / "climatology_pressure_levels.nc"
EVENT_SINGLE_FILE = DATA_DIR / "single_levels.nc"
EVENT_PRESSURE_FILE = DATA_DIR / "pressure_levels.nc"


# =========================================================
# Load datasets
# =========================================================
dataset_clima = xr.open_dataset(CLIM_SINGLE_FILE)
dataset_clima_circ = xr.open_dataset(CLIM_PRESSURE_FILE)
dataset_sl = xr.open_dataset(EVENT_SINGLE_FILE)
dataset_pl = xr.open_dataset(EVENT_PRESSURE_FILE)


# =========================================================
# Helpers
# =========================================================
def setup_map(ax, extent=(-180, 15, -70, 10)):
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.5)
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    gl.xlabel_style = {"size": 9, "color": "gray"}
    gl.ylabel_style = {"size": 9, "color": "gray"}
    gl.xlocator = mticker.FixedLocator(np.arange(-180, 181, 20))
    gl.ylocator = mticker.FixedLocator(np.arange(-90, 91, 10))


def to_2d_coords(lons, lats):
    if lons.ndim == 1 and lats.ndim == 1:
        return np.meshgrid(lons, lats)
    return lons, lats


def get_flip_flag(lat2d, shape):
    flip_flag = np.zeros(shape)
    flip_flag[lat2d < 0] = 1
    return flip_flag


def weighted_monthly_climatology(clim_list, month_counter):
    total = sum(month_counter.values())
    result = np.zeros_like(clim_list[0], dtype=float)

    for month, count in month_counter.items():
        result += clim_list[month - 1] * (count / total)

    return result


def save_figure(fig, filename):
    fig.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


# =========================================================
# Event data
# =========================================================
time_event_sl = pd.to_datetime(dataset_sl["valid_time"].values)
datas_18h = [t for t in time_event_sl if t.hour == 18]

lats = dataset_sl["latitude"].values
lons = dataset_sl["longitude"].values
lon2d, lat2d = to_2d_coords(lons, lats)

t2m = dataset_sl["t2m"].values - 273.15
sst = dataset_sl["sst"].values - 273.15

time_event_pl = pd.to_datetime(dataset_pl["valid_time"].values)

t_850 = dataset_pl.sel(pressure_level=850)["t"].values - 273.15
z_850 = dataset_pl.sel(pressure_level=850)["z"].values / 100.0
u_850 = dataset_pl.sel(pressure_level=850)["u"].values
v_850 = dataset_pl.sel(pressure_level=850)["v"].values

t_500 = dataset_pl.sel(pressure_level=500)["t"].values - 273.15
z_500 = dataset_pl.sel(pressure_level=500)["z"].values / 100.0
u_500 = dataset_pl.sel(pressure_level=500)["u"].values
v_500 = dataset_pl.sel(pressure_level=500)["v"].values


# =========================================================
# Climatology data
# =========================================================
mslc = dataset_clima["msl"].values / 100.0
t2mc = dataset_clima["t2m"].values - 273.15
u10c = dataset_clima["u10"].values
v10c = dataset_clima["v10"].values
sstc = dataset_clima["sst"].values - 273.15

zc = dataset_clima_circ["z"].values / 100.0
tc = dataset_clima_circ["t"].values - 273.15
uc = dataset_clima_circ["u"].values
vc = dataset_clima_circ["v"].values

time_clim_sl = pd.to_datetime(dataset_clima["valid_time"].values)
time_clim_pl = pd.to_datetime(dataset_clima_circ["valid_time"].values)

df_tempo_sl = pd.DataFrame({"time": time_clim_sl})
df_tempo_sl["year"] = df_tempo_sl["time"].dt.year
df_tempo_sl["month"] = df_tempo_sl["time"].dt.month

df_tempo_pl = pd.DataFrame({"time": time_clim_pl})
df_tempo_pl["year"] = df_tempo_pl["time"].dt.year
df_tempo_pl["month"] = df_tempo_pl["time"].dt.month

ano_ini = 1991
ano_fim = 2020

clim_msl = []
clim_t2m = []
clim_u10 = []
clim_v10 = []
clim_sst = []

for month in range(1, 13):
    idx = df_tempo_sl[
        (df_tempo_sl["month"] == month)
        & (df_tempo_sl["year"] >= ano_ini)
        & (df_tempo_sl["year"] <= ano_fim)
    ].index

    clim_msl.append(np.nanmean(mslc[idx, :, :], axis=0))
    clim_t2m.append(np.nanmean(t2mc[idx, :, :], axis=0))
    clim_u10.append(np.nanmean(u10c[idx, :, :], axis=0))
    clim_v10.append(np.nanmean(v10c[idx, :, :], axis=0))
    clim_sst.append(np.nanmean(sstc[idx, :, :], axis=0))

clim_z = {}
clim_t = {}
clim_u = {}
clim_v = {}

for level in [850, 500]:
    z_level = dataset_clima_circ.sel(pressure_level=level)["z"].values / 100.0
    t_level = dataset_clima_circ.sel(pressure_level=level)["t"].values - 273.15
    u_level = dataset_clima_circ.sel(pressure_level=level)["u"].values
    v_level = dataset_clima_circ.sel(pressure_level=level)["v"].values

    monthly_z = []
    monthly_t = []
    monthly_u = []
    monthly_v = []

    for month in range(1, 13):
        idx = df_tempo_pl[
            (df_tempo_pl["month"] == month)
            & (df_tempo_pl["year"] >= ano_ini)
            & (df_tempo_pl["year"] <= ano_fim)
        ].index

        monthly_z.append(np.nanmean(z_level[idx, :, :], axis=0))
        monthly_t.append(np.nanmean(t_level[idx, :, :], axis=0))
        monthly_u.append(np.nanmean(u_level[idx, :, :], axis=0))
        monthly_v.append(np.nanmean(v_level[idx, :, :], axis=0))

    clim_z[level] = monthly_z
    clim_t[level] = monthly_t
    clim_u[level] = monthly_u
    clim_v[level] = monthly_v


MONTH_NAMES = [
    "Janeiro", "Fevereiro", "Março", "Abril", "Maio", "Junho",
    "Julho", "Agosto", "Setembro", "Outubro", "Novembro", "Dezembro"
]


# =========================================================
# Monthly climatology maps
# =========================================================
def plot_monthly_t2m_climatology():
    levels = np.arange(-40, 45, 5)

    for i in range(12):
        fig, ax = plt.subplots(figsize=(15, 10), subplot_kw={"projection": ccrs.PlateCarree()})
        setup_map(ax)

        cf = ax.contourf(
            lon2d, lat2d, clim_t2m[i],
            levels=levels, cmap="coolwarm",
            transform=ccrs.PlateCarree()
        )
        cbar = plt.colorbar(cf, ax=ax, orientation="horizontal", pad=0.05, aspect=50)
        cbar.set_label("Temperatura a 2 m (°C)")
        cbar.set_ticks(levels)

        ax.set_title(f"Climatologia {MONTH_NAMES[i]} (1991–2020): temperatura a 2 m às 18 UTC")

        filename = f"climatology_{i+1:02d}_t2m.png"
        save_figure(fig, filename)


def plot_monthly_t2m_sst_climatology():
    levels = np.arange(-40, 45, 5)

    for i in range(12):
        fig, ax = plt.subplots(figsize=(15, 10), subplot_kw={"projection": ccrs.PlateCarree()})
        setup_map(ax)

        t2m_masked = np.where(np.isnan(clim_sst[i]), clim_t2m[i], np.nan)

        ax.contourf(
            lon2d, lat2d, clim_sst[i],
            levels=levels, cmap="coolwarm",
            transform=ccrs.PlateCarree()
        )
        cf = ax.contourf(
            lon2d, lat2d, t2m_masked,
            levels=levels, cmap="coolwarm",
            transform=ccrs.PlateCarree()
        )

        cbar = plt.colorbar(cf, ax=ax, orientation="horizontal", pad=0.05, aspect=50)
        cbar.set_label("Temperatura (°C)")
        cbar.set_ticks(levels)

        ax.set_title(
            f"Climatologia {MONTH_NAMES[i]} (1991–2020): temperatura a 2 m e temperatura da superfície do mar"
        )

        filename = f"climatology_{i+1:02d}_t2m_sst.png"
        save_figure(fig, filename)


# =========================================================
# Daily anomaly maps
# =========================================================
def plot_daily_t2m_anomaly():
    levels = np.arange(-10, 11, 1)

    for timestamp in datas_18h:
        idx = int((time_event_sl == timestamp).argmax())
        month = timestamp.month

        anomaly = t2m[idx] - clim_t2m[month - 1]

        fig, ax = plt.subplots(figsize=(15, 10), subplot_kw={"projection": ccrs.PlateCarree()})
        setup_map(ax, extent=(-130, 0, -70, 10))

        cf = ax.contourf(
            lon2d, lat2d, anomaly,
            levels=levels, cmap="coolwarm",
            transform=ccrs.PlateCarree()
        )

        contour_3 = ax.contour(
            lon2d, lat2d, anomaly,
            levels=[3], colors="red", linewidths=1,
            transform=ccrs.PlateCarree()
        )
        contour_5 = ax.contour(
            lon2d, lat2d, anomaly,
            levels=[5], colors="purple", linewidths=1,
            transform=ccrs.PlateCarree()
        )

        custom_lines = [
            Line2D([0], [0], color="red", lw=2, label="Anomalia > 3°C"),
            Line2D([0], [0], color="purple", lw=2, label="Anomalia > 5°C"),
        ]
        ax.legend(handles=custom_lines, loc="lower left")

        cbar = plt.colorbar(cf, ax=ax, orientation="horizontal", pad=0.05, aspect=50)
        cbar.set_label("Anomalia (°C)")

        ax.set_title(f"Anomalia de temperatura a 2 m (°C) - {timestamp.strftime('%d/%m/%Y %H:%M UTC')}")

        filename = f"{timestamp.strftime('%Y.%m.%d')}_anomaly_t2m.png"
        save_figure(fig, filename)


def plot_daily_t2m_sst_anomaly():
    levels = np.arange(-10, 11, 1)

    for timestamp in datas_18h:
        idx = int((time_event_sl == timestamp).argmax())
        month = timestamp.month

        anomaly_t2m = t2m[idx] - clim_t2m[month - 1]
        anomaly_sst = sst[idx] - clim_sst[month - 1]
        anomaly_combined = np.where(np.isnan(sst[idx]), anomaly_t2m, anomaly_sst)

        fig, ax = plt.subplots(figsize=(15, 10), subplot_kw={"projection": ccrs.PlateCarree()})
        setup_map(ax)

        cf = ax.contourf(
            lon2d, lat2d, anomaly_combined,
            levels=levels, cmap="coolwarm",
            transform=ccrs.PlateCarree()
        )

        ax.contour(
            lon2d, lat2d, anomaly_combined,
            levels=[3], colors="red", linewidths=1,
            transform=ccrs.PlateCarree()
        )
        ax.contour(
            lon2d, lat2d, anomaly_combined,
            levels=[5], colors="purple", linewidths=1,
            transform=ccrs.PlateCarree()
        )

        cbar = plt.colorbar(cf, ax=ax, orientation="horizontal", pad=0.05, aspect=50)
        cbar.set_label("Anomalia (°C)")

        ax.set_title(
            f"Anomalia de temperatura a 2 m e temperatura da superfície do mar - "
            f"{timestamp.strftime('%d/%m/%Y %H:%M UTC')}"
        )

        filename = f"{timestamp.strftime('%Y.%m.%d')}_anomaly_t2m_sst.png"
        save_figure(fig, filename)


# =========================================================
# Mean event anomaly maps
# =========================================================
def get_event_indices(event_dates, pressure_time):
    pressure_time = pd.to_datetime(pressure_time)
    return [i for i, t in enumerate(pressure_time) if pd.Timestamp(t) in event_dates]


def plot_mean_anomaly_map(level, event_dates, filename_suffix):
    idxs = get_event_indices(event_dates, time_event_pl)
    month_counter = Counter([d.month for d in event_dates])

    if level == 850:
        t_event = np.nanmean(t_850[idxs], axis=0)
        z_event = np.nanmean(z_850[idxs], axis=0)
        u_event = np.nanmean(u_850[idxs], axis=0)
        v_event = np.nanmean(v_850[idxs], axis=0)
        temp_levels = np.arange(-8, 9, 1)
        z_levels = np.arange(-20, 21, 3)
        temp_label = "Anomalia de Temperatura a 850 hPa (°C)"
        title_level = "850 hPa"
    elif level == 500:
        t_event = np.nanmean(t_500[idxs], axis=0)
        z_event = np.nanmean(z_500[idxs], axis=0)
        u_event = np.nanmean(u_500[idxs], axis=0)
        v_event = np.nanmean(v_500[idxs], axis=0)
        temp_levels = np.arange(-8, 9, 1)
        z_levels = np.arange(-18, 22, 3)
        temp_label = "Anomalia de Temperatura a 500 hPa (°C)"
        title_level = "500 hPa"
    else:
        raise ValueError("Level must be 850 or 500.")

    t_clim = weighted_monthly_climatology(clim_t[level], month_counter)
    z_clim = weighted_monthly_climatology(clim_z[level], month_counter)
    u_clim = weighted_monthly_climatology(clim_u[level], month_counter)
    v_clim = weighted_monthly_climatology(clim_v[level], month_counter)

    anom_t = np.squeeze(np.asarray(t_event) - np.asarray(t_clim))
    anom_z = np.squeeze(np.asarray(z_event) - np.asarray(z_clim))
    anom_u = np.squeeze(np.asarray(u_event) - np.asarray(u_clim))
    anom_v = np.squeeze(np.asarray(v_event) - np.asarray(v_clim))

    fig, ax = plt.subplots(figsize=(15, 10), subplot_kw={"projection": ccrs.PlateCarree()})
    setup_map(ax)

    ax.add_feature(cfeature.LAND, facecolor="lightgray", alpha=0.3)

    cf = ax.contourf(
        lon2d, lat2d, anom_t,
        levels=temp_levels, cmap="RdBu_r",
        transform=ccrs.PlateCarree()
    )
    cbar = plt.colorbar(cf, ax=ax, orientation="horizontal", pad=0.05, aspect=50)
    cbar.set_label(temp_label)

    cs = ax.contour(
        lon2d, lat2d, anom_z,
        levels=z_levels, colors="black", linewidths=1,
        transform=ccrs.PlateCarree()
    )
    ax.clabel(cs, fmt="%d", fontsize=8)

    skip = 30
    flip_flag = get_flip_flag(lat2d, anom_u.shape)
    ax.barbs(
        lon2d[::skip, ::skip], lat2d[::skip, ::skip],
        anom_u[::skip, ::skip], anom_v[::skip, ::skip],
        length=6,
        linewidth=0.7,
        transform=ccrs.PlateCarree(),
        sizes=dict(emptybarb=0.0, spacing=0.2, height=0.5),
        pivot="middle",
        barbcolor="black",
        flip_barb=flip_flag[::skip, ::skip]
    )

    ax.set_title(
        f"{title_level}: anomalias médias de altura geopotencial (dam), temperatura (°C) e vento\n"
        f"Período: {event_dates[0].strftime('%d/%m/%Y')} a {event_dates[-1].strftime('%d/%m/%Y')}"
    )

    filename = f"{event_dates[0].strftime('%Y.%m')}_{filename_suffix}.png"
    save_figure(fig, filename)


# =========================================================
# Main
# =========================================================
def main():
    # climatologia mensal
    plot_monthly_t2m_climatology()
    plot_monthly_t2m_sst_climatology()

    # anomalias diárias
    plot_daily_t2m_anomaly()
    plot_daily_t2m_sst_anomaly()

    # período médio do evento
    # ajuste esse recorte conforme o evento analisado
    event_dates = datas_18h[5:-1]

    plot_mean_anomaly_map(850, event_dates, "mean_anomaly_850hpa")
    plot_mean_anomaly_map(500, event_dates, "mean_anomaly_500hpa")


if __name__ == "__main__":
    main()

