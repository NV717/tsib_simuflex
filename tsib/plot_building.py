# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

PROFILES = {
    "Heating Load":    {"label": "Heating Load [kW]",        "color": "#CC071E"},
    "Hot Water Load":  {"label": "Hot Water Load [kW]",      "color": "#F6A800"},
    "Electricity Load":{"label": "Electricity Load [kW]",    "color": "#57AB27"},
    "Cooling Load":    {"label": "Cooling Load [kW]",        "color": "#00549F"},
    "Photovoltaic 1":  {"label": "PV 1 [kW/kWp]",           "color": "#F6A800"},
    "Photovoltaic 2":  {"label": "PV 2 [kW/kWp]",           "color": "#E8820C"},
    "Solar thermal 1": {"label": "Solar Thermal 1 [kW/m²]", "color": "#CC071E"},
    "Solar thermal 2": {"label": "Solar Thermal 2 [kW/m²]", "color": "#A11035"},
    "Heat pump":       {"label": "Heat Pump COP [-]",        "color": "#612158"},
    "T":               {"label": "Temperature [°C]",         "color": "#00549F"},
}


def _get_series(timeseries, profile_name):
    if profile_name not in timeseries.columns:
        raise ValueError(
            f"Profile '{profile_name}' not found. "
            f"Available: {list(timeseries.columns)}"
        )
    return timeseries[profile_name].values


def plot_timeseries(timeseries, profiles=("Heating Load", "Hot Water Load",
                                           "Electricity Load", "Cooling Load"),
                    figsize=(14, 3)):
    """
    Plot full-year time series for selected profiles.

    Parameters
    ----------
    timeseries : pd.DataFrame
        bdg.timeseries after generateDemands().
    profiles : list of str
        Profiles to plot. See PROFILES for available keys.
    figsize : tuple
        Figure size per subplot (width, height).
    """
    times = timeseries.index

    n = len(profiles)
    fig, axes = plt.subplots(n, 1, figsize=(figsize[0], figsize[1] * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, name in zip(axes, profiles):
        y = _get_series(timeseries, name)
        meta = PROFILES.get(name, {"label": name, "color": "#404040"})
        ax.plot(times, y, color=meta["color"], linewidth=0.6)
        ax.set_ylabel(meta["label"], fontsize=10)
        ax.set_xlim(times[0], times[-1])
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Annual Time Series", fontsize=12, y=1.01)
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()


def plot_day(timeseries, day_of_year,
             profiles=("Heating Load", "Hot Water Load", "Electricity Load", "Cooling Load"),
             figsize=(12, 3)):
    """
    Plot profiles for a single day.

    Parameters
    ----------
    timeseries : pd.DataFrame
        bdg.timeseries after generateDemands().
    day_of_year : int
        Day to plot (1 = Jan 1, 365 = Dec 31).
    profiles : list of str
        Profiles to plot.
    figsize : tuple
        Figure size per subplot (width, height).
    """
    dt_seconds = int((timeseries.index[1] - timeseries.index[0]).total_seconds())
    steps_per_day = int(86400 / dt_seconds)
    start = (day_of_year - 1) * steps_per_day
    end = start + steps_per_day

    hours = np.arange(steps_per_day) * dt_seconds / 3600
    date_str = timeseries.index[start].strftime("%A, %d %B %Y")

    n = len(profiles)
    fig, axes = plt.subplots(n, 1, figsize=(figsize[0], figsize[1] * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, name in zip(axes, profiles):
        y = _get_series(timeseries, name)
        meta = PROFILES.get(name, {"label": name, "color": "#404040"})
        ax.plot(hours, y[start:end], color=meta["color"], linewidth=1.5)
        ax.set_ylabel(meta["label"], fontsize=10)
        ax.set_xlim(0, 24)
        ax.set_xticks(range(0, 25, 4))
        ax.grid(alpha=0.3)

    fig.suptitle(f"Daily Profile  {date_str}", fontsize=12, y=1.01)
    axes[-1].set_xlabel("Hour of Day", fontsize=10)
    plt.tight_layout()
    plt.show()


def plot_heatmap(timeseries,
                 profiles=("Heating Load", "Hot Water Load", "Electricity Load", "Cooling Load"),
                 figsize=(12, 3)):
    """
    Plot annual heatmaps (hours × days) for selected profiles.

    Parameters
    ----------
    timeseries : pd.DataFrame
        bdg.timeseries after generateDemands().
    profiles : list of str
        Profiles to plot.
    figsize : tuple
        Figure size per subplot (width, height).
    """
    dt_seconds = int((timeseries.index[1] - timeseries.index[0]).total_seconds())
    steps_per_day = int(86400 / dt_seconds)
    steps_per_hour = int(3600 / dt_seconds)

    month_starts = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    n = len(profiles)
    fig, axes = plt.subplots(n, 1, figsize=(figsize[0], figsize[1] * n))
    if n == 1:
        axes = [axes]

    for ax, name in zip(axes, profiles):
        y = _get_series(timeseries, name)
        meta = PROFILES.get(name, {"label": name, "color": "#404040"})

        n_days = int(len(y) / steps_per_day)
        grid = y[:n_days * steps_per_day].reshape(n_days, steps_per_day)

        if steps_per_hour > 1:
            grid = grid.reshape(n_days, 24, steps_per_hour).mean(axis=2)

        # rows = hours (0-23), cols = days (0-364)
        grid = grid.T

        im = ax.imshow(grid, aspect="auto", origin="lower",
                       cmap="YlOrRd", interpolation="nearest")
        plt.colorbar(im, ax=ax, label=meta["label"], fraction=0.02, pad=0.01)
        ax.set_ylabel("Hour of Day", fontsize=9)
        ax.set_yticks(range(0, 24, 6))
        ax.set_xticks(month_starts)
        ax.set_xticklabels(month_labels, fontsize=8)
        ax.set_title(meta["label"], fontsize=10)

    fig.suptitle("Annual Heatmap", fontsize=12)
    plt.tight_layout()
    plt.show()


def plot_building(timeseries, mode="timeseries", day_of_year=1,
                  profiles=("Heating Load", "Hot Water Load", "Electricity Load", "Cooling Load")):
    """
    Unified plotting function for tsib building simulation results.

    Parameters
    ----------
    timeseries : pd.DataFrame
        bdg.timeseries after getHeatLoad() / getOccupancy() / getRenewables().
    mode : str
        Plot mode:
        - 'timeseries' : full-year time series
        - 'day'        : single day profile
        - 'heatmap'    : annual heatmap (hours x days)
    day_of_year : int
        Day to plot (only for mode='day'). 1 = Jan 1.
    profiles : list of str
        Profiles to include. Any subset of:
        ['Heating Load', 'Hot Water Load', 'Electricity Load', 'Cooling Load',
         'Photovoltaic 1', 'Photovoltaic 2', 'Solar thermal 1', 'Solar thermal 2',
         'Heat pump', 'T']

    Examples
    --------
    >>> plot_building(bdg.timeseries)
    >>> plot_building(bdg.timeseries, mode="day", day_of_year=180)
    >>> plot_building(bdg.timeseries, mode="heatmap", profiles=["Heating Load", "Electricity Load"])
    """
    if mode == "timeseries":
        plot_timeseries(timeseries, profiles=profiles)
    elif mode == "day":
        plot_day(timeseries, day_of_year=day_of_year, profiles=profiles)
    elif mode == "heatmap":
        plot_heatmap(timeseries, profiles=profiles)
    else:
        raise ValueError(f"Unknown mode '{mode}'. Use 'timeseries', 'day' or 'heatmap'.")
