# -*- coding: utf-8 -*-
"""
Created on Thu May 12 10:22:47 2016

Read in weather data

@author: l.kotzur
"""
import os
import pvlib
import math

import numpy as np
import pandas as pd

import tsib.data
import tsib

# ignore pv lib warnings
np.seterr(divide="ignore")
np.seterr(invalid="ignore")



def calcGHI(timeSeries, longitude, latitude):
    """
        Calculates the global horizontal irradiation (GHI) by time, diffuse
        horizontal irradiation (DHI) and direct normal irradiation (DNI).
        """
    solarPos = pvlib.solarposition.get_solarposition(
        timeSeries.index, latitude, longitude
    )
    if "DNI" in timeSeries:
        timeSeries["GHI"] = timeSeries["DHI"] + timeSeries["DNI"] * math.cos(
            math.radians(solarPos["apparent_zenith"])
        )
    elif "B" in timeSeries:  # Direct horizontal irradiation TRY
        timeSeries["GHI"] = timeSeries["DHI"] + timeSeries["B"]
    return timeSeries


def readTMY(filepath=os.path.join("TMY", "Germany DEU Koln (INTL).csv")):
    """
    Reads a typical meteorological year file and gets the GHI, DHI and DNI from it.
    """
    # get data
    data = pd.read_csv(
        os.path.join(tsib.data.PATH, "weatherdata", filepath),
        skiprows=([0, 1]),
        sep=",",
    )
    data.index = pd.date_range(
        "2010-01-01 00:30:00", periods=8760, freq="h", tz="Europe/Berlin"
    )
    data = data.rename(
        columns={"Beam": "DNI", "Diffuse": "DHI", "Tdry": "T", "Wspd": "WS"}
    )
    location_data = pd.read_csv(
        os.path.join(tsib.data.PATH, "profiles", filepath), nrows=1, sep=","
    )

    location = {
        "name": location_data["City"].values[0],
        "latitude": location_data["Latitude"].values[0],
        "longitude": location_data["Longitude"].values[0],
    }
    return data, location

def _find_data_start(filepath):
    with open(filepath, encoding="utf-8") as fp:
        for i, line in enumerate(fp):
            if line.strip().startswith("***"):
                return i
    raise ValueError(f"No '***' separator found in {filepath}")

def _parse_latlon_from_file(filename):
    code = filename.split("_")[1]
    lat = int(code[:6]) / 10000
    lon = int(code[6:]) / 10000
    return lat, lon

def _pick_station_file(region, year_type, future, seed):
    variant = {"average": "Jahr", "hot": "Somm", "cold": "Wint"}[year_type]
    folder = os.path.join(
        tsib.data.PATH, "weatherdata",
        "TRY2045" if future else "TRY2015", f"{int(region):02d}",
    )
    candidates = sorted(f for f in os.listdir(folder) if f.endswith(f"_{variant}.dat"))
    if not candidates:
        raise FileNotFoundError(f"No TRY files for region {region}, variant {variant} in {folder}")
    chosen = np.random.RandomState(seed).choice(candidates)
    return os.path.join(folder, chosen[:-4])

def readTRYnew(filepath, latitude, longitude):
    data_start = _find_data_start(filepath + ".dat")
    header_row = data_start - 1
    data = pd.read_csv(
        filepath + ".dat", sep=r"\s+",
        skiprows=[i for i in range(header_row)] + [data_start],
    )
    data.index = pd.date_range(
        "2010-01-01 00:30:00", periods=8760, freq="h", tz="Europe/Berlin"
    )
    data["GHI"] = data["D"] + data["B"]
    data = data.rename(columns={"D": "DHI", "t": "T", "WG": "WS"})
    data["DNI"] = calculateDNI(data["B"], longitude, latitude)
    return data


def readTRY(try_num=4, year=2010):
    """
    Reads a test refence year file and gets the GHI, DHI and DNI from it.
    
    Parameters
    -------
    try_num: int (default: 4)
        The region number of the test reference year.
    year: int (default: 2010)
        The year. Only data for 2010 and 2030 available
    """
    # get the correct file path
    filepath = os.path.join(
        tsib.data.PATH,
        "weatherdata",
        "TRY",
        "TRY" + str(year) + "_" + str(try_num).zfill(2) + "_Jahr",
    )

    # get the geoposition
    with open(filepath + ".dat", encoding="utf-8") as fp:
        lines = fp.readlines()
        location_name = lines[1][9:-18].encode("utf-8").rstrip()
        lat = float(lines[2][6:8]) + float(lines[2][9:11]) / 60.0
        lon = float(lines[2][21:23]) + float(lines[2][24:26]) / 60.0
    location = {"name": location_name, "latitude": lat, "longitude": lon}

    # check if time series data already exists as .csv with DNI
    if os.path.isfile(filepath + ".csv"):
        data = pd.read_csv(filepath + ".csv", index_col=0, parse_dates=True)
        data.index = pd.to_datetime(data.index, utc=True).tz_convert("Europe/Berlin")
    # else read from .dat and calculate DNI etc.
    else:
        # get data
        data = pd.read_csv(
            filepath + ".dat", sep=r"\s+", skiprows=([i for i in range(0, 36)] + [37])
        )
        data.index = pd.date_range(
            "2010-01-01 00:30:00", periods=8760, freq="h", tz="Europe/Berlin" # 2010 hardcoded as generic placeholder year since the actual year is unimportant
        )
        data["GHI"] = data["D"] + data["B"]
        data = data.rename(columns={"D": "DHI", "t": "T", "WG": "WS"})

        # calculate direct normal
        data["DNI"] = calculateDNI(data["B"], lon, lat)

        # save as .csv
        data.to_csv(filepath + ".csv")
    return data, location

def targetdaterange(data, freq):
    default_delt = data.index[1] - data.index[0]
    target_delt = pd.Timedelta(pd.tseries.frequencies.to_offset(freq))

    if target_delt == default_delt:
        return data.index

    total = pd.Timedelta(hours=8760)

    if total % target_delt != pd.Timedelta(hours=0):
        raise ValueError(f"{freq} doesnt work")

    n_per = int(total/target_delt)
    target = pd.date_range(
        start=pd.Timestamp(
            year = data.index[0].year, month=1, day=1, tz = data.index.tz
        ),
        periods=n_per,
        freq=freq
    )

    return target

def resampletoindex(data, target_index):

    if data.index.equals(target_index):
        return data

    new_index = data.index.union(target_index)
    return data.reindex(new_index).interpolate(method="time").reindex(target_index).ffill().bfill()

def calculateDNI(directHI, lon, lat, zenith_tol=87.0):
    """
    Calculates the direct NORMAL irradiance from the direct horizontal irradiance with the help of the PV lib.

    Parameters
    ----------
    directHI: pd.Series with time index
        Direct horizontal irradiance
    lon: float
        Longitude of the loaction
    lat: float
        Latitude of the location
    zenith_tol: float, optional
        Avoid cosinus of values above a certain zenith angle of in order to avoid division by zero.

    Returns
    -------
    DNI: pd.Series
    """
    solarPos = pvlib.solarposition.get_solarposition(directHI.index, lat, lon)
    solarPos.loc[solarPos.apparent_zenith > zenith_tol, "apparent_zenith"] = zenith_tol
    DNI = directHI.div(solarPos["apparent_zenith"].apply(math.radians).apply(math.cos))
    if DNI.isnull().values.any():
        raise ValueError("Something went wrong...")
    return DNI


def TRY2TM2(trydata):
    """
    Takes a pd.DataFrame from the readTRY function and translate the column
    names to the TM2 format.
    """
    trydata["Year"] = 2010
    trydata = trydata.rename(
        columns={
            "MM": "Month",
            "DD": "Day",
            "HH": "Hour",
            "T": "Tdry",
            "p": "Pres",
            "WR": "Wdir",
            "WS": "Wspd",
        }
    )
    trydata["Tdew"] = trydata["Tdry"]

    return trydata[
        [
            "Year",
            "Month",
            "Day",
            "Hour",
            "GHI",
            "DHI",
            "Tdry",
            "Tdew",
            "Pres",
            "Wdir",
            "Wspd",
        ]
    ]


def TRY2TMY(trydata):
    """
    Takes a pd.DataFrame from the readTRY function and translate the column
    names to the TM2 format.
    """
    return trydata.rename(
        columns={
            "MM": "Month",
            "DD": "Day",
            "HH": "Hour",
            "T": "DryBulb",
            "p": "Pressure",
            "WR": "Wdir",
            "WS": "Wspd",
        }
    )

def getISO12831weather(longitude=None, latitude=None, year_type="average",
                        future=False, climate_region=None, seed=None):
    wzones = pd.read_csv(
        os.path.join(tsib.data.PATH, "weatherdata", "ISO12831", "T_zones_Ger_final.csv"),
        index_col=0, encoding="ISO-8859-1",
    )

    if climate_region is None:
        dist = ((wzones["Lat"] - latitude) ** 2 + (wzones["Lng"] - longitude) ** 2) ** 0.5
        if min(dist) > 5:
            raise NotImplementedError("The weather data is at the moment only implemented for Germany")
        loc_w = wzones.loc[dist.idxmin(), :]
        climate_region = loc_w["Climate Zone"]
        design_T_min = loc_w["Min T"]
    else:
        design_T_min = wzones.loc[wzones["Climate Zone"] == climate_region, "Min T"].iloc[0]

    if seed is None:
        seed = int(abs(latitude * 1000) + abs(longitude * 1000)) if latitude is not None else int(climate_region)

    filepath = _pick_station_file(climate_region, year_type, future, seed)
    station_lat, station_lon = _parse_latlon_from_file(os.path.basename(filepath))
    weather = readTRYnew(filepath, station_lat, station_lon)
    if latitude is None:
        latitude, longitude = station_lat, station_lon
    weatherID = f"TRY{2045 if future else 2015}_{climate_region}_{year_type}_{os.path.basename(filepath)}"

    return weather, design_T_min, weatherID, longitude, latitude, filepath, climate_region

# def getISO12831weather(longitude, latitude, year=2010):
#     """
#
#     Gets the test reference year location and the design temperatures for
#     the heating system based on the ISO12831.
#     Parameters
#     ----------
#     longitude: float
#     latitude: float
#     year: int, optional (default: 2010)
#     Returns
#     -------
#     weather (DataFrame with TRY weather)
#     T_min (design temperature for heating),
#     weatherID (str with climate zone)
#     """
#
#     # read weather zones
#     wzones = pd.read_csv(
#         os.path.join(tsib.data.PATH, "weatherdata", "ISO12831", "T_zones_Ger_final.csv"),
#         index_col=0,
#         encoding="ISO-8859-1",
#     )
#
#     # get distance to all reference weather station points
#     dist = ((wzones["Lat"] - latitude) ** 2 + (wzones["Lng"] - longitude) ** 2) ** 0.5
#
#     # if distance to next reference position is to high.
#     if min(dist) > 5:
#         raise NotImplementedError(
#             "The weather data is at the moment" + " only implemented for Germany"
#         )
#
#     # get the data from the one with the minimal distance
#     loc_w = wzones.loc[dist.idxmin(), :]
#     design_T_min = loc_w["Min T"]
#
#     # read weather data of related try region
#     weatherID = "TRY_" + str(loc_w["Climate Zone"])
#     weather, loc = readTRY(try_num=loc_w["Climate Zone"], year=year)
#
#     return weather, design_T_min, weatherID


