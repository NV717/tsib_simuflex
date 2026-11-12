import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
from tqdm import tqdm
import numpy as np
import itertools
import json
import tables
import tsib
import time
import datetime
import warnings
import calendar

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "tsib" / "data"

EPISCOPE_PATH = DATA_DIR / "episcope" / "episcope.csv"

RESULTS_DIR = PROJECT_ROOT / "Generation" / "results"

MANIFEST_PATH = PROJECT_ROOT / "Generation"/ "manifest.parquet"

SPLITS = {"train": 0, "val": 1, "test": 2, "building_holdout": 3,"region_holdout": 4, "station_holdout": 5, "year_holdout": 6}

HOLDOUT_REGIONS = {11}

HOLDOUT_ARCHETYPES = {("TH", "DE.08", "Semi")}

SPLIT_STRIDE = 100000000
SEED_BASE = 10000000
APP_SEED_BASE = 20000000
WEATHER_SEED_BASE = 30000000


FIXED_SEED_FRACTION = {
    "train": 0.2,
    "val": 1.0,
    "test": 1.0,
    "building_holdout": 1.0,
    "region_holdout": 1.0,
    "station_holdout": 1.0,
    "year_holdout":1.0,
}

NONFIXED_OFFSET = 5000000

FAIL_LOG = PROJECT_ROOT / "Generation" / "failures.log"
STARTED_LOG = PROJECT_ROOT / "Generation" / "started.log"

SFH_MAX_AREA = 280.0 #cap for SFH area since there is a weird tabula archetype in the episcope csv with 2 flats

EPISCOPE_META = {"a0": "A0", "n0": "n0", "sh_spec": "SH_spec"}

def split_paths(split):
    return PROJECT_ROOT / "Generation" / f"manifest_{split}.parquet",RESULTS_DIR / split

def episcope_combis(path: str, country: str = "DE", building_types = None)-> dict:
    if building_types is None:
        building_types = ["SFH", "MFH", "TH", "AB"]

    df = pd.read_csv(path, delimiter=",")
    df = df.copy()
    df = df[(df["Code_Country"] == country) & (df["Code_BuildingSizeClass"].isin(building_types)) & (df["Code_DataType_Building"] == "ReEx") & (df["Code_BuildingVariant"].str.contains(".N.", regex=False))]
    cons = {
        "B_N2": "Terraced",
        "B_N1": "Semi",
        "B_Alone": "Detached"
    }
    df["Connections"] = df["Code_AttachedNeighbours"].map(cons)

    test = df[["Code_BuildingSizeClass", "Code_ConstructionYearClass", "Connections"]].drop_duplicates()
    #print(test.index)
    tzu = df.loc[test.index]
    tzu = tzu[["Code_BuildingSizeClass","Connections", "Code_ConstructionYearClass", "A_C_Ref","n_Apartment","q_h_nd"]]
    tzu.columns = ["Type", "Surrounding","AgeBin", "A0", "n0", "SH_spec"]
    res_dict = tzu.to_dict("records")
    return res_dict

def gen_area_ratio(rng, sigma=0.2, lo=0.8, hi=1.3):
    while True:
        r = rng.lognormal(0.0, sigma)
        if lo <= r <= hi:
            return r


#https://episcope.eu/building-typology/country/de/  Statistics of the German Building Stock Source[2]
def gen_flat_count(buildingtype: str, rng: np.random.Generator) -> int:
    if buildingtype == "SFH":
        return 1
    elif buildingtype == "TH":
        return 1
    elif buildingtype == "MFH":
        return int(rng.integers(low=2, high=13)) #exclusive
    elif buildingtype == "AB":
        return int(rng.integers(low=13, high=21))
    else:
        raise ValueError

#https://www.destatis.de/DE/Themen/Gesellschaft-Umwelt/Wohnen/Tabellen/tabelle-wo2-mietwohnungen.html
#https://www.destatis.de/DE/Themen/Gesellschaft-Umwelt/Wohnen/Tabellen/tabelle-wo2-eigentuemerwohnungen.html
#probs sind für haushalte
def gen_occs(buildingtype: str, apartment_count: int, rng: np.random.Generator) ->list:
    if buildingtype in ("SFH", "TH"):
        return [int(rng.integers(low=1, high=6)) for _ in range(apartment_count)]

    occ_probs = {
        #"SFH": (0.245114, 0.402323, 0.154148, 0.148869, 0.04954599999999998), #equal draw insted
        "MFH": (0.490622, 0.307419, 0.101949, 0.074417, 0.025592999999999977),
        "AB": (0.490622, 0.307419, 0.101949, 0.074417, 0.025592999999999977),
        #"TH": (0.236817, 0.400092, 0.157261, 0.154371, 0.05145899999999992),
    }
    probs = occ_probs[buildingtype]

    num_occs =[]

    for _ in range(apartment_count):
        draw = rng.random()
        x = 0.0
        for i,v in enumerate(probs, start=1):
            x += v
            if x > draw:
                num_occs.append(i)
                break
        else:
            num_occs.append(len(probs))



    return num_occs

#only for AB and MFH, SFH and TH are based on the episcope ranges
#https://www.destatis.de/DE/Themen/Gesellschaft-Umwelt/Wohnen/Tabellen/tabelle-wo4-wohnflaeche.html
def gen_flat_area(occs: list ,rng: np.random.Generator)-> list:
    area_occ_probs = {
        1: {
            "area": [(20, 39), (40, 59), (60, 79), (80, 99), (100, 119), (120, 139), (140, 200)],
            "probs": [0.09605,0.30524,0.27605,0.13575,0.07434,0.05302, 0.05955]
        },
        2: {
            "area": [(20, 39), (40, 59), (60, 79), (80, 99), (100, 119), (120, 139), (140, 200)],
            "probs": [0.00677,0.08684,0.23555,0.19991,0.15465,0.13864, 0.17764]
        },
        3: {
            "area": [(20, 39), (40, 59), (60, 79), (80, 99), (100, 119), (120, 139), (140, 200)],
            "probs": [0.0,0.03780,0.19027,0.20132,0.15757,0.16097, 0.25207]
        },
        4: {
            "area": [(20, 39), (40, 59), (60, 79), (80, 99), (100, 119), (120, 139), (140, 200)],
            "probs": [0.0,0.01389,0.11820,0.17035,0.14979,0.17815, 0.36962]
        },
    }
    areas = []

    for occ in occs:
        if occ > 4:
            occ = 4

        res = rng.choice(area_occ_probs[occ]["area"], p=area_occ_probs[occ]["probs"])
        floor = rng.uniform(low=res[0],high= res[1])
        areas.append(floor)

    return areas


def non_leap_years(n, start=2010):
    years, y = [], start
    while len(years) < n:
        if not calendar.isleap(y):
            years.append(y)
        y += 1
    return years

def manifest(scenario_reps,seed_reps,episcope_path ,
             building_types = None,seed=42, out_path=MANIFEST_PATH,
             overwrite=False, years=None, split="train",fixed_seed_reps=None, years_per_building=None):
    off = SPLITS[split] * SPLIT_STRIDE

    if fixed_seed_reps is None:
        fixed_seed_reps = seed_reps

    if years is None:
        years = non_leap_years(1)

    years = [int(y) for y in years]
    if len(set(years)) != len(years):
        raise ValueError(f"years must be distinct, got {years}")
    if any(calendar.isleap(y) for y in years):
        raise ValueError("years must be non-leap (weather data has 8760 rows)")

    if years_per_building is None:
        years_per_building = len(years)  # old behaviour
    if years_per_building != len(years) and years_per_building != 1:
        raise ValueError("years_per_building must be 1 or len(years)")

    if building_types is None:
        building_types = ["SFH", "MFH", "TH", "AB"]
    episcope_path = Path(episcope_path)
    out_path = Path(out_path)

    building_combis = episcope_combis(episcope_path, building_types=building_types)

    area_range = {}
    for t in ("SFH","TH"):
        a0s = [c["A0"] for c in building_combis if c["Type"] == t]
        if not a0s:
            continue
        lo, hi = 0.9 * min(a0s), 1.2 * max(a0s)
        if t == "SFH":
            hi = min(hi, SFH_MAX_AREA)
        assert hi > lo, f"{t}: empty area range [{lo:.1f}, {hi:.1f}]"
        area_range[t] = (lo, hi)

    is_holdout = lambda c: (c["Type"], c["AgeBin"], c["Surrounding"]) in HOLDOUT_ARCHETYPES
    if split == "holdout":
        building_combis = [c for c in building_combis if is_holdout(c)]
    else:
        building_combis = [c for c in building_combis if not is_holdout(c)]
    if not building_combis:
        raise ValueError(f"No archetypes left for split '{split}'")


    year_types = ["average", "hot", "cold"]
    futures = [False, True]
    weather_combis = [(f, y) for f in futures for y in year_types]
    rng = np.random.default_rng([seed, SPLITS[split]])
    rows = []
    building_id = 0

    def building_type_reps(building_type):
        return scenario_reps[building_type] if isinstance(scenario_reps, dict) else scenario_reps

    total_buildings = sum(building_type_reps(c["Type"]) for c in building_combis)
    n_fixed = int(round(FIXED_SEED_FRACTION[split] * total_buildings))
    fixed_ids = set(np.random.default_rng([seed, SPLITS[split], 1]).choice(total_buildings, size=n_fixed, replace=False).tolist())



    for x in building_combis:
        building_type = x["Type"]
        surrounding = x["Surrounding"]
        buildingAgeBin = x["AgeBin"]

        n_scenarios = building_type_reps(building_type)

        base = weather_combis * (n_scenarios // len(weather_combis))
        remainder = n_scenarios % len(weather_combis)
        extra_idx = rng.choice(len(weather_combis),size = remainder,replace = False,)
        weather_scenario = base + [weather_combis[i] for i in extra_idx]
        rng.shuffle(weather_scenario)

        all_regions = np.arange(1, 16)
        if split == "region_holdout":
            climate_regions = np.array(sorted(HOLDOUT_REGIONS))
        else:
            climate_regions = np.array([r for r in all_regions if r not in HOLDOUT_REGIONS])
        base = np.tile(climate_regions,n_scenarios // len(climate_regions))
        remainder = n_scenarios % len(climate_regions)
        extra = rng.choice(climate_regions,size=remainder,replace=False)
        region_scenario = np.concatenate([base, extra])
        rng.shuffle(region_scenario)

        if years_per_building == 1:
            year_scenario = list(years) * (n_scenarios // len(years))
            year_scenario += [years[i] for i in rng.choice(len(years), size=n_scenarios % len(years), replace=False)]
            rng.shuffle(year_scenario)


        for i in range(n_scenarios):
            n_flats = gen_flat_count(buildingtype=building_type, rng=rng)
            occs = gen_occs(buildingtype=building_type, apartment_count=n_flats, rng=rng)
            if building_type in ("SFH", "TH"):
                total_area = rng.uniform(*area_range[building_type])

            else:
                flat_areas = gen_flat_area(occs=occs,rng=rng)
                total_area = sum(flat_areas)
            clima_region = region_scenario[i]
            future, year_type = weather_scenario[i]

            bid = off + building_id
            is_fixed = building_id in fixed_ids
            n_reps = fixed_seed_reps if is_fixed else seed_reps

            building_years = [year_scenario[i]] if years_per_building == 1 else years

            for year in building_years:
                for rep in range(n_reps):
                    unit = building_id if is_fixed else NONFIXED_OFFSET + len(rows)

                    rows.append(dict(
                        country = "DE",
                        buildingType = building_type,
                        surrounding = surrounding,
                        buildingAgeBin = buildingAgeBin,
                        a_ref = round(total_area,3),
                        **{col: x[key] for col, key in EPISCOPE_META.items()},
                        n_apartments = n_flats,
                        year_type = year_type,
                        future = future,
                        climateRegion = clima_region,
                        n_persons = occs,
                        freq = "1min",
                        hasFirePlace = False,
                        varyoccupancy = 1,
                        mean_load = False,
                        nightReduction = True,
                        capControl = True,
                        occControl = False,
                        comfortT_lb = 20.0,
                        comfortT_ub = 24.0,
                        cores = 1,
                        useCache=False,
                        year=year,
                        rep=rep,
                        weather_seed=off + WEATHER_SEED_BASE + unit,
                        station_pool="holdout" if split == "station_holdout" else "train",
                        app_ownership_seed=off + APP_SEED_BASE + unit,
                        fixed_seeds=is_fixed,
                        building_id = bid,
                        split=split,

                    )
                     )
            building_id += 1

    manifest = pd.DataFrame(rows)
    manifest["run_id"] = manifest.index.map(lambda i: f"{split}_run_{i:06d}")
    manifest["seed"] = off + SEED_BASE + manifest.index
    if out_path.exists() and not overwrite:
        raise FileExistsError(out_path)
    manifest.to_parquet(out_path)
    return manifest

#Validation

def validate_manifest(manifest: pd.DataFrame, seed_reps: int, split="train", years=None,fixed_seed_reps=None,years_per_building=None) -> None:
    off = SPLITS[split] * SPLIT_STRIDE

    if fixed_seed_reps is None:
        fixed_seed_reps = seed_reps


    required_columns = {
        "country",
        "buildingType",
        "surrounding",
        "buildingAgeBin",
        "a_ref",
        "n_apartments",
        "year_type",
        "future",
        "climateRegion",
        "n_persons",
        "freq",
        "hasFirePlace",
        "cores",
        "useCache",
        "building_id",
        "run_id",
        "seed",
        "varyoccupancy",
        "mean_load",
        "nightReduction",
        "capControl",
        "occControl",
        "comfortT_lb",
        "comfortT_ub",
        "year",
        *EPISCOPE_META,
        "weather_seed",
        "app_ownership_seed",
        "split",
        "station_pool",
        "rep",
        "fixed_seeds",
    }

    missing_columns = required_columns - set(manifest.columns)

    assert not missing_columns, (f"Manifest is missing required columns: {sorted(missing_columns)}")

    assert len(manifest) > 0, "Manifest is empty"

    assert not manifest.isna().any().any(), "Manifest contains missing values"

    assert manifest["run_id"].is_unique, "run_id is not unique"

    assert manifest["seed"].is_unique, "seed is not unique"

    b_fixed = manifest.groupby("building_id")["fixed_seeds"].first()
    n_buildings = len(b_fixed)
    n_fixed_obs = int(b_fixed.sum())
    assert n_fixed_obs == int(round(FIXED_SEED_FRACTION[split] * n_buildings)), "wrong number of fixed-seed buildings"

    nonfixed_rows = int((~manifest["fixed_seeds"]).sum())
    expected_unique = n_fixed_obs + nonfixed_rows
    assert manifest["weather_seed"].nunique() == expected_unique
    assert manifest["app_ownership_seed"].nunique() == expected_unique

    for col in ("weather_seed", "app_ownership_seed"):
        fx = manifest[manifest["fixed_seeds"]].groupby("building_id")[col].nunique()
        assert (fx == 1).all(), f"{col} must be constant for fixed-seed buildings"
        nf = manifest[~manifest["fixed_seeds"]]
        assert nf[col].is_unique, f"{col} must be unique per row for non-fixed buildings"

    all_seed_values = pd.concat([manifest["seed"], manifest["weather_seed"], manifest["app_ownership_seed"]])
    assert all_seed_values.max() < 2**32, "seed exceeds 2**32."
    assert manifest["seed"].between(off + SEED_BASE, off + APP_SEED_BASE - 1).all()
    assert manifest["app_ownership_seed"].between(off + APP_SEED_BASE, off + WEATHER_SEED_BASE - 1).all()
    assert manifest["weather_seed"].between(off + WEATHER_SEED_BASE, off + SPLIT_STRIDE - 1).all()

    if years is not None:
        assert set(manifest["year"]) == set(years)

    reps = manifest.groupby("building_id").size()
    n_years = manifest["year"].nunique()
    yb = years_per_building or n_years

    exp_reps = b_fixed.map(lambda f: yb * (fixed_seed_reps if f else seed_reps))
    assert (reps == exp_reps).all(), f"Unexpected number of rows per building:\n{reps.value_counts()}"
    expected_rows = int(exp_reps.sum())

    yrs_obs = manifest.groupby("building_id")["year"].nunique()
    assert (yrs_obs == yb).all(), "Wrong number of years per building."

    assert len(manifest) == expected_rows, (
        f"Row count mismatch: got {len(manifest)}, "
        f"expected {expected_rows}."
    )

    valid_building_types = {"SFH", "TH", "MFH", "AB"}
    valid_year_types = {"average", "hot", "cold"}

    invalid_building_types = (
        set(manifest["buildingType"].unique())
        - valid_building_types
    )

    assert not invalid_building_types, (
        f"Invalid building types: {invalid_building_types}"
    )

    invalid_year_types = (
        set(manifest["year_type"].unique())
        - valid_year_types
    )

    assert not invalid_year_types, (
        f"Invalid year types: {invalid_year_types}"
    )


    assert manifest["climateRegion"].between(1, 15).all(), (
        "climateRegion contains values outside 1...15."
    )

    assert (manifest["a_ref"] > 0).all(), (
        "a_ref contains non-positive values."
    )

    assert (manifest["n_apartments"] >= 1).all(), (
        "n_apartments contains values below 1."
    )

    assert (manifest["cores"] >= 1).all(), (
        "cores contains values below 1."
    )


    assert not manifest["year"].map(calendar.isleap).any(), ("year contains leap years.")

    per_by = manifest.groupby(["building_id", "year"]).size()
    exp_per_by = per_by.index.get_level_values("building_id").map(b_fixed).map(lambda f: fixed_seed_reps if f else seed_reps)

    assert (per_by.to_numpy() == np.asarray(exp_per_by)).all(), "Wrong reps per building/year."

    type_rules = {
        "SFH": (1, 1),
        "TH":  (1, 1),
        "MFH": (2, 12),
        "AB":  (13, 20),
    }


    sfh = manifest[manifest["buildingType"] == "SFH"]
    assert (sfh["a_ref"] <= SFH_MAX_AREA + 0.05).all(), "SFH a_ref above cap"

    b = manifest.drop_duplicates("building_id")
    rows = b[["buildingType", "buildingAgeBin", "n_persons"]].explode("n_persons")
    rows["n_persons"] = rows["n_persons"].astype(int)
    for t, g in rows.groupby("buildingType"):
        m = g.groupby("buildingAgeBin")["n_persons"].mean()
        if (m - m.mean()).abs().max() > 0.5:
            warnings.warn(f"{t}: mean persons per flat varies by age bin:\n{m.round(2)}")


    for building_type, (low, high) in type_rules.items():

        subset = manifest[
            manifest["buildingType"] == building_type
        ]

        if len(subset) == 0:
            continue

        valid = subset["n_apartments"].between(low, high)

        assert valid.all(), (
            f"{building_type} contains invalid apartment counts. "
            f"Expected {low}...{high}.\n"
            f"Invalid values:\n"
            f"{subset.loc[~valid, ['building_id', 'n_apartments']]}"
        )

    def check_n_persons(row):
        persons = row["n_persons"]

        if not isinstance(persons, (list, tuple, np.ndarray)):
            return False

        return len(persons) == row["n_apartments"]

    occupancy_lengths_valid = manifest.apply(
        check_n_persons,
        axis=1,
    )

    assert occupancy_lengths_valid.all(), (
        "At least one row has a different number of occupancy "
        "entries than n_apartments."
    )

    def occupancy_values_valid(persons):
        return all(
            isinstance(x, (int, np.integer))
            and 1 <= x <= 5
            for x in persons
        )

    occ_valid = manifest["n_persons"].apply(
        occupancy_values_valid
    )

    assert manifest["future"].isin([True, False]).all(), (
        "future contains values other than True/False."
    )

    assert occ_valid.all(), (
        "n_persons contains invalid occupancy values. "
        "Expected 1...5 persons per apartment."
    )

    assert manifest["comfortT_lb"].lt(manifest["comfortT_ub"]).all()

    assert manifest["freq"].eq("1min").all()

    assert manifest["varyoccupancy"].eq(1).all()

    assert manifest["nightReduction"].eq(True).all()

    assert manifest["capControl"].eq(True).all()

    assert manifest["occControl"].eq(False).all()

    building_parameter_columns = [
        "country",
        "buildingType",
        "surrounding",
        "buildingAgeBin",
        "a_ref",
        "n_apartments",
        "year_type",
        "future",
        "climateRegion",
        "n_persons",
        "freq",
        "hasFirePlace",
        "cores",
        "useCache",
        "varyoccupancy",
        "mean_load",
        "nightReduction",
        "capControl",
        "occControl",
        "comfortT_lb",
        "comfortT_ub",
        *EPISCOPE_META,
        "station_pool",
        "fixed_seeds",
    ]


    for column in building_parameter_columns:

        unique_per_building = (
            manifest
            .groupby("building_id")[column]
            .apply(lambda x: x.astype(str).nunique())
        )

        assert (unique_per_building == 1).all(), (
            f"Column '{column}' changes between seed repetitions "
            "of the same building_id."
        )


    expected_run_ids = [
        f"{split}_run_{i:06d}"
        for i in range(len(manifest))
    ]

    assert manifest["run_id"].tolist() == expected_run_ids, (
        "run_id sequence is not as expected."
    )

    expected_seeds = (
        off + SEED_BASE + np.arange(len(manifest))
    )

    assert np.array_equal(
        manifest["seed"].to_numpy(),
        expected_seeds,
    ), (
        "Seed sequence is not as expected."
    )

    n_runs = len(manifest)
    n_buildings = manifest["building_id"].nunique()

    print("Manifest validation successful.")
    print(f"Runs:              {n_runs:,}")
    print(f"Unique buildings:  {n_buildings:,}")
    print(f"Seed repetitions:  {seed_reps}")
    print()
    print("Buildings by type:")
    print(
        manifest
        .drop_duplicates("building_id")
        ["buildingType"]
        .value_counts()
        .sort_index()
    )
    print()
    print("Climate regions:")
    print(
        manifest
        .drop_duplicates("building_id")
        ["climateRegion"]
        .value_counts()
        .sort_index()
    )

# generator

def single_run(row: dict, res_dir=RESULTS_DIR)-> tuple[str, str, tuple[str, str]| None]:
    run_id = row["run_id"]
    try:
        res_dir = Path(res_dir)
        res_dir.mkdir(parents=True, exist_ok=True)
        warnings.filterwarnings("ignore", message="Maximal heat load exceeded.*")
        t0 = time.time()
        row = row.copy()
        with STARTED_LOG.open("a") as f:
            f.write(f"{run_id}\n")
        final_path = res_dir / f"{run_id}.h5"
        params = row
        params.pop("run_id", None)
        params.pop("building_id", None)
        meta = {c: params.pop(c, None) for c in EPISCOPE_META}
        split = params.pop("split", None)
        rep = params.pop("rep", None)
        fixed_seeds = params.pop("fixed_seeds", None)

        cfg = tsib.BuildingConfiguration(params)
        bdg = tsib.Building(configurator=cfg)
        bdg.getOccupancy()
        bdg.getHeatLoad()
        violation = bdg.thermalmodel.M.bMaxLoadViolation.value
        resolved = {
            "weatherFilepath": bdg.cfg.get("weatherFilepath"),
            "weatherID": bdg.cfg.get("weatherID"),
            "resolved_climateRegion": bdg.cfg.get("climateRegion"),
            "resolved_latitude": bdg.cfg.get("latitude"),
            "resolved_longitude": bdg.cfg.get("longitude"),
            "state_seed": bdg.cfg.get("state_seed"),
            "apartment_seeds": bdg.cfg.get("apartment_seeds"),
            "appliances": bdg.cfg.get("appliances"),
            "maxLoadViolation_kW": violation,
            "runtime_seconds": time.time() - t0,
            "resolved_year": int(bdg.timeseries.index[0].year),
            "app_ownership_seed": bdg.cfg.get("app_ownership_seed"),
        }

        full_params = {**params, **resolved}
        full_params["n_persons"] = json.dumps(full_params["n_persons"])
        full_params["apartment_seeds"] = json.dumps(full_params["apartment_seeds"])
        full_params["appliances"] = json.dumps(full_params["appliances"])
        full_params["split"] = split
        full_params["rep"] = rep
        full_params["fixed_seeds"] = fixed_seeds
        full_params.update(meta)

        timeseries = bdg.timeseries[["T", "Occupancy Home Active", "Occupancy Home Not Active", "Electricity Load", "Hot Water Load", "Heating Load",]].astype("float32")

        day_types = bdg.cfg.get("day_types")

        if day_types is None:
            raise ValueError("day_types missing (profiles came from cache?)")
          # one entry per calendar day -> expand to every timestep via dayofyear
        is_offday = (np.asarray(day_types) == "we")[timeseries.index.dayofyear - 1]
        timeseries["Off Day"] = is_offday.astype("int8")

        if timeseries.isna().any().any():
            raise ValueError("NaN in timeseries")

        if not (
                timeseries[
                    ["Electricity Load", "Hot Water Load", "Heating Load"]
                ] >= 0
        ).all().all():
            raise ValueError("Negative load values")

        if violation >= 10.0:
            raise ValueError(f"{violation} kW overload")

        tmp_path = final_path.with_suffix(final_path.suffix + ".tmp")

        with pd.HDFStore(tmp_path, mode="w",complevel=9, complib="blosc:zstd") as store:
            store.put("timeseries", timeseries, format="table")
            store.put("params", pd.DataFrame([full_params]), format="table")
        os.replace(tmp_path, final_path)
        return run_id, "done", None

    except Exception as e:
        if "tmp_path" in locals():
            tmp_path.unlink(missing_ok=True)
        return run_id, "failed", (str(e), traceback.format_exc())

def progress_checker(manifest: pd.DataFrame, res_dir = RESULTS_DIR)-> list:
    res_dir = Path(res_dir)
    todo = []
    for _, row in manifest.iterrows():
        f_path = res_dir / f"{row['run_id']}.h5"
        if not f_path.exists() or not is_valid_result(f_path):
            todo.append(row.to_dict())
    return todo

def is_valid_result(path):
    try:
        with pd.HDFStore(path, mode="r") as store:
            if "/timeseries" not in store or "/params" not in store:
                return False

            ts = store["timeseries"]
            params = store["params"]

            return (
                len(ts) > 0
                and len(params) == 1
                and not ts.isna().any().any()
            )
    except Exception:
        return False

def manifest_run(manifest_path = MANIFEST_PATH,res_dir = RESULTS_DIR, fail_log = FAIL_LOG,max_cores=20):
    manifest_path = Path(manifest_path)
    res_dir = Path(res_dir)
    fail_log = Path(fail_log)
    res_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_parquet(manifest_path)
    todo = progress_checker(manifest, res_dir)
    print(f"Total:{len(manifest)}, Remain: {len(todo)}, Done: {len(manifest) - len(todo)}")
    done = failed = 0
    with ProcessPoolExecutor(max_workers=max_cores) as ex:
        futures = {ex.submit(single_run, row, res_dir) for row in todo}
        with tqdm(total=len(futures), unit="runs", desc="Running Simulations") as pbar:
            for future in as_completed(futures):
                run_id, status, err = future.result()
                if status == "done":
                    done +=1
                else:
                    failed +=1
                    with fail_log.open("a") as f:
                        f.write(f"{run_id}\t{err[0]}\n{err[1]}\n---\n")
                pbar.set_postfix(done=done, failed=failed)
                pbar.update(1)

    print(f"Finished: {done} done, {failed} failed ->  {fail_log}")

if __name__ == "__main__":
    split = "train"
    YEARS = [2021, 2022, 2023, 2025, 2026, 2029, 2030]
    mpath, rdir = split_paths(split)
    man = manifest(scenario_reps=1,seed_reps=1,fixed_seed_reps=1,episcope_path=EPISCOPE_PATH,building_types=["SFH", "MFH", "TH", "AB"],
                   seed=42, overwrite=True, years=YEARS,split=split, out_path=mpath, years_per_building=1) #, "MFH", "TH", "AB"
    validate_manifest(man, seed_reps=1, fixed_seed_reps=1, years=YEARS,
                      years_per_building=1, split=split)
    start = time.time()
    manifest_run(mpath,res_dir=rdir, fail_log=FAIL_LOG, max_cores=19)
    elapsed = time.time() - start
    print(f"Total runtime: {datetime.timedelta(seconds=round(elapsed))}")