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

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "tsib" / "data"

EPISCOPE_PATH = DATA_DIR / "episcope" / "episcope.csv"

RESULTS_DIR = PROJECT_ROOT / "Generation" / "results"

MANIFEST_PATH = PROJECT_ROOT / "Generation"/ "manifest.parquet"
FAIL_LOG = PROJECT_ROOT / "Generation" / "failures.log"
STARTED_LOG = PROJECT_ROOT / "Generation" / "started.log"


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
    tzu = tzu[["Code_BuildingSizeClass","Connections", "Code_ConstructionYearClass"]]
    tzu.columns = ["Type", "Surrounding","AgeBin"]
    res_dict = tzu.to_dict("records")
    return res_dict

#https://episcope.eu/building-typology/country/de/  Statistics of the German Building Stock Source[2]
def gen_flat_count(buildingtype: str, rng: np.random.Generator) -> int:
    if buildingtype == "SFH":
        return 1
    elif buildingtype == "TH":
        return int(rng.integers(low=1, high=3))
    elif buildingtype == "MFH":
        return int(rng.integers(low=3, high=13)) #exclusive
    elif buildingtype == "AB":
        return int(rng.integers(low=13, high=21))
    else:
        raise ValueError

#https://www.destatis.de/DE/Themen/Gesellschaft-Umwelt/Wohnen/Tabellen/tabelle-wo2-mietwohnungen.html
#https://www.destatis.de/DE/Themen/Gesellschaft-Umwelt/Wohnen/Tabellen/tabelle-wo2-eigentuemerwohnungen.html
#probs sind für haushalte
def gen_occs(buildingtype: str, apartment_count: int, rng: np.random.Generator) ->list:

    occ_probs = {
        "SFH": (0.245114, 0.402323, 0.154148, 0.148869, 0.04954599999999998),
        "MFH": (0.490622, 0.307419, 0.101949, 0.074417, 0.025592999999999977),
        "AB": (0.490622, 0.307419, 0.101949, 0.074417, 0.025592999999999977),
        "TH": (0.236817, 0.400092, 0.157261, 0.154371, 0.05145899999999992),
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


def manifest(scenario_reps,seed_reps,episcope_path , building_types = None,seed=42, out_path=MANIFEST_PATH, overwrite=False):
    if building_types is None:
        building_types = ["SFH", "MFH", "TH", "AB"]
    episcope_path = Path(episcope_path)
    out_path = Path(out_path)

    building_combis = episcope_combis(episcope_path, building_types=building_types)
    year_types = ["average", "hot", "cold"]
    futures = [False, True]
    weather_combis = [(f, y) for f in futures for y in year_types]
    rng = np.random.default_rng(seed)
    rows = []
    building_id = 0

    def building_type_reps(building_type):
        return scenario_reps[building_type] if isinstance(scenario_reps, dict) else scenario_reps

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

        climate_regions = np.arange(1, 16)
        base = np.tile(climate_regions,n_scenarios // len(climate_regions))
        remainder = n_scenarios % len(climate_regions)
        extra = rng.choice(climate_regions,size=remainder,replace=False)
        region_scenario = np.concatenate([base, extra])
        rng.shuffle(region_scenario)


        for i in range(n_scenarios):
            n_flats = gen_flat_count(buildingtype=building_type, rng=rng)
            occs = gen_occs(buildingtype=building_type, apartment_count=n_flats, rng=rng)
            flat_areas = gen_flat_area(occs=occs,rng=rng)
            total_area = sum(flat_areas)
            clima_region = region_scenario[i]
            future, year_type = weather_scenario[i]
            for _ in range(seed_reps):
                rows.append(dict(
                    country = "DE",
                    buildingType = building_type,
                    surrounding = surrounding,
                    buildingAgeBin = buildingAgeBin,
                    a_ref = round(total_area,1),
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
                    comfortT_lb = 21.0,
                    comfortT_ub = 24.0,
                    cores = 1,
                    useCache=False,
                    building_id = building_id,

                )
                 )
            building_id += 1

    manifest = pd.DataFrame(rows)
    manifest["run_id"] = manifest.index.map(lambda i: f"run_{i:06d}")
    manifest["seed"]= 1000000 + manifest.index
    if out_path.exists() and not overwrite:
        raise FileExistsError(out_path)
    manifest.to_parquet(out_path)
    return manifest

#Validation

def validate_manifest(manifest: pd.DataFrame, seed_reps: int) -> None:
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
    }

    missing_columns = required_columns - set(manifest.columns)

    assert not missing_columns, (f"Manifest is missing required columns: {sorted(missing_columns)}")

    assert len(manifest) > 0, "Manifest is empty."

    assert not manifest.isna().any().any(), ("Manifest contains missing values.")

    assert manifest["run_id"].is_unique, ("run_id is not unique.")

    assert manifest["seed"].is_unique, ("seed is not unique.")

    reps = manifest.groupby("building_id").size()

    assert (reps == seed_reps).all(), (
        "Not every building_id has the expected number of "
        f"seed repetitions ({seed_reps}).\n"
        f"Observed repetition counts:\n{reps.value_counts().sort_index()}"
    )

    expected_rows = manifest["building_id"].nunique() * seed_reps

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

    type_rules = {
        "SFH": (1, 1),
        "TH":  (1, 2),
        "MFH": (3, 12),
        "AB":  (13, 20),
    }

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
        f"run_{i:06d}"
        for i in range(len(manifest))
    ]

    assert manifest["run_id"].tolist() == expected_run_ids, (
        "run_id sequence is not as expected."
    )

    expected_seeds = (
        1_000_000 + np.arange(len(manifest))
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
            "maxLoadViolation_kW": violation,
            "runtime_seconds": time.time() - t0,
        }

        full_params = {**params, **resolved}
        full_params["n_persons"] = json.dumps(full_params["n_persons"])
        full_params["apartment_seeds"] = json.dumps(full_params["apartment_seeds"])


        timeseries = bdg.timeseries[["T", "Occupancy Home Active", "Occupancy Home Not Active", "Electricity Load", "Hot Water Load", "Heating Load",]].astype("float32")

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
    man = manifest(1,1,EPISCOPE_PATH,building_types=["SFH", "MFH", "TH", "AB"],seed=42, overwrite=True)
    validate_manifest(man, seed_reps=1)
    start = time.time()
    manifest_run(MANIFEST_PATH,res_dir=RESULTS_DIR, fail_log=FAIL_LOG, max_cores=19)
    elapsed = time.time() - start
    print(f"Total runtime: {datetime.timedelta(seconds=round(elapsed))}")