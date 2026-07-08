# -*- coding: utf-8 -*-
"""
Created on Fri Apr 08 11:33:01 2016

@author: Leander Kotzur
"""


import tsib
import pandas as pd

def test_configuration_1():
    # parameterize a building
    bdgcfg = tsib.BuildingConfiguration(
        {
            "refurbishment": False,
            "nightReduction": False,
            "occControl": False,
            "capControl": True,
            "n_persons": 2,
            "roofOrientation": 0.0,
            "n_apartments": 1,
            "latitude": 49.,
            "longitude": 12.,
        }
    )
    test = bdgcfg.getBdgCfg()
    return


def test_configuration_2():
    # parameterize a building
    bdgcfg = tsib.BuildingConfiguration(
        {
            "buildingYear": 1980,
            "n_persons": 2,
            "roofOrientation": 0.0,
            "n_apartments": 2,
            "a_ref": 300.,
            "surrounding": "Detached",
            "latitude": 52.,
            "longitude": 13.,
        }
    )
    test = bdgcfg.getBdgCfg()
    return


def test_configuration_3():
    # parameterize a building
    kwgs = {
        "buildingYear": 1990,
        "latitude": 52.0,
        "longitude": 13.0,
        "comfortT_lb": 21.,
        "comfortT_ub": 24.,
        "WACC": 0.03,
        "roofTilt": 45.0,
        "surrounding": "Semi",
        "n_apartments": 2,
        "a_ref_app": 100.,
        "n_persons": 2,
        "roofOrientation": 135.0,
        "costdata": "default_2016",
        "capControl": True,
    }

    bdgcfg = tsib.BuildingConfiguration(kwgs)

    test = bdgcfg.getBdgCfg()

    return


def test_configuration_4():
    # parameterize a building
    kwgs = {
        "country": "DE",
        "n_apartments": 1,
        "a_ref": 150.,
        "buildingYear": 1990,
        "surrounding": "Detached",
    }

    bdgcfg = tsib.BuildingConfiguration(kwgs)

    test = bdgcfg.getBdgCfg()

    return


def test_configuration_other_countries():
    # parameterize a building
    bdgcfg = tsib.BuildingConfiguration(
        {
            "buildingYear": 1980,
            "country": "BE",
            "n_persons": 2,
            "roofOrientation": 0.0,
            "n_apartments": 1,
            "surrounding": "Detached",
            "latitude": 52.,
            "longitude": 13.,
        }
    )
    test = bdgcfg.getBdgCfg()

    # Golden value for the archetype selected by _get_typ_building's
    # country/year/surrounding tie-break (BE.N.SFH.03.Gen.ReEx.001.001).
    # Updated for the pandas 3.x / numpy 2.x migration: the previous value
    # (227) corresponded to an Austrian archetype, i.e. the country filter
    # was silently not being honored under the old dependency versions.
    assert round(test["q_h_nd"]) == 185.

def test_seed_override():
    # two configurations with identical physical parameters but different
    # explicit seeds should get independent state_seeds ...
    kwgs = {
        "buildingYear": 1980,
        "n_persons": 2,
        "roofOrientation": 0.0,
        "n_apartments": 1,
        "surrounding": "Detached",
        "latitude": 52.,
        "longitude": 13.,
    }

    cfg_default = tsib.BuildingConfiguration(dict(kwgs)).getBdgCfg()
    cfg_seed_1 = tsib.BuildingConfiguration(dict(kwgs, seed=1)).getBdgCfg()
    cfg_seed_2 = tsib.BuildingConfiguration(dict(kwgs, seed=2)).getBdgCfg()

    assert cfg_seed_1["state_seed"] == 1
    assert cfg_seed_2["state_seed"] == 2
    # ... while all other physical parameters remain unaffected
    for key in ("A_ref", "n_persons", "longitude", "latitude", "q_h_nd"):
        assert cfg_seed_1[key] == cfg_default[key]
        assert cfg_seed_2[key] == cfg_default[key]


def test_surround_weather_error_with_dummy():
    # parameterize a building
    bdgcfg = tsib.BuildingConfiguration(
        {
            "buildingYear": 1980,
            "country": "BE",
            "n_persons": 2,
            "roofOrientation": 0.0,
            "n_apartments": 1,
            "weatherData": pd.DataFrame([[0,0,0,]], columns=["DHI", "T", "DNI"]),
            "weatherID": "Dummy",
            "surrounding": "Detached",
            "latitude": 50.,
            "longitude": 1.,
        }
    )
    bdg = tsib.Building(configurator=bdgcfg)
    return

