# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 12:40:17 2016

@author: Leander Kotzur
"""

import os
import warnings
import logging

import tinydb
import pandas as pd
import numpy as np

import tsib
import tsib.data


TOTAL_PROFILE_NUM = 20



class Building(object):

    def __init__(
        self,
        configurator=None,        
    ):
        """
        A building model which uses the IWU-Buildingtopology for parameterizing
        the physical building of a model. This can then be used to first run an
        occupancy simulation based on the CREST model (tsorb) and then a thermal
        simulation based on a 5R1C model to predict the heat loads. 
        
        Parameters
        ----------
        configurator: tsib.BuildingConfiguration, optional (default: None)
            Configuration dictionary which includes all parameters required
            for parameterizing a building.
        """

        if configurator is None:
            self.configurator = tsib.BuildingConfiguration({})
        else:
            if isinstance(configurator, tsib.BuildingConfiguration):
                self.configurator = configurator
            else:
                raise ValueError(
                    "'configurator' needs to be of type "
                    + '"buildingconfig.BuildingConfiguration"'
                )

        self.cfg = self.configurator.getBdgCfg(includeSupply=True)

        self.IDentries = self.configurator.IDentries

        self.thermalmodel = tsib.Building5R1C(self.cfg)

        # status if the profiles have already been initialized
        self._has_occupancy_profiles = False
        self._occupancy_profile_names = []
        self._has_heat_profiles = False
        self._heat_profile_names = []
        self._has_renewable_potential_profiles = False
        self._renewable_profile_names = []

        # define identifier for results
        self._ID = None

        # initialize time series for the building with relevant weather data
        #self.timeseries = self.cfg['weather'][[key for key in self.cfg['weatherUnits']]]
        weather_range = tsib.targetdaterange(self.cfg["weather"],self.cfg["freq"])
        weather_freq = tsib.resampletoindex(self.cfg["weather"], weather_range)
        self.timeseries = weather_freq[[key for key in self.cfg['weatherUnits']]]
        # initialize a dictionary to save the units
        self.units = self.cfg["weatherUnits"]

        return


    @property
    def ID(self):
        '''
        Returns an identifier of the building configuration as string.
        '''

        if self._ID is None:
            db = tinydb.TinyDB(os.path.join(tsib.data.PATH, "results","db.json"))

            # check if building exists in database
            def predicate(obj, requirements):
                for k,v in requirements.items():
                    if k not in obj or obj[k] != v:
                        return False
                return True
            
            # avoid json data format conflict with numpy
            db_entry =  {}
            for field, obj in self.IDentries.items():
                if isinstance(obj, np.generic):
                    db_entry[field] = obj.item()
                else:
                    db_entry[field] = obj
            
            # request db entry
            db_id = db.get(lambda obj: predicate(obj, db_entry))

            if db_id:
                logging.info('Building already exists under ID: '
                    +str(db_id.doc_id) + '. If you do not want to overwrite the results, define a separate ID.')
                self._ID = db_id.doc_id
            else:
                self._ID = db.insert(db_entry)

        return self._ID

    @ID.setter
    def ID(self,val):
        if not isinstance(val,str):
            raise ValueError('ID needs to be of type str')
        self._ID = val

        return


    @property
    def ID_attr(self):
        """'Returns a dictionary of all attributes which make the building
        unique"""

        return self.IDentries

    def _saveResults(self):
        datapath1 = os.path.join(
            tsib.data.PATH, "results", "buildingprofiles", str(self.ID) + ".csv"
        )
        datapath2 = os.path.join(
            tsib.data.PATH, "results", "buildingstaticresults", str(self.ID) + ".csv"
        )
        self.timeseries.to_csv(datapath1)
        pd.Series(self.static_results).to_csv(datapath2)
        return

    def _loadResults(self):
        datapath1 = os.path.join(
            tsib.data.PATH, "results", "buildingprofiles", str(self.ID) + ".csv"
        )
        datapath2 = os.path.join(
            tsib.data.PATH, "results", "buildingstaticresults", str(self.ID) + ".csv"
        )
        if os.path.isfile(datapath1):
            self.timeseries = pd.read_csv(datapath1, index_col=0)
            self.timeseries.index = pd.to_datetime(
                self.timeseries.index, utc=True
            )
            self.thermalmodel.static_results = pd.read_csv(
                datapath2, index_col=0, header=None
            ).squeeze().to_dict()
            return True
        else:
            return False

    def _get_renewable_profile(self, cfg):
        """
        Gets all time series for the renewable technologies to supply the building with.
        """

        # TODO tidy up the whole roof irradiance part - introduce index for different roofs
        # sim pv for tilted roof (TR) upper floor ceiling (UC) - assumed tilted
        tmy_data = tsib.TRY2TMY(cfg['weather'])

        # init profiles
        profiles = pd.DataFrame(index = tmy_data.index)

        # get photovoltaic and solar thermal profiles
        if cfg['roofTilt'] > 10:
            profiles['Photovoltaic 1'], pv_cov = tsib.simPhotovoltaic(tmy_data, 
                                            surface_tilt = cfg['roofTilt'],
                                            surface_azimuth = cfg['roofOrientation'], 
                                            latitude = cfg['latitude'],
                                            longitude = cfg['longitude'], 
                                            losses = 0.1,)
            profiles['Solar thermal 1'] = tsib.simSolarThermal(tmy_data, 
                                            surface_tilt = cfg['roofTilt'],
                                            surface_azimuth = cfg['roofOrientation'], 
                                            latitude = cfg['latitude'],
                                            longitude = cfg['longitude'], )
            profiles['Photovoltaic 2'], pv_cov = tsib.simPhotovoltaic(tmy_data, 
                                            surface_tilt = cfg['roofTilt'],
                                            surface_azimuth = cfg['roofOrientation'] + 180., 
                                            latitude = cfg['latitude'],
                                            longitude = cfg['longitude'], 
                                            losses = 0.1,)
            profiles['Solar thermal 2'] = tsib.simSolarThermal(tmy_data, 
                                            surface_tilt = cfg['roofTilt'],
                                            surface_azimuth = cfg['roofOrientation'] + 180., 
                                            latitude = cfg['latitude'],
                                            longitude = cfg['longitude'])
        else: # flat roof 
            profiles['Photovoltaic 1'], pv_cov = tsib.simPhotovoltaic(tmy_data,
                                            surface_tilt = 30., # optimal tilt
                                            surface_azimuth = 180., #and orient
                                            latitude = cfg['latitude'],
                                            longitude = cfg['longitude'], 
                                            losses = 0.2,) # higher losses 
                                            # due shadowing between panels
            profiles['Solar thermal 1'] = tsib.simSolarThermal(tmy_data, 
                                            surface_tilt = 30., # optimal tilt
                                            surface_azimuth = 180., #and orient
                                            latitude = cfg['latitude'],
                                            longitude = cfg['longitude'], )

        # get heat pump coefficient of performance
        profiles['Heat pump'] = tsib.simHeatpump(cfg['weather']['T'], T_hot = cfg['T_sup'], efficiency = 0.45, T_limit = -20.)

        self._has_renewable_potential_profiles = True

        self._renewable_profile_names = profiles.columns.values

        self.units.update({'Heat pump':'kW_{th}/kW_{el}', 'Photovoltaic 1':'kW/kWp', 
                            'Photovoltaic 2':'kW/kWp','Solar thermal 1':'kW_{th}/m^2',
                            'Solar thermal 2':'kW_{th}/m^2',})

        profiles = tsib.resampletoindex(profiles, self.timeseries.index)

        self.timeseries = self.timeseries.join(profiles)

        return profiles
                                            

    def _get_occupancy_profile(self, cfg):
        """
        Get all the occupancy related data, like internal heat gain,
        electricity load and occupants activity.
        """

        logging.info('Occupancy profiles are simulated. ' 
                    + 'This can take a few minutes.')

        # get a number of random seeds to generate the profiles
        seeds = np.random.RandomState(cfg['state_seed']).randint(
            0, TOTAL_PROFILE_NUM, size=int(cfg["varyoccupancy"] * cfg["n_apartments"])
        )

        # get the profiles
        hh_profiles = tsib.getHouseholdProfiles(
            cfg["n_persons"],
            cfg["weather_native"],
            self.IDentries["weather"],
            seeds=seeds,
            ignore_weather=True, #ToDo prüfen wieso ignore hier true ist
            mean_load=cfg["mean_load"],
            freq=cfg["freq"],
            target_index=self.timeseries.index,
        )

        # get short form apartments
        n_app = int(cfg["n_apartments"])
        # abs number of occupants
        n_occs = int(cfg["n_persons"]) * n_app

        # get from the household profiles buildingprofiles
        bdg_profiles = {}

        # get a profile for each occupancy variation
        for i in range(cfg["varyoccupancy"]):
            if n_app == 1:
                occData = hh_profiles[i]
            else:
                # merge profiles for multiple appartments
                occData = pd.concat(
                    hh_profiles[i * n_app : (i + 1) * n_app],
                    keys=range(i * n_app, (i + 1) * n_app),
                )
                occData = occData.groupby(level=[1]).sum()

            bdg_profiles[i] = {}

            # heat gain values relative (Master thesis cheng feng)
            OccNotActiveHeatGain = 100
            OccActiveHeatGain = 150

            # internal heat gain [kW]
            bdg_profiles[i]["Q_ig"] = (
                occData["AppHeatGain"].values
                + occData["OccActive"].values * OccActiveHeatGain
                + occData["OccNotActive"].values * OccNotActiveHeatGain
            ) / 1000
            #        bdg_profiles[i]['occ_active'] = occData['OccActive'] > 0.0

            #add base occupancy to bdg_profiles
            bdg_profiles[i]["OccActive"] = occData["OccActive"]
            bdg_profiles[i]["OccNotActive"] = occData["OccNotActive"]

            # the share of occupants which is not at home
            bdg_profiles[i]["occ_nothome"] = (
                n_occs - occData["OccActive"] - occData["OccNotActive"]
            ) / n_occs

            # the share of occupants which is sleeping
            bdg_profiles[i]["occ_sleeping"] = occData["OccNotActive"].div(n_occs)
            if cfg['tsorb_device_load']:
                bdg_profiles[i]["elecLoad"] = occData["Load"] / 1000
            else:
                bdg_profiles[i]["elecLoad"] = cfg["elecLoad"]

            #ToDo swap in OpenDHW here
            opendhwresult = tsib.getOpenDHWProfiles(
                cfg["n_persons"],
                n_app,
                cfg["buildingType"],
                cfg["weather"].index[0].year,
                cfg["freq"],
                occupancy_series=occData["OccActive"] #+ occData["OccNotActive"],
            )

            if len(opendhwresult) == len(self.timeseries.index):
                opendhwresult = opendhwresult.set_axis(self.timeseries.index)
            else:
                raise ValueError("Index of OpenDHW is misaligned")

            bdg_profiles[i]["hotWaterLoad"] = opendhwresult["Heat_kW"]

            # get hot water load
            #bdg_profiles[i]["hotWaterLoad"] = occData["HotWater"] / 1000




            # get fireplace profile
            if cfg["hasFirePlace"]:
                pot_filename = os.path.join(
                    tsib.data.PATH,
                    "results",
                    "fireplaceprofiles",
                    "Profile"
                    + "_apps"
                    + str(int(n_app))
                    + "_occ"
                    + str(int(cfg["n_persons"]))
                    + "_wea"
                    + str(cfg["weatherID"])
                    + "_seed"
                    + str(cfg['state_seed'])
                    + "_var"
                    + str(i)
                    + ".csv",
                )
                if os.path.isfile(pot_filename):
                    fireplaceLoad = pd.read_csv(
                        pot_filename,
                        index_col=0,
                        header=None,
                    ).iloc[:, 0]
                    fireplaceLoad.index = pd.to_datetime(
                        fireplaceLoad.index, utc=True
                    )
                else:
                    # get oven profile depending on activity and outside temperature
                    fireplaceLoad = tsib.simFireplace(
                        self.timeseries["T"],
                        occData["OccActive"] / n_occs,
                        n_ovens=n_app,
                        T_oven_on=5,
                        t_cool=5.0,
                        fullloadSteps=450,
                        seed=int(seeds[i * n_app]),
                    )
                    fireplaceLoad.to_csv(pot_filename, header=False)

                bdg_profiles[i]["fireplaceLoad"] = fireplaceLoad

        # give building config the first profile
        cfg.update(bdg_profiles[0])
        #old
        # hourly_index = cfg["weather"].index
        #
        # Q_ig = pd.Series(cfg["Q_ig"], index=self.timeseries.index)
        # cfg["Q_ig"] = tsib.resampletoindex(Q_ig, hourly_index).values
        # for key in ("occ_nothome", "occ_sleeping", "elecLoad"):
        #     cfg[key] = tsib.resampletoindex(cfg[key], hourly_index)

        #data given to the optimizer is averaged so that the hourly resolution contains all the information of Q_ig etc. and not just the one on the full hour mark
        hourly_index = cfg["weather"].index
        freq_delta = pd.Timedelta(pd.tseries.frequencies.to_offset(cfg["freq"]))

        # def to_hourly(series):
        #     if freq_delta <= pd.Timedelta(hours=1):
        #         return series.resample("h").mean().reindex(hourly_index)
        #     return tsib.resampletoindex(series, hourly_index)

        def to_hourly(series):
            if freq_delta <= pd.Timedelta(hours=1):
                return series.resample("h", origin=hourly_index[0]).mean().reindex(hourly_index)
            return tsib.resampleToIndex(series, hourly_index)

        Q_ig = pd.Series(cfg["Q_ig"], index=self.timeseries.index)
        cfg["Q_ig"] = to_hourly(Q_ig).values
        for key in ("occ_nothome", "occ_sleeping", "elecLoad"):
            cfg[key] = to_hourly(cfg[key])

        # give the other profiles to the configuration file as well
        if cfg["varyoccupancy"] > 1:
            cfg["vary_profiles"] = bdg_profiles

        # collect all profiles
        profiles = pd.DataFrame(index = self.timeseries.index)

        # TODO: improve the structure of this code to drop this step and dictionary
        profileDict = {'elecLoad': 'Electricity Load', 'hotWaterLoad': 'Hot Water Load',
                        'fireplaceLoad': 'Fireplace Load', "OccActive": "Occupancy Home Active","OccNotActive": "Occupancy Home Not Active",'occ_nothome': 'Occupancy Not Home',}
        for key in profileDict:
            # TODO: add other bdg profiles
            if key in bdg_profiles[0]:
                profiles.loc[:,profileDict[key]] = bdg_profiles[0][key]

        self.units.update({'Electricity Load':'kW_{el}', 'Hot Water Load':'kW_{th}',
                            'Fireplace Load':'kW/kWp', "Occupancy Home Active": "-","Occupancy Home Not Active": "-","Occupancy Not Home": "-", })

        # define relevant time series 
        self._occupancy_profile_names = profiles.columns.values

        # append to timeseries
        self.timeseries = self.timeseries.join(profiles)
        
        self._has_occupancy_profiles = True

        return cfg


    def _get_heatload_profile(self):
        '''
        Simulates the heat load based on a 5R1C model
        '''
       # occupancy profiles are required for thermal load determination
        if not self._has_occupancy_profiles:
            self._get_occupancy_profile(
                    self.cfg
                )

        logging.info('Heat load profiles are simulated. ' 
        + 'This can take a few minutes.')
        # get thermal load with 5R1C model
        if self.cfg["refurbishment"]:
            # save refurbishment satus
            befRef = True
            warnings.warn(
                "For the simulation the refurbishment decisions"
                + " are deactivated",
                UserWarning,
            )
            self.cfg["refurbishment"] = False
        else:
            befRef = False

        # run simulation
        self.thermalmodel.sim5R1C(tee=False)

        # overwrite refurbishment options again
        self.cfg["refurbishment"] = befRef

        self._has_heat_profiles = True
        
        # define relevant time series 
        self._heat_profile_names = ['Heating Load', 'Cooling Load']

        self.units.update({'Heating Load':'kW_{th}', 'Cooling Load':'kW_{th}', })

        # # append simulation (TODO improve this call)
        #alt
        # self.timeseries = self.timeseries.join(self.thermalmodel.detailedResults[self._heat_profile_names])


        #resmaple SH to freq after hourly optimization
        heat_load = tsib.resampletoindex(
            self.thermalmodel.detailedResults[self._heat_profile_names],
            self.timeseries.index,
        )
        self.timeseries = self.timeseries.join(heat_load)
        return self.timeseries[self._heat_profile_names]

    def getHeatingSystem(self):
        """
        Determines the heat transfer coefficient between the heating system
        (e.q. the radiator) and the room based on following assumption:
            The system is designed such that it is able to supply a room
            temperature of 20°C at the outside design temperature with a 
            defined supply temperature. This supply temperature is an 
            assumption for each building.
        """
        logging.warning('Method to generate the nominal heat transfer coefficient of the heating system has not been validated."')
        # get design heat load
        self.design_Q = self.thermalmodel.calcDesignHeatLoad()

        # derive the heat transfer coefficient of the heating system kW/K
        self.design_H_heat = self.design_Q / (self.cfg['T_sup'] - 20.)
        
        return


    def toCSV(self, filename="result"):
        """
        Writes the results to 2 csv files:
            filename + "_dynamic"
            filename + "_static"
        
        Parameters
        ----------
        filename: str, optional, default: "result"
            Filepath and name.
        """
        if filename[-4:] == ".csv":
            filename = filename[:-4]
        self.timeseries.to_csv(filename + "_dynamic.csv", sep=",")
        pd.Series(self.static_results).to_csv(filename + "_static.csv", sep=",")
        return

    def readCSV(self, filename="result"):
        """
        Reads the results from 2 csv files:
            filename + "_dynamic"
            filename + "_static" 
        
        Parameters
        ----------
        filename: str, optional, default: "result"
            Filepath and name.
        """
        if filename[-4:] == ".csv":
            filename = filename[:-4]
        self.timeseries = pd.read_csv(
            filename + "_dynamic.csv", sep=",", index_col=0
        )
        self.timeseries.index = pd.to_datetime(self.timeseries.index, utc=True)
        self.results = pd.read_csv(filename + "_static.csv", sep=",").to_dict()
        return


    def getOccupancy(self):
        '''
        Returns the occupancy profiles, and creates those, in case that they do
        not exist.
        '''
        if not self._has_occupancy_profiles:
            self._get_occupancy_profile(
                    self.cfg
                )
        
        return self.timeseries[self._occupancy_profile_names]


    def getHeatLoad(self):
        '''
        Returns the thermal load profile, and creates those, in case that they do
        not exist.
        '''
    
        if not self._has_heat_profiles:
            self._get_heatload_profile()
        
        return self.timeseries[self._heat_profile_names]



    def getRenewables(self):
        '''
        Returns the renewable potenital profiles, such as solar
        thermal, photovoltaic or  load profile, and creates those, in case that they do
        not exist.
        '''
        if not self._has_renewable_potential_profiles:
            logging.info('Occupancy profiles are simulated. ' 
                        + 'This can take a few minutes.')
            self._get_renewable_profile(self.cfg)
        
        return self.timeseries[self._renewable_profile_names]


    def getLoad(self):
        '''
        Returns all time series of the building.
        It manages existing load profiles, and if existing, 
        the profile is just read. Otherwise it simulates a new ones
        and saves those.
        
        Returns
        -------
        timeseries as pandas.DataFrame
        """

        '''

        resultsExist = self._loadResults()
        if resultsExist:
            return self.timeseries
        else:

            # Get occupancy profile
            self.getOccupancy()

            # Calculate renewable generation
            self.getRenewables()

            # Calculate heat profile
            self.getHeatLoad()

            # stores the time series
            self._saveResults()
        return self.timeseries


    @property
    def static_results(self):
        warnings.warn(
            "static_results is deprecated, use timeseries instead",
                    DeprecationWarning
        )
        return self.thermalmodel.static_results 

    @property
    def detailedResults(self):
        warnings.warn(
            "detailedResults is deprecated, use timeseries instead",
                    DeprecationWarning
        )
        return self.thermalmodel.detailedResults 

    def sim5R1C(self):
        '''
        Runs the 5R1C simulation.
        '''
        warnings.warn(
            "sim5R1C is deprecated, use getHeatLoad instead",
                    DeprecationWarning
        )
        self.getHeatLoad()

        return self.timeseries