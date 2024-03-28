# Buildings in OCHRE™ Gym

A key advantage of OCHRE™ Gym over related building RL Gyms is the ease with which new buildings can be added. Buildings in OCHRE™ Gym are specified with the Home Performance eXtensible Markup Language (HPXML) format. Since OCHRE™ Gym wraps around OCHRE™, you can use any building that is supported by OCHRE™. Please read OCHRE™'s documentation for [adding a custom building](https://ochre-docs-final.readthedocs.io/en/latest/InputsAndArguments.html). 

To quote from these docs, HPXML and occupancy schedule input files can be generated from:


- [BEopt 3.0 or later](https://www.nrel.gov/buildings/beopt.html): best for designing a single building model. Includes a user interface to select building features. Note that the occupancy schedule file is optional; users must specify stochastic occupancy in BEopt. To generate input files from BEopt, run your model as usual. The input files you need for OCHRE™ (in.hpxml and schedules.csv) will be automatically generated and are located in ‘C:/Users/your_username/Documents/BEopt_3.0.x/TEMP1/1/run’. BEopt generates several xml files as part of the workflow, but the one OCHRE™ is looking for is always within the run directory.
- [End-Use Load Profiles Database](https://www.nrel.gov/buildings/end-use-load-profiles.html): best for using pre-existing building models
- [ResStock](https://resstock.nrel.gov/): best for existing ResStock users and for users in need of a large sample of building models.

We provide a number of example buildings in `ochre_gym/buildings/` created with ResStock.

### Building config details

For each building, the following files should be located in the same directory under `ochre_gym/buildings/your-env-name`:

- The HPXML file
- a schedules.csv file for occupancy usage patterns
- a weather file (TMY .epw or AMY)

See an example in `ochre_gym/buildings/basic-v0`. 
The weather and schedules files should contain a sufficiently long time range spanning the entire episode start/end timestamps.


### Removing uncontrollable loads

Remove uncontrollable "scheduled" loads like lighting and appliances from the building by setting `disable_uncontrollable_loads=True` in `ochre_gym.load()`.
