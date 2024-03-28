# OCHRE™ Gym Observations

The OCHRE™ Gym observation space is highly customizable to allow users to focus on the most relevant information for their use case. The core set of observations are those provided by the OCHRE™ building simulator. The OCHRE™ building simulator records a large number of state variables for the Dwelling---the exact number of variables returned in the OCHRE™ control result depends on the Dwelling `verbosity` (default = 7, which we do not recommend changing).

The observation space can be configured in the following ways:

- Choose a vector or dictionary space
- Use all or a predefined reduced subset of OCHRE™ Dwelling state variables
- Override the observation space to use only a subset of the OCHRE™ Dwelling state variables by specifying a list of keys 
- Add timestamp and energy price information to the observation space
- Add static metadata about the Dwelling to the observation space (e.g., # of bedrooms, square footage, etc.)
- Add custom values derived from the observation space values by defining the `customize_observation_space_hook` and `customize_observations_from_ochre_hook` methods in a custom `OchreObservationSpaceBaseConfig` class

Vectorized observations sort the keys alphabetically.

### OCHRE™ keys (OCHRE™ Dwelling verbosity = 7)

| Name (units) |
| --- | 
| Air Changes per Hour - Indoor (1/hour) |
| Air Density - Indoor (kg/m^3) |
| Forced Ventilation Flow Rate - Indoor (m^3/s) |
| Forced Ventilation Heat Gain - Indoor (W) |
| Grid Voltage (-) |
| HVAC Cooling Electric Power (kW) | 
| HVAC Cooling Reactive Power (kVAR) |
| HVAC Heating Electric Power (kW) |  
| HVAC Heating Gas Power (therms/hour) |
| HVAC Heating Reactive Power (kVAR) |
| Humidity Ratio - Indoor (-) |
| Infiltration Flow Rate - Attic (m^3/s) |
| Infiltration Flow Rate - Garage (m^3/s) |
| Infiltration Flow Rate - Indoor (m^3/s) |
| Infiltration Heat Gain - Attic (W) |
| Infiltration Heat Gain - Garage (W) |
| Infiltration Heat Gain - Indoor (W) |
| Internal Heat Gain - Indoor (W) |
| Lighting Electric Power (kW) |
| Lighting Reactive Power (kVAR) |
| Natural Ventilation Flow Rate - Indoor (m^3/s) |
| Natural Ventilation Heat Gain - Indoor (W) |
| Net Latent Heat Gain - Indoor (W) |
| Net Sensible Heat Gain - Attic (W) |
| Net Sensible Heat Gain - Garage (W) |
| Net Sensible Heat Gain - Indoor (W) |
| Occupancy (Persons) |
| Occupancy Heat Gain - Indoor (W) |
| Other Electric Power (kW) |
| Other Gas Power (therms/hour) |
| Other Reactive Power (kVAR) |
| Radiation Heat Gain - Attic (W) |
| Radiation Heat Gain - Garage (W) |
| Radiation Heat Gain - Indoor (W) |
| Relative Humidity - Indoor (-) |
| Temperature - Attic (C) |
| Temperature - Garage (C) |
| Temperature - Ground (C) |
| Temperature - Indoor (C) | 
| Temperature - Outdoor (C) | 
| Total Electric Energy (kWh) | 
| Total Electric Power (kW) | 
| Total Gas Energy (therms) |
| Total Gas Power (therms/hour) |
| Total Reactive Energy (kVARh) |
| Total Reactive Power (kVAR) |
| Unmet HVAC Load (C) |
| Water Heating Gas Power (therms/hour) |
| Wet Bulb - Indoor (C) |
| Window Transmitted Solar Gain (W) |

### Custom keys

Added by the default `TimeAndEnergyPriceObservationSpaceConfig` class:

| Name  | Extra info |
| --- | --- |
| Day of month |  A scalar (int) for the day of month index. |
| Day of week |  A scalar (int) for the day of week index. |
| Hour of day | A scalar (int) for the hour of day index.  |
| Energy Price ($) | If the demand response problem is TOU, this is a scalar (float) of the price of energy (USD $). If the demand response problem is RTP, this is a vector with `lookahead_steps` dims of the day-ahead price (USD \$).  |
| Power Limit (kW) | A scalar (float) of the maximum power consumption for the Dwelling in kW. |

Added by the `MetadataObservationSpaceConfig` class:

| Name | Extra info |
| --- | --- |
| Has Attic | A binary value (0 or 1) indicating presence of an attic. |
| Has Garage | A binary value (0 or 1) indicating presence of a garage. |
| Number of Bedrooms | A scalar (int) counting number of bedrooms in the Dwelling. |    
| Square Footage | A scalar of the total square footage of the Dwelling. |

### Using specific observations

To only use a few specific observations, list the desired keys in the `override_ochre_observations_with_keys` keyword argument in `ochre_gym.load()`:

```python
env = ochre_gym.load(
    env_name="basic-v0",
    seed=42,
    override_ochre_observations_with_keys=[
        'Temperature - Indoor (C)',
        'Temperature - Outdoor (C)',
        'Hour of day',
        'HVAC Cooling Electric Power (kW)'
    ],
)
```

### Customizing the observation space

For the most flexibility, create a subclass of `OchreObservationSpaceBaseConfig` and override the `customize_observation_space_hook` and `customize_observations_from_ochre_hook` methods.

As an example, here is the `MetadataObservationSpaceConfig` class, which adds metadata about the Dwelling *on top of the additional time and energy price observations*:

```python

class MetadataObservationSpaceConfig(TimeAndEnergyPriceObservationSpaceConfig):
    """
    A configuration for the observation space that:

    - Adds time and energy price observations
    - Adds metadata:
        - Has Attic (binary)
        - Has Garage (binary)
        - Square Footage (float)
        - Number of Bedrooms (int)
    """

    def __init__(self, 
                 observation_args,
                 vectorize_observations,
                 use_all_ochre_observations,
                 override_ochre_observations_with_keys):
        """Initialize the OCHRE™ observation space configuration.
        
        Args:
            observation_args (Dict): A dictionary of keyword arguments.
            vectorize_observations (bool): Whether to vectorize observations.
            use_all_ochre_observations (bool): Whether to use all OCHRE™ observations or not.
            override_ochre_observations_with_keys (Optional[List[str]]): A list of keys to override the OCHRE™ observations with. If None, then use all OCHRE™ observations.
        """                      
        assert 'dwelling_metadata' in observation_args, '"dwelling_metadata" must be a key in observation_args'
        super().__init__(observation_args,
                         vectorize_observations,
                         use_all_ochre_observations,
                         override_ochre_observations_with_keys)

    def customize_observation_space_hook(self, observations: Dict) -> Dict:
        """Customize the observation space by adding metadata observations.
        Args:
            observations (Dict): A dictionary of observations to customize.
        
        Returns:
            observations (Dict): The customized dictionary of observations.
        """
        observations = super().customize_observation_space_hook(observations)
        if 'Garage' in self.observation_args['dwelling_metadata']['zones']:
            observations['Has Garage'] = spaces.MultiBinary(1)
        if 'Attic' in self.observation_args['dwelling_metadata']['zones']:
            observations['Has Attic'] = spaces.MultiBinary(1)

        metadata_observations = {
            'Square Footage': spaces.Box(low=0., high=10000., shape=(1,)),
            'Number of Bedrooms': spaces.Discrete(10),
        }
        observations = {**observations, **metadata_observations}
        return observations
    

    def customize_observations_from_ochre_hook(self, observations, control_result, args):
        """Customize the observations from OCHRE™ by adding metadata observations.

        Args:
            observations (Dict): A dictionary of observations.
        
        Returns:
            observations (Dict): The customized dictionary of observations.
        """        
        observations = super().customize_observations_from_ochre_hook(observations, control_result, args)
        has_attic = 0
        has_garage = 0
        total_sq_ft = args['dwelling_metadata']['zones']['Indoor']['Zone Area (m^2)']
        if 'Garage' in args['dwelling_metadata']['zones']:
            has_garage = 1
            total_sq_ft += args['dwelling_metadata']['zones']['Garage']['Zone Area (m^2)']
        if 'Attic' in args['dwelling_metadata']['zones']:
            has_attic = 1
            total_sq_ft += args['dwelling_metadata']['zones']['Attic']['Zone Area (m^2)']

        observations['Square Footage'] = total_sq_ft
        observations['Has Attic'] = has_attic
        observations['Has Garage'] = has_garage
        observations['Number of Bedrooms'] = args['dwelling_metadata'][
            'construction']['Number of Bedrooms (-)']
        
        return observations
```
