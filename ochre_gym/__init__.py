from ochre_gym.ochre_env import OchreEnv
import tomli 
import pprint
from typing import Dict
import datetime as dt
from pathlib import Path 
from gymnasium.wrappers import ClipAction
from ochre_gym.spaces.act_spaces import ClipActionComposite
from ochre_gym.ochre_env import Reward
from ochre_gym import log 

__version__ = "0.1.0a1"

THIS_PATH = Path(__file__).parent.absolute()


def parse_dwelling_args(
    dwelling_dict: Dict,
    building_path: Dict):
    """Parse the dwelling arguments.

    Args:
        dwelling_dict (Dict): Dictionary of dwelling arguments.
        building_path (Dict): Path to the building.

    Returns:
        dwelling_dict (Dict): Dictionary of dwelling arguments.
    """
    dwelling_dict['start_time'] = dt.datetime.strptime(dwelling_dict['start_time'], '%Y-%m-%d %H:%M:%S')
    
    time = dt.datetime.strptime(dwelling_dict['time_res'], '%H:%M')
    dwelling_dict['time_res'] = dt.timedelta(hours=time.hour, minutes=time.minute)

    if 'end_time' in dwelling_dict:
        end_time = dt.datetime.strptime(dwelling_dict['end_time'], '%Y-%m-%d %H:%M:%S')
        dwelling_dict['duration'] = end_time - dwelling_dict['start_time']

    elif 'episode_duration' in dwelling_dict:
        time, units = dwelling_dict['episode_duration'].split(' ')
        assert units == 'hours' or units == 'days', 'Episode duration must be in hours or days.'
        dwelling_dict['duration'] = dt.timedelta(**{units: int(time)})
        dwelling_dict.pop('episode_duration')

    if dwelling_dict['initialization_time'] == '':
        dwelling_dict['initialization_time'] = None
    else:
        time, units = dwelling_dict['initialization_time'].split(' ')
        dwelling_dict['initialization_time'] = dt.timedelta(**{units: int(time)})
    
    dwelling_dict['hpxml_file'] = str(building_path / 'home.xml')
    dwelling_dict['schedule_input_file'] = str(building_path / 'schedules.csv')
    dwelling_dict['weather_file'] = str(building_path / dwelling_dict['weather_file'])

    return dwelling_dict



def load(
    env_name: str,
    **kwargs
):
    """Load an ochre_gym Env by name. Override default arguments with kwargs.
    Default arguments can be found in the defaults.toml file in ochre_gym/buildings.

    Args:
        env_name (str): Name of the building environment.

    Keyword Args:
        seed (int): The random seed used to initialize the environment.
        start_time (str): Time to start controlling the building. Format: "YYYY-MM-DD HH:MM:SS".
        time_res (str): Time resolution for OCHRE. String. Format: "HH:MM".
        end_time (str): Time to stop controlling the building. Format: "YYYY-MM-DD HH:MM:SS".
        episode_duration (str): Amount of time to run each episode for. 
            Number of episode steps is episode_duration / time_res. Ignored if end_time is specified.
            Format: "X days" or "X hours", where X > 0.      
        initialization_time (str): Amount of time that OCHRE should run before starting to control the building,
            to initialize the physical building states. Format: "X days" or "X hours", where X > 0.
        hpxml_file (str): Path to the building properties file.
        schedule_input_file (str): Path to the building schedule file.
        weather_file (str): Path to the building weather file.
        from_beopt (bool): Whether using a BEopt building model.
        verbosity (int): OCHRE internal verbosity level. Default: 7.
        log_to_file (bool): Log to file. Default: False.
        log_to_console (bool): Log to console. Default: False.
        log_output_filepath (str): Path to a log file. Default: './ochre_gym.log'.
        disable_uncontrollable_loads (bool): Disable load due to uncontrolled appliances 
            such as washer/dryer. Default: False.        
        vectorize_observations (bool): Vectorize the observation space. If False, the observation space
            is a composite spaces.Dict. If True, it is a spaces.Box. Default: True.
        use_all_ochre_observations (bool): Whether to use all OCHRE observations or 
            a reduced set of defaults. Default: True.
        override_ochre_observations_with_keys (List[str]): Only take these observations from OCHRE.
        override_equipment_controls (Dict): Override the default action space to use the 
            provided equipment and control types. Dictionary with keys "OCHRE equipment type"
            and *list* of values "control type".
            For example, {"HVAC Heating": ["Setpoint"]} or {"HVAC Cooling": ["Duty Cycle"]}.
        clip_actions (bool): Clip the actions to the action space bounds. Default: False.
        vectorize_actions (bool): Vectorize the action space. If False, the action space is a
            composite spaces.Dict. If True, it is a spaces.Box. Default: True.
        lookahead (str): Length of price lookahead provided as part of observation, in "hour:minute" format.
            Default: 00:00.
        thermal_comfort_unit_penalty (float): Unit penalty for being outside the comfort band.
        thermal_comfort_band_low (float): Lower bound of comfort band for thermal comfort.
        thermal_comfort_band_high (float): Upper bound of comfort band for thermal comfort.
        flat_energy_price (float): Price of energy in $/kWh, used in PC program.
        reward_scale (float): Scale the reward by a constant.
        dr_type (str): The type of DR program considered, could be "TOU", "PC" or "RTP".
        dr_subfolder (str): The name of the subfolder in `ochre_gym/energy_price` containing the DR files. Default: env_name.
        tou_price_file (str): File name of the TOU price daily signal.
        rtp_price_file (str): File name of the RTP price signal.
        pc_power_file (str): File name of the PC power limit signal.
        pc_unit_penalty (float): Unit penalty for violating the power constraint.
                         
    Returns:
        env (OchreEnv): The OCHRE Gym environment.
    """
    exp_config = {}
    config_dir = THIS_PATH / 'buildings' / env_name
    energy_price_dir = THIS_PATH / 'energy_price' / kwargs.get('dr_subfolder', env_name)
    with open(THIS_PATH / 'buildings' / 'defaults.toml', 'rb') as f:
        default_args = tomli.load(f)['defaults']

    exp_config['env_name'] = env_name

    #############################
    # Parse kwargs
    #############################

    # Override default arguments with kwargs
    for k, v in kwargs.items():
        if k in default_args and k != 'actions':
            default_args[k] = v

    # Store actions in the user config, grab defaults, override them 
    # Rename actions to correct format
    new_keys = []
    for k,v in default_args['actions'].items():
        new_keys += [ (k,k.replace('_', ' ')) ]
    for k, k_ in new_keys:
        default_args['actions'][k_] = default_args['actions'].pop(k)

    exp_config['actions'] = {}
    if not 'override_equipment_controls' in kwargs:
        for k, v in default_args['actions'].items():
            exp_config['actions'][k] = v
    else:
        exp_config['actions'] = kwargs['override_equipment_controls']

    default_args.pop('actions')
    exp_config['vectorize_actions'] = kwargs.get('vectorize_actions', True)

    exp_config['reward'] = {'dr_subfolder': energy_price_dir}
    for k, v in default_args['reward'].items():
        if not k in kwargs:
            exp_config['reward'][k] = v
        else: # override
            exp_config['reward'][k] = kwargs[k]
    default_args.pop('reward')

    # Validate DR type
    if exp_config['reward']['dr_type'] not in list(Reward.DR_TYPES_AND_DEFAULT_OBS.keys()):
        raise ValueError('Invalid DR Program type: {}'.format(exp_config['reward']['dr_type']))
    
    exp_config['lookahead'] = kwargs.get('lookahead', 
                                         default_args['observations']['lookahead'])
    default_args.pop('observations')

    # Parse args
    if default_args['weather_file'] == '':
        # return all files ending in .epw in config_dir
        weather_file = [f for f in config_dir.glob('*.epw')][0]
        default_args['weather_file'] = weather_file.name
    default_args = parse_dwelling_args(default_args, config_dir)

    #############################
    # Configure logging
    #############################
    log_output = kwargs.get('log_output_filepath', 'ochre_gym.log')
    logger, handler_types = None, []
    log_to_file = kwargs.get('log_to_file', default_args['log_to_file'])
    log_to_console = kwargs.get('log_to_console', default_args['log_to_console'])
    default_args.pop('log_to_file')
    default_args.pop('log_to_console')
    if log_to_file:
        handler_types += ['file']
    if log_to_console:
        handler_types += ['stream']
    if log_to_file or log_to_console:
        logger = log.get_logger(__name__, handler_types=handler_types, log_file = log_output)
    
    ##################################
    # Create and configure environment
    ##################################
    env = OchreEnv(exp_config['env_name'],
                   default_args,
                   exp_config['actions'],
                   exp_config['vectorize_actions'],
                   exp_config['lookahead'],
                   exp_config['reward'],
                   default_args['disable_uncontrollable_loads'],  
                   vectorize_observations=kwargs.get('vectorize_observations', True),
                   use_all_ochre_observations=kwargs.get('use_all_ochre_observations', True),
                   override_ochre_observations_with_keys=kwargs.get('override_ochre_observations_with_keys', None), 
                   logger = logger)
    
    # Clip Box actions within bounds
    do_clip = kwargs.get('clip_actions', False)
    if do_clip and exp_config['vectorize_actions']:
        env = ClipAction(env)
    elif do_clip:
        env = ClipActionComposite(env)
        
    if logger:
        # Log the configuration by pretty printing the config dicts
        logger.info('User configuration:\n{}'.format(
            pprint.pformat(exp_config, indent=4)
        ))
        logger.info('Dwelling configuration:\n{}'.format(
            pprint.pformat(default_args, indent=4)
        ))
    
    return env
