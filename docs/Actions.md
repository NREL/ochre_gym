# OCHRE Gym Actions

In OCHRE Gym, agents control different types of equipment in the OCHRE Dwelling. Currently, we support controlling HVAC (heating and cooling are separate equipments) and a water heater. OCHRE Gym supports vector action spaces (default) and **nested dictionary** action spaces.For nested dictionary actions, the top-level keys are the names of the OCHRE Dwelling equipment and the second-level keys are the names of the
available controls for each equipment. The values are the corresponding gym.spaces objects.

!!! note
    The OCHRE simulator internally works with (unordered) nested dictionaries. If a vector action is provided, the OchreEnv will unflatten it, assuming the following order, and convert it into an ordered dictionary. 

### Equipment types

To summarize, the following OCHRE equipment types are currently supported:

- HVAC Cooling
- HVAC Heating
- Water Heating

We have future plans to add PV and Battery equipment types.

### Equipment controls

The following ways to control equipment in OCHRE are provided:

- Load Fraction (discrete)
- Duty Cycle (continuous)
- Setpoint (continuous)
- Deadband (continuous)

If you are unfamiliar with some of the control types listed above, **we recommend starting with only using Setpoint** and ignoring the rest (default).

### Customizing the action space


Users are able to customize the equipment and equipment controls in the action space
if desired with a dictionary config via the `override_equipment_controls` keyword argument in 
`ochre_gym.load()`.

For example:

```python
env = ochre_gym.load(
    "basic-v0",
    override_equipment_controls={
        'HVAC Cooling': ['Setpoint'],
        'HVAC Heating': ['Setpoint']
    }
)
```

Will remove the `Water Heating` equipment from the Dwelling and only use `Setpoint` controls for HVAC.

### As Nested Dict

```python
{
    'HVAC Cooling': {
        'Setpoint': np.array([23.]), # or 23.
    },
    'HVAC Heating': {
        ...
    },
    'Water Heating': {
        ...
    }
}
```

### Vectorized Array

**Vector actions** are numpy arrays arranged as follows:

```python
np.array([
    HVAC Cooling - Load Fraction,
    HVAC Cooling - Duty Cycle,
    HVAC Cooling - Setpoint,
    HVAC Cooling - Deadband,
    HVAC Heating - Load Fraction,
    HVAC Heating - Duty Cycle,
    HVAC Heating - Setpoint,
    HVAC Heating - Deadband,
    Water Heating - Load Fraction,
    Water Heating - Duty Cycle,
    Water Heating - Setpoint,
    Water Heating - Deadband
])
```

If any equipment or equipment control-type is not available or not used, the flattened
action space should be dynamically resized while maintaining the same ordering.

For example, without `HVAC Heating`:

```python
np.array([
    HVAC Cooling - Load Fraction,
    HVAC Cooling - Duty Cycle,
    HVAC Cooling - Setpoint,
    HVAC Cooling - Deadband,
    Water Heating - Load Fraction,
    Water Heating - Duty Cycle,
    Water Heating - Setpoint,
    Water Heating - Deadband
])
```

or, without using `Load Fraction`:

```python
np.array([
    HVAC Cooling - Duty Cycle,
    HVAC Cooling - Setpoint,
    HVAC Cooling - Deadband,
    HVAC Heating - Duty Cycle,
    HVAC Heating - Setpoint,
    HVAC Heating - Deadband,
    Water Heating - Duty Cycle,
    Water Heating - Setpoint,
    Water Heating - Deadband
])
```

--- 