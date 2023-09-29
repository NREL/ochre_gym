# ochre_gym.ochre_env

## OchreEnv
::: ochre_gym.ochre_env.OchreEnv
    options:
        show_source: false
        heading_level: 4
        show_root_heading: true

---

## Reward

The reward function at each time step is calculated as:

$$
r = -(\texttt{energy_used} * \texttt{energy_price} + \texttt{discomfort_penalty}).
$$

We current support three demand response programs with different energy prices:

- `TOU`: Time-of-use pricing
- `RTP`: Real-time pricing
- `PC`: Power constraint

The discomfort penalty is calculated as:

```python
deviation = max(max(0.0, indoor_temp - self.thermal_comfort_band_high),
                        max(0.0, self.thermal_comfort_band_low - indoor_temp),
                        0.0)
discomfort = self.thermal_discomfort_unit_cost * deviation ** 2
```

!!! note "Reward configuration CSV files"

     The `time_of_use_price.csv` and `dr_power_limit.csv` files only have entries for 1 day. Hence, every day in an episode, which may extend over months, will use the same TOU and PC. The `real_time_price.csv` file has entries for every 5 minutes for 2 months. We will need to consider a general solution for obtaining the price for any time step in an episode, and whether we want these to be fixed or stochastic.

---

::: ochre_gym.ochre_env.Reward
    options:
        show_source: false
        heading_level: 4
        show_root_heading: true