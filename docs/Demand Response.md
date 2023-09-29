# Demand Response

OCHRE Gym supports three different demand response (DR) reward functions: Real-Time Pricing (RTP), Time-of-Use (TOU), and Power Constraints (PC). Our goal is to train agents that can respond to DR signals and reduce the cost of energy use while maintaining comfort. The three DR programs are inspired by the experiments in [this paper](https://arxiv.org/abs/2210.10203).

The reward function at each time step is calculated as:

$$
r = -(\texttt{energy_used} * \texttt{energy_price} + \texttt{discomfort_penalty}).
$$

Pricing information for each DR type is stored in a .csv file in the `ochre_gym/energy_price/{DR_SUBFOLDER}` directory, where `DR_SUBFOLDER` is a custom name (e.g., `basic-v0`). This allows users to mix and match buildings with different DR programs. Use the keyword argument `dr_subfolder` (e.g., `dr_subfolder=basic-v0`) in `ochre_gym.load()`. The time resolution of these files should match or be higher than control step time resolution. For example, if the control step time resolution is 15 minutes, the pricing info should be 15 minutes or finer.

### Real-Time Pricing (RTP)

Electricity price changes every hour based on the RTP from the wholesale market, which is known by the end of the hour. The day-ahead price (DAP) is given at the end of the previous day and provides a prediction of the next day’s RTP. We use the RTP and DAP data from a real-world RTP program.

RTP info should be stored in a `real_time_price.csv` file. The columns should be `Datetime,DAP,RTP` and the first row should be the column names. The `Datetime` column should be in the format `YYYY-MM-DD HH:MM:SS`. The `DAP` and `RTP` columns should be in $/kWh.

Example:

```csv
Datetime,DAP,RTP
2018-06-01 00:00:00,3.6,3.7
```

### Time-of-Use (TOU)

Electricity price varies by time of day according to a predetermined schedule. The price is either peak or off-peak, where the peak price is 2-10x higher than the off-peak price. For example, during peak hours (12:00 - 18:00) the price may be $10/kWh, while during off-peak hours (18:00 - 12:00 next day) the price may be $1/kWh. 

TOU info should be stored in a `time_of_use_price.csv` file. The columns should be `Datetime,TOU` and the first row should be the column names. The `Datetime` column should be in the format `MM/DD/YY HH:MM`. The `TOU` column is the energy price in $/kWh.

Example:

```csv
Datetime,TOU
6/1/18 0:00,1
```

### Power Constraints (PC)

Customers receive a lower electricity price for participating in the program. In exchange, during specified times called DR events, their building must reduce its power below a predetermined limit for a period of time—otherwise a penalty is applied. The limit is typically calculated from the building’s baseline consumption and load reduction potential.

PC info should be stored in a `dr_power_limit.csv` file. The columns should be `Datetime,power_limit` and the first row should be the column names. The `Datetime` column should be in the format `MM/DD/YY HH:MM`. The `power_limit` column is the power limit in kW.

Example: 

```csv
Datetime,power_limit
6/1/18 0:00,20
```

