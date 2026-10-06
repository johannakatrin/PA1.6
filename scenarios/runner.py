import os
import time
from typing import Dict, List, Optional
import numpy as np

from utils.config import load_config
from utils.rng import RNG
from sensors.temp_sensor import TempSensor
from sensors.filters import hold_last, MovingAverageFilter
from controllers.onoff import OnOffThermostat
from controllers.predictive_onoff import PredictiveOnOff
from simulations.room_model import step_room
from simulations.environment import Environment
from plotting.plots import plot_timeseries, plot_error, plot_duty, plot_predictive, plot_heater

def simulate_once(scenario_path: str, seed: Optional[int] = None):
    scenario = load_config(scenario_path)
    rng = RNG(scenario.sim.seed if seed is None else seed)

    env = Environment(
        base=scenario.env.base,
        amplitude=scenario.env.amplitude,
        period_s=scenario.env.period_s,
        door_drop_C=scenario.env.door_drop_C,
        door_start_s=scenario.env.door_start_s,
        door_duration_s=scenario.env.door_duration_s,
    )

    sensor = TempSensor(sigma=scenario.sensor.sigma, bias=scenario.sensor.bias,
                        dropout_prob=scenario.sensor.dropout_prob, rng=rng)

    # Choose controller
    if getattr(scenario.controller, 'type', 'predictive_onoff') == 'onoff':
        ctrl = OnOffThermostat(
            setpoint=scenario.controller.setpoint,
            deadband=scenario.controller.deadband,
            safety_high=scenario.controller.safety_high,
            state=0
        )
        use_predictive = False
    else:
        ctrl = PredictiveOnOff(
            setpoint=scenario.controller.setpoint,
            deadband=scenario.controller.deadband,
            tau=scenario.controller.tau,
            safety_high=scenario.controller.safety_high,
            state=0
        )
        use_predictive = True

    dt = scenario.sim.dt
    steps = int(scenario.sim.duration_s / dt)
    T = scenario.sim.init_T
    last_valid: Optional[float] = T

    # Logs
    keys = ["t","T_true","T_meas","T_out","setpoint","heater","error","T_pred","lower","upper"]
    log: Dict[str, List[float]] = {k: [] for k in keys}

    # Filter
    ma = MovingAverageFilter(window=5)

    for k in range(steps):
        t = k * dt
        T_out = env.T_out(t)
        meas = sensor.read(T)
        meas = hold_last(meas, last_valid)
        if meas is None:
            meas = T
        last_valid = meas
        filt = ma.update(meas)

        if use_predictive:
            heater = ctrl.update(filt, dt)
            T_pred = ctrl.last_pred if ctrl.last_pred is not None else filt
            lower = ctrl.lower_threshold if ctrl.lower_threshold is not None else (ctrl.setpoint - ctrl.deadband/2)
            upper = ctrl.upper_threshold if ctrl.upper_threshold is not None else (ctrl.setpoint + ctrl.deadband/2)
        else:
            heater = ctrl.update(filt)
            T_pred = filt
            lower = ctrl.setpoint - ctrl.deadband/2
            upper = ctrl.setpoint + ctrl.deadband/2

        error = ctrl.setpoint - filt

        log["t"].append(t)
        log["T_true"].append(T)
        log["T_meas"].append(filt)
        log["T_out"].append(T_out)
        log["setpoint"].append(ctrl.setpoint)
        log["heater"].append(heater)
        log["error"].append(error)
        log["T_pred"].append(T_pred)
        log["lower"].append(lower)
        log["upper"].append(upper)

        T = step_room(T, heater, T_out, scenario.model.R, scenario.model.C, scenario.model.P,
                      dt, scenario.model.process_sigma, rng)

    return log, use_predictive


def run_scenario(scenario_path: str):
    log, use_predictive = simulate_once(scenario_path)

    # Write CSV
    ts = time.strftime("%Y%m%d-%H%M%S")
    base = os.path.splitext(os.path.basename(scenario_path))[0]
    log_dir = os.path.join("outputs","logs")
    fig_dir = os.path.join("outputs","figures")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    csv_path = os.path.join(log_dir, f"{base}-{ts}.csv")
    with open(csv_path, "w") as f:
        header = ",".join(log.keys()) + "\n"
        f.write(header)
        for i in range(len(log["t"])):
            row = ",".join(str(log[k][i]) for k in log.keys()) + "\n"
            f.write(row)

    # Plots
    plot_timeseries(log, os.path.join(fig_dir, f"{base}-temps-{ts}.png"))
    plot_heater(log, os.path.join(fig_dir, f"{base}-heater-{ts}.png"))
    plot_error(log, os.path.join(fig_dir, f"{base}-error-{ts}.png"))
    plot_duty(log, os.path.join(fig_dir, f"{base}-duty-{ts}.png"))
    if use_predictive:
        plot_predictive(log, os.path.join(fig_dir, f"{base}-predictive-{ts}.png"))

    print(f"Wrote log to {csv_path}")
    print(f"Figures saved to {fig_dir}")


def run_monte_carlo(scenario_path: str, runs: int = 100, start_seed: int = 0):
    if runs < 1:
        raise ValueError("runs must be at least 1")

    final_temperatures = []
    minimum_temperatures = []
    heater_duty_cycles = []

    for seed in range(start_seed, start_seed + runs):
        log, _ = simulate_once(scenario_path, seed=seed)
        temperatures = np.asarray(log["T_true"])
        heater = np.asarray(log["heater"])
        final_temperatures.append(temperatures[-1])
        minimum_temperatures.append(temperatures.min())
        heater_duty_cycles.append(heater.mean())

    metrics = {
        "final_temperature": final_temperatures,
        "minimum_temperature": minimum_temperatures,
        "heater_duty_cycle": heater_duty_cycles,
    }
    summary = {}
    for name, values in metrics.items():
        summary[f"{name}_mean"] = float(np.mean(values))
        summary[f"{name}_std"] = float(np.std(values))

    ts = time.strftime("%Y%m%d-%H%M%S")
    base = os.path.splitext(os.path.basename(scenario_path))[0]
    log_dir = os.path.join("outputs", "logs")
    os.makedirs(log_dir, exist_ok=True)
    csv_path = os.path.join(log_dir, f"{base}-monte-carlo-{ts}.csv")
    with open(csv_path, "w") as f:
        f.write("metric,mean,std\n")
        for name, values in metrics.items():
            f.write(f"{name},{np.mean(values)},{np.std(values)}\n")

    print(f"Monte Carlo runs: {runs}")
    for name in metrics:
        print(f"{name}: mean={summary[f'{name}_mean']:.4f}, std={summary[f'{name}_std']:.4f}")
    print(f"Summary saved to {csv_path}")
    return summary
