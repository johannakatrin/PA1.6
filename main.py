import argparse
from scenarios.runner import run_monte_carlo, run_scenario

def main():
    parser = argparse.ArgumentParser(description="Run a room temperature scenario")
    parser.add_argument("--scenario", required=True, help="Path to YAML scenario file")
    parser.add_argument(
        "--monte-carlo",
        type=int,
        metavar="N",
        help="Run N simulations with different random seeds",
    )
    parser.add_argument(
        "--start-seed",
        type=int,
        default=0,
        help="Seed for the first Monte Carlo run",
    )
    args = parser.parse_args()
    if args.monte_carlo is None:
        run_scenario(args.scenario)
    else:
        run_monte_carlo(args.scenario, args.monte_carlo, args.start_seed)

if __name__ == "__main__":
    main()
