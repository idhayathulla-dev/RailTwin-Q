import sys
from services.synthetic_data.dataset_exporter import DatasetExporter

def main():
    num_scenarios = 100
    ticks_per_scenario = 180
    output_dir = "datasets"

    if len(sys.argv) > 1:
        try:
            num_scenarios = int(sys.argv[1])
        except ValueError:
            print("Usage: python generate_synthetic_data.py [num_scenarios] [ticks_per_scenario]")
            sys.exit(1)
            
    if len(sys.argv) > 2:
        try:
            ticks_per_scenario = int(sys.argv[2])
        except ValueError:
            print("Usage: python generate_synthetic_data.py [num_scenarios] [ticks_per_scenario]")
            sys.exit(1)

    # Execute the modular dataset generation pipeline
    DatasetExporter.run_pipeline(
        num_scenarios=num_scenarios,
        ticks_per_scenario=ticks_per_scenario,
        output_dir=output_dir
    )

if __name__ == "__main__":
    main()
