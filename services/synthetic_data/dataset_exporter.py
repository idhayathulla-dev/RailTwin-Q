import os
import csv
import random
import networkx as nx
from services.data_loader import DataLoader
from services.graph_builder import GraphBuilder
from services.state_engine import StateEngine
from services.movement_engine import MovementEngine
from services.event_system import (
    SignalFailureEvent,
    HeavyRainEvent,
    FestivalRushEvent,
    MaintenanceEvent,
    TrackBlockageEvent,
    PowerFailureEvent
)
from models.train import Train

# Import generators
from services.synthetic_data.feature_engineering import FeatureEngineering
from services.synthetic_data.train_dataset_generator import TrainDatasetGenerator
from services.synthetic_data.station_dataset_generator import StationDatasetGenerator
from services.synthetic_data.track_dataset_generator import TrackDatasetGenerator
from services.synthetic_data.network_dataset_generator import NetworkDatasetGenerator
from services.synthetic_data.propagation_dataset_generator import PropagationDatasetGenerator

def format_time(minutes):
    hrs = (8 + (minutes // 60)) % 24
    mins = minutes % 60
    return f"{hrs:02d}:{mins:02d}"

class DatasetExporter:
    @staticmethod
    def run_pipeline(num_scenarios=50, ticks_per_scenario=180, output_dir="datasets"):
        print("\n" + "=" * 60)
        print("    RAILTWIN-Q: MODULAR SYNTHETIC DATA GENERATION PIPELINE")
        print("=" * 60)

        os.makedirs(output_dir, exist_ok=True)

        all_trains_data = []
        all_stations_data = []
        all_tracks_data = []
        all_networks_data = []
        all_propagations_data = []

        # Run scenarios
        for sc in range(1, num_scenarios + 1):
            if sc % 10 == 0 or sc == 1:
                print(f" -> Running Scenario {sc}/{num_scenarios}...")

            # 1. Generate scenario profile
            day_type = random.choice(["WEEKDAY", "WEEKEND", "FESTIVAL"])
            
            # Load basic network
            network = DataLoader.load_network("data")
            
            # Setup dynamic graph for centralities
            graph = GraphBuilder.build_graph(network)
            graph_metrics = FeatureEngineering.compute_graph_metrics(graph)
            
            # Inject centrality attributes into station objects
            for station in network.stations:
                metrics = graph_metrics.get(station.station_id, {})
                station.node_degree = metrics.get("node_degree", 0)
                station.betweenness_centrality = metrics.get("betweenness_centrality", 0.0)
                station.closeness_centrality = metrics.get("closeness_centrality", 0.0)
                station.station_connectivity = metrics.get("station_connectivity", 0.0)

            # Diversity injection: randomize train properties
            train_types = ["PASSENGER", "EXPRESS", "SUPERFAST", "FREIGHT"]
            for train in network.trains:
                train.train_type = random.choice(train_types)
                train.is_priority_train = train.train_type in ["EXPRESS", "SUPERFAST"]
                
                # Assign speed profile based on train type
                if train.train_type == "SUPERFAST":
                    train.base_speed = 120
                    train.max_speed = 130
                elif train.train_type == "EXPRESS":
                    train.base_speed = 100
                    train.max_speed = 110
                elif train.train_type == "FREIGHT":
                    train.base_speed = 70
                    train.max_speed = 75
                else: # PASSENGER
                    train.base_speed = 80
                    train.max_speed = 90
                
                train.speed = train.base_speed
                train.delay = random.choice([0, 0, 0, 5, 10]) # initial random delays

            # Dynamically schedule random events for this scenario
            active_events = []
            
            # 1. Rain (50% chance)
            if random.random() < 0.5:
                start_tick = random.randint(10, 45)
                duration = random.randint(30, 90)
                intensity = random.uniform(0.3, 0.95)
                rain_ev = HeavyRainEvent(intensity=intensity, duration=duration)
                rain_ev.start_tick = start_tick
                rain_ev.active = False
                active_events.append(rain_ev)

            # 2. Signal Failure (30% chance)
            if random.random() < 0.3:
                start_tick = random.randint(30, 90)
                duration = random.randint(15, 50)
                station_id = random.choice([1, 2, 3])
                sig_ev = SignalFailureEvent(station_id=station_id, duration=duration)
                sig_ev.start_tick = start_tick
                sig_ev.active = False
                active_events.append(sig_ev)

            # 3. Power Failure (15% chance)
            if random.random() < 0.15:
                start_tick = random.randint(20, 80)
                duration = random.randint(10, 30)
                station_id = random.choice([2, 3, 4])
                pwr_ev = PowerFailureEvent(station_id=station_id, duration=duration)
                pwr_ev.start_tick = start_tick
                pwr_ev.active = False
                active_events.append(pwr_ev)

            # 4. Maintenance (25% chance)
            if random.random() < 0.25:
                start_tick = random.randint(40, 100)
                duration = random.randint(30, 70)
                track_id = random.choice([1, 2, 3])
                maint_ev = MaintenanceEvent(track_id=track_id, duration=duration)
                maint_ev.start_tick = start_tick
                maint_ev.active = False
                active_events.append(maint_ev)

            # 5. Track Blockage (15% chance)
            if random.random() < 0.15:
                start_tick = random.randint(15, 60)
                duration = random.randint(20, 45)
                track_id = random.choice([1, 2, 3])
                block_ev = TrackBlockageEvent(track_id=track_id, duration=duration)
                block_ev.start_tick = start_tick
                block_ev.active = False
                active_events.append(block_ev)

            # 6. Festival Rush (30% chance)
            festival_rush_active = False
            festival_start = 0
            if day_type == "FESTIVAL" or random.random() < 0.3:
                festival_rush_active = True
                festival_start = random.randint(5, 30)

            # Track active events pool
            running_events = []
            StateEngine.clear_history()

            # Run ticks
            for tick in range(ticks_per_scenario):
                current_time_str = format_time(tick)

                # Trigger scheduled events
                for event in active_events:
                    if getattr(event, "start_tick", 0) == tick:
                        event.active = True
                        running_events.append(event)

                # Festival rush extra train injection
                if festival_rush_active and tick == festival_start:
                    extra_train_1 = Train(
                        train_no=99501,
                        name="Festival Special 1",
                        route_id=1,
                        current_station_id=1,
                        speed=85,
                        delay=0,
                        train_type="SUPERFAST",
                        max_speed=130
                    )
                    extra_train_2 = Train(
                        train_no=99502,
                        name="Festival Special 2",
                        route_id=1,
                        current_station_id=2,
                        speed=75,
                        delay=0,
                        train_type="EXPRESS",
                        max_speed=110
                    )
                    network.add_train(extra_train_1)
                    network.add_train(extra_train_2)

                # Tick active events
                for event in running_events:
                    if event.active:
                        event.tick()

                # Movement Engine advance
                MovementEngine.tick(network, running_events, tick)

                # Update State Engine occupancies
                StateEngine.update_occupancies(network)

                # Record snapshot
                StateEngine.record_snapshot(network, current_time_str, running_events)

            # Load helper map lookups
            stations_map = {s.station_id: s.name for s in network.stations}
            tracks_map = {t.track_id: t for t in network.tracks}

            # Generate structured datasets for this scenario
            scenario_trains = TrainDatasetGenerator.generate(
                scenario_id=sc,
                history_records=list(enumerate(StateEngine.history)),
                network_routes=network.routes,
                stations_map=stations_map,
                tracks_map=tracks_map,
                graph=graph
            )
            all_trains_data.extend(scenario_trains)

            scenario_stations = StationDatasetGenerator.generate(
                scenario_id=sc,
                history_records=list(enumerate(StateEngine.history)),
                stations_map=stations_map
            )
            all_stations_data.extend(scenario_stations)

            scenario_tracks = TrackDatasetGenerator.generate(
                scenario_id=sc,
                history_records=list(enumerate(StateEngine.history)),
                stations_map=stations_map
            )
            all_tracks_data.extend(scenario_tracks)

            scenario_networks = NetworkDatasetGenerator.generate(
                scenario_id=sc,
                history_records=list(enumerate(StateEngine.history)),
                day_type=day_type
            )
            all_networks_data.extend(scenario_networks)

            scenario_propagations = PropagationDatasetGenerator.generate(
                scenario_id=sc,
                history_records=list(enumerate(StateEngine.history)),
                stations_map=stations_map
            )
            all_propagations_data.extend(scenario_propagations)

        # 3. Export CSV files
        DatasetExporter.write_csv("train_dataset.csv", all_trains_data, output_dir)
        DatasetExporter.write_csv("station_dataset.csv", all_stations_data, output_dir)
        DatasetExporter.write_csv("track_dataset.csv", all_tracks_data, output_dir)
        DatasetExporter.write_csv("network_state_dataset.csv", all_networks_data, output_dir)
        DatasetExporter.write_csv("delay_propagation_dataset.csv", all_propagations_data, output_dir)

        print("\n[PIPELINE CSV EXPORTS COMPLETE]")
        print(f" -> Output Directory: {os.path.abspath(output_dir)}")
        print(f" -> train_dataset.csv: {len(all_trains_data)} rows")
        print(f" -> station_dataset.csv: {len(all_stations_data)} rows")
        print(f" -> track_dataset.csv: {len(all_tracks_data)} rows")
        print(f" -> network_state_dataset.csv: {len(all_networks_data)} rows")
        print(f" -> delay_propagation_dataset.csv: {len(all_propagations_data)} rows")

        # 4. Run automated validation report
        try:
            print("\n[Step 4] Running validation report generator...")
            from generate_validation_report import run_report_pipeline
            run_report_pipeline(output_dir)
        except Exception as e:
            print(f"Warning: Could not automatically run validation report: {e}")
        print("=" * 60)

    @staticmethod
    def write_csv(filename, data, output_dir):
        if not data:
            print(f"Warning: No data to write for {filename}")
            return
            
        filepath = os.path.join(output_dir, filename)
        headers = list(data[0].keys())
        
        with open(filepath, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(data)
