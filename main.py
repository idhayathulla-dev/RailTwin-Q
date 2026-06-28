import os
import json
import time
import pandas as pd
import numpy as np
from services.data_loader import DataLoader
from services.graph_builder import GraphBuilder
from services.state_engine import StateEngine
from services.movement_engine import MovementEngine
from services.event_system import SignalFailureEvent, HeavyRainEvent

def format_time(minutes):
    hrs = (8 + (minutes // 60)) % 24
    mins = minutes % 60
    return f"{hrs:02d}:{mins:02d}"

def render_ascii_map(network):
    # Retrieve station and track occupancies for rendering
    c_central = network.get_station_by_id(1)
    arakkonam = network.get_station_by_id(2)
    katpadi = network.get_station_by_id(3)
    jolarpettai = network.get_station_by_id(4)
    
    # Get tracks
    t1 = network.get_track_by_id(1)
    t2 = network.get_track_by_id(2)
    t3 = network.get_track_by_id(3)
    
    # Helper to color codes (ASCII-safe)
    def get_station_status(s):
        if s.platforms_occupied == 0:
            return "Empty"
        elif s.platforms_occupied >= s.platforms * 0.8:
            return "CONGESTED"
        return "Busy"

    def get_track_status(t):
        if t.current_trains == 0:
            return "[0]"
        elif t.occupancy_percent >= 80.0:
            return f"[{t.current_trains}] (CONGESTED)"
        return f"[{t.current_trains}]"

    print("\n   [LIVE SCHEMATIC RAILWAY MAP]")
    print("   " + "="*80)
    print(f"   [Chennai Central] ===={get_track_status(t1)}====> [Arakkonam] ===={get_track_status(t2)}====> [Katpadi] ===={get_track_status(t3)}====> [Jolarpettai]")
    print(f"   Platforms: {c_central.platforms_occupied}/{c_central.platforms}          Platforms: {arakkonam.platforms_occupied}/{arakkonam.platforms}          Platforms: {katpadi.platforms_occupied}/{katpadi.platforms}          Platforms: {jolarpettai.platforms_occupied}/{jolarpettai.platforms}")
    print(f"   Status: {get_station_status(c_central)}        Status: {get_station_status(arakkonam)}        Status: {get_station_status(katpadi)}        Status: {get_station_status(jolarpettai)}")
    print("   " + "="*80)

def generate_web_dashboard(network, tick: int, sim_time_str: str, active_events: list, preds: list):
    """
    Generates a premium dark-themed HTML visual web dashboard updated in real-time.
    Written to datasets/dashboard.html
    """
    os.makedirs("datasets", exist_ok=True)
    dashboard_path = os.path.join("datasets", "dashboard.html")

    # 1. Format weather & events list
    active_ev_list = [e.name for e in active_events if e.active]
    events_badges = ""
    for ev in active_events:
        if ev.active:
            color = "red" if "Failure" in ev.name or "Block" in ev.name or "Maint" in ev.name else "yellow"
            events_badges += f"<span class='badge badge-{color}'>{ev.name}</span>"
    if not events_badges:
        events_badges = "<span class='badge badge-green'>Normal Operations</span>"

    # 2. Compute train positions on the schematic line (0% to 100%)
    train_markers = ""
    trains_cards = ""
    for train in network.trains:
        # Calculate visual offset on map
        # Chennai Central (0%), Arakkonam (30%), Katpadi (65%), Jolarpettai (100%)
        pos_pct = 0.0
        loc_desc = ""
        if train.progress == 0.0:
            if train.current_station_id == 1:
                pos_pct = 0.0
                loc_desc = "At Chennai Central"
            elif train.current_station_id == 2:
                pos_pct = 30.0
                loc_desc = "At Arakkonam"
            elif train.current_station_id == 3:
                pos_pct = 65.0
                loc_desc = "At Katpadi"
            elif train.current_station_id == 4:
                pos_pct = 100.0
                loc_desc = "At Jolarpettai"
        else:
            p = train.progress / 100.0
            if train.current_track_id == 1: # Chennai -> Arakkonam
                pos_pct = p * 30.0
                loc_desc = f"Moving Chennai Central -> Arakkonam ({train.progress:.1f}%)"
            elif train.current_track_id == 2: # Arakkonam -> Katpadi
                pos_pct = 30.0 + p * (65.0 - 30.0)
                loc_desc = f"Moving Arakkonam -> Katpadi ({train.progress:.1f}%)"
            elif train.current_track_id == 3: # Katpadi -> Jolarpettai
                pos_pct = 65.0 + p * (100.0 - 65.0)
                loc_desc = f"Moving Katpadi -> Jolarpettai ({train.progress:.1f}%)"

        # Position marker HTML
        # Offsets slightly to prevent train overlays
        offset_y = (train.train_no % 3) * 12 - 6
        train_markers += f"""
        <div class='train-marker' style='left: {pos_pct}%; transform: translate(-50%, {offset_y}px);' title='Train {train.train_no}'>
            🚆<span class='marker-label'>{train.train_no}</span>
        </div>
        """

        # Fetch predictions
        pred_item = None
        for p in preds:
            if p["train_id"] == train.train_no:
                pred_item = p
                break

        if pred_item:
            del_15 = pred_item["delay_predictions"]["15"]
            del_30 = pred_item["delay_predictions"]["30"]
            del_60 = pred_item["delay_predictions"]["60"]
            conf = int(pred_item["confidence"] * 100)
            factors = ", ".join(pred_item["top_factors"])
            
            conf_color = "var(--accent-green)" if conf >= 80 else ("var(--accent-yellow)" if conf >= 65 else "var(--accent-red)")
            
            trains_cards += f"""
            <div class='train-card'>
                <div class='card-header'>
                    <h3>Train {train.train_no} - {train.name}</h3>
                    <span class='badge badge-blue'>{train.train_type}</span>
                </div>
                <div class='card-body'>
                    <div class='card-info'>
                        <p><strong>Status:</strong> {train.status}</p>
                        <p><strong>Speed:</strong> {train.speed} km/h</p>
                        <p><strong>Position:</strong> {loc_desc}</p>
                        <p><strong>Current Delay:</strong> <span class='delay-high'>{train.delay:.1f} mins</span></p>
                    </div>
                    <div class='predictions-box'>
                        <h4>AI Delay Projections</h4>
                        <div class='pred-row'><span>+15 mins:</span><strong>{del_15:.1f}m</strong></div>
                        <div class='pred-row'><span>+30 mins:</span><strong>{del_30:.1f}m</strong></div>
                        <div class='pred-row'><span>+60 mins:</span><strong>{del_60:.1f}m</strong></div>
                    </div>
                    <div class='conf-box' style='border-color: {conf_color};'>
                        <div class='conf-val' style='color: {conf_color};'>{conf}%</div>
                        <div class='conf-lbl'>Confidence</div>
                    </div>
                </div>
                <div class='card-footer'>
                    <strong>Top Delay Factors:</strong> {factors}
                </div>
            </div>
            """

    # Station occupancy indicators
    c_central = network.get_station_by_id(1)
    arakkonam = network.get_station_by_id(2)
    katpadi = network.get_station_by_id(3)
    jolarpettai = network.get_station_by_id(4)

    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta http-equiv="refresh" content="1">
    <title>RailTwin-Q Live Digital Twin Dashboard</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-color: #0f172a;
            --card-bg: rgba(30, 41, 59, 0.7);
            --border-color: rgba(255, 255, 255, 0.1);
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --accent-purple: #8b5cf6;
            --accent-indigo: #6366f1;
            --accent-green: #10b981;
            --accent-red: #ef4444;
            --accent-yellow: #f59e0b;
        }}
        
        body {{
            background-color: var(--bg-color);
            color: var(--text-main);
            font-family: 'Outfit', sans-serif;
            margin: 0;
            padding: 30px;
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}

        header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 15px;
            margin-bottom: 25px;
        }}

        h1 {{
            margin: 0;
            font-size: 1.8rem;
            background: linear-gradient(135deg, #a855f7, #6366f1);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}

        .sub-header {{
            color: var(--text-muted);
            margin: 3px 0 0 0;
            font-size: 0.95rem;
        }}

        .clock-box {{
            text-align: right;
        }}

        .clock-val {{
            font-size: 1.6rem;
            font-weight: 700;
            color: var(--accent-purple);
        }}

        .clock-tick {{
            font-size: 0.85rem;
            color: var(--text-muted);
        }}

        .badge {{
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 600;
            display: inline-block;
            margin-right: 5px;
        }}

        .badge-green {{ background: rgba(16, 185, 129, 0.15); color: var(--accent-green); border: 1px solid var(--accent-green); }}
        .badge-yellow {{ background: rgba(245, 158, 11, 0.15); color: var(--accent-yellow); border: 1px solid var(--accent-yellow); }}
        .badge-red {{ background: rgba(239, 68, 68, 0.15); color: var(--accent-red); border: 1px solid var(--accent-red); }}
        .badge-blue {{ background: rgba(99, 102, 241, 0.15); color: var(--accent-indigo); border: 1px solid var(--accent-indigo); }}

        /* Schematic Map Styles */
        .map-card {{
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 30px 40px;
            backdrop-filter: blur(10px);
            margin-bottom: 25px;
            position: relative;
        }}

        .map-line-container {{
            position: relative;
            height: 60px;
            margin: 30px 0;
        }}

        .map-line {{
            position: absolute;
            top: 25px;
            left: 0;
            right: 0;
            height: 8px;
            background: rgba(255,255,255,0.1);
            border-radius: 4px;
        }}

        .station-node {{
            position: absolute;
            top: 15px;
            transform: translateX(-50%);
            display: flex;
            flex-direction: column;
            align-items: center;
        }}

        .node-dot {{
            width: 26px;
            height: 26px;
            border-radius: 50%;
            background: var(--accent-indigo);
            border: 4px solid var(--bg-color);
            z-index: 2;
        }}

        .node-label {{
            margin-top: 8px;
            font-size: 0.85rem;
            font-weight: 600;
            color: var(--text-main);
        }}

        .node-plat {{
            font-size: 0.75rem;
            color: var(--text-muted);
            margin-top: 2px;
        }}

        .train-marker {{
            position: absolute;
            top: 18px;
            font-size: 1.4rem;
            z-index: 10;
            cursor: pointer;
            transition: left 0.5s ease-in-out;
        }}

        .marker-label {{
            position: absolute;
            top: -15px;
            left: 50%;
            transform: translateX(-50%);
            font-size: 0.65rem;
            background: rgba(0,0,0,0.8);
            padding: 1px 4px;
            border-radius: 3px;
            font-weight: 600;
        }}

        /* Train Cards Grid */
        .cards-grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 20px;
        }}

        .train-card {{
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 20px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }}

        .card-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 10px;
            margin-bottom: 15px;
        }}

        .card-header h3 {{
            margin: 0;
            font-size: 1.1rem;
            font-weight: 600;
        }}

        .card-body {{
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}

        .card-info {{
            flex-grow: 1;
        }}

        .card-info p {{
            margin: 5px 0;
            font-size: 0.85rem;
            color: var(--text-muted);
        }}

        .card-info p strong {{
            color: var(--text-main);
        }}

        .predictions-box {{
            background: rgba(0,0,0,0.2);
            border-radius: 6px;
            padding: 8px 12px;
            margin-left: 15px;
            min-width: 110px;
        }}

        .predictions-box h4 {{
            margin: 0 0 5px 0;
            font-size: 0.75rem;
            text-transform: uppercase;
            color: var(--accent-purple);
            letter-spacing: 0.05em;
        }}

        .pred-row {{
            display: flex;
            justify-content: space-between;
            font-size: 0.8rem;
            margin-bottom: 3px;
        }}

        .pred-row strong {{
            color: var(--accent-yellow);
        }}

        .conf-box {{
            border: 2px solid;
            border-radius: 50%;
            width: 54px;
            height: 54px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            margin-left: 15px;
        }}

        .conf-val {{
            font-size: 0.95rem;
            font-weight: 700;
        }}

        .conf-lbl {{
            font-size: 0.5rem;
            text-transform: uppercase;
            color: var(--text-muted);
        }}

        .card-footer {{
            border-top: 1px solid var(--border-color);
            margin-top: 15px;
            padding-top: 10px;
            font-size: 0.8rem;
            color: var(--text-muted);
        }}

        .delay-high {{
            color: var(--accent-red);
            font-weight: 600;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div>
                <h1>RailTwin-Q Live Digital Twin</h1>
                <p class="sub-header">Live Railway Map & AI Delay Predictions Dashboard</p>
            </div>
            <div class="clock-box">
                <div class="clock-val">{sim_time_str}</div>
                <div class="clock-tick">Simulation Minute: {tick}</div>
            </div>
        </header>

        <!-- Live Schematic Map -->
        <div class="map-card">
            <h2>Live Schematic Railway Line</h2>
            <div style="margin-bottom: 10px;">
                <strong>Active Anomalies:</strong> {events_badges}
            </div>
            <div class="map-line-container">
                <div class="map-line"></div>
                
                <!-- Chennai Central Node -->
                <div class="station-node" style="left: 0%;">
                    <div class="node-dot"></div>
                    <div class="node-label">Chennai Central</div>
                    <div class="node-plat">Plats: {c_central.platforms_occupied}/{c_central.platforms}</div>
                </div>

                <!-- Arakkonam Node -->
                <div class="station-node" style="left: 30%;">
                    <div class="node-dot"></div>
                    <div class="node-label">Arakkonam</div>
                    <div class="node-plat">Plats: {arakkonam.platforms_occupied}/{arakkonam.platforms}</div>
                </div>

                <!-- Katpadi Node -->
                <div class="station-node" style="left: 65%;">
                    <div class="node-dot"></div>
                    <div class="node-label">Katpadi</div>
                    <div class="node-plat">Plats: {katpadi.platforms_occupied}/{katpadi.platforms}</div>
                </div>

                <!-- Jolarpettai Node -->
                <div class="station-node" style="left: 100%;">
                    <div class="node-dot"></div>
                    <div class="node-label">Jolarpettai</div>
                    <div class="node-plat">Plats: {jolarpettai.platforms_occupied}/{jolarpettai.platforms}</div>
                </div>

                <!-- Dynamic Train Markers -->
                {train_markers}
            </div>
        </div>

        <!-- Train Cards -->
        <h2>Active Train Predictions</h2>
        <div class="cards-grid">
            {trains_cards}
        </div>
    </div>
</body>
</html>
"""

    with open(dashboard_path, "w", encoding="utf-8") as f:
        f.write(html_content)

def main():
    print("=" * 80)
    print("           RAILTWIN-Q: DIGITAL TWIN SIMULATION ENGINE")
    print("=" * 80)

    # 1. Load data
    print("\n[Step 1] Loading railway network data...")
    network = DataLoader.load_network("data")
    print(f" -> Loaded {len(network.stations)} stations.")
    print(f" -> Loaded {len(network.tracks)} tracks.")
    print(f" -> Loaded {len(network.trains)} trains.")

    # 2. Build Railway Graph
    print("\n[Step 2] Building topology graph...")
    graph = GraphBuilder.build_graph(network)
    print(f" -> Nodes (Stations): {graph.number_of_nodes()}")
    print(f" -> Edges (Tracks): {graph.number_of_edges()}")

    # 3. Setup Simulation Timeline & Events
    # Scheduled events:
    # - Heavy Rain: starts at minute 15, duration 40
    # - Signal Failure: starts at Arakkonam (ID: 2), minute 45, duration 25
    scheduled_rain = HeavyRainEvent(intensity=0.8, duration=40)
    scheduled_rain.start_tick = 15
    scheduled_rain.active = False
    
    scheduled_signal = SignalFailureEvent(station_id=2, duration=25)
    scheduled_signal.start_tick = 45
    scheduled_signal.active = False

    active_events = []
    
    # 4. Initialize Prediction Service
    print("\n[Step 4] Initializing AI Delay Prediction Subsystem...")
    from ai.delay_prediction.predictor import PredictionService
    prediction_service = PredictionService()

    # 5. Initialize Prediction History Log files
    history_log_path = os.path.join("datasets", "prediction_history.csv")
    decision_log_path = os.path.join("datasets", "prediction_decisions.jsonl")
    
    # If file exists, delete to record fresh run, or preserve
    if os.path.exists(history_log_path):
        os.remove(history_log_path)
    if os.path.exists(decision_log_path):
        os.remove(decision_log_path)

    # We will log: [tick, train_id, pred_15, pred_30, pred_60, actual_15, actual_30, actual_60]
    # We maintain a buffer to map actual delays as simulation ticks advance
    # buffer format: (train_id, target_tick) -> predicted_value
    pred_buffer_15 = {}
    pred_buffer_30 = {}
    pred_buffer_60 = {}
    
    history_records_to_write = []

    print("\n[Step 5] Starting simulation loop (120 minutes)...")
    time.sleep(1)

    for tick in range(121):
        sim_time_str = format_time(tick)
        
        # Trigger scheduled events
        if tick == scheduled_rain.start_tick:
            scheduled_rain.active = True
            active_events.append(scheduled_rain)
            print(f"\n[EVENT] DISRUPTION EVENT TRIGGERED: Heavy Rain (Intensity: {scheduled_rain.intensity:.1f})")
            
        if tick == scheduled_signal.start_tick:
            scheduled_signal.active = True
            active_events.append(scheduled_signal)
            print(f"\n[EVENT] DISRUPTION EVENT TRIGGERED: Signal Failure at Arakkonam (Station ID: 2)")

        # Update active events
        for event in active_events:
            if event.active:
                event.tick()

        # Advance movement
        MovementEngine.tick(network, active_events, tick)

        # Update occupancy calculations
        StateEngine.update_occupancies(network)

        # Record snapshot
        snapshot = StateEngine.record_snapshot(network, sim_time_str, active_events)

        # Run delay prediction for this tick
        preds = prediction_service.get_predictions_for_tick(network, tick, sim_time_str, active_events)

        # Check and resolve historical actuals
        # If tick = T, we check if we had predictions for T - 15, T - 30, or T - 60
        train_map = {t.train_no: t for t in network.trains}
        
        # Record current tick predictions into history buffers and log decisions
        for p in preds:
            train_id = p["train_id"]
            p_15 = p["delay_predictions"]["15"]
            p_30 = p["delay_predictions"]["30"]
            p_60 = p["delay_predictions"]["60"]
            conf = p["confidence"]
            factors = p["top_factors"]
            state_id = p["state_id"]

            # Store in buffer
            pred_buffer_15[(train_id, tick + 15)] = p_15
            pred_buffer_30[(train_id, tick + 30)] = p_30
            pred_buffer_60[(train_id, tick + 60)] = p_60

            # Write to JSONL Decision Log
            decision_entry = {
                "tick": tick,
                "state_id": state_id,
                "train_id": train_id,
                "predictions": {
                    "15": p_15,
                    "30": p_30,
                    "60": p_60
                },
                "confidence": conf,
                "top_factors": factors,
                "model_version": p["model_version"]
            }
            with open(decision_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(decision_entry) + "\n")

        # Now, resolve historical values
        for train_id, train_obj in train_map.items():
            current_actual_delay = train_obj.delay
            
            # Resolve delay_15 prediction made at tick - 15
            t_15 = tick - 15
            if (train_id, tick) in pred_buffer_15:
                pred_val = pred_buffer_15[(train_id, tick)]
                history_records_to_write.append({
                    "tick": t_15,
                    "train_id": train_id,
                    "prediction_horizon": 15,
                    "predicted_delay": pred_val,
                    "actual_delay": current_actual_delay
                })
                del pred_buffer_15[(train_id, tick)]

            # Resolve delay_30 prediction made at tick - 30
            t_30 = tick - 30
            if (train_id, tick) in pred_buffer_30:
                pred_val = pred_buffer_30[(train_id, tick)]
                history_records_to_write.append({
                    "tick": t_30,
                    "train_id": train_id,
                    "prediction_horizon": 30,
                    "predicted_delay": pred_val,
                    "actual_delay": current_actual_delay
                })
                del pred_buffer_30[(train_id, tick)]

            # Resolve delay_60 prediction made at tick - 60
            t_60 = tick - 60
            if (train_id, tick) in pred_buffer_60:
                pred_val = pred_buffer_60[(train_id, tick)]
                history_records_to_write.append({
                    "tick": t_60,
                    "train_id": train_id,
                    "prediction_horizon": 60,
                    "predicted_delay": pred_val,
                    "actual_delay": current_actual_delay
                })
                del pred_buffer_60[(train_id, tick)]

        # Generate HTML visual web dashboard
        generate_web_dashboard(network, tick, sim_time_str, active_events, preds)

        # Render status update every 15 minutes (or on events)
        if tick % 15 == 0 or tick in [15, 45, 55, 80]:
            print("\n" + "-"*80)
            print(f"Time: {sim_time_str} | Active Events: {[e.name for e in active_events if e.active]}")
            print("-"*80)
            
            # Print train list status
            print("Train Statuses & Delay Predictions:")
            for train in network.trains:
                loc_str = ""
                if train.progress == 0.0:
                    station = network.get_station_by_id(train.current_station_id)
                    loc_str = f"At {station.name}"
                    if train.dwell_time_remaining > 0:
                        loc_str += f" (Dwelling, remaining: {train.dwell_time_remaining}m)"
                    elif train.status == "WAITING":
                        loc_str += " (Waiting to depart)"
                else:
                    track = network.get_track_by_id(train.current_track_id)
                    src = network.get_station_by_id(track.source_station_id).name
                    dest = network.get_station_by_id(track.destination_station_id).name
                    loc_str = f"Moving {src} -> {dest} (Progress: {train.progress:.1f}%)"

                # Find predictions
                pred_str = ""
                for p in preds:
                    if p["train_id"] == train.train_no:
                        pred_str = f" | AI Pred: +15m: {p['delay_predictions']['15']}m, +30m: {p['delay_predictions']['30']}m, +60m: {p['delay_predictions']['60']}m (Conf: {int(p['confidence']*100)}%)"
                        break

                print(f" * Train {train.train_no} ({train.name}): Status: {train.status} | Speed: {train.speed} km/h | {loc_str} | Delay: {train.delay} min{pred_str}")

            # Render Schematic
            render_ascii_map(network)
            time.sleep(0.1) # short pause for readability

    # Write prediction history logs to CSV
    if history_records_to_write:
        df_hist = pd.DataFrame(history_records_to_write)
        # Sort values
        df_hist.sort_values(by=["tick", "train_id", "prediction_horizon"], inplace=True)
        df_hist.to_csv(history_log_path, index=False)
        print(f"\n[Step 6] Saved prediction history to {history_log_path} ({len(df_hist)} rows logged).")

    # 4. Summarize results
    print("\n" + "=" * 80)
    print("           SIMULATION COMPLETED SUCCESSFULLY")
    print(f" -> Recorded {len(StateEngine.history)} chronological state snapshots.")
    print(f" -> Final average train delay: {StateEngine.history[-1]['average_delay']} mins.")
    print(f" -> Visual dashboard generated at datasets/dashboard.html (open in browser).")
    print(f" -> Decision logs generated at datasets/prediction_decisions.jsonl.")
    print("=" * 80)

if __name__ == "__main__":
    main()
