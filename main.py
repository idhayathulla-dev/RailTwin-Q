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

def generate_web_dashboard(network, tick: int, sim_time_str: str, active_events: list, preds: list, preds_cong: dict, preds_prop: dict = None):
    """
    Generates a premium dark-themed HTML visual web dashboard updated in real-time.
    Written to datasets/dashboard.html
    """
    os.makedirs("datasets", exist_ok=True)
    dashboard_path = os.path.join("datasets", "dashboard.html")

    # 1. Format weather & events list
    events_badges = ""
    for ev in active_events:
        if ev.active:
            color = "red" if "Failure" in ev.name or "Block" in ev.name or "Maint" in ev.name else "yellow"
            events_badges += f"<span class='badge badge-{color}'>{ev.name}</span>"
    if not events_badges:
        events_badges = "<span class='badge badge-green'>Normal Operations</span>"

    # 2. Extract Layer 3 Predictions for Node/Track Colorings (+30m horizon)
    p_st = preds_cong.get(30, {}).get("predicted_stations", {})
    p_tr = preds_cong.get(30, {}).get("predicted_tracks", {})
    p_net = preds_cong.get(30, {}).get("predicted_network", {})

    # Helper to return visual colors based on predicted +30m congestion levels
    def get_color_class(val):
        if val >= 75.0: return "cong-high"
        elif val >= 45.0: return "cong-medium"
        return "cong-low"

    def get_color_hex(val):
        if val >= 75.0: return "var(--accent-red)"
        elif val >= 45.0: return "var(--accent-yellow)"
        return "var(--accent-green)"

    # Get station predictions +30m
    st_c_1 = p_st.get(1, {}).get("congestion", 0.0)
    st_c_2 = p_st.get(2, {}).get("congestion", 0.0)
    st_c_3 = p_st.get(3, {}).get("congestion", 0.0)
    st_c_4 = p_st.get(4, {}).get("congestion", 0.0)

    # Get track predictions +30m
    tr_o_1 = p_tr.get(1, {}).get("occupancy", 0.0)
    tr_o_2 = p_tr.get(2, {}).get("occupancy", 0.0)
    tr_o_3 = p_tr.get(3, {}).get("occupancy", 0.0)

    # Get station prediction intervals for map tags
    st_pi_1 = p_st.get(1, {}).get("prediction_interval", [0, 0])
    st_pi_2 = p_st.get(2, {}).get("prediction_interval", [0, 0])
    st_pi_3 = p_st.get(3, {}).get("prediction_interval", [0, 0])
    st_pi_4 = p_st.get(4, {}).get("prediction_interval", [0, 0])

    # 3. Compute train positions on the schematic line (0% to 100%)
    train_markers = ""
    trains_cards = ""
    for train in network.trains:
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

        # Position marker HTML (offsets vertically to prevent overlaps)
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

    # Format Network predictions targets
    net_c_15 = p_net.get("network_congestion", 0.0)
    net_c_30 = p_net.get("network_congestion_30", 0.0)
    net_c_60 = p_net.get("network_congestion_60", 0.0)
    
    net_p_30 = p_net.get("platform_utilization_30", 0.0)
    net_t_30 = p_net.get("track_utilization_30", 0.0)
    net_d_30 = p_net.get("average_delay_30", 0.0)
    net_si_30 = p_net.get("stress_index_30", 0.0)
    net_conf = int(p_net.get("confidence", 0.8) * 100)
    net_pi = p_net.get("prediction_interval", [0, 0])

    # 4. Spatio-Temporal Disruption Propagation Parsing (Layer 4)
    if preds_prop:
        f_state = preds_prop.get("future_state", {})
        csi_val = f_state.get("cascade_severity_index", 0.0)
        severity_level = f_state.get("severity_level", "Normal")
        
        expected_rec = preds_prop.get("expected_recovery", {})
        rec_time = expected_rec.get("expected_recovery_time_mins", 0)
        prop_conf = int(expected_rec.get("confidence", 95.0))
        uncertainty_val = preds_prop.get("uncertainty", {}).get("uncertainty_score", 0.05)
        
        # Color based on CSI
        if csi_val >= 60.0: csi_color = "var(--accent-red)"
        elif csi_val >= 40.0: csi_color = "var(--accent-yellow)"
        else: csi_color = "var(--accent-green)"
        
        # Critical nodes list
        c_trains = preds_prop.get("critical_nodes", {}).get("critical_trains", [])[:2]
        c_stations = preds_prop.get("critical_nodes", {}).get("critical_stations", [])[:2]
        c_tracks = preds_prop.get("critical_nodes", {}).get("critical_tracks", [])[:2]
        
        critical_nodes_html = ""
        for ct in c_trains:
            critical_nodes_html += f"<li>🚂 Train {ct['id']}: Score <strong>{ct['criticality_score']}</strong> (Delay: {ct['delay']:.0f}m)</li>"
        for cs in c_stations:
            critical_nodes_html += f"<li>🚉 Station {cs['name']}: Score <strong>{cs['criticality_score']}</strong> (Congestion: {cs['congestion']:.0f}%)</li>"
        for ck in c_tracks:
            critical_nodes_html += f"<li>🛤️ Track {ck['id']}: Score <strong>{ck['criticality_score']}</strong> (Occupancy: {ck['occupancy']:.0f}%)</li>"
            
        # Risk scores heatmap proxy
        train_risks = preds_prop.get("risk_scores", {}).get("train_risks", [])[:2]
        station_risks = preds_prop.get("risk_scores", {}).get("station_risks", [])[:2]
        track_risks = preds_prop.get("risk_scores", {}).get("track_risks", [])[:2]
        
        risk_nodes_html = ""
        for tr in train_risks:
            risk_nodes_html += f"<li>🚂 Train {tr['id']}: Risk <strong>{tr['risk_score']}</strong></li>"
        for sr in station_risks:
            risk_nodes_html += f"<li>🚉 Station {sr['name']}: Risk <strong>{sr['risk_score']}</strong></li>"
        for tk in track_risks:
            risk_nodes_html += f"<li>🛤️ Track {tk['id']}: Risk <strong>{tk['risk_score']}</strong></li>"
            
        # Format Causal path explorer list
        timeline_html = ""
        for exp in preds_prop.get("root_cause_tree", []):
            timeline_html += f"<div style='margin-bottom: 8px; border-left: 2px solid var(--accent-purple); padding-left: 6px;'>{exp['root_cause']}</div>"
        if not timeline_html:
            timeline_html = "<div style='color: var(--text-muted);'>No disruption anomalies active. Network operating normally.</div>"

        # 1. Decision Reasoning trace list (Improvement 1)
        reasoning = preds_prop.get("decision_reasoning", {})
        reasoning_html = ""
        for aid, item in list(reasoning.items())[:2]:
            chain_text = " &rarr; ".join(item["reasoning_chain"])
            reasoning_html += f"<div style='margin-bottom: 8px; font-size:0.75rem; border-left: 2px solid var(--accent-indigo); padding-left: 6px;'><strong>Action {aid} Reason:</strong> {chain_text}</div>"
        if not reasoning_html:
            reasoning_html = "<div style='color: var(--text-muted);'>No active reasoning chain.</div>"

        # 2. Counterfactual what-if analysis (Improvement 2)
        cf_scenarios = preds_prop.get("counterfactual_analysis", {})
        cf_html = "<table style='width: 100%; border-collapse: collapse; font-size: 0.8rem; text-align: left;'>"
        cf_html += "<tr style='border-bottom: 1px solid var(--border-color); color: var(--text-muted);'><th>What-If Option</th><th>Recovery</th><th>Saving</th></tr>"
        for name, details in list(cf_scenarios.items())[:4]:
            cf_html += f"<tr style='border-bottom: 1px dashed rgba(255,255,255,0.05);'><td>{name.replace('Scenario ', '')}</td><td><strong>{details['recovery_time_mins']}m</strong></td><td style='color: var(--accent-green);'>-{details['delay_reduction_minutes']:.1f}m</td></tr>"
        cf_html += "</table>"

        # 3. Passenger impact stats (Improvement 4)
        p_impact = preds_prop.get("passenger_impact", {})
        passenger_html = "<ul style='list-style: none; padding: 0; margin: 0; font-size: 0.8rem; line-height: 1.5; color: var(--text-muted);'>"
        for aid, detail in list(p_impact.items())[:2]:
            passenger_html += f"<li>Action {aid}: Delayed: <strong>{detail['passengers_delayed']}</strong> | Saved: <strong style='color:var(--accent-green);'>{detail['passengers_saved']}</strong></li>"
        passenger_html += "</ul>"

        # 4. Pareto optimal frontier list (Improvement 8)
        pareto = preds_prop.get("pareto_front", {})
        pareto_html = "<ul style='list-style: none; padding: 0; margin: 0; font-size: 0.8rem; line-height: 1.5; color: var(--text-muted);'>"
        for item in pareto.get("pareto_solutions", [])[:2]:
            pareto_html += f"<li>Sol {item['action_id']} ({item['action']}): Risk: <strong>{item['risk']:.2f}</strong> | Energy: <strong>{item['energy']:.2f}</strong></li>"
        pareto_html += "</ul>"

        # 5. Robustness rating (Improvement 6)
        robust = preds_prop.get("robustness_report", {})
        robustness_html = "<ul style='list-style: none; padding: 0; margin: 0; font-size: 0.8rem; line-height: 1.5; color: var(--text-muted);'>"
        for aid, detail in list(robust.items())[:2]:
            robustness_html += f"<li>Action {aid}: Robustness: <strong style='color:var(--accent-green);'>{detail['robustness_score']}%</strong></li>"
        robustness_html += "</ul>"

        # 6. Optimization Search Space constraints & variables count (Improvement 12)
        search_space = preds_prop.get("optimization_search_space", {})
        variables_count = len(search_space.get("decision_variables", {}))
        readiness_status = "READY" if variables_count > 0 else "PENDING"

        # Format Candidate Actions & Cost Vectors (Improvement 3 & 8)
        impacts = preds_prop.get("decision_impact_graph", {})
        actions_html = "<table style='width: 100%; border-collapse: collapse; font-size: 0.8rem; text-align: left;'>"
        actions_html += "<tr style='border-bottom: 1px solid var(--border-color); color: var(--text-muted);'><th>Action</th><th>Target</th><th>Saving</th></tr>"
        for aid, details in list(impacts.items())[:3]:
            actions_html += f"<tr style='border-bottom: 1px dashed rgba(255,255,255,0.05);'><td><span class='badge badge-blue' style='padding: 2px 6px; font-size: 0.7rem;'>{details['action']}</span></td><td>{details['target']}</td><td style='color: var(--accent-green);'>-{details['expected_effects']['delay_reduction_minutes']:.1f}m</td></tr>"
        actions_html += "</table>"
    else:
        csi_val = 0.0
        severity_level = "Normal"
        rec_time = 0
        prop_conf = 95
        uncertainty_val = 0.05
        csi_color = "var(--accent-green)"
        critical_nodes_html = "<li>No critical bottlenecks.</li>"
        risk_nodes_html = "<li>No risk elements.</li>"
        timeline_html = "<div style='color: var(--text-muted);'>No propagation trace active.</div>"
        cf_html = "<div style='color: var(--text-muted);'>No scenario counterfactuals.</div>"
        actions_html = "<div style='color: var(--text-muted);'>No actions generated.</div>"
        reasoning_html = "<div style='color: var(--text-muted);'>No reasoning trace.</div>"
        passenger_html = "<li>No passenger statistics.</li>"
        pareto_html = "<li>No pareto solutions.</li>"
        robustness_html = "<li>No robustness scores.</li>"
        variables_count = 0
        readiness_status = "PENDING"

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
            max-width: 1250px;
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
            height: 90px;
            margin: 40px 0;
        }}

        .map-line-segment {{
            position: absolute;
            top: 25px;
            height: 8px;
            border-radius: 4px;
            z-index: 1;
        }}

        .station-node {{
            position: absolute;
            top: 15px;
            transform: translateX(-50%);
            display: flex;
            flex-direction: column;
            align-items: center;
            z-index: 3;
        }}

        .node-dot {{
            width: 26px;
            height: 26px;
            border-radius: 50%;
            border: 4px solid var(--bg-color);
        }}

        .node-label {{
            margin-top: 8px;
            font-size: 0.85rem;
            font-weight: 600;
            color: var(--text-main);
            text-align: center;
        }}

        .node-plat {{
            font-size: 0.75rem;
            color: var(--text-muted);
            margin-top: 2px;
        }}

        .node-pred-cong {{
            font-size: 0.75rem;
            margin-top: 3px;
            font-weight: 600;
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

        /* Congestion Color Indicators */
        .cong-low {{ background-color: var(--accent-green); color: var(--accent-green); }}
        .cong-medium {{ background-color: var(--accent-yellow); color: var(--accent-yellow); }}
        .cong-high {{ background-color: var(--accent-red); color: var(--accent-red); }}

        /* Top Indicators Grid */
        .top-stats-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 20px;
            margin-bottom: 25px;
        }}

        .stat-card {{
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            position: relative;
        }}

        .stat-val {{
            font-size: 2rem;
            font-weight: 700;
            color: var(--accent-purple);
            margin-top: 5px;
        }}

        .stat-lbl {{
            color: var(--text-muted);
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-top: 5px;
        }}

        .stat-interval {{
            font-size: 0.75rem;
            color: var(--accent-yellow);
            margin-top: 3px;
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
                <p class="sub-header">Hierarchical AI Congestion & Delay Optimization Dashboard</p>
            </div>
            <div class="clock-box">
                <div class="clock-val">{sim_time_str}</div>
                <div class="clock-tick">Simulation Minute: {tick}</div>
            </div>
        </header>

        <!-- Top Predicted Network Metrics -->
        <div class="top-stats-grid">
            <div class="stat-card">
                <div class="stat-val" style="color: {get_color_hex(net_c_30)};">{net_c_30:.1f}%</div>
                <div class="stat-lbl">Future Congestion (+30m)</div>
                <div class="stat-interval">Interval: [{net_pi[0]:.1f}% - {net_pi[1]:.1f}%]</div>
            </div>
            <div class="stat-card">
                <div class="stat-val">{net_p_30:.1f}%</div>
                <div class="stat-lbl">Platform Util (+30m)</div>
                <div class="stat-interval">Active Platforms: {c_central.platforms_occupied + arakkonam.platforms_occupied + katpadi.platforms_occupied + jolarpettai.platforms_occupied}</div>
            </div>
            <div class="stat-card">
                <div class="stat-val">{net_t_30:.1f}%</div>
                <div class="stat-lbl">Track Util (+30m)</div>
                <div class="stat-interval">Active Segments: 3</div>
            </div>
            <div class="stat-card">
                <div class="stat-val" style="color: var(--accent-purple);">{net_si_30:.1f}</div>
                <div class="stat-lbl">Network Stress Index (+30m)</div>
                <div class="stat-interval">Ensemble Confidence: {net_conf}%</div>
            </div>
        </div>

        <!-- Live Schematic Map -->
        <div class="map-card">
            <h2>Live Spatial Congestion Mapping (+30m AI predictions)</h2>
            <div style="margin-bottom: 10px;">
                <strong>Active Anomalies:</strong> {events_badges}
            </div>
            
            <div class="map-line-container">
                <!-- Track Line segments (colored dynamically by predicted occupancy +30m) -->
                <div class="map-line-segment {get_color_class(tr_o_1)}" style="left: 0%; width: 30%;"></div>
                <div class="map-line-segment {get_color_class(tr_o_2)}" style="left: 30%; width: 35%;"></div>
                <div class="map-line-segment {get_color_class(tr_o_3)}" style="left: 65%; width: 35%;"></div>
                
                <!-- Chennai Central Node -->
                <div class="station-node" style="left: 0%;">
                    <div class="node-dot {get_color_class(st_c_1)}"></div>
                    <div class="node-label">Chennai Central</div>
                    <div class="node-plat">Plats: {c_central.platforms_occupied}/{c_central.platforms}</div>
                    <div class="node-pred-cong" style="color: {get_color_hex(st_c_1)};">AI: {st_c_1:.0f}% [{st_pi_1[0]:.0f}%-{st_pi_1[1]:.0f}%]</div>
                </div>

                <!-- Arakkonam Node -->
                <div class="station-node" style="left: 30%;">
                    <div class="node-dot {get_color_class(st_c_2)}"></div>
                    <div class="node-label">Arakkonam</div>
                    <div class="node-plat">Plats: {arakkonam.platforms_occupied}/{arakkonam.platforms}</div>
                    <div class="node-pred-cong" style="color: {get_color_hex(st_c_2)};">AI: {st_c_2:.0f}% [{st_pi_2[0]:.0f}%-{st_pi_2[1]:.0f}%]</div>
                </div>

                <!-- Katpadi Node -->
                <div class="station-node" style="left: 65%;">
                    <div class="node-dot {get_color_class(st_c_3)}"></div>
                    <div class="node-label">Katpadi</div>
                    <div class="node-plat">Plats: {katpadi.platforms_occupied}/{katpadi.platforms}</div>
                    <div class="node-pred-cong" style="color: {get_color_hex(st_c_3)};">AI: {st_c_3:.0f}% [{st_pi_3[0]:.0f}%-{st_pi_3[1]:.0f}%]</div>
                </div>

                <!-- Jolarpettai Node -->
                <div class="station-node" style="left: 100%;">
                    <div class="node-dot {get_color_class(st_c_4)}"></div>
                    <div class="node-label">Jolarpettai</div>
                    <div class="node-plat">Plats: {jolarpettai.platforms_occupied}/{jolarpettai.platforms}</div>
                    <div class="node-pred-cong" style="color: {get_color_hex(st_c_4)};">AI: {st_c_4:.0f}% [{st_pi_4[0]:.0f}%-{st_pi_4[1]:.0f}%]</div>
                </div>

                <!-- Dynamic Train Markers -->
                {train_markers}
            </div>
        </div>

        <!-- Disruption Propagation & Train Cards Grid -->
        <div class="main-layout-grid" style="display: grid; grid-template-columns: 2fr 1fr; gap: 20px; margin-top: 25px;">
            <div>
                <h2>Active Train Delay Forecasts</h2>
                <div class="cards-grid" style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
                    {trains_cards}
                </div>
            </div>
            
            <div style="background: var(--card-bg); border: 1px solid var(--border-color); border-radius: 12px; padding: 20px; display: flex; flex-direction: column; overflow-y: auto; max-height: 800px;">
                <h2 style="margin-top: 0; border-bottom: 1px solid var(--border-color); padding-bottom: 10px; color: var(--accent-indigo);">Layer 4: Decision Intelligence</h2>
                
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 15px;">
                    <div style="background: rgba(255,255,255,0.02); padding: 10px; border-radius: 8px; border: 1px solid var(--border-color); text-align: center;">
                        <div style="font-size: 1.3rem; font-weight: 700; color: {csi_color};">{csi_val:.1f}%</div>
                        <div style="font-size: 0.6rem; color: var(--text-muted); text-transform: uppercase; margin-top: 2px;">Severity Index (CSI)</div>
                        <div style="font-size: 0.7rem; font-weight: 600; color: {csi_color}; margin-top: 2px;">{severity_level}</div>
                    </div>
                    <div style="background: rgba(255,255,255,0.02); padding: 10px; border-radius: 8px; border: 1px solid var(--border-color); text-align: center;">
                        <div style="font-size: 1.3rem; font-weight: 700; color: var(--accent-purple);">{rec_time} mins</div>
                        <div style="font-size: 0.6rem; color: var(--text-muted); text-transform: uppercase; margin-top: 2px;">Est. Recovery</div>
                        <div style="font-size: 0.7rem; color: var(--text-muted); margin-top: 2px;">Uncertainty: {uncertainty_val:.1%}</div>
                    </div>
                </div>

                <div style="background: rgba(99, 102, 241, 0.1); border: 1px solid var(--accent-indigo); border-radius: 6px; padding: 8px; margin-bottom: 15px; text-align: center; font-size: 0.8rem; font-weight: 600;">
                    Optimization Readiness: <span style="color:#10b981;">{readiness_status}</span> ({variables_count} variables compiled)
                </div>

                <h3 style="margin-top: 0; font-size: 0.95rem; color: var(--text-main); margin-bottom: 6px;">Top Critical Bottlenecks</h3>
                <ul style="list-style: none; padding: 0; margin: 0 0 15px 0; font-size: 0.8rem; line-height: 1.5; color: var(--text-muted);">
                    {critical_nodes_html}
                </ul>

                <h3 style="margin-top: 0; font-size: 0.95rem; color: var(--text-main); margin-bottom: 6px;">Decision Reasoning Pathways</h3>
                <div style="margin-bottom: 15px; background: rgba(0,0,0,0.15); padding: 8px; border-radius: 6px;">
                    {reasoning_html}
                </div>

                <h3 style="margin-top: 0; font-size: 0.95rem; color: var(--text-main); margin-bottom: 6px;">What-If Counterfactual Scenarios</h3>
                <div style="margin-bottom: 15px; background: rgba(0,0,0,0.15); padding: 8px; border-radius: 6px;">
                    {cf_html}
                </div>

                <h3 style="margin-top: 0; font-size: 0.95rem; color: var(--text-main); margin-bottom: 6px;">Passenger Impact Projections</h3>
                <div style="margin-bottom: 15px; background: rgba(0,0,0,0.15); padding: 8px; border-radius: 6px;">
                    {passenger_html}
                </div>

                <h3 style="margin-top: 0; font-size: 0.95rem; color: var(--text-main); margin-bottom: 6px;">Robustness & Resilience Ratings</h3>
                <div style="margin-bottom: 15px; background: rgba(0,0,0,0.15); padding: 8px; border-radius: 6px;">
                    {robustness_html}
                </div>

                <h3 style="margin-top: 0; font-size: 0.95rem; color: var(--text-main); margin-bottom: 6px;">Pareto Optimal Frontier</h3>
                <div style="margin-bottom: 15px; background: rgba(0,0,0,0.15); padding: 8px; border-radius: 6px;">
                    {pareto_html}
                </div>

                <h3 style="margin-top: 0; font-size: 0.95rem; color: var(--text-main); margin-bottom: 6px;">Candidate Optimization Actions</h3>
                <div style="margin-bottom: 15px; background: rgba(0,0,0,0.15); padding: 8px; border-radius: 6px;">
                    {actions_html}
                </div>

                <h3 style="margin-top: 0; font-size: 0.95rem; color: var(--text-main); margin-bottom: 6px;">Causal Disruption Path</h3>
                <div style="background: rgba(0,0,0,0.2); padding: 8px; border-radius: 6px; font-size: 0.75rem; overflow-y: auto; max-height: 120px; line-height: 1.3;">
                    {timeline_html}
                </div>
            </div>
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
    
    # 4. Initialize Prediction Services
    print("\n[Step 4] Initializing AI Delay & Congestion Prediction Subsystems...")
    from ai.delay_prediction.predictor import PredictionService as DelayPredictionService
    from ai.congestion_prediction.predictor import PredictionService as CongestionPredictionService
    from ai.delay_propagation.predictor import DelayPropagationPredictor
    
    delay_predictor = DelayPredictionService()
    congestion_predictor = CongestionPredictionService()
    propagation_predictor = DelayPropagationPredictor()

    # 5. Initialize Prediction History Log files
    history_log_path = os.path.join("datasets", "prediction_history.csv")
    decision_log_path = os.path.join("datasets", "prediction_decisions.jsonl")
    
    congestion_history_path = os.path.join("datasets", "congestion_history.csv")
    congestion_decision_path = os.path.join("datasets", "congestion_decisions.jsonl")
    
    if os.path.exists(history_log_path): os.remove(history_log_path)
    if os.path.exists(decision_log_path): os.remove(decision_log_path)
    if os.path.exists(congestion_history_path): os.remove(congestion_history_path)
    if os.path.exists(congestion_decision_path): os.remove(congestion_decision_path)

    # Delay tracking buffers
    pred_buffer_15 = {}
    pred_buffer_30 = {}
    pred_buffer_60 = {}
    
    # Congestion tracking buffers
    cong_st_buffer_15 = {}
    cong_st_buffer_30 = {}
    cong_st_buffer_60 = {}
    
    cong_tr_buffer_15 = {}
    cong_tr_buffer_30 = {}
    cong_tr_buffer_60 = {}
    
    cong_net_buffer_15 = {}
    cong_net_buffer_30 = {}
    cong_net_buffer_60 = {}
    
    history_records_to_write = []
    congestion_history_records = []

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

        # -------------------------------------------------------------
        # RUN AI DELAY & HIERARCHICAL CONGESTION PREDICTORS
        # -------------------------------------------------------------
        preds_delay = delay_predictor.get_predictions_for_tick(network, tick, sim_time_str, active_events)
        preds_congestion = congestion_predictor.get_predictions_for_tick(network, tick, sim_time_str, active_events, preds_delay)
        preds_propagation = propagation_predictor.get_predictions_for_tick(network, tick, sim_time_str, active_events, preds_delay, preds_congestion)

        # Log Delay Decisions
        for p in preds_delay:
            train_id = p["train_id"]
            p_15 = p["delay_predictions"]["15"]
            p_30 = p["delay_predictions"]["30"]
            p_60 = p["delay_predictions"]["60"]
            conf = p["confidence"]
            factors = p["top_factors"]
            state_id = p["state_id"]

            pred_buffer_15[(train_id, tick + 15)] = p_15
            pred_buffer_30[(train_id, tick + 30)] = p_30
            pred_buffer_60[(train_id, tick + 60)] = p_60

            decision_entry = {
                "tick": tick,
                "state_id": state_id,
                "train_id": train_id,
                "predictions": {"15": p_15, "30": p_30, "60": p_60},
                "confidence": conf,
                "top_factors": factors,
                "model_version": p["model_version"]
            }
            with open(decision_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(decision_entry) + "\n")

        # Log Hierarchical Congestion Decisions (FutureNetworkState dict output contract)
        # Store future predictions in buffers to align with historical actuals later
        for horizon in [15, 30, 60]:
            f_state = preds_congestion.get(horizon, {})
            # Write structured FutureNetworkState directly to decisions log
            with open(congestion_decision_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(f_state) + "\n")

            # Buffer stations
            for s_id, p_det in f_state.get("predicted_stations", {}).items():
                if horizon == 15: cong_st_buffer_15[(s_id, tick + 15)] = p_det["congestion"]
                elif horizon == 30: cong_st_buffer_30[(s_id, tick + 30)] = p_det["congestion"]
                elif horizon == 60: cong_st_buffer_60[(s_id, tick + 60)] = p_det["congestion"]

            # Buffer tracks
            for tr_id, p_det in f_state.get("predicted_tracks", {}).items():
                if horizon == 15: cong_tr_buffer_15[(tr_id, tick + 15)] = p_det["occupancy"]
                elif horizon == 30: cong_tr_buffer_30[(tr_id, tick + 30)] = p_det["occupancy"]
                elif horizon == 60: cong_tr_buffer_60[(tr_id, tick + 60)] = p_det["occupancy"]

            # Buffer network
            net_c = f_state.get("predicted_network", {}).get("network_congestion", 0.0)
            if horizon == 15: cong_net_buffer_15[tick + 15] = net_c
            elif horizon == 30: cong_net_buffer_30[tick + 30] = net_c
            elif horizon == 60: cong_net_buffer_60[tick + 60] = net_c

        # -------------------------------------------------------------
        # ALIGN HISTORICAL PREDICTIONS WITH OBSERVED ACTUALS
        # -------------------------------------------------------------
        train_map = {t.train_no: t for t in network.trains}
        for train_id, train_obj in train_map.items():
            current_actual_delay = train_obj.delay
            if (train_id, tick) in pred_buffer_15:
                history_records_to_write.append({"tick": tick - 15, "train_id": train_id, "prediction_horizon": 15, "predicted_delay": pred_buffer_15[(train_id, tick)], "actual_delay": current_actual_delay})
                del pred_buffer_15[(train_id, tick)]
            if (train_id, tick) in pred_buffer_30:
                history_records_to_write.append({"tick": tick - 30, "train_id": train_id, "prediction_horizon": 30, "predicted_delay": pred_buffer_30[(train_id, tick)], "actual_delay": current_actual_delay})
                del pred_buffer_30[(train_id, tick)]
            if (train_id, tick) in pred_buffer_60:
                history_records_to_write.append({"tick": tick - 60, "train_id": train_id, "prediction_horizon": 60, "predicted_delay": pred_buffer_60[(train_id, tick)], "actual_delay": current_actual_delay})
                del pred_buffer_60[(train_id, tick)]

        # Align Congestions
        for s in network.stations:
            act_c = s.station_congestion_score
            if (s.station_id, tick) in cong_st_buffer_15:
                congestion_history_records.append({"tick": tick - 15, "entity_type": "station", "entity_id": s.station_id, "horizon": 15, "predicted_val": cong_st_buffer_15[(s.station_id, tick)], "actual_val": act_c})
                del cong_st_buffer_15[(s.station_id, tick)]
            if (s.station_id, tick) in cong_st_buffer_30:
                congestion_history_records.append({"tick": tick - 30, "entity_type": "station", "entity_id": s.station_id, "horizon": 30, "predicted_val": cong_st_buffer_30[(s.station_id, tick)], "actual_val": act_c})
                del cong_st_buffer_30[(s.station_id, tick)]
            if (s.station_id, tick) in cong_st_buffer_60:
                congestion_history_records.append({"tick": tick - 60, "entity_type": "station", "entity_id": s.station_id, "horizon": 60, "predicted_val": cong_st_buffer_60[(s.station_id, tick)], "actual_val": act_c})
                del cong_st_buffer_60[(s.station_id, tick)]

        for tr in network.tracks:
            act_o = tr.occupancy_percent
            if (tr.track_id, tick) in cong_tr_buffer_15:
                congestion_history_records.append({"tick": tick - 15, "entity_type": "track", "entity_id": tr.track_id, "horizon": 15, "predicted_val": cong_tr_buffer_15[(tr.track_id, tick)], "actual_val": act_o})
                del cong_tr_buffer_15[(tr.track_id, tick)]
            if (tr.track_id, tick) in cong_tr_buffer_30:
                congestion_history_records.append({"tick": tick - 30, "entity_type": "track", "entity_id": tr.track_id, "horizon": 30, "predicted_val": cong_tr_buffer_30[(tr.track_id, tick)], "actual_val": act_o})
                del cong_tr_buffer_30[(tr.track_id, tick)]
            if (tr.track_id, tick) in cong_tr_buffer_60:
                congestion_history_records.append({"tick": tick - 60, "entity_type": "track", "entity_id": tr.track_id, "horizon": 60, "predicted_val": cong_tr_buffer_60[(tr.track_id, tick)], "actual_val": act_o})
                del cong_tr_buffer_60[(tr.track_id, tick)]

        # Align global network congestion (represented by network congestion score snapshot)
        net_c_score = snapshot.get("average_delay", 0.0) # actual observed delay metric proxy
        if tick in cong_net_buffer_15:
            congestion_history_records.append({"tick": tick - 15, "entity_type": "network", "entity_id": 0, "horizon": 15, "predicted_val": cong_net_buffer_15[tick], "actual_val": net_c_score})
            del cong_net_buffer_15[tick]
        if tick in cong_net_buffer_30:
            congestion_history_records.append({"tick": tick - 30, "entity_type": "network", "entity_id": 0, "horizon": 30, "predicted_val": cong_net_buffer_30[tick], "actual_val": net_c_score})
            del cong_net_buffer_30[tick]
        if tick in cong_net_buffer_60:
            congestion_history_records.append({"tick": tick - 60, "entity_type": "network", "entity_id": 0, "horizon": 60, "predicted_val": cong_net_buffer_60[tick], "actual_val": net_c_score})
            del cong_net_buffer_60[tick]

        # Generate HTML visual web dashboard
        generate_web_dashboard(network, tick, sim_time_str, active_events, preds_delay, preds_congestion, preds_propagation)

        # Render statuses every 15 minutes
        if tick % 15 == 0 or tick in [15, 45, 55, 80]:
            print("\n" + "-"*80)
            print(f"Time: {sim_time_str} | Active Events: {[e.name for e in active_events if e.active]}")
            print("-"*80)
            
            # Print network level predictions
            p30_net = preds_congestion[30]["predicted_network"]
            print(f"AI Global Projections (+30m): Network Congestion: {p30_net['network_congestion']}% | Platform Util: {p30_net['platform_utilization']}% | Stress Index: {p30_net['stress_index']:.1f} (Conf: {int(p30_net['confidence']*100)}%)")
            
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
                for p in preds_delay:
                    if p["train_id"] == train.train_no:
                        pred_str = f" | AI Pred: +15m: {p['delay_predictions']['15']}m, +30m: {p['delay_predictions']['30']}m, +60m: {p['delay_predictions']['60']}m (Conf: {int(p['confidence']*100)}%)"
                        break

                print(f" * Train {train.train_no} ({train.name}): Status: {train.status} | Speed: {train.speed} km/h | {loc_str} | Delay: {train.delay} min{pred_str}")

            # Render Schematic
            render_ascii_map(network)
            time.sleep(0.1)

    # Write prediction history logs to CSV
    if history_records_to_write:
        df_hist = pd.DataFrame(history_records_to_write)
        df_hist.sort_values(by=["tick", "train_id", "prediction_horizon"], inplace=True)
        df_hist.to_csv(history_log_path, index=False)
        print(f"\n[Step 6] Saved delay prediction history to {history_log_path} ({len(df_hist)} rows logged).")

    if congestion_history_records:
        df_cong_hist = pd.DataFrame(congestion_history_records)
        df_cong_hist.sort_values(by=["tick", "entity_type", "entity_id", "horizon"], inplace=True)
        df_cong_hist.to_csv(congestion_history_path, index=False)
        print(f" -> Saved congestion prediction history to {congestion_history_path} ({len(df_cong_hist)} rows logged).")

    # Generate Layer 4 Disruption Propagation Evaluation HTML Report
    from ai.delay_propagation.evaluation import PropagationEvaluationEngine
    rep_path = PropagationEvaluationEngine.generate_evaluation_report(propagation_predictor.history_csv, "reports")
    print(f" -> Saved disruption propagation evaluation report to {rep_path}.")

    # Generate Layer 4 Decision Space Evaluation HTML Report
    from ai.decision_space.evaluation import DecisionEvaluationEngine
    dec_rep_path = DecisionEvaluationEngine.generate_decision_report("datasets", "reports")
    print(f" -> Saved decision space evaluation report to {dec_rep_path}.")

    # 4. Summarize results
    print("\n" + "=" * 80)
    print("           SIMULATION COMPLETED SUCCESSFULLY")
    print(f" -> Recorded {len(StateEngine.history)} chronological state snapshots.")
    print(f" -> Final average train delay: {StateEngine.history[-1]['average_delay']} mins.")
    print(f" -> Visual dashboard generated at datasets/dashboard.html (open in browser).")
    print(f" -> Congestion decision logs generated at datasets/congestion_decisions.jsonl.")
    print("=" * 80)

if __name__ == "__main__":
    main()
