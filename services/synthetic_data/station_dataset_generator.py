class StationDatasetGenerator:
    @staticmethod
    def generate(scenario_id: int, history_records: list, stations_map: dict) -> list:
        rows = []
        num_ticks = len(history_records)

        for i, (tick, snap) in enumerate(history_records):
            state_id = f"state_S{scenario_id:03d}_T{tick:03d}"
            active_events = snap.get("active_events", [])
            rain_intensity = 0.0
            
            for ev in active_events:
                if ev["name"] == "Heavy Rain":
                    rain_intensity = ev.get("intensity", 0.0)
                    break
            
            weather_type = "RAINY" if rain_intensity > 0.0 else "CLEAR"

            for station in snap["stations"]:
                station_id = station["station_id"]
                name = station["name"]
                platforms_total = station["platforms_total"]
                platforms_occupied = station["platforms_occupied"]
                trains_waiting = station["trains_waiting"]
                incoming_trains = station["incoming_trains"]
                outgoing_trains = station["outgoing_trains"]
                congestion_score = station["station_congestion_score"]
                utilization = station["station_utilization_percent"]
                avg_delay = station["average_station_delay"]
                
                platforms_available = max(0, platforms_total - platforms_occupied)
                
                # Active disruptions
                disruptions = []
                for ev in active_events:
                    if ev["name"] == "Signal Failure" and ev.get("station_id") == station_id:
                        disruptions.append("SIGNAL_FAILURE")
                    elif ev["name"] == "Power Failure" and ev.get("station_id") == station_id:
                        disruptions.append("POWER_FAILURE")
                
                disruptions_str = ",".join(disruptions) if disruptions else "NONE"
                
                # Lookahead target (+30 mins)
                future_congestion = congestion_score
                future_idx = min(i + 30, num_ticks - 1)
                for f_station in history_records[future_idx][1]["stations"]:
                    if f_station["station_id"] == station_id:
                        future_congestion = f_station["station_congestion_score"]
                        break

                rows.append({
                    "scenario_id": scenario_id,
                    "tick": tick,
                    "state_id": state_id,
                    "station_id": station_id,
                    "station_name": name,
                    "platforms_total": platforms_total,
                    "platforms_available": platforms_available,
                    "platforms_occupied": platforms_occupied,
                    "waiting_trains": trains_waiting,
                    "incoming_trains": incoming_trains,
                    "outgoing_trains": outgoing_trains,
                    "station_capacity": platforms_total,
                    "station_congestion_score": congestion_score,
                    "station_utilization_percent": utilization,
                    "average_station_delay": avg_delay,
                    "weather_type": weather_type,
                    "active_disruptions": disruptions_str,
                    "node_degree": station["node_degree"],
                    "betweenness_centrality": station["betweenness_centrality"],
                    "closeness_centrality": station["closeness_centrality"],
                    "station_connectivity": station["station_connectivity"],
                    "future_station_congestion": round(future_congestion, 2)
                })

        return rows
