from services.synthetic_data.feature_engineering import FeatureEngineering

class NetworkDatasetGenerator:
    @staticmethod
    def generate(scenario_id: int, history_records: list, day_type: str = "WEEKDAY") -> list:
        rows = []
        num_ticks = len(history_records)

        # Precompute network metrics
        precomputed_metrics = []
        for tick, snap in history_records:
            metrics = FeatureEngineering.calculate_network_metrics(snap)
            precomputed_metrics.append(metrics)

        for i, (tick, snap) in enumerate(history_records):
            time_str = snap["time"]
            state_id = f"state_S{scenario_id:03d}_T{tick:03d}"
            time_of_day = FeatureEngineering.get_time_of_day(time_str)
            
            metrics = precomputed_metrics[i]
            network_congestion = metrics["network_congestion_score"]
            platform_util = metrics["platform_utilization_percent"]
            track_util = metrics["track_utilization_percent"]

            # Events
            active_events = snap.get("active_events", [])
            rain_intensity = 0.0
            disruption_names = []
            
            critical_track_count = 0
            critical_station_count = 0
            active_event_durations = []
            
            for ev in active_events:
                disruption_names.append(ev["name"].upper().replace(" ", "_"))
                active_event_durations.append(ev.get("duration", 0))
                
                if ev["name"] == "Heavy Rain":
                    rain_intensity = ev.get("intensity", 0.0)
                elif ev["name"] in ["Signal Failure", "Power Failure"]:
                    critical_station_count += 1
                elif ev["name"] in ["Maintenance", "Track Blockage"]:
                    critical_track_count += 1

            weather_type = "RAINY" if rain_intensity > 0.0 else "CLEAR"
            disruptions_str = ",".join(set(disruption_names)) if disruption_names else "NONE"

            # Train totals
            total_waiting = sum(s["trains_waiting"] for s in snap["stations"])
            total_platforms_used = sum(s["platforms_occupied"] for s in snap["stations"])
            total_tracks_used = sum(1 for t in snap["tracks"] if t["trains_on_track"] > 0)

            # Resilience Metrics
            network_resilience = round(max(0.0, 100.0 - network_congestion), 2)
            
            # Recovery time estimate: max train delay + max active event remaining duration
            max_train_delay = max([t["delay"] for t in snap["trains"]] + [0.0])
            max_event_rem_duration = max(active_event_durations + [0])
            recovery_time_estimate = round(max_train_delay + max_event_rem_duration, 2)
            
            # Bottlenecks: stations or tracks with occupancy/utilization >= 80%
            congested_stations = snap["congested_stations"]
            congested_tracks = snap["congested_tracks"]
            bottleneck_count = congested_stations + congested_tracks

            # Lookaheads (+30 ticks)
            future_idx = min(i + 30, num_ticks - 1)
            future_metrics = precomputed_metrics[future_idx]
            future_snap = history_records[future_idx][1]
            
            future_network_congestion = future_metrics["network_congestion_score"]
            future_platform_util = future_metrics["platform_utilization_percent"]
            future_avg_delay = future_snap["average_delay"]

            rows.append({
                "scenario_id": scenario_id,
                "tick": tick,
                "state_id": state_id,
                "simulation_time_minutes": tick,
                "weather_type": weather_type,
                "time_of_day": time_of_day,
                "day_type": day_type,
                "network_congestion_score": network_congestion,
                "average_network_delay": snap["average_delay"],
                "platform_utilization_percent": platform_util,
                "track_utilization_percent": track_util,
                "active_disruptions": disruptions_str,
                "num_active_trains": snap["active_trains"],
                "congested_stations": congested_stations,
                "congested_tracks": congested_tracks,
                "average_train_speed": snap["average_speed"],
                "total_waiting_trains": total_waiting,
                "total_platforms_used": total_platforms_used,
                "total_tracks_used": total_tracks_used,
                "network_resilience_score": network_resilience,
                "recovery_time_estimate": recovery_time_estimate,
                "bottleneck_count": bottleneck_count,
                "critical_track_count": critical_track_count,
                "critical_station_count": critical_station_count,
                "future_network_congestion": round(future_network_congestion, 2),
                "future_platform_utilization": round(future_platform_util, 2),
                "future_average_network_delay": round(future_avg_delay, 2)
            })

        return rows
