class TrackDatasetGenerator:
    @staticmethod
    def generate(scenario_id: int, history_records: list, stations_map: dict) -> list:
        rows = []
        num_ticks = len(history_records)

        for i, (tick, snap) in enumerate(history_records):
            state_id = f"state_S{scenario_id:03d}_T{tick:03d}"
            
            for track in snap["tracks"]:
                track_id = track["track_id"]
                src_id = track["source"]
                dest_id = track["destination"]
                distance = track.get("distance", 69.0)
                
                src_name = stations_map.get(src_id, f"Station {src_id}")
                dest_name = stations_map.get(dest_id, f"Station {dest_id}")
                
                capacity = track["track_capacity"]
                trains_on_track = track["trains_on_track"]
                occupancy = track["occupancy_percent"]
                avg_speed = track["average_speed"]
                travel_time = track["travel_time"]
                blocked = 1 if track["blocked"] else 0
                maintenance = 1 if track["maintenance"] else 0

                # Lookahead target (+30 mins)
                future_occupancy = occupancy
                future_idx = min(i + 30, num_ticks - 1)
                for f_track in history_records[future_idx][1]["tracks"]:
                    if f_track["track_id"] == track_id:
                        future_occupancy = f_track["occupancy_percent"]
                        break

                rows.append({
                    "scenario_id": scenario_id,
                    "tick": tick,
                    "state_id": state_id,
                    "track_id": track_id,
                    "source_station_id": src_id,
                    "source_station": src_name,
                    "destination_station_id": dest_id,
                    "destination_station": dest_name,
                    "distance": distance,
                    "track_capacity": capacity,
                    "trains_on_track": trains_on_track,
                    "occupancy_percent": occupancy,
                    "average_speed": avg_speed,
                    "travel_time": travel_time,
                    "blocked": blocked,
                    "maintenance": maintenance,
                    "future_track_congestion": round(future_occupancy, 2)
                })

        return rows
