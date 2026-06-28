from services.synthetic_data.feature_engineering import FeatureEngineering

class PropagationDatasetGenerator:
    @staticmethod
    def generate(scenario_id: int, history_records: list, stations_map: dict) -> list:
        rows = []
        num_ticks = len(history_records)

        for i, (tick, snap) in enumerate(history_records):
            state_id = f"state_S{scenario_id:03d}_T{tick:03d}"
            active_trains = snap["trains"]
            
            net_metrics = FeatureEngineering.calculate_network_metrics(snap)
            network_congestion = net_metrics["network_congestion_score"]

            train_by_no = {t["train_no"]: t for t in active_trains}
            links = []

            for train in active_trains:
                train_no = train["train_no"]
                curr_station_id = train["current_station"]
                status = train["status"]
                progress = train["progress"]
                primary_reason = train["primary_delay_reason"]
                
                # Case A: WAITING_FOR_PLATFORM
                if primary_reason == "WAITING_FOR_PLATFORM":
                    platform_occupants = []
                    for other in active_trains:
                        if other["train_no"] != train_no and other["current_station"] == curr_station_id:
                            if other["progress"] == 0.0 and other["status"] in ["ARRIVED", "WAITING", "DELAYED"]:
                                if other["primary_delay_reason"] != "WAITING_FOR_PLATFORM":
                                    platform_occupants.append(other)
                    
                    if platform_occupants:
                        source_train = max(platform_occupants, key=lambda o: o["delay"])
                        links.append({
                            "source_train": source_train["train_no"],
                            "affected_train": train_no,
                            "source_station_id": curr_station_id,
                            "affected_station_id": curr_station_id,
                            "intermediate_station_id": curr_station_id,
                            "cause": "WAITING_FOR_PLATFORM"
                        })

                # Case B: FOLLOWING_TRAIN
                elif primary_reason == "FOLLOWING_TRAIN" and train["current_track_id"] is not None:
                    other_trains_ahead = []
                    for other in active_trains:
                        if other["train_no"] != train_no and other["current_track_id"] == train["current_track_id"]:
                            if other["progress"] > progress:
                                other_trains_ahead.append(other)
                    
                    if other_trains_ahead:
                        source_train = min(other_trains_ahead, key=lambda o: o["progress"] - progress)
                        links.append({
                            "source_train": source_train["train_no"],
                            "affected_train": train_no,
                            "source_station_id": source_train["current_station"],
                            "affected_station_id": curr_station_id,
                            "intermediate_station_id": -1, # N/A for tracks
                            "cause": "FOLLOWING_TRAIN"
                        })

            if links:
                adj = {t["train_no"]: [] for t in active_trains}
                indegree = {t["train_no"]: 0 for t in active_trains}
                
                for link in links:
                    adj[link["source_train"]].append(link["affected_train"])
                    indegree[link["affected_train"]] += 1
                
                depth = {}
                def get_depth(node):
                    if node in depth:
                        return depth[node]
                    if indegree[node] == 0:
                        depth[node] = 0
                        return 0
                    parents = [link["source_train"] for link in links if link["affected_train"] == node]
                    if not parents:
                        depth[node] = 0
                        return 0
                    max_parent_depth = max(get_depth(p) for p in parents)
                    depth[node] = max_parent_depth + 1
                    return depth[node]

                for t in active_trains:
                    get_depth(t["train_no"])

                chain_length = {}
                def get_chain_length(node):
                    if node in chain_length:
                        return chain_length[node]
                    children = adj[node]
                    if not children:
                        chain_length[node] = 1
                        return 1
                    max_child_chain = max(get_chain_length(c) for c in children)
                    chain_length[node] = max_child_chain + 1
                    return chain_length[node]

                for t in active_trains:
                    get_chain_length(t["train_no"])

                for link in links:
                    aff_no = link["affected_train"]
                    src_no = link["source_train"]
                    
                    aff_train = train_by_no[aff_no]
                    
                    # Lookahead target (+30 mins)
                    future_delay = aff_train["delay"]
                    future_idx = min(i + 30, num_ticks - 1)
                    for f_train in history_records[future_idx][1]["trains"]:
                        if f_train["train_no"] == aff_no:
                            future_delay = f_train["delay"]
                            break

                    src_name = stations_map.get(link["source_station_id"], f"Station {link['source_station_id']}")
                    aff_name = stations_map.get(link["affected_station_id"], f"Station {link['affected_station_id']}")
                    
                    int_id = link["intermediate_station_id"]
                    int_name = stations_map.get(int_id, "N/A") if int_id != -1 else "N/A"

                    rows.append({
                        "scenario_id": scenario_id,
                        "tick": tick,
                        "state_id": state_id,
                        "parent_train": src_no,
                        "child_train": aff_no,
                        "source_station_id": link["source_station_id"],
                        "source_station": src_name,
                        "affected_station_id": link["affected_station_id"],
                        "affected_station": aff_name,
                        "intermediate_station_id": int_id,
                        "intermediate_station": int_name,
                        "delay_transferred": round(aff_train["delay_change_last_tick"], 2),
                        "propagation_depth": depth[aff_no],
                        "delay_chain_length": chain_length[src_no],
                        "cause": link["cause"],
                        "network_congestion_score": network_congestion,
                        "future_delay": round(future_delay, 2)
                    })

        return rows
