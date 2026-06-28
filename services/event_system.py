class DisruptionEvent:
    def __init__(self, name, duration):
        self.name = name
        self.duration = duration  # Duration in minutes (simulation ticks)
        self.active = True

    def tick(self):
        if self.duration > 0:
            self.duration -= 1
        if self.duration <= 0:
            self.active = False


class SignalFailureEvent(DisruptionEvent):
    def __init__(self, station_id, duration):
        super().__init__("Signal Failure", duration)
        self.station_id = station_id

    def __repr__(self):
        return f"SignalFailure(station_id={self.station_id}, duration={self.duration})"


class HeavyRainEvent(DisruptionEvent):
    def __init__(self, intensity, duration):
        super().__init__("Heavy Rain", duration)
        self.intensity = intensity  # 0.0 (no rain) to 1.0 (extreme downpour)

    def __repr__(self):
        return f"HeavyRain(intensity={self.intensity:.2f}, duration={self.duration})"


class FestivalRushEvent(DisruptionEvent):
    def __init__(self, duration, extra_train_count=2):
        super().__init__("Festival Rush", duration)
        self.extra_train_count = extra_train_count

    def __repr__(self):
        return f"FestivalRush(extra_trains={self.extra_train_count}, duration={self.duration})"


class MaintenanceEvent(DisruptionEvent):
    def __init__(self, track_id, duration):
        super().__init__("Maintenance", duration)
        self.track_id = track_id

    def __repr__(self):
        return f"Maintenance(track_id={self.track_id}, duration={self.duration})"


class TrackBlockageEvent(DisruptionEvent):
    def __init__(self, track_id, duration):
        super().__init__("Track Blockage", duration)
        self.track_id = track_id

    def __repr__(self):
        return f"TrackBlockage(track_id={self.track_id}, duration={self.duration})"


class PowerFailureEvent(DisruptionEvent):
    def __init__(self, station_id, duration):
        super().__init__("Power Failure", duration)
        self.station_id = station_id

    def __repr__(self):
        return f"PowerFailure(station_id={self.station_id}, duration={self.duration})"
