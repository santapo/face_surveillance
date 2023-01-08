
TRACKER = {
    "sort": "",
    "deepsort": ""
}


def get_tracker(tracker_name: str = "sort"):
    tracker = TRACKER.get(tracker_name)
    return tracker