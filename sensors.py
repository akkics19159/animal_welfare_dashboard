import random


def read_sensors(use_simulation=True):
    """Read sensor values.
    Set use_simulation=False when real sensor integration is available.
    """
    if not use_simulation:
        raise NotImplementedError("Real sensor integration is not configured in this prototype.")

    return {
        "temp": random.uniform(20, 35),
        "humidity": random.uniform(40, 70),
        "heart_rate": random.randint(60, 120),
    }
