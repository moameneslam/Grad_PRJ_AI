import json
import time
import random
import paho.mqtt.client as mqtt

client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.connect("localhost", 1883, 60)

print("[Simulator] Publishing microgrid telemetry every 1 second (Ctrl+C to stop)...")

cycle = 0
try:
    while True:
        cycle += 1
        # Inject an anomaly every 10 cycles (e.g., severe voltage sag / current surge)
        if cycle % 10 == 0:
            voltage = round(random.uniform(160.0, 180.0), 2)  # Severe voltage drop
            current = round(random.uniform(22.0, 28.0), 2)    # Fault current surge
            freq = round(random.uniform(48.2, 48.9), 2)       # Frequency sag
            print("--> Injecting simulated fault!")
        else:
            voltage = round(random.uniform(218.0, 222.5), 2)
            current = round(random.uniform(9.5, 10.5), 2)
            freq = round(random.uniform(49.95, 50.05), 2)

        payload = {
            "bus_id": "bus_1",
            "voltage": voltage,
            "current": current,
            "frequency": freq,
            "active_power": round(voltage * current * 0.95, 2)
        }

        client.publish("grid/telemetry/bus_1", json.dumps(payload))
        time.sleep(1.0)

except KeyboardInterrupt:
    print("\n[Simulator] Stopped.")
    client.disconnect()
