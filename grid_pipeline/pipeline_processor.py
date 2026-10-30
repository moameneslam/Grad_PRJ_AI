#!/usr/bin/env python3
"""
Real-Time Microgrid AI Inference Pipeline Processor
---------------------------------------------------
Subscribes to live telemetry from Simulink via Mosquitto MQTT, cleans 128-byte 
buffer padding/concatenation, runs dual-model inference (XGBoost + Isolation Forest),
and publishes enriched diagnostic telemetry for Node-RED / InfluxDB / Grafana.
"""

import json
import logging
import os
import pickle
import re
import sys
import numpy as np
import paho.mqtt.client as mqtt
import xgboost as xgb

# Set up clean logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Configuration Parameters
BROKER_HOST  = "127.0.0.1"
BROKER_PORT  = 1883
TOPIC_INPUT  = "grid/telemetry/sim_bus_1"
TOPIC_OUTPUT = "grid/telemetry/sim_bus_1/inference"

XGB_MODEL_PATH = "xgboost_fault_classifier.json"
ISO_MODEL_PATH = "iso_forest_model.pkl"


class MicrogridInferenceEngine:
    def __init__(self, xgb_path: str, iso_path: str):
        """Load and initialize machine learning models."""
        self.xgb_model = None
        self.iso_model = None
        self._load_models(xgb_path, iso_path)

    def _load_models(self, xgb_path: str, iso_path: str):
        if not os.path.exists(xgb_path):
            raise FileNotFoundError(f"XGBoost model file not found at: {xgb_path}")
        if not os.path.exists(iso_path):
            raise FileNotFoundError(f"Isolation Forest model file not found at: {iso_path}")

        # Load XGBoost Classifier
        self.xgb_model = xgb.XGBClassifier()
        self.xgb_model.load_model(xgb_path)
        logging.info(f"Loaded XGBoost model from '{xgb_path}'")

        # Load Isolation Forest Model
        with open(iso_path, "rb") as f:
            self.iso_model = pickle.load(f)
        logging.info(f"Loaded Isolation Forest model from '{iso_path}'")

    def predict(self, voltage: float, current: float, frequency: float, active_power: float) -> dict:
        """Execute dual-model inference on numerical feature vector."""
        features = np.array([[voltage, current, frequency, active_power]])

        # 1. XGBoost Supervised Classification (Known Faults)
        xgb_pred = int(self.xgb_model.predict(features)[0])
        xgb_probs = self.xgb_model.predict_proba(features)[0]
        xgb_confidence = float(np.max(xgb_probs) * 100.0)

        # 2. Isolation Forest Unsupervised Anomaly Detection (Novel Outliers)
        iso_raw = self.iso_model.predict(features)[0]
        iso_pred = 1 if iso_raw == -1 else 0  # Convert -1 (outlier) to 1 (anomaly)

        # 3. Decision Logic & System Status Formatting
        if xgb_pred == 1:
            status = "🚨 KNOWN FAULT DETECTED (XGBoost)"
            alert_level = "CRITICAL"
        elif iso_pred == 1:
            status = "⚠️ UNKNOWN ANOMALY DETECTED (IsoForest)"
            alert_level = "WARNING"
        else:
            status = "✅ NORMAL OPERATING GRID"
            alert_level = "NORMAL"

        return {
            "xgb_fault_pred": xgb_pred,
            "xgb_confidence_pct": round(xgb_confidence, 2),
            "iso_anomaly_pred": iso_pred,
            "status_text": status,
            "alert_level": alert_level
        }


def on_connect(client, userdata, flags, rc, properties=None):
    """Callback triggered upon MQTT broker connection."""
    if rc == 0:
        logging.info(f"Connected to Mosquitto MQTT Broker at {BROKER_HOST}:{BROKER_PORT}")
        client.subscribe(TOPIC_INPUT)
        logging.info(f"Subscribed to topic: '{TOPIC_INPUT}'")
    else:
        logging.error(f"Failed to connect to MQTT broker with return code: {rc}")


def on_message(client, userdata, msg):
    """Process incoming telemetry, clean ASCII padding, and run inference."""
    engine = userdata["engine"]

    try:
        # Step 1: Decode binary payload & strip 128-byte null padding (\x00)
        raw_payload = msg.payload.decode("utf-8", errors="ignore").replace("\x00", "").strip()

        # Step 2: Extract JSON string objects via regex (handles concatenated frames)
        json_frames = re.findall(r"\{.*?\}", raw_payload)

        if not json_frames:
            return

        for frame_str in json_frames:
            telemetry = json.loads(frame_str)

            # Step 3: Extract features safely with fallback defaults
            bus_id = telemetry.get("bus_id", "sim_bus_1")
            voltage = float(telemetry.get("voltage", 0.0))
            current = float(telemetry.get("current", 0.0))
            frequency = float(telemetry.get("frequency", 50.0))
            active_power = float(telemetry.get("active_power", 0.0))
            sim_anomaly_flag = int(telemetry.get("anomaly_flag", 0))

            # Step 4: Run Inference
            inference = engine.predict(voltage, current, frequency, active_power)

            # Step 5: Construct enriched output payload for downstream tools
            output_payload = {
                "bus_id": bus_id,
                "voltage": voltage,
                "current": current,
                "frequency": frequency,
                "active_power": active_power,
                "sim_anomaly_flag": sim_anomaly_flag,
                "xgb_fault_pred": inference["xgb_fault_pred"],
                "xgb_confidence_pct": inference["xgb_confidence_pct"],
                "iso_anomaly_pred": inference["iso_anomaly_pred"],
                "alert_level": inference["alert_level"],
                "status_text": inference["status_text"]
            }

            # Step 6: Publish result to MQTT topic
            client.publish(TOPIC_OUTPUT, json.dumps(output_payload))

            # Terminal log output
            logging.info(
                f"[{inference['status_text']}] V: {voltage:.1f}V | I: {current:.1f}A | "
                f"P: {active_power:.1f}W | Conf: {inference['xgb_confidence_pct']:.1f}%"
            )

    except Exception as e:
        logging.error(f"Error processing frame: {e}")


def main():
    # Initialize Machine Learning Models
    try:
        engine = MicrogridInferenceEngine(XGB_MODEL_PATH, ISO_MODEL_PATH)
    except Exception as e:
        logging.critical(f"Initialization error: {e}")
        sys.exit(1)

    # Initialize MQTT Client (compatible across paho-mqtt v1 and v2)
    try:
        client = mqtt.Client(
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
            userdata={"engine": engine}
        )
    except AttributeError:
        client = mqtt.Client(userdata={"engine": engine})

    client.on_connect = on_connect
    client.on_message = on_message

    logging.info(f"Connecting to MQTT Broker at {BROKER_HOST}:{BROKER_PORT}...")
    try:
        client.connect(BROKER_HOST, BROKER_PORT, keepalive=60)
        logging.info("Inference pipeline running. Press Ctrl+C to stop.")
        client.loop_forever()
    except KeyboardInterrupt:
        logging.info("\nPipeline shut down cleanly.")
        client.disconnect()
    except Exception as e:
        logging.critical(f"Broker connection failed: {e}")


if __name__ == "__main__":
    main()