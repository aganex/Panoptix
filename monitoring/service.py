"""Panoptix integrated monitoring service."""

import csv
import socket
import threading
import time
import re

from collections import deque
from pathlib import Path

import psutil

from monitoring import monitor, collector


class PanoptixMonitoringService:

    def __init__(self, interval=2.0):

        self.interval = interval

        self.base_dir = Path(__file__).resolve().parent.parent
        self.dataset_dir = self.base_dir / "dataset" / "raw"

        self.output_file = self.dataset_dir / "hardware_metrics.csv"
        self.devices_file = self.dataset_dir / "devices.csv"

        self.device_id = re.sub(
            r"[^A-Za-z0-9_-]+",
            "_",
            socket.gethostname()
        ).strip("_").upper()

        self.session_id = None

        self.stop_event = threading.Event()
        self.thread = None
        self.gpu_monitor = None

        self.lock = threading.Lock()

        self.latest_metrics = None
        self.history = deque(maxlen=300)

        self.sample_count = 0
        self.last_error = None
        self.started_at = None

    @property
    def is_running(self):

        return (
            self.thread is not None
            and self.thread.is_alive()
            and not self.stop_event.is_set()
        )

    def start(self):

        if self.is_running:
            return

        self.stop_event.clear()
        self.last_error = None
        self.sample_count = 0

        self.session_id = (
            f"SESSION_{time.strftime('%Y%m%dT%H%M%S')}"
        )

        self.dataset_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        collector.initialize_dataset(self.output_file)

        collector.ensure_device(
            self.devices_file,
            self.device_id
        )

        self.gpu_monitor = collector.initialize_gpu_monitor()

        self.started_at = time.time()

        self.thread = threading.Thread(
            target=self._run,
            name="Panoptix-Monitoring-Service",
            daemon=True
        )

        self.thread.start()

    def stop(self):

        self.stop_event.set()

        if self.thread is not None:
            self.thread.join(timeout=5)

        if self.gpu_monitor is not None:
            try:
                self.gpu_monitor.stop()
            except Exception:
                pass

        self.gpu_monitor = None

    def get_snapshot(self):

        with self.lock:

            return {
                "running": self.is_running,
                "metrics": (
                    dict(self.latest_metrics)
                    if self.latest_metrics
                    else None
                ),
                "history": list(self.history),
                "sample_count": self.sample_count,
                "last_error": self.last_error,
                "device_id": self.device_id,
                "session_id": self.session_id
            }

    def _run(self):

        try:

            previous = monitor.initialize_monitoring()

            next_sample = time.monotonic()

            with self.output_file.open(
                "a",
                newline="",
                encoding="utf-8"
            ) as csv_file:

                writer = csv.DictWriter(
                    csv_file,
                    fieldnames=collector.CSV_FIELDS
                )

                while not self.stop_event.is_set():

                    next_sample += self.interval

                    wait_time = max(
                        0,
                        next_sample - time.monotonic()
                    )

                    if self.stop_event.wait(wait_time):
                        break

                    current = monitor.get_hardware_metrics()

                    current_time = time.monotonic()

                    elapsed = (
                        current_time - previous["timestamp"]
                    )

                    if elapsed <= 0:
                        continue

                    metrics = collector.collect_metrics(
                        previous,
                        current,
                        elapsed,
                        self.gpu_monitor
                    )

                    row = collector.write_metrics(
                        writer,
                        csv_file,
                        metrics,
                        self.device_id,
                        self.session_id
                    )

                    snapshot = {
                        **metrics,
                        "timestamp": row["timestamp"]
                    }

                    with self.lock:

                        self.latest_metrics = snapshot
                        self.history.append(snapshot)
                        self.sample_count += 1

                    previous = {
                        "disk": current["disk"],
                        "network": current["network"],
                        "timestamp": current_time
                    }

        except Exception as error:

            with self.lock:
                self.last_error = str(error)

        finally:

            if self.gpu_monitor is not None:

                try:
                    self.gpu_monitor.stop()
                except Exception:
                    pass

            self.gpu_monitor = None