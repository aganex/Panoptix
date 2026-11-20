
"""Panoptix cross-platform hardware monitoring."""

import json
import math
import platform
import re
import shutil
import subprocess
import threading
import time

import psutil


# ==================================================
# CONFIGURATION
# ==================================================

MONITOR_INTERVAL = 2.0

GPU_INTERVAL = 5.0
GPU_TIMEOUT = 15.0
GPU_MAX_AGE = 20.0
GPU_RETRIES = 2

WINDOWS_POWERSHELL = (
    shutil.which("powershell")
    or shutil.which("pwsh")
)


# ==================================================
# UTILITY
# ==================================================

def valid_percentage(value):
    """Validate a percentage value between 0 and 100."""

    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return None

    if not math.isfinite(value):
        return None

    if value < 0 or value > 100:
        return None

    return round(value, 2)


def safe_float(value):
    """Convert a value to a finite float."""

    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None

    if not math.isfinite(result):
        return None

    return result


# ==================================================
# GPU MONITOR
# ==================================================

class GPUMonitor:
    """
    Windows GPU Engine Counter monitor.

    Reads GPU engine utilization and exposes:
    - Overall busiest-engine proxy
    - 3D
    - Copy
    - Video Decode
    - Video Processing
    - Compute

    GPU monitoring is optional. Unsupported systems
    return None instead of fabricated values.
    """

    ENGINE_TYPES = {
        "3d": "3d",
        "copy": "copy",
        "videodecode": "video_decode",
        "video_decode": "video_decode",
        "videoprocessing": "video_processing",
        "video_processing": "video_processing",
        "compute": "compute",
    }

    def __init__(self):
        self.value = None
        self.timestamp = None

        self.details = {
            "gpu_usage": None,
            "gpu_3d_usage": None,
            "gpu_copy_usage": None,
            "gpu_video_decode_usage": None,
            "gpu_video_processing_usage": None,
            "gpu_compute_usage": None,
        }

        self.last_error = None

        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = None

        self.pattern = re.compile(
            r"(?:^|_)luid_(0x[0-9a-f]+)_"
            r"(0x[0-9a-f]+)_"
            r"phys_(\d+)_"
            r"eng_(\d+)_"
            r"engtype_(.*)$",
            re.IGNORECASE,
        )

    def read_gpu_details(self):
        """
        Read Windows GPU Engine Counter.

        Returns a dictionary of engine utilization.
        """

        if platform.system() != "Windows":
            raise RuntimeError(
                "GPU Engine Counter hanya didukung "
                "oleh backend Windows ini."
            )

        if not WINDOWS_POWERSHELL:
            raise RuntimeError("PowerShell tidak ditemukan.")

        command = (
            "$ErrorActionPreference = 'Stop'; "
            "$samples = (Get-Counter "
            "'\\GPU Engine(*)\\Utilization Percentage' "
            "-ErrorAction Stop).CounterSamples; "
            "$values = @($samples | ForEach-Object { "
            "[PSCustomObject]@{ "
            "Instance = $_.InstanceName; "
            "Value = [double]$_.CookedValue "
            "} }); "
            "ConvertTo-Json -InputObject $values "
            "-Depth 3 -Compress"
        )

        result = subprocess.run(
            [
                WINDOWS_POWERSHELL,
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                command,
            ],
            capture_output=True,
            text=True,
            timeout=GPU_TIMEOUT,
            check=False,
        )

        if result.returncode != 0:
            message = result.stderr.strip() or (
                f"PowerShell exit code: {result.returncode}"
            )
            raise RuntimeError(message)

        output = result.stdout.strip()

        if not output:
            raise RuntimeError("GPU Counter menghasilkan output kosong.")

        # PowerShell may emit warning text before JSON.
        json_start = output.find("[")
        if json_start < 0:
            json_start = output.find("{")

        if json_start < 0:
            raise RuntimeError("Output GPU bukan JSON yang valid.")

        samples = json.loads(output[json_start:])

        if isinstance(samples, dict):
            samples = [samples]

        if not isinstance(samples, list):
            raise RuntimeError("Format GPU Counter tidak dikenal.")

        # Group by physical adapter, engine ID and engine type.
        engine_totals = {}

        for sample in samples:
            if not isinstance(sample, dict):
                continue

            instance = str(sample.get("Instance", ""))
            value = safe_float(sample.get("Value"))

            if value is None or value < 0:
                continue

            match = self.pattern.search(instance)

            if not match:
                continue

            luid_high = match.group(1).lower()
            luid_low = match.group(2).lower()
            physical_gpu = match.group(3)
            engine_id = match.group(4)
            engine_type = match.group(5).lower()

            engine_key = (
                luid_high,
                luid_low,
                physical_gpu,
                engine_id,
                engine_type,
            )

            engine_totals[engine_key] = (
                engine_totals.get(engine_key, 0.0) + value
            )

        if not engine_totals:
            raise RuntimeError(
                "Tidak ada GPU Engine Counter yang cocok "
                "dengan pola instance."
            )

        # Aggregate process instances belonging to the same engine.
        # Clamp each engine to 100%.
        by_type = {}

        for key, total in engine_totals.items():
            engine_type = key[4]

            if engine_type not in self.ENGINE_TYPES:
                continue

            category = self.ENGINE_TYPES[engine_type]
            usage = min(100.0, total)

            by_type.setdefault(category, []).append(usage)

        def busiest(category):
            values = by_type.get(category, [])
            return round(max(values), 2) if values else None

        details = {
            "gpu_3d_usage": busiest("3d"),
            "gpu_copy_usage": busiest("copy"),
            "gpu_video_decode_usage": busiest("video_decode"),
            "gpu_video_processing_usage": busiest("video_processing"),
            "gpu_compute_usage": busiest("compute"),
        }

        available_engines = [
            value
            for value in details.values()
            if value is not None
        ]

        if not available_engines:
            raise RuntimeError(
                "Counter terbaca, tetapi engine GPU yang dikenal "
                "tidak ditemukan."
            )

        # This is a busiest-engine proxy, NOT total GPU utilization.
        details["gpu_usage"] = max(available_engines)

        return details

    def read_gpu(self):
        """Compatibility method: return busiest-engine proxy."""

        details = self.read_gpu_details()
        return details["gpu_usage"]

    def worker(self):
        """Background GPU polling worker."""

        while not self.stop_event.is_set():
            start_time = time.monotonic()
            successful_read = False
            error_message = None

            for attempt in range(GPU_RETRIES + 1):
                if self.stop_event.is_set():
                    break

                try:
                    details = self.read_gpu_details()

                    gpu_value = valid_percentage(
                        details.get("gpu_usage")
                    )

                    if gpu_value is None:
                        raise RuntimeError(
                            "GPU usage tidak valid."
                        )

                    with self.lock:
                        self.value = gpu_value
                        self.details = details
                        self.timestamp = time.monotonic()
                        self.last_error = None

                    successful_read = True
                    break

                except Exception as error:
                    error_message = str(error)

                    if attempt < GPU_RETRIES:
                        self.stop_event.wait(0.5)

            if not successful_read:
                with self.lock:
                    self.last_error = error_message

            elapsed = time.monotonic() - start_time
            remaining = max(0.0, GPU_INTERVAL - elapsed)

            self.stop_event.wait(remaining)

    def start(self):
        """Start GPU monitoring thread."""

        if platform.system() != "Windows":
            self.last_error = (
                "GPU Engine Counter backend ini hanya mendukung Windows."
            )
            return

        if self.thread is not None and self.thread.is_alive():
            return

        self.stop_event.clear()

        self.thread = threading.Thread(
            target=self.worker,
            name="Panoptix-GPU-Monitor",
            daemon=True,
        )

        self.thread.start()

    def get_data(self):
        """
        Return (GPU usage, data age).

        Existing collector.py remains compatible.
        """

        with self.lock:
            value = self.value
            timestamp = self.timestamp

        if value is None or timestamp is None:
            return None, None

        age = time.monotonic() - timestamp

        if age > GPU_MAX_AGE:
            return None, age

        return value, age

    def get_details(self):
        """Return detailed engine data and its age."""

        with self.lock:
            details = dict(self.details)
            timestamp = self.timestamp

        if timestamp is None:
            return {
                key: None for key in self.details
            }, None

        age = time.monotonic() - timestamp

        if age > GPU_MAX_AGE:
            return {
                key: None for key in self.details
            }, age

        return details, age

    def get_status(self):
        """Return GPU monitoring status for diagnostics."""

        with self.lock:
            timestamp = self.timestamp
            error = self.last_error

        age = (
            time.monotonic() - timestamp
            if timestamp is not None
            else None
        )

        return {
            "available": (
                age is not None and age <= GPU_MAX_AGE
            ),
            "age": age,
            "last_error": error,
        }

    def stop(self):
        """Stop GPU monitoring thread."""

        self.stop_event.set()

        if self.thread is not None:
            self.thread.join(
                timeout=GPU_TIMEOUT + 2
            )


# ==================================================
# HARDWARE METRICS
# ==================================================

def initialize_monitoring():
    """
    Initialize CPU sampling and hardware baselines.

    The first CPU reading initializes psutil's counter.
    """

    psutil.cpu_percent(interval=None)
    psutil.cpu_percent(interval=None, percpu=True)

    return {
        "disk": psutil.disk_io_counters(),
        "network": psutil.net_io_counters(),
        "timestamp": time.monotonic(),
    }


def get_hardware_metrics():
    """
    Collect current hardware metrics.

    CPU interval=None measures utilization since the
    previous call. The collector controls the sampling interval.
    """

    memory = psutil.virtual_memory()

    cpu_total = psutil.cpu_percent(interval=None)
    cpu_per_core = psutil.cpu_percent(
        interval=None,
        percpu=True,
    )

    try:
        frequency = psutil.cpu_freq()
        cpu_frequency = (
            frequency.current
            if frequency is not None
            else None
        )
    except (NotImplementedError, OSError):
        cpu_frequency = None

    return {
        # Existing collector compatibility
        "cpu_usage": valid_percentage(cpu_total),
        "ram_usage": valid_percentage(memory.percent),
        "disk": psutil.disk_io_counters(),
        "network": psutil.net_io_counters(),

        # Additional detailed metrics
        "cpu_per_core": [
            valid_percentage(value)
            for value in cpu_per_core
        ],
        "cpu_frequency_mhz": safe_float(cpu_frequency),
        "ram_used_bytes": memory.used,
        "ram_available_bytes": memory.available,
        "ram_total_bytes": memory.total,
        "ram_used_gb": round(
            memory.used / (1024 ** 3), 2
        ),
        "ram_available_gb": round(
            memory.available / (1024 ** 3), 2
        ),
        "ram_total_gb": round(
            memory.total / (1024 ** 3), 2
        ),
    }


# ==================================================
# DISK AND NETWORK SPEED
# ==================================================

def calculate_speed(current, previous, attribute, elapsed_time):
    """
    Calculate I/O throughput in bytes per second.
    """

    if elapsed_time <= 0:
        return None

    if current is None or previous is None:
        return None

    current_value = getattr(current, attribute, None)
    previous_value = getattr(previous, attribute, None)

    if current_value is None or previous_value is None:
        return None

    difference = current_value - previous_value

    if difference < 0:
        return None

    return difference / elapsed_time


# ==================================================
# FORMATTING
# ==================================================

def format_metric(value, unit=""):
    """Format metric for terminal output."""

    if value is None:
        return "Not Available"

    return f"{value:.2f}{unit}"


# ==================================================
# MAIN MONITOR
# ==================================================

def monitor():
    """Run standalone Panoptix hardware monitor."""

    print("=" * 55)
    print("             PANOPTIX HARDWARE MONITOR")
    print("=" * 55)

    print(
        f"Operating System : "
        f"{platform.system()} {platform.release()}"
    )

    print(f"Processor        : {platform.processor()}")

    print("=" * 55)

    previous = initialize_monitoring()

    gpu_monitor = GPUMonitor()
    gpu_monitor.start()

    try:
        while True:
            time.sleep(MONITOR_INTERVAL)

            current = get_hardware_metrics()
            current_time = time.monotonic()

            elapsed_time = (
                current_time - previous["timestamp"]
            )

            # CPU
            print("\n--- HARDWARE METRICS ---")

            print(
                f"CPU Total        : "
                f"{format_metric(current['cpu_usage'], '%')}"
            )

            print(
                f"CPU Frequency    : "
                f"{format_metric(current['cpu_frequency_mhz'], ' MHz')}"
            )

            print(
                f"RAM Usage        : "
                f"{format_metric(current['ram_usage'], '%')}"
            )

            print(
                f"RAM Used         : "
                f"{current['ram_used_gb']:.2f} GB / "
                f"{current['ram_total_gb']:.2f} GB"
            )

            print(
                f"RAM Available    : "
                f"{current['ram_available_gb']:.2f} GB"
            )

            print("CPU Per Core     :")

            for index, value in enumerate(
                current["cpu_per_core"],
                start=1,
            ):
                print(
                    f"  Core {index:<2}: "
                    f"{format_metric(value, '%')}"
                )

            # GPU
            gpu_details, gpu_age = gpu_monitor.get_details()

            print("\nGPU ENGINE METRICS")

            gpu_labels = {
                "gpu_3d_usage": "3D",
                "gpu_copy_usage": "Copy",
                "gpu_video_decode_usage": "Video Decode",
                "gpu_video_processing_usage": "Video Processing",
                "gpu_compute_usage": "Compute",
                "gpu_usage": "Busiest Engine Proxy",
            }

            for key, label in gpu_labels.items():
                print(
                    f"{label:<22}: "
                    f"{format_metric(gpu_details.get(key), '%')}"
                )

            if gpu_age is not None:
                print(f"GPU Data Age          : {gpu_age:.1f}s")

            status = gpu_monitor.get_status()

            if not status["available"]:
                print(
                    f"GPU Status            : N/A "
                    f"({status['last_error'] or 'Waiting for data'})"
                )

            # Disk
            disk_read = calculate_speed(
                current["disk"],
                previous["disk"],
                "read_bytes",
                elapsed_time,
            )

            disk_write = calculate_speed(
                current["disk"],
                previous["disk"],
                "write_bytes",
                elapsed_time,
            )

            # Network
            network_upload = calculate_speed(
                current["network"],
                previous["network"],
                "bytes_sent",
                elapsed_time,
            )

            network_download = calculate_speed(
                current["network"],
                previous["network"],
                "bytes_recv",
                elapsed_time,
            )

            print("\nDISK AND NETWORK")

            print(
                f"Disk Read        : "
                f"{format_metric(disk_read, ' B/s')}"
            )

            print(
                f"Disk Write       : "
                f"{format_metric(disk_write, ' B/s')}"
            )

            print(
                f"Network Upload   : "
                f"{format_metric(network_upload, ' B/s')}"
            )

            print(
                f"Network Download : "
                f"{format_metric(network_download, ' B/s')}"
            )

            # Update baseline
            previous = {
                "disk": current["disk"],
                "network": current["network"],
                "timestamp": current_time,
            }

    except KeyboardInterrupt:
        print("\nPanoptix monitoring dihentikan.")

    finally:
        gpu_monitor.stop()
        print("GPU monitoring dihentikan.")


# ==================================================
# ENTRY POINT
# ==================================================

if __name__ == "__main__":
    monitor()
