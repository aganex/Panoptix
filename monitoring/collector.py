
"""Collect Panoptix hardware metrics and automatically label system load."""

import csv
import math
import platform
import re
import socket
import subprocess
import threading
import time
import uuid

from datetime import datetime
from pathlib import Path

import psutil

try:
    from . import monitor
except ImportError:
    import monitor


# ==================================================
# CONFIGURATION
# ==================================================

CSV_FIELDS = (
    "timestamp",
    "device_id",
    "session_id",
    "cpu_usage",
    "ram_usage",
    "gpu_usage",
    "disk_read_bytes_per_sec",
    "disk_write_bytes_per_sec",
    "network_upload_bytes_per_sec",
    "network_download_bytes_per_sec",
    "system_load_score",
    "load_category",
)

DEVICE_FIELDS = (
    "device_id",
    "CPU",
    "GPU",
    "RAM",
    "OS",
)

GPU_INITIAL_WAIT = 15.0


# ==================================================
# SYSTEM LOAD CLASSIFICATION
# ==================================================

def calculate_system_load(cpu_usage, ram_usage):
    """
    Calculate system load score using CPU and RAM.

    CPU weight: 60%
    RAM weight: 40%
    """

    score = (0.6 * cpu_usage) + (0.4 * ram_usage)
    score = max(0.0, min(100.0, score))

    if score < 40:
        category = "Low"
    elif score < 70:
        category = "Moderate"
    else:
        category = "High"

    return round(score, 2), category


# ==================================================
# GPU NORMALIZATION
# ==================================================

def normalize_gpu_usage(value):
    """
    Return a valid GPU percentage.
    Return None if GPU data is unavailable.
    """

    if value is None:
        return None

    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return None

    if not math.isfinite(value):
        return None

    if value < 0 or value > 100:
        return None

    return round(value, 2)


# ==================================================
# DEVICE METADATA
# ==================================================

def get_gpu_name():
    """
    Retrieve GPU name when supported.
    GPU name is optional and does not affect collection.
    """

    if platform.system() != "Windows":
        return "Unknown"

    commands = ("powershell", "pwsh")

    for command_name in commands:
        try:
            result = subprocess.run(
                [
                    command_name,
                    "-NoProfile",
                    "-NonInteractive",
                    "-Command",
                    (
                        "(Get-CimInstance Win32_VideoController "
                        "| ForEach-Object Name) -join '; '"
                    ),
                ],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )

            if result.returncode == 0:
                names = result.stdout.strip()

                if names:
                    return names

        except (OSError, subprocess.SubprocessError):
            continue

    return "Unknown"


def device_metadata(device_id):
    """
    Collect hardware and operating system information.
    """

    cpu = platform.processor() or "Unknown"

    ram_gb = (
        psutil.virtual_memory().total
        / (1024 ** 3)
    )

    return {
        "device_id": device_id,
        "CPU": cpu,
        "GPU": get_gpu_name(),
        "RAM": f"{ram_gb:.1f} GB",
        "OS": f"{platform.system()} {platform.release()}",
    }


# ==================================================
# DEVICE REGISTRATION
# ==================================================

def ensure_device(devices_file, device_id):
    """
    Register device if it does not exist.
    Existing device records are preserved.
    """

    devices_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    existing_ids = set()

    if devices_file.exists() and devices_file.stat().st_size > 0:
        with devices_file.open(
            "r",
            newline="",
            encoding="utf-8-sig",
        ) as file:

            reader = csv.DictReader(file)

            if tuple(reader.fieldnames or ()) != DEVICE_FIELDS:
                raise RuntimeError(
                    "Struktur devices.csv tidak sesuai.\n"
                    f"Expected: {DEVICE_FIELDS}\n"
                    f"Found: {reader.fieldnames}\n"
                    "File tidak diubah untuk menjaga data lama."
                )

            existing_ids = {
                row.get("device_id", "")
                for row in reader
            }

    write_header = (
        not devices_file.exists()
        or devices_file.stat().st_size == 0
    )

    if device_id in existing_ids:
        return

    metadata = device_metadata(device_id)

    with devices_file.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=DEVICE_FIELDS,
        )

        if write_header:
            writer.writeheader()

        writer.writerow(metadata)


# ==================================================
# DATASET VALIDATION
# ==================================================

def validate_dataset_schema(output_file):
    """
    Validate hardware_metrics.csv.

    Empty files are allowed and will receive a new header.
    Existing non-empty datasets with different schemas
    are never overwritten.
    """

    if not output_file.exists():
        return

    if output_file.stat().st_size == 0:
        return

    with output_file.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as file:

        reader = csv.reader(file)

        existing_fields = None

        for row in reader:
            if row:
                existing_fields = tuple(row)
                break

    if existing_fields is None:
        return

    if existing_fields != CSV_FIELDS:
        raise RuntimeError(
            "\nStruktur hardware_metrics.csv tidak sesuai.\n\n"
            f"Expected:\n{CSV_FIELDS}\n\n"
            f"Found:\n{existing_fields}\n\n"
            "Collector dihentikan untuk melindungi dataset lama."
        )


def initialize_dataset(output_file):
    """
    Create dataset file and header if necessary.
    """

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    validate_dataset_schema(output_file)

    if not output_file.exists() or output_file.stat().st_size == 0:

        with output_file.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=CSV_FIELDS,
            )

            writer.writeheader()

        return

    # Handle files containing only blank lines.
    with output_file.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as file:

        has_header = any(
            row for row in csv.reader(file)
        )

    if not has_header:
        with output_file.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=CSV_FIELDS,
            )

            writer.writeheader()


# ==================================================
# GPU MONITORING
# ==================================================

class UnavailableGPU:
    """
    Fallback GPU monitor when GPU monitoring is unsupported.
    """

    def start(self):
        pass

    def get_data(self):
        return None, None

    def stop(self):
        pass


def initialize_gpu_monitor():
    """
    Initialize GPU monitoring without making it mandatory.
    """

    try:
        gpu_monitor = monitor.GPUMonitor()
        gpu_monitor.start()

        return gpu_monitor

    except Exception as error:
        print(f"GPU monitoring tidak tersedia: {error}")

        return UnavailableGPU()


def wait_for_gpu(gpu_monitor, timeout=GPU_INITIAL_WAIT):
    """
    Wait for the first valid GPU reading.
    Maximum waiting time is 15 seconds.
    """

    if isinstance(gpu_monitor, UnavailableGPU):
        return False

    print(
        "Menunggu pembacaan GPU pertama",
        end="",
        flush=True,
    )

    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:

        try:
            gpu_usage, _gpu_age = gpu_monitor.get_data()
            gpu_usage = normalize_gpu_usage(gpu_usage)

            if gpu_usage is not None:
                print(f" OK ({gpu_usage:.2f}%)")
                return True

        except Exception:
            pass

        print(".", end="", flush=True)
        time.sleep(0.5)

    print(" timeout.")
    print("Collector tetap berjalan tanpa data GPU.")

    return False


# ==================================================
# METRIC COLLECTION
# ==================================================

def collect_metrics(previous, current, elapsed, gpu_monitor):
    """
    Collect hardware metrics and calculate system load.
    """

    cpu_usage = float(current["cpu_usage"])
    ram_usage = float(current["ram_usage"])

    # ----------------------------------------------
    # GPU
    # ----------------------------------------------

    try:
        gpu_usage, _gpu_age = gpu_monitor.get_data()
        gpu_usage = normalize_gpu_usage(gpu_usage)

    except Exception:
        gpu_usage = None

    # ----------------------------------------------
    # DISK
    # ----------------------------------------------

    disk_read = monitor.calculate_speed(
        current["disk"],
        previous["disk"],
        "read_bytes",
        elapsed,
    )

    disk_write = monitor.calculate_speed(
        current["disk"],
        previous["disk"],
        "write_bytes",
        elapsed,
    )

    # ----------------------------------------------
    # NETWORK
    # ----------------------------------------------

    network_upload = monitor.calculate_speed(
        current["network"],
        previous["network"],
        "bytes_sent",
        elapsed,
    )

    network_download = monitor.calculate_speed(
        current["network"],
        previous["network"],
        "bytes_recv",
        elapsed,
    )

    # ----------------------------------------------
    # SYSTEM LOAD
    # ----------------------------------------------

    load_score, load_category = calculate_system_load(
        cpu_usage,
        ram_usage,
    )

    return {
        "cpu_usage": round(cpu_usage, 2),
        "ram_usage": round(ram_usage, 2),
        "gpu_usage": gpu_usage,
        "disk_read_bytes_per_sec": disk_read,
        "disk_write_bytes_per_sec": disk_write,
        "network_upload_bytes_per_sec": network_upload,
        "network_download_bytes_per_sec": network_download,
        "system_load_score": load_score,
        "load_category": load_category,
    }


# ==================================================
# DATASET WRITER
# ==================================================

def write_metrics(writer, csv_file, metrics, device_id, session_id):
    """
    Write one hardware sample to CSV.
    """

    row = {
        "timestamp": datetime.now()
        .astimezone()
        .isoformat(timespec="seconds"),

        "device_id": device_id,
        "session_id": session_id,

        **metrics,
    }

    writer.writerow(row)

    # Flush each sample to reduce data loss if interrupted.
    csv_file.flush()

    return row


# ==================================================
# MAIN COLLECTOR
# ==================================================

def collect():
    """
    Start continuous hardware monitoring and CSV collection.
    """

    output_dir = (
        Path(__file__).resolve().parent.parent
        / "dataset"
        / "raw"
    )

    output_file = output_dir / "hardware_metrics.csv"
    devices_file = output_dir / "devices.csv"

    print("=" * 55)
    print("              PANOPTIX DATA COLLECTOR")
    print("=" * 55)

    # ----------------------------------------------
    # DATASET INITIALIZATION
    # ----------------------------------------------

    initialize_dataset(output_file)

    # ----------------------------------------------
    # DEVICE INITIALIZATION
    # ----------------------------------------------

    default_device_id = re.sub(
        r"[^A-Za-z0-9_-]+",
        "_",
        socket.gethostname(),
    ).strip("_").upper()

    device_id = (
        input(f"Device ID [{default_device_id}]: ").strip()
        or default_device_id
    )

    ensure_device(
        devices_file,
        device_id,
    )

    # ----------------------------------------------
    # SESSION INITIALIZATION
    # ----------------------------------------------

    session_id = (
        f"SESSION_"
        f"{datetime.now().astimezone():%Y%m%dT%H%M%S}_"
        f"{uuid.uuid4().hex[:8]}"
    )

    # ----------------------------------------------
    # HARDWARE INITIALIZATION
    # ----------------------------------------------

    previous = monitor.initialize_monitoring()

    # ----------------------------------------------
    # GPU INITIALIZATION
    # ----------------------------------------------

    gpu_monitor = initialize_gpu_monitor()

    try:
        wait_for_gpu(gpu_monitor)

    except Exception as error:
        print(f"GPU initialization warning: {error}")

    # ----------------------------------------------
    # COLLECTOR INFORMATION
    # ----------------------------------------------

    print("\n=== PANOPTIX SYSTEM LOAD COLLECTOR ===")
    print(f"Device ID : {device_id}")
    print(f"Session ID: {session_id}")
    print(f"Dataset   : {output_file}")
    print(f"Interval  : {monitor.MONITOR_INTERVAL:g} seconds")
    print("Label     : Low / Moderate / High")
    print("GPU       : Optional")
    print("Tekan Ctrl+C untuk menghentikan monitoring.\n")

    sample_count = 0

    # ----------------------------------------------
    # COLLECTION LOOP
    # ----------------------------------------------

    try:

        with output_file.open(
            "a",
            newline="",
            encoding="utf-8",
        ) as csv_file:

            writer = csv.DictWriter(
                csv_file,
                fieldnames=CSV_FIELDS,
            )

            while True:

                time.sleep(monitor.MONITOR_INTERVAL)

                current = monitor.get_hardware_metrics()

                current_time = time.monotonic()

                elapsed = (
                    current_time - previous["timestamp"]
                )

                if elapsed <= 0:
                    continue

                metrics = collect_metrics(
                    previous,
                    current,
                    elapsed,
                    gpu_monitor,
                )

                row = write_metrics(
                    writer,
                    csv_file,
                    metrics,
                    device_id,
                    session_id,
                )

                sample_count += 1

                # ----------------------------------
                # TERMINAL OUTPUT
                # ----------------------------------

                gpu_display = (
                    f"{metrics['gpu_usage']:.1f}%"
                    if metrics["gpu_usage"] is not None
                    else "N/A"
                )

                print(
                    f"[{row['timestamp']}] "
                    f"CPU: {metrics['cpu_usage']:5.1f}% | "
                    f"RAM: {metrics['ram_usage']:5.1f}% | "
                    f"GPU: {gpu_display:>8} | "
                    f"Score: {metrics['system_load_score']:5.2f} | "
                    f"Load: {metrics['load_category']}"
                )

                # ----------------------------------
                # UPDATE BASELINE
                # ----------------------------------

                previous = {
                    "disk": current["disk"],
                    "network": current["network"],
                    "timestamp": current_time,
                }

    except KeyboardInterrupt:

        print("\n\nMonitoring dihentikan.")

    except Exception as error:

        print(f"\nCollector error: {error}")
        raise

    finally:

        try:
            gpu_monitor.stop()
        except Exception:
            pass

        print("\nGPU monitoring dihentikan.")
        print(f"Total sampel: {sample_count}")
        print("Collector selesai.")


# ==================================================
# ENTRY POINT
# ==================================================

if __name__ == "__main__":
    collect()




