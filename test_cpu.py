
import time
import psutil

print("PANOPTIX CPU DIAGNOSTIC")
print("Logical processors:", psutil.cpu_count(logical=True))
print("Physical cores:", psutil.cpu_count(logical=False))

# Initialize CPU counters
psutil.cpu_percent(interval=None)
psutil.cpu_percent(interval=None, percpu=True)

print("\nSampling every 1 second...\n")

try:
    while True:
        time.sleep(1)

        total = psutil.cpu_percent(interval=None)
        per_core = psutil.cpu_percent(
            interval=None,
            percpu=True
        )

        print(
            f"Total: {total:5.1f}% | "
            f"Cores: {[round(x, 1) for x in per_core]}"
        )

except KeyboardInterrupt:
    print("\nDiagnostic stopped.")
