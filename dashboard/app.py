"""Panoptix Dashboard — Minimal Modern Red & Black."""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from monitoring.service import PanoptixMonitoringService
from prediction.predict import PanoptixPredictor

RAW_DIR = BASE_DIR / "dataset" / "raw"
METRICS_FILE = RAW_DIR / "hardware_metrics.csv"
DEVICES_FILE = RAW_DIR / "devices.csv"
MODEL_FILE = BASE_DIR / "model" / "panoptix_ann.keras"

st.set_page_config(
    page_title="Panoptix",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def monitoring_service():
    return PanoptixMonitoringService(interval=2.0)


@st.cache_resource
def ann_predictor():
    return PanoptixPredictor()


service = monitoring_service()


# -------------------- STYLE --------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root {
  --background: #0b0b0c;
  --surface: #121214;
  --surface-soft: #18181b;
  --line: #27272a;
  --red: #e33131;
  --red-soft: #ff5a5f;
  --text: #f4f4f5;
  --muted: #929298;
}

.stApp { background: var(--background); color: var(--text); }
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1, h2, h3, h4 { font-family: 'Manrope', sans-serif !important; letter-spacing: -0.6px; }
[data-testid="stSidebar"] {
  background: #101012;
  border-right: 1px solid var(--line);
}
[data-testid="stSidebar"] > div { padding-top: 1.6rem; }
.block-container { padding-top: 2.1rem; padding-bottom: 2.5rem; max-width: 1500px; }

.brand {
  font-family: 'Manrope', sans-serif;
  font-weight: 800;
  font-size: 1.65rem;
  letter-spacing: -1.2px;
  color: #fff;
}
.brand-mark { color: var(--red); }
.brand-caption {
  color: #77777e;
  font: 10px 'IBM Plex Mono', monospace;
  letter-spacing: 1.1px;
  margin-top: 3px;
}
.section-kicker {
  color: var(--red-soft);
  font: 10px 'IBM Plex Mono', monospace;
  letter-spacing: 1.5px;
  text-transform: uppercase;
  margin-bottom: 8px;
}
.page-heading {
  color: #fafafa;
  font: 700 29px 'Manrope', sans-serif;
  letter-spacing: -1.1px;
  margin: 0;
}
.page-description {
  color: var(--muted);
  font-size: 13px;
  margin-top: 6px;
  margin-bottom: 25px;
}
.metric-label {
  color: #8e8e95;
  font: 10px 'IBM Plex Mono', monospace;
  letter-spacing: 1px;
  text-transform: uppercase;
}
.metric-value {
  color: #fafafa;
  font: 700 32px 'Manrope', sans-serif;
  letter-spacing: -1.2px;
  margin-top: 5px;
}
.metric-unit { color: #88888f; font-size: 12px; font-weight: 500; }
.metric-wrap {
  padding: 16px 17px 15px;
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 10px;
  min-height: 100px;
}
.metric-wrap.red { border-top: 2px solid var(--red); }
.panel {
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 10px;
  padding: 17px 18px;
}
.panel-title {
  color: #e7e7e9;
  font: 600 13px 'Manrope', sans-serif;
  letter-spacing: -.1px;
  margin-bottom: 14px;
}
.muted { color: var(--muted); font-size: 12px; }
.mono { font-family: 'IBM Plex Mono', monospace; }
.status-dot {
  display: inline-block;
  width: 7px; height: 7px; border-radius: 50%;
  background: #22c55e; margin-right: 7px;
}
.status-text {
  color: #b4b4ba;
  font: 10px 'IBM Plex Mono', monospace;
  letter-spacing: .8px;
}
.big-prediction {
  font: 800 30px 'Manrope', sans-serif;
  letter-spacing: -1px;
}
div[data-testid="stMetric"] {
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 10px;
  padding: 12px 15px;
}
[data-testid="stMetricLabel"] { color: #96969d !important; font-size: 10px !important; letter-spacing: .8px; }
[data-testid="stMetricValue"] { font-family: 'Manrope', sans-serif !important; font-size: 25px !important; }
div.stButton > button {
  border-radius: 7px;
  border: 1px solid #343438;
  font-weight: 600;
  min-height: 38px;
}
div.stButton > button[kind="primary"] {
  background: var(--red);
  border-color: var(--red);
  color: white;
}
div.stButton > button[kind="primary"]:hover { background: #c92828; border-color: #c92828; }
[data-testid="stProgressBar"] > div > div > div { background: var(--red); }
[data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 8px; }
hr { border-color: var(--line) !important; }
code, pre { font-family: 'IBM Plex Mono', monospace !important; }
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# -------------------- HELPERS --------------------

def page_header(section, title, description):
    st.markdown(f'<div class="section-kicker">{section}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="page-heading">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="page-description">{description}</div>', unsafe_allow_html=True)


def metric_block(label, value, unit="", accent=False):
    cls = "metric-wrap red" if accent else "metric-wrap"
    st.markdown(
        f'<div class="{cls}"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}<span class="metric-unit"> {unit}</span></div></div>',
        unsafe_allow_html=True,
    )


def read_csv(path):
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(
                df["timestamp"], errors="coerce", utc=True
            ).dt.tz_convert(None)
        return df
    except (OSError, UnicodeDecodeError, pd.errors.ParserError):
        return pd.DataFrame()


def format_rate(value):
    if value is None or pd.isna(value):
        return "N/A"
    value = float(value)
    for unit in ("B/s", "KB/s", "MB/s", "GB/s"):
        if value < 1024 or unit == "GB/s":
            return f"{value:.2f} {unit}"
        value /= 1024
    return "N/A"


def get_prediction(metrics):
    return ann_predictor().predict(metrics["cpu_usage"], metrics["ram_usage"])


def prediction_color(label):
    return {"Low": "#4ade80", "Moderate": "#fbbf24", "High": "#ff5a5f"}.get(label, "#fff")


# -------------------- SIDEBAR --------------------

with st.sidebar:
    st.markdown(
        '<div class="brand">panoptix<span class="brand-mark">.</span></div>'
        '<div class="brand-caption">HARDWARE INTELLIGENCE</div>',
        unsafe_allow_html=True,
    )
    st.write("")
    st.write("")
    page = st.radio(
        "Workspace",
        [
            "Overview",
            "Hardware",
            "Network & Disk",
            "AI Analytics",
            "History",
            "Dataset",
            "System",
        ],
        label_visibility="collapsed",
    )
    st.divider()

    sidebar_snapshot = service.get_snapshot()
    if sidebar_snapshot["running"]:
        st.markdown('<span class="status-dot"></span><span class="status-text">MONITORING ACTIVE</span>', unsafe_allow_html=True)
        if st.button("Stop monitoring", use_container_width=True):
            service.stop()
            st.rerun()
    else:
        st.markdown('<span class="status-text">MONITORING PAUSED</span>', unsafe_allow_html=True)
        if st.button("Start monitoring", type="primary", use_container_width=True):
            try:
                service.start()
                st.rerun()
            except Exception as exc:
                st.error(f"Could not start: {exc}")

    st.divider()
    st.markdown('<div class="metric-label">DEVICE</div>', unsafe_allow_html=True)
    st.caption(service.device_id)
    st.markdown('<div class="metric-label">SAMPLING INTERVAL</div>', unsafe_allow_html=True)
    st.caption("2 seconds")
    st.markdown('<div class="metric-label">MODEL</div>', unsafe_allow_html=True)
    st.caption("Panoptix ANN")


# -------------------- PAGE CONTENT --------------------

def overview(snapshot):
    page_header("WORKSPACE / OVERVIEW", "System overview", "A concise view of current hardware health and system activity.")
    m = snapshot["metrics"]
    if m is None:
        st.info("Monitoring belum aktif. Gunakan tombol Start monitoring pada sidebar.")
        return

    cols = st.columns(4, gap="medium")
    values = [
        ("CPU", f'{m["cpu_usage"]:.1f}', "%"),
        ("MEMORY", f'{m["ram_usage"]:.1f}', "%"),
        ("GPU", f'{m["gpu_usage"]:.1f}' if m.get("gpu_usage") is not None else "N/A", "%" if m.get("gpu_usage") is not None else ""),
        ("LOAD SCORE", f'{m["system_load_score"]:.2f}', "/ 100"),
    ]
    for col, (label, value, unit) in zip(cols, values):
        with col:
            metric_block(label, value, unit, accent=(label == "LOAD SCORE"))

    st.write("")
    left, right = st.columns([1.55, 1], gap="large")
    history = pd.DataFrame(snapshot["history"])
    with left:
        st.markdown('<div class="panel"><div class="panel-title">Resource utilization</div>', unsafe_allow_html=True)
        if not history.empty:
            history["timestamp"] = pd.to_datetime(history["timestamp"], errors="coerce")
            st.line_chart(history.set_index("timestamp")[["cpu_usage", "ram_usage"]], height=275)
        else:
            st.caption("Waiting for telemetry samples.")
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="panel"><div class="panel-title">ANN classification</div>', unsafe_allow_html=True)
        try:
            result = get_prediction(m)
            label = result["predicted_class"]
            st.markdown(
                f'<div class="big-prediction" style="color:{prediction_color(label)}">{label}</div>',
                unsafe_allow_html=True,
            )
            st.caption(f'Model confidence · {result["confidence"] * 100:.2f}%')
            for name, probability in result["probabilities"].items():
                st.caption(f"{name}  {probability * 100:.1f}%")
                st.progress(float(probability))
        except Exception as exc:
            st.error(f"ANN prediction error: {exc}")
        st.markdown('</div>', unsafe_allow_html=True)

    st.write("")
    st.markdown("**I/O throughput**")
    cols = st.columns(4)
    for col, (label, key) in zip(cols, [
        ("Disk read", "disk_read_bytes_per_sec"),
        ("Disk write", "disk_write_bytes_per_sec"),
        ("Network upload", "network_upload_bytes_per_sec"),
        ("Network download", "network_download_bytes_per_sec"),
    ]):
        with col:
            st.metric(label, format_rate(m.get(key)))

    st.caption(f'Session {snapshot["session_id"] or "—"} · {snapshot["sample_count"]} samples collected')


def hardware(snapshot):
    page_header("TELEMETRY / HARDWARE", "Hardware", "Live processor, memory and graphics utilization.")
    m = snapshot["metrics"]
    if m is None:
        st.info("Start monitoring to view hardware telemetry.")
        return
    cols = st.columns(3)
    cols[0].metric("CPU utilization", f'{m["cpu_usage"]:.2f}%')
    cols[1].metric("Memory utilization", f'{m["ram_usage"]:.2f}%')
    gpu = m.get("gpu_usage")
    cols[2].metric("GPU utilization", f"{gpu:.2f}%" if gpu is not None else "N/A")
    hist = pd.DataFrame(snapshot["history"])
    if not hist.empty:
        hist["timestamp"] = pd.to_datetime(hist["timestamp"], errors="coerce")
        st.markdown("**Utilization over time**")
        st.line_chart(hist.set_index("timestamp")[["cpu_usage", "ram_usage"]], height=310)
        if "gpu_usage" in hist and hist["gpu_usage"].notna().any():
            st.markdown("**GPU engine utilization**")
            st.line_chart(hist.set_index("timestamp")[["gpu_usage"]].dropna(), height=220)
        st.markdown("**Recent samples**")
        cols = [c for c in ["timestamp", "cpu_usage", "ram_usage", "gpu_usage"] if c in hist]
        st.dataframe(hist[cols].tail(20).iloc[::-1], use_container_width=True, hide_index=True)


def network_disk(snapshot):
    page_header("TELEMETRY / I-O", "Network & disk", "Current throughput and recent storage/network activity.")
    m = snapshot["metrics"]
    if m is None:
        st.info("Start monitoring to view I/O telemetry.")
        return
    cols = st.columns(4)
    for col, (label, key) in zip(cols, [
        ("Disk read", "disk_read_bytes_per_sec"),
        ("Disk write", "disk_write_bytes_per_sec"),
        ("Upload", "network_upload_bytes_per_sec"),
        ("Download", "network_download_bytes_per_sec"),
    ]):
        col.metric(label, format_rate(m.get(key)))
    hist = pd.DataFrame(snapshot["history"])
    if hist.empty:
        return
    hist["timestamp"] = pd.to_datetime(hist["timestamp"], errors="coerce")
    indexed = hist.set_index("timestamp")
    st.markdown("**Disk throughput**")
    keys = [k for k in ["disk_read_bytes_per_sec", "disk_write_bytes_per_sec"] if k in indexed]
    if keys:
        st.line_chart(indexed[keys], height=250)
    st.markdown("**Network throughput**")
    keys = [k for k in ["network_upload_bytes_per_sec", "network_download_bytes_per_sec"] if k in indexed]
    if keys:
        st.line_chart(indexed[keys], height=250)


def analytics(snapshot):
    page_header("INTELLIGENCE / ANN", "AI analytics", "ANN classification based on CPU and RAM utilization.")
    m = snapshot["metrics"]
    if m is None:
        st.info("Start monitoring to run ANN inference.")
        return
    try:
        result = get_prediction(m)
    except Exception as exc:
        st.error(f"ANN prediction error: {exc}")
        return
    left, right = st.columns([1, 1.5], gap="large")
    with left:
        st.markdown('<div class="panel"><div class="panel-title">Predicted system load</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="big-prediction" style="color:{prediction_color(result["predicted_class"])}">{result["predicted_class"]}</div>', unsafe_allow_html=True)
        st.metric("Confidence", f'{result["confidence"] * 100:.2f}%')
        st.caption(f'CPU {result["cpu_usage"]:.2f}% · RAM {result["ram_usage"]:.2f}%')
        st.markdown('</div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="panel"><div class="panel-title">Class probabilities</div>', unsafe_allow_html=True)
        for name, probability in result["probabilities"].items():
            st.caption(f"{name} · {probability * 100:.2f}%")
            st.progress(float(probability))
        st.markdown('</div>', unsafe_allow_html=True)
    st.caption("Confidence is the model's Softmax output and is not separately probability-calibrated.")


def history_page():
    page_header("DATA / HISTORY", "Monitoring history", "Explore and export recorded hardware telemetry.")
    df = read_csv(METRICS_FILE)
    if df.empty:
        st.info("No historical samples found in hardware_metrics.csv.")
        return
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    if df.empty:
        st.info("No valid timestamps found.")
        return
    date_range = st.date_input("Date range", value=(df.timestamp.min().date(), df.timestamp.max().date()))
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start, end = date_range
        df = df[(df.timestamp.dt.date >= start) & (df.timestamp.dt.date <= end)]
    cols = st.columns(3)
    cols[0].metric("Samples", f"{len(df):,}")
    cols[1].metric("Sessions", str(df.session_id.nunique()) if "session_id" in df else "—")
    cols[2].metric("Devices", str(df.device_id.nunique()) if "device_id" in df else "—")
    keys = [k for k in ["cpu_usage", "ram_usage", "gpu_usage"] if k in df]
    if keys and not df.empty:
        st.markdown("**Resource history**")
        st.line_chart(df.set_index("timestamp")[keys], height=320)
    if "load_category" in df:
        st.markdown("**Load category distribution**")
        st.bar_chart(df.load_category.value_counts().rename_axis("Category").to_frame("Samples"))
    st.markdown("**Recorded samples**")
    st.dataframe(df.tail(200).iloc[::-1], use_container_width=True, hide_index=True)
    st.download_button("Export filtered CSV", df.to_csv(index=False).encode("utf-8"), "panoptix_history.csv", "text/csv")


def dataset_page():
    page_header("DATA / DATASET", "Dataset", "Sample volume, class distribution and registered devices.")
    df = read_csv(METRICS_FILE)
    devices = read_csv(DEVICES_FILE)
    if df.empty:
        st.info("Dataset is currently empty.")
    else:
        cols = st.columns(4)
        cols[0].metric("Total samples", f"{len(df):,}")
        cols[1].metric("Devices", str(df.device_id.nunique()) if "device_id" in df else "—")
        cols[2].metric("Sessions", str(df.session_id.nunique()) if "session_id" in df else "—")
        cols[3].metric("GPU missing", str(df.gpu_usage.isna().sum()) if "gpu_usage" in df else "—")
        if "load_category" in df:
            st.markdown("**Class distribution**")
            st.bar_chart(df.load_category.value_counts().rename_axis("Category").to_frame("Samples"))
        if "session_id" in df:
            st.markdown("**Session summary**")
            summary = df.groupby("session_id").agg(
                samples=("session_id", "size"),
                first_sample=("timestamp", "min"),
                last_sample=("timestamp", "max"),
            ).reset_index()
            st.dataframe(summary, use_container_width=True, hide_index=True)
    st.markdown("**Registered devices**")
    if devices.empty:
        st.caption("No device metadata available.")
    else:
        st.dataframe(devices, use_container_width=True, hide_index=True)


def system_page(snapshot):
    page_header("SYSTEM / CONFIGURATION", "System information", "Device metadata, application runtime and model availability.")
    devices = read_csv(DEVICES_FILE)
    device = None
    if not devices.empty and "device_id" in devices:
        found = devices[devices.device_id == service.device_id]
        if not found.empty:
            device = found.iloc[-1].to_dict()
    cols = st.columns(3)
    cols[0].metric("Monitoring", "Active" if snapshot["running"] else "Stopped")
    cols[1].metric("Session samples", str(snapshot["sample_count"]))
    cols[2].metric("ANN model", "Available" if MODEL_FILE.exists() else "Missing")
    with st.container(border=True):
        st.markdown("**Device profile**")
        if device:
            for key in ["device_id", "CPU", "GPU", "RAM", "OS"]:
                st.write(f"**{key}:** {device.get(key, 'N/A')}")
        else:
            st.info("Device metadata has not been registered yet.")
    with st.container(border=True):
        st.markdown("**Runtime**")
        st.write(f"Sampling interval: {service.interval} seconds")
        st.write(f"Dataset path: `{METRICS_FILE}`")
        st.write(f"Model path: `{MODEL_FILE}`")
        st.write(f'Session ID: `{snapshot["session_id"] or "—"}`')
        st.write(f'Last error: `{snapshot["last_error"] or "None"}`')


@st.fragment(run_every="2s")
def live_page(selected):
    snapshot = service.get_snapshot()
    if selected == "Overview":
        overview(snapshot)
    elif selected == "Hardware":
        hardware(snapshot)
    elif selected == "Network & Disk":
        network_disk(snapshot)
    elif selected == "AI Analytics":
        analytics(snapshot)
    elif selected == "System":
        system_page(snapshot)


if page in ["Overview", "Hardware", "Network & Disk", "AI Analytics", "System"]:
    live_page(page)
elif page == "History":
    history_page()
elif page == "Dataset":
    dataset_page()

st.markdown(
    '<div style="border-top:1px solid #27272a;margin-top:32px;padding-top:12px;color:#666;font:10px IBM Plex Mono,monospace;">PANOPTIX <span style="float:right">RED / BLACK</span></div>',
    unsafe_allow_html=True,
)
