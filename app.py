"""
Industrial Quality Control Station // Defect Detection Entrypoint.
High-throughput, budget-aware anomaly detection system for manufacturing lines.
"""
import sys
from pathlib import Path
import streamlit as st
import torch
from torch.utils.data import DataLoader

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from config import (
    CATEGORIES,
    DEFAULT_CATEGORY,
    DEFAULT_BATCH_SIZE,
    DEFAULT_REVIEW_CAPACITY_PCT,
    DEFAULT_DEFECT_RATE_PCT,
    DEFAULT_COST_REVIEW,
    DEFAULT_COST_MISSED,
    DATA_RAW_DIR,
    ASSETS_DIR
)
from src.dataset import MVTecCategoryDataset, get_default_transforms
from src.model import ResNetAnomalyDetector
from ui.components import render_header_bar, render_glossary_expander
from ui.tabs.inspection import render_inspection_tab
from ui.tabs.simulation import render_simulation_tab
from ui.tabs.benchmark import render_benchmark_tab

# Page Configuration
st.set_page_config(
    page_title="Industrial Quality Control // Anomaly Detection",
    page_icon="■",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load CSS
css_file = ASSETS_DIR / "style.css"
if css_file.exists():
    with open(css_file, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ----------------- SESSION STATE & RESET LOGIC -----------------
def reset_to_defaults():
    st.session_state["slider_batch_size"] = DEFAULT_BATCH_SIZE
    st.session_state["slider_review_capacity"] = DEFAULT_REVIEW_CAPACITY_PCT
    st.session_state["slider_defect_rate"] = DEFAULT_DEFECT_RATE_PCT
    st.session_state["num_cost_review"] = DEFAULT_COST_REVIEW
    st.session_state["num_cost_missed"] = DEFAULT_COST_MISSED

if "slider_batch_size" not in st.session_state:
    reset_to_defaults()

if "audience_mode" not in st.session_state:
    st.session_state["audience_mode"] = "Operator view"

# ----------------- CACHED MODEL LOADING -----------------
@st.cache_resource
def load_category_data(cat: str):
    """Caches model, embeddings, and test evaluations on demand per category."""
    cat_dir = DATA_RAW_DIR / cat
    transform = get_default_transforms()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    train_ds = MVTecCategoryDataset(cat_dir, split="train", transform=transform)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=False)
    
    detector = ResNetAnomalyDetector(device=device, backbone="resnet18")
    detector.fit(train_loader)
    
    test_ds = MVTecCategoryDataset(cat_dir, split="test", transform=transform)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)
    eval_res = detector.evaluate(test_loader)
    
    return {
        "detector": detector,
        "eval_res": eval_res,
        "test_ds": test_ds,
        "transform": transform
    }

def get_all_categories_cache():
    """Retrieves all cached category detectors for the multi-category benchmark."""
    return {cat: load_category_data(cat) for cat in CATEGORIES}

# ----------------- SIDEBAR CONTROLS -----------------
st.sidebar.markdown("### SYSTEM CONTROL // MODE")
audience_mode = st.sidebar.radio(
    "Select Operating Perspective:",
    ["Operator view", "Engineer view"],
    index=0 if st.session_state["audience_mode"] == "Operator view" else 1,
    help="Operator view simplifies technical terms into actionable decision language. Engineer view exposes raw scores, calibration thresholds, and mathematical formulas."
)
st.session_state["audience_mode"] = audience_mode

st.sidebar.markdown("### PRODUCTION DOMAIN")
selected_category = st.sidebar.selectbox(
    "Active Inspection Line:",
    CATEGORIES,
    index=CATEGORIES.index(DEFAULT_CATEGORY),
    help="Select the active manufacturing category to inspect and simulate."
)

st.sidebar.markdown("### SHIFT BATCH CONFIGURATION")
batch_size = st.sidebar.slider(
    "Shift Production Batch (Units):",
    min_value=200,
    max_value=2000,
    value=st.session_state["slider_batch_size"],
    step=100,
    key="slider_batch_size",
    help="Total number of parts produced and passed under the optical sensor during one work shift."
)

st.sidebar.markdown("### INSPECTION CAPACITY")
review_capacity_pct = st.sidebar.slider(
    "Human Review Quota (% of Batch):",
    min_value=2,
    max_value=30,
    value=st.session_state["slider_review_capacity"],
    step=1,
    key="slider_review_capacity",
    help="The maximum percentage of total produced items that human inspectors have the physical capacity to examine."
)

defect_rate_pct = st.sidebar.slider(
    "True Defect Prevalence (%):",
    min_value=1.0,
    max_value=15.0,
    value=st.session_state["slider_defect_rate"],
    step=0.5,
    key="slider_defect_rate",
    help="The expected baseline defect rate in manufacturing for this production shift."
)

st.sidebar.markdown("### FINANCIAL ASSUMPTIONS")
cost_review = st.sidebar.number_input(
    "Inspector Labor Cost per Item ($):",
    min_value=0.10,
    max_value=10.0,
    value=st.session_state["num_cost_review"],
    step=0.10,
    key="num_cost_review",
    help="The labor cost required for an inspector to conduct a manual visual check on one flagged product."
)

cost_miss = st.sidebar.number_input(
    "Cost of an Escaped Defect ($):",
    min_value=5.0,
    max_value=250.0,
    value=st.session_state["num_cost_missed"],
    step=5.0,
    key="num_cost_missed",
    help="The commercial penalty incurred if an uninspected defective product escapes to the customer (warranty, rework, returns)."
)

st.sidebar.button("Reset Parameters to Defaults", on_click=reset_to_defaults, use_container_width=True)

# ----------------- MAIN APP EXECUTION -----------------
active_data = load_category_data(selected_category)
detector = active_data["detector"]
eval_res = active_data["eval_res"]
test_ds = active_data["test_ds"]
transform = active_data["transform"]

normal_scores = eval_res["scores"][eval_res["labels"] == 0]
defect_scores = eval_res["scores"][eval_res["labels"] == 1]
device_str = "NVIDIA RTX 3050 (CUDA)" if torch.cuda.is_available() else "CPU HOST"

# Top Industrial Status Bar
render_header_bar(
    category=selected_category,
    mode=audience_mode,
    device=device_str,
    auroc=eval_res["auroc"] * 100.0
)

# Tab Navigation
tab_inspect, tab_sim, tab_bench = st.tabs([
    "INSPECTION STATION",
    "FACTORY SHIFT SIMULATION",
    "MULTI-CATEGORY BENCHMARK"
])

with tab_inspect:
    render_inspection_tab(
        detector=detector,
        eval_res=eval_res,
        test_ds=test_ds,
        transform=transform,
        selected_category=selected_category,
        audience_mode=audience_mode
    )

with tab_sim:
    render_simulation_tab(
        normal_scores=normal_scores,
        defect_scores=defect_scores,
        batch_size=batch_size,
        review_capacity_pct=review_capacity_pct,
        defect_rate_pct=defect_rate_pct,
        cost_review=cost_review,
        cost_miss=cost_miss,
        threshold=eval_res["threshold"],
        selected_category=selected_category,
        audience_mode=audience_mode
    )

with tab_bench:
    render_benchmark_tab(
        category_cache=get_all_categories_cache(),
        audience_mode=audience_mode
    )

# Bottom Model Card & Glossary Expander
st.markdown("---")
render_glossary_expander()

if audience_mode == "Engineer view":
    with st.expander("MODEL CARD & SYSTEM SPECIFICATION // MVTec AD BENCHMARK"):
        st.markdown(f"""
        - **Architecture:** `ResNet-18` (Feature backbone with classification head stripped via `nn.Identity()`).
        - **Feature Dimension:** 512-dimensional continuous latent space (`AdaptiveAvgPool2d`).
        - **Nominal Archetype:** Vector centroid $\\mathbf{{\\mu}} = \\frac{{1}}{{N}}\\sum_{{i=1}}^{{N}} \\mathbf{{f}}_i$ computed on defect-free training images.
        - **Threshold Calibration:** 95th percentile of normal training distances: `tau = {eval_res['threshold']:.4f}`.
        - **Display Review Buffer:** 12% relative margin below threshold: `buffer = {eval_res['threshold'] * 0.88:.4f}`.
        - **Memory Footprint:** `< 450 MB VRAM` during batched inference on NVIDIA RTX 3050 GPU.
        - **Hardware:** Local host execution on Ubuntu Linux with PyTorch 2.6.0+cu124.
        """)
