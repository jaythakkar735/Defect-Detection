"""
Automated Playwright Screenshot Capture.
Captures all 3 tabs in both Operator and Engineer views at 1366x850 resolution.
"""
import time
from pathlib import Path
from playwright.sync_api import sync_playwright

OUTPUT_DIR = Path("/home/asus/.gemini/antigravity/brain/26a46698-4be0-4b92-99f7-07803c8d8f1e/screenshots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def capture_all():
    with sync_playwright() as p:
        browser = p.chromium.launch(
            executable_path="/usr/bin/google-chrome",
            headless=True,
            args=["--no-sandbox", "--disable-setuid-sandbox", "--disable-dev-shm-usage"]
        )
        page = browser.new_page(viewport={"width": 1366, "height": 850})
        
        print("Navigating to http://localhost:8501...")
        page.goto("http://localhost:8501", wait_until="networkidle", timeout=30000)
        
        # Wait for header bar
        page.wait_for_selector(".scada-header-bar", timeout=20000)
        time.sleep(2)
        
        # ------------------- 1. OPERATOR VIEW: TAB 1 (INSPECTION) -------------------
        print("1. Capturing Tab 1: Inspection (Operator View)...")
        page.screenshot(path=str(OUTPUT_DIR / "tab1_inspection_operator.png"))
        
        # ------------------- 2. OPERATOR VIEW: TAB 2 (SIMULATION) -------------------
        print("2. Switching to Tab 2: Factory Shift Simulation...")
        tab2 = page.get_by_role("tab", name="FACTORY SHIFT SIMULATION")
        tab2.click()
        time.sleep(3)
        print("   Capturing Tab 2: Shift Simulation (Operator View)...")
        page.screenshot(path=str(OUTPUT_DIR / "tab2_simulation_operator.png"))
        
        # ------------------- 3. OPERATOR VIEW: TAB 3 (BENCHMARK) -------------------
        print("3. Switching to Tab 3: Multi-Category Benchmark...")
        tab3 = page.get_by_role("tab", name="MULTI-CATEGORY BENCHMARK")
        tab3.click()
        # Wait for benchmark header text to appear
        page.get_by_text("CROSS-DOMAIN PORTFOLIO").wait_for(state="visible", timeout=60000)
        time.sleep(3)
        print("   Capturing Tab 3: Benchmark (Operator View)...")
        page.screenshot(path=str(OUTPUT_DIR / "tab3_benchmark_operator.png"))
        
        # ------------------- 4. ENGINEER VIEW: TAB 3 (BENCHMARK) -------------------
        print("4. Switching to 'Engineer view' in sidebar...")
        eng_radio = page.get_by_text("Engineer view")
        eng_radio.click()
        time.sleep(3)
        print("   Capturing Tab 3: Benchmark (Engineer View)...")
        page.screenshot(path=str(OUTPUT_DIR / "tab3_benchmark_engineer.png"))
        
        # ------------------- 5. ENGINEER VIEW: TAB 2 (SIMULATION) -------------------
        print("5. Switching to Tab 2: Shift Simulation (Engineer View)...")
        tab2 = page.get_by_role("tab", name="FACTORY SHIFT SIMULATION")
        tab2.click()
        time.sleep(2)
        print("   Capturing Tab 2: Shift Simulation (Engineer View)...")
        page.screenshot(path=str(OUTPUT_DIR / "tab2_simulation_engineer.png"))
        
        # ------------------- 6. ENGINEER VIEW: TAB 1 (INSPECTION) -------------------
        print("6. Switching to Tab 1: Inspection (Engineer View)...")
        tab1 = page.get_by_role("tab", name="INSPECTION STATION")
        tab1.click()
        time.sleep(2)
        print("   Capturing Tab 1: Inspection (Engineer View)...")
        page.screenshot(path=str(OUTPUT_DIR / "tab1_inspection_engineer.png"))
        
        browser.close()
        print(f"SUCCESS: All 6 screenshots captured in {OUTPUT_DIR}!")

if __name__ == "__main__":
    capture_all()
