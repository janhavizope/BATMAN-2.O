import sys, os, time, json, subprocess, asyncio, base64
import urllib.request
import websockets

EDGE_PATH = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
PORT = 9222

PAGES = [
    ("Overview", "outputs/streamlit_screenshots/page_overview.png"),
    ("Transaction Analysis", "outputs/streamlit_screenshots/page_transaction_analysis.png"),
    ("Network Graph", "outputs/streamlit_screenshots/page_network_graph.png"),
    ("Entity Investigation", "outputs/streamlit_screenshots/page_entity_investigation.png"),
    ("Alerts", "outputs/streamlit_screenshots/page_alerts.png"),
    ("Explainability", "outputs/streamlit_screenshots/page_explainability.png"),
]

async def capture_all_pages():
    req = urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json")
    targets = json.loads(req.read().decode('utf-8'))
    ws_url = None
    for t in targets:
        if t.get("type") == "page":
            ws_url = t.get("webSocketDebuggerUrl")
            break
            
    if not ws_url:
        req = urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json/new")
        target = json.loads(req.read().decode('utf-8'))
        ws_url = target.get("webSocketDebuggerUrl")

    async with websockets.connect(ws_url, max_size=20*1024*1024) as ws:
        msg_id = 1
        # Set viewport metrics override to full desktop resolution
        await ws.send(json.dumps({
            "id": msg_id,
            "method": "Emulation.setDeviceMetricsOverride",
            "params": {
                "width": 1400,
                "height": 900,
                "deviceScaleFactor": 1,
                "mobile": False
            }
        }))
        msg_id += 1
        # 1. Navigate to Streamlit root
        await ws.send(json.dumps({"id": msg_id, "method": "Page.navigate", "params": {"url": "http://127.0.0.1:8501"}}))
        await asyncio.sleep(5) # wait for initial load

        for nav_name, out_path in PAGES:
            msg_id += 1
            # Click the radio button label if not on Overview
            if nav_name != "Overview":
                js_click = f"""
                (() => {{
                    const labels = Array.from(document.querySelectorAll('[data-testid="stRadio"] label'));
                    for (const l of labels) {{
                        if (l.innerText.includes("{nav_name}")) {{
                            l.click();
                            return true;
                        }}
                    }}
                    return false;
                }})()
                """
                await ws.send(json.dumps({
                    "id": msg_id,
                    "method": "Runtime.evaluate",
                    "params": {"expression": js_click}
                }))
                await asyncio.sleep(3) # wait for page to render

            # Capture screenshot
            msg_id += 1
            await ws.send(json.dumps({"id": msg_id, "method": "Page.captureScreenshot", "params": {"format": "png"}}))
            while True:
                resp = json.loads(await ws.recv())
                if resp.get("id") == msg_id:
                    img_data = base64.b64decode(resp["result"]["data"])
                    with open(out_path, "wb") as f:
                        f.write(img_data)
                    print(f"Captured: {out_path}")
                    break

def main():
    user_data = os.path.join(os.environ.get("TEMP", r"C:\temp"), "edge_cdp_profile_pages")
    proc = subprocess.Popen([
        EDGE_PATH,
        "--headless=new",
        f"--remote-debugging-port={PORT}",
        f"--user-data-dir={user_data}",
        "--window-size=1400,950",
        "--force-device-scale-factor=1",
        "--disable-gpu",
        "about:blank"
    ])
    time.sleep(2)

    try:
        os.makedirs("outputs/streamlit_screenshots", exist_ok=True)
        asyncio.run(capture_all_pages())
        print("All pages successfully captured!")
    finally:
        proc.kill()

if __name__ == "__main__":
    main()
