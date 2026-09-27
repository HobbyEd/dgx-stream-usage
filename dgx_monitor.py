#!/usr/bin/env python3
"""
DGX Spark (GB10) Home Assistant Telemetry Service
NVIDIA DGX Style Dashboard with Home Assistant Dark/Amber Theme
"""

import os
import sys
import time
import json
import datetime
import threading
import subprocess
from collections import deque
from io import BytesIO
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn

import psutil
from PIL import Image, ImageDraw, ImageFont

PORT = int(os.environ.get("PORT", 9199))
FPS = float(os.environ.get("FPS", 2.0))
FRAME_INTERVAL = 1.0 / FPS

# Theme Colors (Matching Home Assistant Amber / Dark Palette)
BG_COLOR = (24, 25, 28)       # #18191c
CARD_BG = (35, 36, 41)        # #232429
CARD_BORDER = (50, 52, 60)    # #32343c
AMBER = (245, 158, 11)        # #f59e0b primary warm amber
AMBER_LIGHT = (251, 191, 36)  # #fbbf24
ORANGE = (241, 157, 56)       # #f19d38
RED = (239, 68, 68)           # Warning red
BLUE_LINE = (33, 150, 243)    # Chart line #2196f3
BLUE_FILL = (19, 56, 110)     # Chart area fill #13386e
GRID_COLOR = (48, 52, 60)
TEXT_WHITE = (255, 255, 255)
TEXT_MUTED = (156, 163, 175)  # #9ca3af

# Fonts (loaded for 2x supersampled canvas)
FONT_DIR = "/usr/share/fonts/truetype/ubuntu"
def load_font(filename, size):
    path = os.path.join(FONT_DIR, filename)
    if os.path.exists(path):
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()

FONTS = {
    "title": load_font("Ubuntu-M.ttf", 52),
    "val_big": load_font("Ubuntu-B.ttf", 86),
    "sub": load_font("Ubuntu-R.ttf", 38),
    "chart_lbl": load_font("Ubuntu-R.ttf", 36),
}

class TelemetryCollector:
    def __init__(self):
        self.lock = threading.Lock()
        self.metrics = {}
        self.current_frame = b""
        self.history_gpu = deque(maxlen=24)
        self.history_mem = deque(maxlen=24)
        self.frame_count = 0

    def query_gpu(self):
        gpu_util = 0.0
        gpu_temp = 0.0
        gpu_power = 0.0
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=utilization.gpu,temperature.gpu,power.draw", "--format=csv,noheader,nounits"],
                text=True, timeout=2
            ).strip().split(", ")
            gpu_util = float(out[0])
            gpu_temp = float(out[1])
            gpu_power = float(out[2])
        except Exception:
            pass
        return gpu_util, gpu_temp, gpu_power

    def sample(self):
        gpu_util, gpu_temp, gpu_power = self.query_gpu()
        vm = psutil.virtual_memory()

        # Decimal GB matching NVIDIA DGX Dashboard
        ram_total_gb = round(vm.total / 1e9, 2)
        # Used memory including buffers/cache active load
        ram_used_gb = round((vm.total - vm.available) / 1e9, 2)
        ram_pct = round((ram_used_gb / ram_total_gb) * 100.0, 1)

        if not self.history_gpu:
            for _ in range(24):
                self.history_gpu.append(gpu_util)
                self.history_mem.append(ram_pct)
        else:
            self.history_gpu.append(gpu_util)
            self.history_mem.append(ram_pct)

        self.frame_count += 1

        metrics = {
            "gpu_utilization_pct": gpu_util,
            "gpu_temperature_c": gpu_temp,
            "gpu_power_w": gpu_power,
            "ram_total_gb": ram_total_gb,
            "ram_used_gb": ram_used_gb,
            "ram_percent": ram_pct,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        frame_bytes = self.render_frame(metrics, list(self.history_mem), list(self.history_gpu))

        with self.lock:
            self.metrics = metrics
            self.current_frame = frame_bytes

    def render_frame(self, m, hist_mem, hist_gpu):
        # 2x supersampled canvas: 1800 x 1360
        W, H = 1800, 1360
        img = Image.new("RGB", (W, H), color=BG_COLOR)
        draw = ImageDraw.Draw(img)

        def draw_card(y_top, y_bot, title, val_str, sub_str, pct, history, max_val_str):
            # Card background with rounded corners
            draw.rounded_rectangle([(60, y_top), (W - 60, y_bot)], radius=32, fill=CARD_BG, outline=CARD_BORDER, width=4)

            # --- LEFT: ARC GAUGE ---
            cx = 450
            cy = y_top + 430
            r_out = 230
            r_in = 165
            gauge_w = r_out - r_in

            # Title centered above arch
            draw.text((cx, y_top + 90), title, fill=TEXT_WHITE, font=FONTS["title"], anchor="mm")

            # Outer thin warning track
            r_thin = r_out + 22
            bbox_thin = [cx - r_thin, cy - r_thin, cx + r_thin, cy + r_thin]
            draw.arc(bbox_thin, start=180, end=360, fill=(55, 60, 70), width=8)
            draw.arc(bbox_thin, start=180, end=180 + int(180 * 0.70), fill=AMBER, width=8)
            draw.arc(bbox_thin, start=180 + int(180 * 0.70), end=180 + int(180 * 0.85), fill=ORANGE, width=8)
            draw.arc(bbox_thin, start=180 + int(180 * 0.85), end=360, fill=RED, width=8)

            # Main thick gauge arch
            bbox_main = [cx - r_out, cy - r_out, cx + r_out, cy + r_out]
            draw.arc(bbox_main, start=180, end=360, fill=(50, 54, 62), width=gauge_w)

            # Fill main arch with warm amber
            if pct > 0:
                angle_end = 180 + int(180 * (min(100.0, max(0.0, pct)) / 100.0))
                draw.arc(bbox_main, start=180, end=angle_end, fill=AMBER, width=gauge_w)

            # Center value & subtitle
            text_y = cy - (r_in // 2) + 20
            draw.text((cx, text_y), val_str, fill=TEXT_WHITE, font=FONTS["val_big"], anchor="mm")
            if sub_str:
                draw.text((cx, text_y + 70), sub_str, fill=TEXT_MUTED, font=FONTS["sub"], anchor="mm")

            # --- RIGHT: CHART ---
            gx = 900
            gy = y_top + 180
            gw = 780
            gh = 380

            # Max label top left of chart
            draw.text((gx, gy - 40), max_val_str, fill=TEXT_MUTED, font=FONTS["chart_lbl"])

            # 4 Horizontal grid lines
            for i in range(4):
                gl_y = gy + int(gh * (i / 3.0))
                draw.line([(gx, gl_y), (gx + gw, gl_y)], fill=GRID_COLOR, width=2)

            if history and len(history) >= 2:
                pts = []
                n = len(history)
                for i, val in enumerate(history):
                    px = gx + int(i * (gw / (n - 1)))
                    py = gy + gh - int(gh * (min(100.0, max(0.0, val)) / 100.0))
                    pts.append((px, py))

                # Blue filled area
                poly = [(gx, gy + gh)] + pts + [(gx + gw, gy + gh)]
                draw.polygon(poly, fill=BLUE_FILL)
                # Blue line
                draw.line(pts, fill=BLUE_LINE, width=6)
                # White nodes on data points
                step = max(1, len(pts) // 10)
                for i in range(0, len(pts), step):
                    px, py = pts[i]
                    draw.ellipse([(px - 9, py - 9), (px + 9, py + 9)], fill=TEXT_WHITE, outline=BLUE_LINE, width=4)
                lpx, lpy = pts[-1]
                draw.ellipse([(lpx - 9, lpy - 9), (lpx + 9, lpy + 9)], fill=TEXT_WHITE, outline=BLUE_LINE, width=4)

        # Card 1: System Memory
        draw_card(
            y_top=50, y_bot=670,
            title="System Memory",
            val_str=f"{m['ram_used_gb']:.2f} GB",
            sub_str=f"{m['ram_total_gb']:.2f} GB total",
            pct=m["ram_percent"],
            history=hist_mem,
            max_val_str=f"{m['ram_total_gb']:.2f} GB"
        )

        # Card 2: GPU Utilization
        draw_card(
            y_top=710, y_bot=1330,
            title="GPU Utilization",
            val_str=f"{int(m['gpu_utilization_pct'])} %",
            sub_str=None,
            pct=m["gpu_utilization_pct"],
            history=hist_gpu,
            max_val_str="100%"
        )

        # Supersampled downscale for crisp anti-aliasing
        img_final = img.resize((900, 680), Image.Resampling.LANCZOS)
        buf = BytesIO()
        img_final.save(buf, format="JPEG", quality=92)
        return buf.getvalue()

    def get_metrics_json(self):
        with self.lock:
            return json.dumps(self.metrics, indent=2).encode("utf-8")

    def get_snapshot(self):
        with self.lock:
            return self.current_frame

collector = TelemetryCollector()

def sampler_worker():
    while True:
        try:
            collector.sample()
        except Exception as e:
            print(f"Sample error: {e}", file=sys.stderr)
        time.sleep(1.0)

class StreamingHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_HEAD(self):
        if self.path in ("/snapshot.jpg", "/stream.mjpg"):
            frame = collector.get_snapshot()
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(frame)))
            self.send_header("Cache-Control", "no-cache, must-revalidate")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
        elif self.path == "/api/metrics":
            data = collector.get_metrics_json()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
        elif self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", "16")
            self.end_headers()
        else:
            self.send_error(404, "File not found")

    def do_GET(self):
        if self.path == "/stream.mjpg":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=--frame")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            try:
                while True:
                    frame = collector.get_snapshot()
                    if frame:
                        self.wfile.write(b"--frame\r\n")
                        self.send_header("Content-Type", "image/jpeg")
                        self.send_header("Content-Length", str(len(frame)))
                        self.end_headers()
                        self.wfile.write(frame)
                        self.wfile.write(b"\r\n")
                    time.sleep(FRAME_INTERVAL)
            except (BrokenPipeError, ConnectionResetError):
                pass
            return

        elif self.path in ("/snapshot.jpg", "/", "/index.html"):
            frame = collector.get_snapshot()
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(frame)))
            self.send_header("Cache-Control", "no-cache, must-revalidate")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(frame)
            return

        elif self.path == "/api/metrics":
            data = collector.get_metrics_json()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(data)
            return

        elif self.path == "/health":
            resp = b'{"status": "ok"}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)
            return

        else:
            self.send_error(404, "File not found")

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True

def main():
    print(f"Starting DGX NVIDIA Dashboard Service on port {PORT}...")
    collector.sample()

    t = threading.Thread(target=sampler_worker, daemon=True)
    t.start()

    server = ThreadedHTTPServer(("0.0.0.0", PORT), StreamingHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()

if __name__ == "__main__":
    main()
