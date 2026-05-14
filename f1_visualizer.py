
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle
import pandas as pd
import random

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Segoe UI', 'Helvetica']

TOTAL_LAPS = 57
BASE_SPEED = 0.008
FPS = 60
DEFAULT_SPEED_MULT = 3.0

# Colors
BG_WHITE = "#ffffff"
BG_PANEL = "#f8f9fa"
PANEL_BORDER = "#dee2e6"
RED_F1 = "#e10600"
TEXT_DARK = "#1a1a2e"
TEXT_GRAY = "#6c757d"
GREEN = "#28a745"
YELLOW = "#ffc107"
ORANGE = "#fd7e14"
BLUE = "#007bff"
PURPLE = "#6f42c1"

# Tyre colors for display
TYRE_COLORS = {
    'S': '#e10600',  # Soft - Red
    'M': '#ffc107',  # Medium - Yellow
    'H': '#ffffff',  # Hard - White
    'I': '#28a745',  # Intermediate - Green
    'W': '#007bff'   # Wet - Blue
}

# Track configurations
TRACKS = {
    "monza": {
        "name": "Monza",
        "country": "Italy",
        "length": "5.793",
        "corners": 11,
        "drs_zones": 3,
        "shape": "monza"
    },
    "bahrain": {
        "name": "Bahrain",
        "country": "Bahrain",
        "length": "5.412",
        "corners": 15,
        "drs_zones": 3,
        "shape": "bahrain"
    },
    "silverstone": {
        "name": "Silverstone",
        "country": "UK",
        "length": "5.891",
        "corners": 18,
        "drs_zones": 3,
        "shape": "silverstone"
    },
    "spa": {
        "name": "Spa",
        "country": "Belgium",
        "length": "7.004",
        "corners": 19,
        "drs_zones": 2,
        "shape": "spa"
    },
    "default": {
        "name": "Grand Prix",
        "country": "International",
        "length": "5.500",
        "corners": 16,
        "drs_zones": 3,
        "shape": "mixed"
    }
}

DRS_ZONES = {
    "monza": [(0.05, 0.12), (0.48, 0.55), (0.72, 0.78)],
    "bahrain": [(0.08, 0.16), (0.42, 0.50), (0.68, 0.76)],
    "silverstone": [(0.12, 0.20), (0.52, 0.60), (0.78, 0.86)],
    "spa": [(0.10, 0.18), (0.60, 0.68)],
    "mixed": [(0.05, 0.12), (0.52, 0.60), (0.75, 0.82)]
}


def build_monza_track():
    t = np.linspace(0, 2*np.pi, 2000)
    x = np.cos(t) * 1.4
    y = np.sin(t) * 0.35
    chicane_positions = [0.25, 0.75]
    for pos in chicane_positions:
        idx = int(pos * len(t))
        for i in range(idx - 30, idx + 30):
            if 0 <= i < len(x):
                factor = 1 - abs(i - idx) / 30
                if pos == 0.25:
                    x[i] -= 0.12 * factor
                    y[i] -= 0.08 * factor
                else:
                    x[i] += 0.12 * factor
                    y[i] -= 0.08 * factor
    return x, y


def build_bahrain_track():
    t = np.linspace(0, 2*np.pi, 2000)
    x = np.cos(t) * 1.1
    y = np.sin(t) * 0.55
    hairpin_idx = int(0.65 * len(t))
    for i in range(hairpin_idx - 50, hairpin_idx + 50):
        if 0 <= i < len(x):
            factor = 1 - abs(i - hairpin_idx) / 50
            x[i] -= 0.18 * factor
            y[i] -= 0.15 * factor
    tech_idx = int(0.35 * len(t))
    for i in range(tech_idx - 40, tech_idx + 40):
        if 0 <= i < len(x):
            factor = 1 - abs(i - tech_idx) / 40
            x[i] += 0.1 * np.sin(factor * np.pi) * 0.15
            y[i] += 0.05 * np.cos(factor * np.pi)
    return x, y


def build_silverstone_track():
    t = np.linspace(0, 2*np.pi, 2000)
    x = np.cos(t)
    y = np.sin(t) * 0.5
    for i, theta in enumerate(t):
        if 0.2 < theta < 0.4:
            factor = (theta - 0.2) / 0.2
            x[i] = 1.0 - factor * 0.3
            y[i] = -0.3 + factor * 0.6
        elif 0.6 < theta < 0.8:
            factor = (theta - 0.6) / 0.2
            x[i] = -0.7 + factor * 1.7
            y[i] = 0.3 - factor * 0.6
    x = (x - x.min()) / (x.max() - x.min())
    y = (y - y.min()) / (y.max() - y.min())
    return x, y


def build_spa_track():
    t = np.linspace(0, 2*np.pi, 2000)
    x = np.cos(t) * 1.2
    y = np.sin(t) * 0.3
    eau_rouge = int(0.3 * len(t))
    for i in range(eau_rouge - 60, eau_rouge + 60):
        if 0 <= i < len(x):
            factor = 1 - abs(i - eau_rouge) / 60
            x[i] += 0.25 * factor
            y[i] += 0.2 * factor * np.sin(factor * np.pi)
    bus_stop = int(0.85 * len(t))
    for i in range(bus_stop - 30, bus_stop + 30):
        if 0 <= i < len(x):
            factor = 1 - abs(i - bus_stop) / 30
            x[i] -= 0.15 * factor
            y[i] -= 0.05 * np.cos(factor * np.pi)
    return x, y


def build_default_track():
    t = np.linspace(0, 2*np.pi, 2000)
    x = np.cos(t)
    y = np.sin(t) * 0.6
    x += 0.15 * np.cos(2*t) - 0.08 * np.cos(3*t) + 0.05 * np.sin(5*t)
    y += 0.12 * np.sin(2*t) + 0.08 * np.cos(4*t) - 0.03 * np.sin(6*t)
    return x, y


def build_track(shape_type="mixed"):
    if shape_type == "monza":
        x, y = build_monza_track()
    elif shape_type == "bahrain":
        x, y = build_bahrain_track()
    elif shape_type == "silverstone":
        x, y = build_silverstone_track()
    elif shape_type == "spa":
        x, y = build_spa_track()
    else:
        x, y = build_default_track()
    x = (x - x.min()) / (x.max() - x.min())
    y = (y - y.min()) / (y.max() - y.min())
    return x, y


def pos_at(t, tx, ty):
    n = len(tx)
    idx = t * n
    i = int(idx) % n
    j = (i + 1) % n
    f = idx - int(idx)
    f = f * f * (3 - 2 * f)
    return tx[i] + (tx[j]-tx[i])*f, ty[i] + (ty[j]-ty[i])*f


class F1Visualizer:
    def __init__(self, drivers, race_name="F1 Race", total_laps=57, track_name="default"):
        self.speed_mult = DEFAULT_SPEED_MULT
        self.race_name = race_name
        self.total_laps = total_laps
        self.race_finished = False
        self.final_results = None
        self.track_name = track_name.lower()
        
        self.track_config = TRACKS.get(self.track_name, TRACKS["default"])
        shape = self.track_config["shape"]
        
        self.tx, self.ty = build_track(shape)
        self.drs_zones = DRS_ZONES.get(shape, DRS_ZONES["mixed"])
        
        offset = 0.018
        self.pit_x = self.tx - offset * np.sin(np.linspace(0, 2*np.pi, len(self.tx)))
        self.pit_y = self.ty + offset * np.cos(np.linspace(0, 2*np.pi, len(self.ty)))
        
        leader_time = min([d["total_time"] for d in drivers])
        
        # Tyre options with degradation rates
        self.tyre_types = ['S', 'M', 'H']
        self.tyre_performance = {'S': 0.98, 'M': 1.00, 'H': 1.02}
        self.tyre_degradation = {'S': 0.015, 'M': 0.008, 'H': 0.004}
        
        self.cars = []
        for i, d in enumerate(drivers):
            pace = d["total_time"] / leader_time
            # Assign random starting tyre
            starting_tyre = random.choice(['S', 'M', 'H'])
            self.cars.append({
                "abbr": d["abbr"],
                "pace": pace,
                "t": (i * 0.02) % 1,
                "lap": 0,
                "total_time": 0,
                "predicted_time": d["total_time"],
                "finished": False,
                "pit": False,
                "pit_timer": 0,
                "pit_duration": np.random.uniform(2.0, 4.5),
                "tyre": starting_tyre,
                "tyre_age": 0,
                "tyre_history": [starting_tyre]
            })
        
        self.colors = plt.cm.tab20(np.linspace(0, 1, len(self.cars)))
        self.overtakes = []
        self.pit_events = []
        self.prev_order = []
        self.flash_text = None
        self.flash_timer = 0
        self.last_overtake = None
        
        self._build_ui()
        self.anim = None
    
    def _build_ui(self):
        self.fig = plt.figure(figsize=(20, 11), facecolor=BG_WHITE)
        
        gs = GridSpec(2, 4, figure=self.fig,
                      height_ratios=[0.07, 0.93],
                      width_ratios=[0.85, 1.9, 0.95, 0.95],
                      hspace=0.05, wspace=0.06)
        
        self.ax_header = self.fig.add_subplot(gs[0, :])
        self.ax_header.set_facecolor(RED_F1)
        self.ax_header.set_xlim(0, 1)
        self.ax_header.set_ylim(0, 1)
        self.ax_header.axis("off")
        
        self.ax_info = self.fig.add_subplot(gs[1, 0])
        self.ax_info.set_facecolor(BG_PANEL)
        self.ax_info.set_xlim(0, 1)
        self.ax_info.set_ylim(0, 1)
        self.ax_info.axis("off")
        
        self.ax_track = self.fig.add_subplot(gs[1, 1])
        self.ax_track.set_facecolor(BG_WHITE)
        
        self.ax_events = self.fig.add_subplot(gs[1, 2])
        self.ax_events.set_facecolor(BG_PANEL)
        self.ax_events.set_xlim(0, 1)
        self.ax_events.set_ylim(0, 1)
        self.ax_events.axis("off")
        
        self.ax_lb = self.fig.add_subplot(gs[1, 3])
        self.ax_lb.set_facecolor(BG_PANEL)
        self.ax_lb.set_xlim(0, 1)
        self.ax_lb.set_ylim(0, 1)
        self.ax_lb.axis("off")
        
        self._init_header()
        self._init_info_panel()
        self._init_track()
        self._init_events_panel()
        self._init_leaderboard()
        
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
    
    def _init_header(self):
        self.ax_header.text(0.02, 0.5, f"{self.race_name.upper()}",
                           fontsize=14, fontweight="bold", color=BG_WHITE, va='center')
        
        self.lap_display = self.ax_header.text(0.5, 0.5, f"LAP 1 / {self.total_laps}",
                                               fontsize=18, fontweight="bold", 
                                               color=YELLOW, ha='center', va='center',
                                               bbox=dict(boxstyle="round,pad=0.3", 
                                                        facecolor=TEXT_DARK, alpha=0.8))
        
        self.speed_display = self.ax_header.text(0.82, 0.5, f"SPEED {self.speed_mult:.1f}X",
                                                 fontsize=13, fontweight="bold", 
                                                 color=YELLOW, ha='center', va='center')
        
        self.ax_header.text(0.97, 0.5, "● LIVE", fontsize=11, fontweight="bold", 
                           color=GREEN, ha='right', va='center')
    
    def _init_info_panel(self):
        y = 0.95
        
        self.ax_info.add_patch(plt.Rectangle((0, 0.92), 1, 0.08, 
                                facecolor=RED_F1, alpha=0.1, transform=self.ax_info.transAxes))
        self.ax_info.text(0.1, y, self.track_config["name"].upper(),
                         fontsize=13, fontweight="bold", color=RED_F1)
        y -= 0.07
        self.ax_info.text(0.1, y, self.track_config["country"],
                         fontsize=11, color=TEXT_GRAY)
        y -= 0.06
        self.ax_info.axhline(y=y, xmin=0.1, xmax=0.9, color=PANEL_BORDER, linewidth=1)
        y -= 0.08
        
        stats = [
            ("LENGTH", f"{self.track_config['length']} km"),
            ("CORNERS", str(self.track_config['corners'])),
            ("DRS ZONES", str(self.track_config['drs_zones'])),
            ("LAP RECORD", "1:19.813")
        ]
        
        for label, value in stats:
            self.ax_info.text(0.1, y, label, fontsize=10, fontweight="bold", color=TEXT_GRAY)
            self.ax_info.text(0.6, y, value, fontsize=11, fontweight="bold", color=TEXT_DARK)
            y -= 0.055
        
        self.ax_info.axhline(y=y, xmin=0.1, xmax=0.9, color=PANEL_BORDER, linewidth=1)
        y -= 0.08
        
        self.ax_info.text(0.1, y, "CONDITIONS", fontsize=11, fontweight="bold", color=RED_F1)
        y -= 0.07
        
        weather = [("AIR", "23°C"), ("TRACK", "34°C"), ("WIND", "12 km/h"), ("HUMIDITY", "0%")]
        for label, value in weather:
            self.ax_info.text(0.1, y, label, fontsize=10, color=TEXT_GRAY)
            self.ax_info.text(0.6, y, value, fontsize=10, fontweight="bold", color=TEXT_DARK)
            y -= 0.045
        
        self.ax_info.text(0.1, 0.05, "37°C  SUNNY", fontsize=11, fontweight="bold", color=ORANGE)
    
    def _init_track(self):
        for alpha, width in [(0.15, 16), (0.25, 10), (0.4, 5)]:
            self.ax_track.plot(self.tx, self.ty, color=TEXT_GRAY, lw=width, alpha=alpha, zorder=1)
        
        self.ax_track.plot(self.tx, self.ty, color=TEXT_DARK, lw=3, zorder=3)
        
        for a, b in self.drs_zones:
            i1 = int(a * len(self.tx))
            i2 = int(b * len(self.tx))
            if i1 < i2:
                self.ax_track.plot(self.tx[i1:i2], self.ty[i1:i2], 
                                  color=GREEN, lw=6, alpha=0.7, zorder=4)
        
        self.ax_track.plot(self.pit_x, self.pit_y, color=ORANGE, 
                          linestyle="--", lw=3, alpha=0.7, zorder=2)
        
        self.ax_track.scatter([self.tx[0]], [self.ty[0]], color=RED_F1, 
                             s=220, zorder=5, marker="s", alpha=0.9,
                             edgecolors=TEXT_DARK, linewidth=2)
        
        self.scat = self.ax_track.scatter([], [], s=170, edgecolors=TEXT_DARK,linewidth=2, zorder=6, alpha=0.95)
        
        margin = 0.12
        self.ax_track.set_xlim(-margin, 1 + margin)
        self.ax_track.set_ylim(-margin, 1 + margin)
        self.ax_track.set_aspect('equal')
        self.ax_track.axis("off")
        
        shape_names = {
            "monza": "HIGH SPEED TEMPLE OF SPEED",
            "bahrain": "DESERT TECHNICAL CIRCUIT",
            "silverstone": "HIGH SPEED ARROW",
            "spa": "LONG FLOWING ARDENNES",
            "mixed": "INTERNATIONAL CIRCUIT"
        }
        shape_text = shape_names.get(self.track_config["shape"], "INTERNATIONAL CIRCUIT")
        
        self.ax_track.text(0.5, 0.98, f"{self.track_config['name']} INTERNATIONAL CIRCUIT - {shape_text}",
                          transform=self.ax_track.transAxes, ha='center', fontsize=10,
                          color=TEXT_GRAY, fontweight='bold')
    
    def _init_events_panel(self):
        self.ax_events.text(0.05, 0.96, "OVERTAKES",
                           fontsize=12, fontweight="bold", color=RED_F1)
        
        self.ax_events.axhline(y=0.48, xmin=0.05, xmax=0.95,
                              color=PANEL_BORDER, linewidth=1.5)
        
        self.ax_events.text(0.05, 0.44, "PIT STOPS",
                           fontsize=12, fontweight="bold", color=RED_F1)
        
        self.overtakes_text = self.ax_events.text(0.05, 0.92, "",
                                                 va="top", fontsize=10, 
                                                 color=TEXT_DARK, linespacing=1.3)
        
        self.pits_text = self.ax_events.text(0.05, 0.40, "",
                                            va="top", fontsize=10, 
                                            color=TEXT_DARK, linespacing=1.3)
    
    def _init_leaderboard(self):
        self.ax_lb.text(0.5, 0.96, "LIVE TIMING",
                       fontsize=13, fontweight="bold", color=RED_F1, ha="center")
        
        headers = [("POS", 0.05), ("DRIVER", 0.25), ("GAP", 0.55), ("TYRE", 0.80)]
        for text, x in headers:
            self.ax_lb.text(x, 0.90, text, fontsize=11, fontweight="bold", color=TEXT_GRAY)
        
        self.ax_lb.axhline(y=0.87, xmin=0.05, xmax=0.95,
                          color=RED_F1, linewidth=1.5, alpha=0.5)
        
        self.lb_text = self.ax_lb.text(0.05, 0.84, "",
                                      va="top", fontsize=11, 
                                      color=TEXT_DARK, linespacing=1.3)
    
    def _show_flash(self, msg, color):
        if self.flash_text:
            try:
                self.flash_text.remove()
            except:
                pass
        self.flash_text = self.fig.text(0.5, 0.5, msg, ha="center", va="center",
                                        fontsize=28, color=color, fontweight="bold",
                                        alpha=0.95, bbox=dict(boxstyle="round,pad=0.6",
                                        facecolor=BG_WHITE, edgecolor=RED_F1, linewidth=3))
        self.flash_timer = 30
    
    def _on_key(self, event):
        if event.key == "up":
            self.speed_mult = min(self.speed_mult + 1.0, 45.0)
            self._show_flash(f"SPEED: {self.speed_mult:.1f}X", GREEN)
            self.speed_display.set_text(f"SPEED {self.speed_mult:.1f}X")
        elif event.key == "down":
            self.speed_mult = max(self.speed_mult - 1.0, 1.0)
            self._show_flash(f"SPEED: {self.speed_mult:.1f}X", ORANGE)
            self.speed_display.set_text(f"SPEED {self.speed_mult:.1f}X")
        elif event.key == "r":
            self.speed_mult = DEFAULT_SPEED_MULT
            self._show_flash("SPEED RESET", RED_F1)
            self.speed_display.set_text(f"SPEED {self.speed_mult:.1f}X")
    
    def _choose_new_tyre(self, current_tyre):
        """Choose a new tyre when pitting (strategic)"""
        options = ['S', 'M', 'H']
        # Remove current tyre to force change
        options.remove(current_tyre)
        # Choose based on race position (simulate strategy)
        return random.choice(options)
    
    def _step(self):
        if self.race_finished:
            return
        
        step = BASE_SPEED * self.speed_mult
        all_finished = True
        
        for car in self.cars:
            if car["finished"]:
                continue
            all_finished = False
            
            # Pit entry with tyre age consideration
            # Soft tyres degrade faster, pit more often
            tyre_degradation_rate = self.tyre_degradation.get(car["tyre"], 0.008)
            should_pit = (not car["pit"] and 
                         car["lap"] > 3 and 
                         car["tyre_age"] > 20 and 
                         np.random.rand() < tyre_degradation_rate * 2)
            
            if not car["pit"] and should_pit:
                car["pit"] = True
                car["pit_timer"] = 0
            
            if car["pit"]:
                car["pit_timer"] += 1
                car["t"] += step * 0.25
                if car["pit_timer"] > 60:  # 1 second pit stop
                    car["pit"] = False
                    # Change tyre when exiting pit
                    old_tyre = car["tyre"]
                    car["tyre"] = self._choose_new_tyre(old_tyre)
                    car["tyre_age"] = 0
                    car["tyre_history"].append(car["tyre"])
                    car["total_time"] += car["pit_duration"]
                    tyre_symbol = car["tyre"]
                    self.pit_events.insert(0, f"{car['abbr']}   {car['pit_duration']:.2f}s  → {tyre_symbol}")
                    if len(self.pit_events) > 8:
                        self.pit_events = self.pit_events[:8]
                continue
            
            # Normal racing with tyre degradation effect
            degradation = 1 + (self.tyre_degradation.get(car["tyre"], 0.008) * car["tyre_age"])
            car["t"] += (step / (car["pace"] * (1/degradation)))
            car["tyre_age"] += 0.05  # Increase tyre age gradually
            
            if car["t"] >= 1.0:
                laps = int(car["t"])
                car["t"] -= laps
                for _ in range(laps):
                    if car["lap"] < self.total_laps:
                        car["lap"] += 1
                        car["tyre_age"] += 1  # Full lap increases tyre age
                car["total_time"] += 90 * car["pace"] * laps
            
            if car["lap"] >= self.total_laps:
                car["finished"] = True
        
        if all_finished and not self.race_finished:
            self.race_finished = True
            self._show_flash("RACE FINISHED!", RED_F1)
            self.final_results = sorted(self.cars, key=lambda c: c["total_time"])
    
    def _detect_overtakes(self):
        if self.race_finished:
            return

    # Get current race order
        current = [
            car["abbr"]
            for car in sorted(self.cars, key=lambda c: (c["lap"], c["t"]), reverse=True)
            if not car["finished"]
        ]

    # Compare with previous frame
        if self.prev_order:
            for i, driver in enumerate(current):
                if driver in self.prev_order:
                    old_pos = self.prev_order.index(driver)

                # Overtake detected
                    if old_pos > i:
                        overtaken = self.prev_order[i]

                        if driver != overtaken:
                            msg = f"{driver} → {overtaken}"

                        # Prevent spam
                            if msg != self.last_overtake:
                                self.overtakes.insert(0, msg)
                                self.last_overtake = msg

                            # Keep only latest 12
                                if len(self.overtakes) > 12:
                                    self.overtakes = self.overtakes[:12]

    # Update AFTER processing
        self.prev_order = current

    def _update_leaderboard(self, leader):
        if self.race_finished and self.final_results:
            cars = self.final_results
        else:
            cars = sorted(self.cars, key=lambda c: (-c["lap"], -c["t"] if not c["pit"] else -999))
        
        lines = []
        leader_time = None
        
        if self.race_finished:
            leader_time = cars[0]["total_time"]
        else:
            for car in cars:
                if not car["pit"] and not car["finished"]:
                    leader_time = car["total_time"] + car["t"] * 90
                    break
        
        for i, car in enumerate(cars[:20]):
            if car["finished"] and not self.race_finished:
                continue
            
            pos = f"{i+1}"
            driver = car["abbr"]
            
            if self.race_finished:
                gap = f"{car['total_time']:.1f}"
            elif car["pit"]:
                gap = "PIT"
            elif i == 0:
                gap = "LEADER"
            else:
                current_time = car["total_time"] + car["t"] * 90
                gap_val = current_time - leader_time if leader_time else 0
                
                if gap_val > 60:
                    gap = f"+{gap_val/60:.1f}m"
                elif gap_val < 1:
                    gap = f"+{gap_val:.2f}s"
                else:
                    gap = f"+{gap_val:.1f}s"
            
            # Show current tyre with color indication
            tyre = car["tyre"]
            # Add a symbol for tyre age
            if car["tyre_age"] > 25:
                tyre += "!"  # Worn tyre indicator
            elif car["tyre_age"] > 15:
                tyre += "•"  # Medium wear
            
            line = f"{pos:>3}   {driver:<6}   {gap:>8}   {tyre:>4}"
            lines.append(line)
        
        if self.race_finished:
            lines.insert(0, "─" * 33)
            lines.insert(0, "FINAL CLASSIFICATION")
        
        self.lb_text.set_text("\n".join(lines) if lines else "Loading...")
    
    def _render(self, frame):
        if self.flash_timer > 0:
            self.flash_timer -= 1
            if self.flash_timer == 0 and self.flash_text:
                try:
                    self.flash_text.remove()
                    self.flash_text = None
                except:
                    pass
        
        self._step()
        self._detect_overtakes()
        
        leader = max(self.cars, key=lambda c: (c["lap"], c["t"]))
        
        xs, ys = [], []
        if not self.race_finished:
            sorted_cars = sorted(self.cars, key=lambda c: (-c["lap"], -c["t"] if not c["pit"] else -999))
            for i, car in enumerate(sorted_cars):
                if car["finished"]:
                    continue
                offset = i * 0.0015
                if car["pit"]:
                    idx = int((car["t"] + offset) % 1 * len(self.pit_x))
                    idx = min(idx, len(self.pit_x) - 1)
                    x, y = self.pit_x[idx], self.pit_y[idx]
                else:
                    t_pos = (car["t"] - offset) % 1
                    x, y = pos_at(t_pos, self.tx, self.ty)
                xs.append(x)
                ys.append(y)
        
        if xs:
            self.scat.set_offsets(np.c_[xs, ys])
            colors = list(self.colors[:len(xs)])

            if len(colors) > 0:
                colors[0] = "#ffd700"  # gold
            if len(colors) > 1:
                colors[1] = "#c0c0c0"  # silver
            if len(colors) > 2:
                colors[2] = "#cd7f32"  # bronze

            for i, car in enumerate(sorted_cars[:len(xs)]):
                for a, b in self.drs_zones:
                    if a < car["t"] < b:
                        colors[i] = "#39ff14"
                        
            self.scat.set_color(colors)

            sizes = [260 if i == 0 else 170 for i in range(len(xs))]
            self.scat.set_sizes(sizes)
        
        overtakes_display = "\n".join(self.overtakes[:12]) if self.overtakes else "No overtakes"
        pits_display = "\n".join(self.pit_events[:8]) if self.pit_events else "No pit stops"
        
        self.overtakes_text.set_text(overtakes_display)
        self.pits_text.set_text(pits_display)
        self._update_leaderboard(leader)
        
        if self.race_finished:
            self.lap_display.set_text("RACE FINISHED")
            self.lap_display.set_color(RED_F1)
        else:
            current_lap = min(leader["lap"] + 1, self.total_laps)
            self.lap_display.set_text(f"LAP {current_lap} / {self.total_laps}")
        
        return (self.scat, self.overtakes_text, self.pits_text, self.lb_text, self.lap_display)
    
    def run(self):
        self.anim = FuncAnimation(
        self.fig,
        self._render,
        interval=1000/FPS,
        blit=False,
        cache_frame_data=False   
        )
        plt.show()


def run_visualization(standings, race_name="F1 Race"):
    """Main entry point"""
    if "driver" not in standings.columns or "total_time" not in standings.columns:
        raise ValueError("Missing required columns: 'driver', 'total_time'")
    
    drivers = []
    for _, row in standings.iterrows():
        drivers.append({
            "abbr": row["driver"],
            "total_time": row["total_time"]
        })
    
    track_name = "default"
    race_lower = race_name.lower()
    track_keywords = ["monza", "bahrain", "silverstone", "spa"]
    for key in track_keywords:
        if key in race_lower:
            track_name = key
            break
    
    total_laps = 57
    
    print("\n" + "="*60)
    print(f"  F1 RACE VISUALIZER - {TRACKS[track_name]['name'].upper()} CIRCUIT")
    print("="*60)
    print("  CONTROLS:  UP/DOWN arrows = Speed  |  R = Reset  |  Close to exit")
    print("="*60 + "\n")
    
    vis = F1Visualizer(drivers, race_name, total_laps, track_name)
    vis.run()


if __name__ == "__main__":
    test = pd.DataFrame({
        "driver": ["HAM", "LEC", "BOT", "NOR", "SAI", "RIC", "PER", "OCO",
                   "STR", "ALO", "GAS", "TSU", "RUS", "LAT", "GIO", "RAI", 
                   "MAZ", "MSC", "VET"],
        "total_time": [5130.0, 5131.2, 5153.3, 5165.6, 5168.6, 5188.8, 5210.2,
                       5217.4, 5220.9, 5223.4, 5226.2, 5232.0, 5233.1, 5241.6,
                       5243.5, 5245.3, 5291.7, 5316.4, 5532.6]
    })

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle
import pandas as pd
import random

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Segoe UI', 'Helvetica']

TOTAL_LAPS = 57
BASE_SPEED = 0.008
FPS = 60
DEFAULT_SPEED_MULT = 3.0

# Colors
BG_WHITE = "#ffffff"
BG_PANEL = "#f8f9fa"
PANEL_BORDER = "#dee2e6"
RED_F1 = "#e10600"
TEXT_DARK = "#1a1a2e"
TEXT_GRAY = "#6c757d"
GREEN = "#28a745"
YELLOW = "#ffc107"
ORANGE = "#fd7e14"
BLUE = "#007bff"
PURPLE = "#6f42c1"

# Tyre colors for display
TYRE_COLORS = {
    'S': '#e10600',  # Soft - Red
    'M': '#ffc107',  # Medium - Yellow
    'H': '#ffffff',  # Hard - White
    'I': '#28a745',  # Intermediate - Green
    'W': '#007bff'   # Wet - Blue
}

# Track configurations
TRACKS = {
    "monza": {
        "name": "Monza",
        "country": "Italy",
        "length": "5.793",
        "corners": 11,
        "drs_zones": 3,
        "shape": "monza"
    },
    "bahrain": {
        "name": "Bahrain",
        "country": "Bahrain",
        "length": "5.412",
        "corners": 15,
        "drs_zones": 3,
        "shape": "bahrain"
    },
    "silverstone": {
        "name": "Silverstone",
        "country": "UK",
        "length": "5.891",
        "corners": 18,
        "drs_zones": 3,
        "shape": "silverstone"
    },
    "spa": {
        "name": "Spa",
        "country": "Belgium",
        "length": "7.004",
        "corners": 19,
        "drs_zones": 2,
        "shape": "spa"
    },
    "default": {
        "name": "Grand Prix",
        "country": "International",
        "length": "5.500",
        "corners": 16,
        "drs_zones": 3,
        "shape": "mixed"
    }
}

DRS_ZONES = {
    "monza": [(0.05, 0.12), (0.48, 0.55), (0.72, 0.78)],
    "bahrain": [(0.08, 0.16), (0.42, 0.50), (0.68, 0.76)],
    "silverstone": [(0.12, 0.20), (0.52, 0.60), (0.78, 0.86)],
    "spa": [(0.10, 0.18), (0.60, 0.68)],
    "mixed": [(0.05, 0.12), (0.52, 0.60), (0.75, 0.82)]
}


def build_monza_track():
    t = np.linspace(0, 2*np.pi, 2000)
    x = np.cos(t) * 1.4
    y = np.sin(t) * 0.35
    chicane_positions = [0.25, 0.75]
    for pos in chicane_positions:
        idx = int(pos * len(t))
        for i in range(idx - 30, idx + 30):
            if 0 <= i < len(x):
                factor = 1 - abs(i - idx) / 30
                if pos == 0.25:
                    x[i] -= 0.12 * factor
                    y[i] -= 0.08 * factor
                else:
                    x[i] += 0.12 * factor
                    y[i] -= 0.08 * factor
    return x, y


def build_bahrain_track():
    t = np.linspace(0, 2*np.pi, 2000)
    x = np.cos(t) * 1.1
    y = np.sin(t) * 0.55
    hairpin_idx = int(0.65 * len(t))
    for i in range(hairpin_idx - 50, hairpin_idx + 50):
        if 0 <= i < len(x):
            factor = 1 - abs(i - hairpin_idx) / 50
            x[i] -= 0.18 * factor
            y[i] -= 0.15 * factor
    tech_idx = int(0.35 * len(t))
    for i in range(tech_idx - 40, tech_idx + 40):
        if 0 <= i < len(x):
            factor = 1 - abs(i - tech_idx) / 40
            x[i] += 0.1 * np.sin(factor * np.pi) * 0.15
            y[i] += 0.05 * np.cos(factor * np.pi)
    return x, y


def build_silverstone_track():
    t = np.linspace(0, 2*np.pi, 2000)
    x = np.cos(t)
    y = np.sin(t) * 0.5
    for i, theta in enumerate(t):
        if 0.2 < theta < 0.4:
            factor = (theta - 0.2) / 0.2
            x[i] = 1.0 - factor * 0.3
            y[i] = -0.3 + factor * 0.6
        elif 0.6 < theta < 0.8:
            factor = (theta - 0.6) / 0.2
            x[i] = -0.7 + factor * 1.7
            y[i] = 0.3 - factor * 0.6
    x = (x - x.min()) / (x.max() - x.min())
    y = (y - y.min()) / (y.max() - y.min())
    return x, y


def build_spa_track():
    t = np.linspace(0, 2*np.pi, 2000)
    x = np.cos(t) * 1.2
    y = np.sin(t) * 0.3
    eau_rouge = int(0.3 * len(t))
    for i in range(eau_rouge - 60, eau_rouge + 60):
        if 0 <= i < len(x):
            factor = 1 - abs(i - eau_rouge) / 60
            x[i] += 0.25 * factor
            y[i] += 0.2 * factor * np.sin(factor * np.pi)
    bus_stop = int(0.85 * len(t))
    for i in range(bus_stop - 30, bus_stop + 30):
        if 0 <= i < len(x):
            factor = 1 - abs(i - bus_stop) / 30
            x[i] -= 0.15 * factor
            y[i] -= 0.05 * np.cos(factor * np.pi)
    return x, y


def build_default_track():
    t = np.linspace(0, 2*np.pi, 2000)
    x = np.cos(t)
    y = np.sin(t) * 0.6
    x += 0.15 * np.cos(2*t) - 0.08 * np.cos(3*t) + 0.05 * np.sin(5*t)
    y += 0.12 * np.sin(2*t) + 0.08 * np.cos(4*t) - 0.03 * np.sin(6*t)
    return x, y


def build_track(shape_type="mixed"):
    if shape_type == "monza":
        x, y = build_monza_track()
    elif shape_type == "bahrain":
        x, y = build_bahrain_track()
    elif shape_type == "silverstone":
        x, y = build_silverstone_track()
    elif shape_type == "spa":
        x, y = build_spa_track()
    else:
        x, y = build_default_track()
    x = (x - x.min()) / (x.max() - x.min())
    y = (y - y.min()) / (y.max() - y.min())
    return x, y


def pos_at(t, tx, ty):
    n = len(tx)
    idx = t * n
    i = int(idx) % n
    j = (i + 1) % n
    f = idx - int(idx)
    f = f * f * (3 - 2 * f)
    return tx[i] + (tx[j]-tx[i])*f, ty[i] + (ty[j]-ty[i])*f


class F1Visualizer:
    def __init__(self, drivers, race_name="F1 Race", total_laps=57, track_name="default"):
        self.speed_mult = DEFAULT_SPEED_MULT
        self.race_name = race_name
        self.total_laps = total_laps
        self.race_finished = False
        self.final_results = None
        self.track_name = track_name.lower()
        
        self.track_config = TRACKS.get(self.track_name, TRACKS["default"])
        shape = self.track_config["shape"]
        
        self.tx, self.ty = build_track(shape)
        self.drs_zones = DRS_ZONES.get(shape, DRS_ZONES["mixed"])
        
        offset = 0.018
        self.pit_x = self.tx - offset * np.sin(np.linspace(0, 2*np.pi, len(self.tx)))
        self.pit_y = self.ty + offset * np.cos(np.linspace(0, 2*np.pi, len(self.ty)))
        
        leader_time = min([d["total_time"] for d in drivers])
        
        # Tyre options with degradation rates
        self.tyre_types = ['S', 'M', 'H']
        self.tyre_performance = {'S': 0.98, 'M': 1.00, 'H': 1.02}
        self.tyre_degradation = {'S': 0.015, 'M': 0.008, 'H': 0.004}
        
        self.cars = []
        for i, d in enumerate(drivers):
            pace = d["total_time"] / leader_time
            # Assign random starting tyre
            starting_tyre = random.choice(['S', 'M', 'H'])
            self.cars.append({
                "abbr": d["abbr"],
                "pace": pace,
                "t": (i * 0.02) % 1,
                "lap": 0,
                "total_time": 0,
                "predicted_time": d["total_time"],
                "finished": False,
                "pit": False,
                "pit_timer": 0,
                "pit_duration": np.random.uniform(2.0, 4.5),
                "tyre": starting_tyre,
                "tyre_age": 0,
                "tyre_history": [starting_tyre]
            })
        
        self.colors = plt.cm.tab20(np.linspace(0, 1, len(self.cars)))
        self.overtakes = []
        self.pit_events = []
        self.prev_order = []
        self.flash_text = None
        self.flash_timer = 0
        self.last_overtake = None
        
        self._build_ui()
        self.anim = None
    
    def _build_ui(self):
        self.fig = plt.figure(figsize=(20, 11), facecolor=BG_WHITE)
        
        gs = GridSpec(2, 4, figure=self.fig,
                      height_ratios=[0.07, 0.93],
                      width_ratios=[0.85, 1.9, 0.95, 0.95],
                      hspace=0.05, wspace=0.06)
        
        self.ax_header = self.fig.add_subplot(gs[0, :])
        self.ax_header.set_facecolor(RED_F1)
        self.ax_header.set_xlim(0, 1)
        self.ax_header.set_ylim(0, 1)
        self.ax_header.axis("off")
        
        self.ax_info = self.fig.add_subplot(gs[1, 0])
        self.ax_info.set_facecolor(BG_PANEL)
        self.ax_info.set_xlim(0, 1)
        self.ax_info.set_ylim(0, 1)
        self.ax_info.axis("off")
        
        self.ax_track = self.fig.add_subplot(gs[1, 1])
        self.ax_track.set_facecolor(BG_WHITE)
        
        self.ax_events = self.fig.add_subplot(gs[1, 2])
        self.ax_events.set_facecolor(BG_PANEL)
        self.ax_events.set_xlim(0, 1)
        self.ax_events.set_ylim(0, 1)
        self.ax_events.axis("off")
        
        self.ax_lb = self.fig.add_subplot(gs[1, 3])
        self.ax_lb.set_facecolor(BG_PANEL)
        self.ax_lb.set_xlim(0, 1)
        self.ax_lb.set_ylim(0, 1)
        self.ax_lb.axis("off")
        
        self._init_header()
        self._init_info_panel()
        self._init_track()
        self._init_events_panel()
        self._init_leaderboard()
        
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
    
    def _init_header(self):
        self.ax_header.text(0.02, 0.5, f"{self.race_name.upper()}",
                           fontsize=14, fontweight="bold", color=BG_WHITE, va='center')
        
        self.lap_display = self.ax_header.text(0.5, 0.5, f"LAP 1 / {self.total_laps}",
                                               fontsize=18, fontweight="bold", 
                                               color=YELLOW, ha='center', va='center',
                                               bbox=dict(boxstyle="round,pad=0.3", 
                                                        facecolor=TEXT_DARK, alpha=0.8))
        
        self.speed_display = self.ax_header.text(0.82, 0.5, f"SPEED {self.speed_mult:.1f}X",
                                                 fontsize=13, fontweight="bold", 
                                                 color=YELLOW, ha='center', va='center')
        
        self.ax_header.text(0.97, 0.5, "● LIVE", fontsize=11, fontweight="bold", 
                           color=GREEN, ha='right', va='center')
    
    def _init_info_panel(self):
        y = 0.95
        
        self.ax_info.add_patch(plt.Rectangle((0, 0.92), 1, 0.08, 
                                facecolor=RED_F1, alpha=0.1, transform=self.ax_info.transAxes))
        self.ax_info.text(0.1, y, self.track_config["name"].upper(),
                         fontsize=13, fontweight="bold", color=RED_F1)
        y -= 0.07
        self.ax_info.text(0.1, y, self.track_config["country"],
                         fontsize=11, color=TEXT_GRAY)
        y -= 0.06
        self.ax_info.axhline(y=y, xmin=0.1, xmax=0.9, color=PANEL_BORDER, linewidth=1)
        y -= 0.08
        
        stats = [
            ("LENGTH", f"{self.track_config['length']} km"),
            ("CORNERS", str(self.track_config['corners'])),
            ("DRS ZONES", str(self.track_config['drs_zones'])),
            ("LAP RECORD", "1:19.813")
        ]
        
        for label, value in stats:
            self.ax_info.text(0.1, y, label, fontsize=10, fontweight="bold", color=TEXT_GRAY)
            self.ax_info.text(0.6, y, value, fontsize=11, fontweight="bold", color=TEXT_DARK)
            y -= 0.055
        
        self.ax_info.axhline(y=y, xmin=0.1, xmax=0.9, color=PANEL_BORDER, linewidth=1)
        y -= 0.08
        
        self.ax_info.text(0.1, y, "CONDITIONS", fontsize=11, fontweight="bold", color=RED_F1)
        y -= 0.07
        
        weather = [("AIR", "23°C"), ("TRACK", "34°C"), ("WIND", "12 km/h"), ("HUMIDITY", "0%")]
        for label, value in weather:
            self.ax_info.text(0.1, y, label, fontsize=10, color=TEXT_GRAY)
            self.ax_info.text(0.6, y, value, fontsize=10, fontweight="bold", color=TEXT_DARK)
            y -= 0.045
        
        self.ax_info.text(0.1, 0.05, "37°C  SUNNY", fontsize=11, fontweight="bold", color=ORANGE)
    
    def _init_track(self):
        for alpha, width in [(0.15, 16), (0.25, 10), (0.4, 5)]:
            self.ax_track.plot(self.tx, self.ty, color=TEXT_GRAY, lw=width, alpha=alpha, zorder=1)
        
        self.ax_track.plot(self.tx, self.ty, color=TEXT_DARK, lw=3, zorder=3)
        
        for a, b in self.drs_zones:
            i1 = int(a * len(self.tx))
            i2 = int(b * len(self.tx))
            if i1 < i2:
                self.ax_track.plot(self.tx[i1:i2], self.ty[i1:i2], 
                                  color=GREEN, lw=6, alpha=0.7, zorder=4)
        
        self.ax_track.plot(self.pit_x, self.pit_y, color=ORANGE, 
                          linestyle="--", lw=3, alpha=0.7, zorder=2)
        
        self.ax_track.scatter([self.tx[0]], [self.ty[0]], color=RED_F1, 
                             s=220, zorder=5, marker="s", alpha=0.9,
                             edgecolors=TEXT_DARK, linewidth=2)
        
        self.scat = self.ax_track.scatter([], [], s=170, edgecolors=TEXT_DARK,linewidth=2, zorder=6, alpha=0.95)
        
        margin = 0.12
        self.ax_track.set_xlim(-margin, 1 + margin)
        self.ax_track.set_ylim(-margin, 1 + margin)
        self.ax_track.set_aspect('equal')
        self.ax_track.axis("off")
        
        shape_names = {
            "monza": "HIGH SPEED TEMPLE OF SPEED",
            "bahrain": "DESERT TECHNICAL CIRCUIT",
            "silverstone": "HIGH SPEED ARROW",
            "spa": "LONG FLOWING ARDENNES",
            "mixed": "INTERNATIONAL CIRCUIT"
        }
        shape_text = shape_names.get(self.track_config["shape"], "INTERNATIONAL CIRCUIT")
        
        self.ax_track.text(0.5, 0.98, f"{self.track_config['name']} INTERNATIONAL CIRCUIT - {shape_text}",
                          transform=self.ax_track.transAxes, ha='center', fontsize=10,
                          color=TEXT_GRAY, fontweight='bold')
    
    def _init_events_panel(self):
        self.ax_events.text(0.05, 0.96, "OVERTAKES",
                           fontsize=12, fontweight="bold", color=RED_F1)
        
        self.ax_events.axhline(y=0.48, xmin=0.05, xmax=0.95,
                              color=PANEL_BORDER, linewidth=1.5)
        
        self.ax_events.text(0.05, 0.44, "PIT STOPS",
                           fontsize=12, fontweight="bold", color=RED_F1)
        
        self.overtakes_text = self.ax_events.text(0.05, 0.92, "",
                                                 va="top", fontsize=10, 
                                                 color=TEXT_DARK, linespacing=1.3)
        
        self.pits_text = self.ax_events.text(0.05, 0.40, "",
                                            va="top", fontsize=10, 
                                            color=TEXT_DARK, linespacing=1.3)
    
    def _init_leaderboard(self):
        self.ax_lb.text(0.5, 0.96, "LIVE TIMING",
                       fontsize=13, fontweight="bold", color=RED_F1, ha="center")
        
        headers = [("POS", 0.05), ("DRIVER", 0.25), ("GAP", 0.55), ("TYRE", 0.80)]
        for text, x in headers:
            self.ax_lb.text(x, 0.90, text, fontsize=11, fontweight="bold", color=TEXT_GRAY)
        
        self.ax_lb.axhline(y=0.87, xmin=0.05, xmax=0.95,
                          color=RED_F1, linewidth=1.5, alpha=0.5)
        
        self.lb_text = self.ax_lb.text(0.05, 0.84, "",
                                      va="top", fontsize=11, 
                                      color=TEXT_DARK, linespacing=1.3)
    
    def _show_flash(self, msg, color):
        if self.flash_text:
            try:
                self.flash_text.remove()
            except:
                pass
        self.flash_text = self.fig.text(0.5, 0.5, msg, ha="center", va="center",
                                        fontsize=28, color=color, fontweight="bold",
                                        alpha=0.95, bbox=dict(boxstyle="round,pad=0.6",
                                        facecolor=BG_WHITE, edgecolor=RED_F1, linewidth=3))
        self.flash_timer = 30
    
    def _on_key(self, event):
        if event.key == "up":
            self.speed_mult = min(self.speed_mult + 1.0, 45.0)
            self._show_flash(f"SPEED: {self.speed_mult:.1f}X", GREEN)
            self.speed_display.set_text(f"SPEED {self.speed_mult:.1f}X")
        elif event.key == "down":
            self.speed_mult = max(self.speed_mult - 1.0, 1.0)
            self._show_flash(f"SPEED: {self.speed_mult:.1f}X", ORANGE)
            self.speed_display.set_text(f"SPEED {self.speed_mult:.1f}X")
        elif event.key == "r":
            self.speed_mult = DEFAULT_SPEED_MULT
            self._show_flash("SPEED RESET", RED_F1)
            self.speed_display.set_text(f"SPEED {self.speed_mult:.1f}X")
    
    def _choose_new_tyre(self, current_tyre):
        """Choose a new tyre when pitting (strategic)"""
        options = ['S', 'M', 'H']
        # Remove current tyre to force change
        options.remove(current_tyre)
        # Choose based on race position (simulate strategy)
        return random.choice(options)
    
    def _step(self):
        if self.race_finished:
            return
        
        step = BASE_SPEED * self.speed_mult
        all_finished = True
        
        for car in self.cars:
            if car["finished"]:
                continue
            all_finished = False
            
            # Pit entry with tyre age consideration
            # Soft tyres degrade faster, pit more often
            tyre_degradation_rate = self.tyre_degradation.get(car["tyre"], 0.008)
            should_pit = (not car["pit"] and 
                         car["lap"] > 3 and 
                         car["tyre_age"] > 20 and 
                         np.random.rand() < tyre_degradation_rate * 2)
            
            if not car["pit"] and should_pit:
                car["pit"] = True
                car["pit_timer"] = 0
            
            if car["pit"]:
                car["pit_timer"] += 1
                car["t"] += step * 0.25
                if car["pit_timer"] > 60:  # 1 second pit stop
                    car["pit"] = False
                    # Change tyre when exiting pit
                    old_tyre = car["tyre"]
                    car["tyre"] = self._choose_new_tyre(old_tyre)
                    car["tyre_age"] = 0
                    car["tyre_history"].append(car["tyre"])
                    car["total_time"] += car["pit_duration"]
                    tyre_symbol = car["tyre"]
                    self.pit_events.insert(0, f"{car['abbr']}   {car['pit_duration']:.2f}s  → {tyre_symbol}")
                    if len(self.pit_events) > 8:
                        self.pit_events = self.pit_events[:8]
                continue
            
            # Normal racing with tyre degradation effect
            degradation = 1 + (self.tyre_degradation.get(car["tyre"], 0.008) * car["tyre_age"])
            car["t"] += (step / (car["pace"] * (1/degradation)))
            car["tyre_age"] += 0.05  # Increase tyre age gradually
            
            if car["t"] >= 1.0:
                laps = int(car["t"])
                car["t"] -= laps
                for _ in range(laps):
                    if car["lap"] < self.total_laps:
                        car["lap"] += 1
                        car["tyre_age"] += 1  # Full lap increases tyre age
                car["total_time"] += 90 * car["pace"] * laps
            
            if car["lap"] >= self.total_laps:
                car["finished"] = True
        
        if all_finished and not self.race_finished:
            self.race_finished = True
            self._show_flash("RACE FINISHED!", RED_F1)
            self.final_results = sorted(self.cars, key=lambda c: c["total_time"])
    
    def _detect_overtakes(self):
        if self.race_finished:
            return

    # Get current race order
        current = [
            car["abbr"]
            for car in sorted(self.cars, key=lambda c: (c["lap"], c["t"]), reverse=True)
            if not car["finished"]
        ]

    # Compare with previous frame
        if self.prev_order:
            for i, driver in enumerate(current):
                if driver in self.prev_order:
                    old_pos = self.prev_order.index(driver)

                # Overtake detected
                    if old_pos > i:
                        overtaken = self.prev_order[i]

                        if driver != overtaken:
                            msg = f"{driver} → {overtaken}"

                        # Prevent spam
                            if msg != self.last_overtake:
                                self.overtakes.insert(0, msg)
                                self.last_overtake = msg

                            # Keep only latest 12
                                if len(self.overtakes) > 12:
                                    self.overtakes = self.overtakes[:12]

    # Update AFTER processing
        self.prev_order = current

    def _update_leaderboard(self, leader):
        if self.race_finished and self.final_results:
            cars = self.final_results
        else:
            cars = sorted(self.cars, key=lambda c: (-c["lap"], -c["t"] if not c["pit"] else -999))
        
        lines = []
        leader_time = None
        
        if self.race_finished:
            leader_time = cars[0]["total_time"]
        else:
            for car in cars:
                if not car["pit"] and not car["finished"]:
                    leader_time = car["total_time"] + car["t"] * 90
                    break
        
        for i, car in enumerate(cars[:20]):
            if car["finished"] and not self.race_finished:
                continue
            
            pos = f"{i+1}"
            driver = car["abbr"]
            
            if self.race_finished:
                gap = f"{car['total_time']:.1f}"
            elif car["pit"]:
                gap = "PIT"
            elif i == 0:
                gap = "LEADER"
            else:
                current_time = car["total_time"] + car["t"] * 90
                gap_val = current_time - leader_time if leader_time else 0
                
                if gap_val > 60:
                    gap = f"+{gap_val/60:.1f}m"
                elif gap_val < 1:
                    gap = f"+{gap_val:.2f}s"
                else:
                    gap = f"+{gap_val:.1f}s"
            
            # Show current tyre with color indication
            tyre = car["tyre"]
            # Add a symbol for tyre age
            if car["tyre_age"] > 25:
                tyre += "!"  # Worn tyre indicator
            elif car["tyre_age"] > 15:
                tyre += "•"  # Medium wear
            
            line = f"{pos:>3}   {driver:<6}   {gap:>8}   {tyre:>4}"
            lines.append(line)
        
        if self.race_finished:
            lines.insert(0, "─" * 33)
            lines.insert(0, "FINAL CLASSIFICATION")
        
        self.lb_text.set_text("\n".join(lines) if lines else "Loading...")
    
    def _render(self, frame):
        if self.flash_timer > 0:
            self.flash_timer -= 1
            if self.flash_timer == 0 and self.flash_text:
                try:
                    self.flash_text.remove()
                    self.flash_text = None
                except:
                    pass
        
        self._step()
        self._detect_overtakes()
        
        leader = max(self.cars, key=lambda c: (c["lap"], c["t"]))
        
        xs, ys = [], []
        if not self.race_finished:
            sorted_cars = sorted(self.cars, key=lambda c: (-c["lap"], -c["t"] if not c["pit"] else -999))
            for i, car in enumerate(sorted_cars):
                if car["finished"]:
                    continue
                offset = i * 0.0015
                if car["pit"]:
                    idx = int((car["t"] + offset) % 1 * len(self.pit_x))
                    idx = min(idx, len(self.pit_x) - 1)
                    x, y = self.pit_x[idx], self.pit_y[idx]
                else:
                    t_pos = (car["t"] - offset) % 1
                    x, y = pos_at(t_pos, self.tx, self.ty)
                xs.append(x)
                ys.append(y)
        
        if xs:
            self.scat.set_offsets(np.c_[xs, ys])
            colors = list(self.colors[:len(xs)])

            if len(colors) > 0:
                colors[0] = "#ffd700"  # gold
            if len(colors) > 1:
                colors[1] = "#c0c0c0"  # silver
            if len(colors) > 2:
                colors[2] = "#cd7f32"  # bronze

            for i, car in enumerate(sorted_cars[:len(xs)]):
                for a, b in self.drs_zones:
                    if a < car["t"] < b:
                        colors[i] = "#39ff14"
                        
            self.scat.set_color(colors)

            sizes = [260 if i == 0 else 170 for i in range(len(xs))]
            self.scat.set_sizes(sizes)
        
        overtakes_display = "\n".join(self.overtakes[:12]) if self.overtakes else "No overtakes"
        pits_display = "\n".join(self.pit_events[:8]) if self.pit_events else "No pit stops"
        
        self.overtakes_text.set_text(overtakes_display)
        self.pits_text.set_text(pits_display)
        self._update_leaderboard(leader)
        
        if self.race_finished:
            self.lap_display.set_text("RACE FINISHED")
            self.lap_display.set_color(RED_F1)
        else:
            current_lap = min(leader["lap"] + 1, self.total_laps)
            self.lap_display.set_text(f"LAP {current_lap} / {self.total_laps}")
        
        return (self.scat, self.overtakes_text, self.pits_text, self.lb_text, self.lap_display)
    
    def run(self):
        self.anim = FuncAnimation(
        self.fig,
        self._render,
        interval=1000/FPS,
        blit=False,
        cache_frame_data=False   
        )
        plt.show()


def run_visualization(standings, race_name="F1 Race"):
    """Main entry point"""
    if "driver" not in standings.columns or "total_time" not in standings.columns:
        raise ValueError("Missing required columns: 'driver', 'total_time'")
    
    drivers = []
    for _, row in standings.iterrows():
        drivers.append({
            "abbr": row["driver"],
            "total_time": row["total_time"]
        })
    
    track_name = "default"
    race_lower = race_name.lower()
    track_keywords = ["monza", "bahrain", "silverstone", "spa"]
    for key in track_keywords:
        if key in race_lower:
            track_name = key
            break
    
    total_laps = 57
    
    print("\n" + "="*60)
    print(f"  F1 RACE VISUALIZER - {TRACKS[track_name]['name'].upper()} CIRCUIT")
    print("="*60)
    print("  CONTROLS:  UP/DOWN arrows = Speed  |  R = Reset  |  Close to exit")
    print("="*60 + "\n")
    
    vis = F1Visualizer(drivers, race_name, total_laps, track_name)
    vis.run()


if __name__ == "__main__":
    test = pd.DataFrame({
        "driver": ["HAM", "LEC", "BOT", "NOR", "SAI", "RIC", "PER", "OCO",
                   "STR", "ALO", "GAS", "TSU", "RUS", "LAT", "GIO", "RAI", 
                   "MAZ", "MSC", "VET"],
        "total_time": [5130.0, 5131.2, 5153.3, 5165.6, 5168.6, 5188.8, 5210.2,
                       5217.4, 5220.9, 5223.4, 5226.2, 5232.0, 5233.1, 5241.6,
                       5243.5, 5245.3, 5291.7, 5316.4, 5532.6]
    })
    run_visualization(test, "Bahrain Grand Prix")