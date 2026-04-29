"""Patch script: upgrades run_simulation in flood_animation_50%.py to 3-panel academic layout."""
from pathlib import Path
import csv as _csv_mod

SRC = Path(__file__).parent / "flood_animation_50%.py"
src = SRC.read_text(encoding="utf-8")

SEP = "─" * 29

# ── 1. Add imports if missing ──────────────────────────────────────────────────
OLD_IMPORTS = "import matplotlib.animation as animation\nimport matplotlib.patches as mpatches\nimport matplotlib.pyplot as plt"
NEW_IMPORTS = "import csv\nimport matplotlib.animation as animation\nimport matplotlib.patches as mpatches\nimport matplotlib.pyplot as plt\nfrom matplotlib.patches import Patch"
if "import csv" not in src:
    src = src.replace(OLD_IMPORTS, NEW_IMPORTS, 1)

# ── 2. Add academic helper functions before run_simulation ─────────────────────
HELPERS_MARKER = "def run_simulation(dem: np.ndarray, cellsize: float,"
if "_estimate_return_period" not in src:
    HELPERS = f'''
# =============================================================================
# ACADEMIC HELPERS  (return period, CSV export)
# =============================================================================

def _estimate_return_period(total_mm: float, duration_h: float) -> str:
    """Rough return-period estimate using PAGASA IDF data for Southern Mindanao."""
    rate = total_mm / max(duration_h, 0.1)
    if rate < 7.5:   return "< 2-year event"
    if rate < 15.0:  return "2 – 5 year event"
    if rate < 25.0:  return "5 – 10 year event"
    if rate < 40.0:  return "10 – 25 year event"
    if rate < 60.0:  return "25 – 50 year event"
    return               "50 – 100 year event"


def _export_sim_csv_50(out_path: str, times_list: list, elapsed_list: list,
                       rate_frames: list, stats: dict, timestep_min: int) -> None:
    """Write full time-series simulation data to CSV for academic reporting."""
    fields = [
        "time", "elapsed_min", "rain_intensity_mmhr",
        "rain_fallen_mm", "flooded_pct", "flooded_ha",
        "max_depth_mm", "river_pct", "max_river_mm", "runoff_vol_m3",
    ]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i, t in enumerate(times_list):
            w.writerow({{
                "time":                 t,
                "elapsed_min":          elapsed_list[i],
                "rain_intensity_mmhr":  f"{{rate_frames[i]:.2f}}",
                "rain_fallen_mm":       f"{{stats['rain_mm'][i]:.2f}}",
                "flooded_pct":          f"{{stats['flooded_pct'][i]:.2f}}",
                "flooded_ha":           f"{{stats['flooded_ha'][i]:.3f}}",
                "max_depth_mm":         f"{{stats['max_depth_mm'][i]:.1f}}",
                "river_pct":            f"{{stats['river_pct'][i]:.2f}}",
                "max_river_mm":         f"{{stats['max_river_mm'][i]:.1f}}",
                "runoff_vol_m3":        f"{{stats['runoff_vol_m3'][i]:.1f}}",
            }})


'''
    src_before = src[:src.index(HELPERS_MARKER)]
    src_after  = src[src.index(HELPERS_MARKER):]
    src = src_before + HELPERS + src_after

# ── 3. Replace the figure/draw/widgets block inside run_simulation ─────────────
# Find where the improved simulation ends and the figure section starts
OLD_FIGURE_START = "    # ── Build infrastructure overlay arrays (for drawing on the map) ─────────"
OLD_FIGURE_END   = "    plt.show()\n"

si = src.index(OLD_FIGURE_START)
ei = src.index(OLD_FIGURE_END, si) + len(OLD_FIGURE_END)

NEW_FIGURE = f'''    # ── Build infrastructure overlay arrays ──────────────────────────────────
    H, W = dem.shape
    ext  = (0, W, H, 0)

    total_area_m2 = H * W * cellsize ** 2

    # Academic metrics (computed once from improved-scenario stats)
    rate_frames = [rate_mmhr * _intensity_factor(i, num_frames, pattern)
                   for i in range(num_frames)]
    ret_period        = _estimate_return_period(rainfall_mm, duration_h)
    peak_intensity_ms = max(rate_frames) / (1000.0 * 3600.0)
    peak_Q_m3s        = 0.55 * peak_intensity_ms * total_area_m2

    # Add derived columns to stats (flooded_ha, runoff_vol_m3) if not present
    if "flooded_ha" not in stats:
        stats["flooded_ha"] = [
            p / 100.0 * total_area_m2 / 10000.0 for p in stats["flooded_pct"]]
    if "runoff_vol_m3" not in stats:
        stats["runoff_vol_m3"] = [
            m / 1000.0 * 0.55 * total_area_m2 for m in stats["rain_mm"]]
    elapsed_list = [i * timestep_min for i in range(num_frames)]

    # Floodwall overlay
    wall_rgba = np.zeros((H, W, 4), dtype=np.float32)
    if wall_mask.any():
        wall_rgba[wall_mask, 0] = 0.90
        wall_rgba[wall_mask, 1] = 0.15
        wall_rgba[wall_mask, 2] = 0.10
        wall_rgba[wall_mask, 3] = 0.32

    # Canal overlay
    canal_rgba = np.zeros((H, W, 4), dtype=np.float32)
    if canal_mask.any():
        canal_rgba[canal_mask, 0] = 0.00
        canal_rgba[canal_mask, 1] = 0.88
        canal_rgba[canal_mask, 2] = 0.95
        canal_rgba[canal_mask, 3] = 0.28

    # ── Background + colormaps ──────────────────────────────────────────────────
    print("\\n  Rendering map background\\u2026")
    bg         = render_jpeg_background(dem)
    cmap_rain  = _rain_cmap()
    cmap_river = _river_cmap()

    DARK     = "#0D1117"
    PANEL    = "#161B22"
    TCLR     = "#E6EDF3"
    ACC      = "#4FC3F7"
    CHART_BG = "#0A0F18"

    # ── Figure: 3-panel layout (map | stats | hydrograph) ──────────────────────
    title_suffix = f"  |  Prevention: {{prevention_str}}" if any_prevention else ""
    fig = plt.figure(figsize=(26, 13), facecolor=DARK)
    fig.suptitle(
        f"JADE VALLEY SUBDIVISION  \\u2014  FLOOD SIMULATION  |  {{scenario_name}}{{title_suffix}}",
        fontsize=14, fontweight="bold", color=TCLR, y=0.983)

    ax_map   = fig.add_axes((0.03, 0.13, 0.61, 0.83))
    ax_stats = fig.add_axes((0.67, 0.38, 0.31, 0.58))
    ax_chart = fig.add_axes((0.67, 0.13, 0.31, 0.21), facecolor=CHART_BG)

    ax_map  .set_facecolor("black")
    ax_stats.set_facecolor(PANEL)
    ax_stats.axis("off")

    # Map: base layers
    ax_map.imshow(bg, extent=ext, aspect="auto", zorder=1, interpolation="bilinear")
    im_rain = ax_map.imshow(
        rain_frames[0] * 1000,
        cmap=cmap_rain, vmin=0, vmax=600,
        extent=ext, aspect="auto", zorder=2, interpolation="bilinear", alpha=0.45)
    total_water0 = rain_frames[0] + river_frames[0]
    im_river = ax_map.imshow(
        np.clip(np.power(total_water0 / 1.1, 0.65), 0, 1),
        cmap=cmap_river, vmin=0, vmax=1,
        extent=ext, aspect="auto", zorder=3, interpolation="nearest", alpha=0.62)

    # Elevation contour lines
    try:
        ax_map.contour(np.flipud(dem),
                       levels=np.linspace(dem.min(), dem.max(), 14),
                       colors="white", alpha=0.10, linewidths=0.4, zorder=4)
    except Exception:
        pass

    # Stream-network overlay
    strm_rgba = np.zeros((H, W, 4), dtype=np.float32)
    strm_rgba[sim.streams, 0] = 0.15
    strm_rgba[sim.streams, 2] = 0.90
    strm_rgba[sim.streams, 3] = 0.65
    ax_map.imshow(strm_rgba, extent=ext, aspect="auto", zorder=5, interpolation="nearest")

    # Prevention overlays (zorder 6 = above stream)
    if wall_mask.any():
        ax_map.imshow(wall_rgba, extent=ext, aspect="auto", zorder=6, interpolation="nearest")
    if canal_mask.any():
        ax_map.imshow(canal_rgba, extent=ext, aspect="auto", zorder=6, interpolation="nearest")

    # Scale bar
    sc_cells = max(3, int(round(50.0 / cellsize)))
    sc_m     = sc_cells * cellsize
    bx0, bx1 = W * 0.05, W * 0.05 + sc_cells
    by, bh   = H * 0.930, H * 0.007
    ax_map.fill_between([bx0, bx1], [by - bh] * 2, [by + bh] * 2, color="white", zorder=15)
    ax_map.text((bx0 + bx1) / 2, by + bh * 3.0, f"{{sc_m:.0f}} m",
                color="white", fontsize=7, ha="center", va="bottom",
                fontweight="bold", zorder=15)

    # North arrow
    nx, ny0, ny1 = W * 0.938, H * 0.115, H * 0.060
    ax_map.annotate("", xy=(nx, ny1), xytext=(nx, ny0),
                    arrowprops=dict(arrowstyle="->", color="white", lw=2.0), zorder=15)
    ax_map.text(nx, ny1 - H * 0.014, "N", color="white",
                fontsize=9, ha="center", va="bottom", fontweight="bold", zorder=15)

    # Map legend
    leg_h = [
        Patch(facecolor="#29B6F6", alpha=0.80, label="Stream channel"),
        Patch(facecolor="#1E90FF", alpha=0.75, label="River overflow"),
        Patch(facecolor="#00FFCC", alpha=0.55, label="Runoff \\u2264 30 mm"),
        Patch(facecolor="#FFFF00", alpha=0.60, label="Runoff \\u2264 100 mm"),
        Patch(facecolor="#FF6600", alpha=0.65, label="Runoff \\u2264 300 mm"),
    ]
    if wall_mask.any():
        leg_h.append(Patch(facecolor=(0.9,0.15,0.1,0.5), edgecolor="red",
                           label=f"Floodwall (+{{wall_height:.1f}} m)"))
    if canal_mask.any():
        leg_h.append(Patch(facecolor=(0,0.88,0.95,0.45), edgecolor="cyan",
                           label="Drainage Canal"))
    ax_map.legend(handles=leg_h, loc="lower right",
                  facecolor="#0D1117", edgecolor=ACC,
                  labelcolor="#E6EDF3", fontsize=6.5,
                  framealpha=0.88, handlelength=1.2, borderpad=0.7)

    ax_map.set_xlim(0, W); ax_map.set_ylim(H, 0)
    ax_map.tick_params(colors=TCLR, labelsize=7)
    for sp in ax_map.spines.values():
        sp.set_edgecolor("#30363D")

    cbar = fig.colorbar(im_rain, ax=ax_map, orientation="vertical", pad=0.01, shrink=0.74)
    cbar.set_label("Rainfall Runoff Depth (mm)", color=TCLR, fontsize=8)
    cbar.set_ticks([0, 30, 100, 200, 400, 600])
    cbar.ax.set_yticklabels(["Dry","30","100","200","400","600+"], color=TCLR, fontsize=7)
    cbar.ax.tick_params(colors=TCLR)

    time_txt = ax_map.text(
        0.015, 0.975, "", transform=ax_map.transAxes,
        fontsize=11, fontweight="bold", color="white", va="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor=PANEL,
                  alpha=0.90, edgecolor=ACC, linewidth=1.5))

    stats_txt = ax_stats.text(
        0.04, 0.98, "", fontsize=8.5, family="monospace",
        color=TCLR, va="top", transform=ax_stats.transAxes,
        bbox=dict(boxstyle="round,pad=0.7", facecolor="#0D1117",
                  edgecolor=ACC, alpha=0.92, linewidth=1.2))

    # ── Hydrograph panel ────────────────────────────────────────────────────────
    time_axis = [i * timestep_min for i in range(num_frames)]

    ax_chart.bar(time_axis, rate_frames, width=timestep_min * 0.85,
                 color="#29B6F6", alpha=0.55, zorder=2)
    ax_chart.set_xlim(-timestep_min * 0.5, time_axis[-1] + timestep_min)
    ax_chart.set_ylim(0, max(rate_frames) * 1.40 + 1)
    ax_chart.set_xlabel("Elapsed Time (min)", color=TCLR, fontsize=7)
    ax_chart.set_ylabel("Intensity (mm/hr)", color="#29B6F6", fontsize=7)
    ax_chart.tick_params(colors=TCLR, labelsize=6.5)
    for sp in ax_chart.spines.values():
        sp.set_edgecolor("#30363D")
    ax_chart.text(0.015, 0.91, "HYDROGRAPH", transform=ax_chart.transAxes,
                  fontsize=6.5, color="#B0BEC5", fontweight="bold", va="top")

    ax_chart2 = ax_chart.twinx()
    ax_chart2.tick_params(colors="#F97316", labelsize=6.5)
    ax_chart2.set_ylabel("Max Depth (mm)", color="#F97316", fontsize=7)
    ax_chart2.set_ylim(0, max(max(stats["max_depth_mm"]) * 1.35, 10))
    ax_chart2.set_xlim(-timestep_min * 0.5, time_axis[-1] + timestep_min)
    depth_line, = ax_chart2.plot([], [], "-", color="#F97316", linewidth=1.8, zorder=3)
    if base_stats:
        base_depth_line, = ax_chart2.plot([], [], "--", color="#888888",
                                           linewidth=1.2, zorder=2, label="Baseline")
        ax_chart2.legend(fontsize=6, labelcolor="#AAAAAA",
                         facecolor=CHART_BG, edgecolor="#30363D")
    else:
        base_depth_line = None
    chart_vline = ax_chart.axvline(0, color="#FFD740", linewidth=1.3, linestyle="--", zorder=4)

    # ── Widgets ──────────────────────────────────────────────────────────────────
    ax_sl_frame = fig.add_axes((0.03, 0.084, 0.61, 0.022), facecolor="#21262D")
    ax_sl_speed = fig.add_axes((0.03, 0.040, 0.23, 0.022), facecolor="#21262D")
    ax_btn_play = fig.add_axes((0.280, 0.018, 0.080, 0.052))
    ax_btn_prev = fig.add_axes((0.366, 0.018, 0.048, 0.052))
    ax_btn_next = fig.add_axes((0.420, 0.018, 0.048, 0.052))
    ax_btn_gif  = fig.add_axes((0.475, 0.018, 0.083, 0.052))
    ax_btn_csv  = fig.add_axes((0.565, 0.018, 0.083, 0.052))

    sl_frame = Slider(ax_sl_frame, "Frame", 0, num_frames - 1,
                      valinit=0, valstep=1, color=ACC)
    sl_speed = Slider(ax_sl_speed, "Speed \\u00d7", 0.25, 4.0,
                      valinit=1.0, color="#FFB74D")
    for sl in (sl_frame, sl_speed):
        sl.label.set_color(TCLR); sl.valtext.set_color(TCLR)
        sl.label.set_fontsize(7.5)

    btn_play = Button(ax_btn_play, "\\u23f8 Pause", color="#1B5E20", hovercolor="#2E7D32")
    btn_prev = Button(ax_btn_prev, "\\u25c4\\u25c4",   color="#0D47A1", hovercolor="#1565C0")
    btn_next = Button(ax_btn_next, "\\u25ba\\u25ba",   color="#0D47A1", hovercolor="#1565C0")
    btn_gif  = Button(ax_btn_gif,  "\\U0001f4be GIF", color="#4A148C", hovercolor="#6A1B9A")
    btn_csv  = Button(ax_btn_csv,  "\\U0001f4ca CSV", color="#1A3A2A", hovercolor="#1B5E20")
    for b in (btn_play, btn_prev, btn_next, btn_gif, btn_csv):
        b.label.set_color("white"); b.label.set_fontsize(9)

    player = {{"playing": True, "frame": 0}}
    DIRS = ["N","NNE","NE","ENE","E","ESE","SE","SSE",
            "S","SSW","SW","WSW","W","WNW","NW","NNW"]
    compass_lbl = DIRS[int((wind_dir + 11.25) / 22.5) % 16]
    wind_str    = (f"{{wind_speed:.0f}} km/h from {{compass_lbl}}"
                   if wind_speed >= 1.0 else "None")

    def _risk_str(max_depth_mm: float, river_pct: float):
        if river_pct > 30 or max_depth_mm > 600:
            return "\\u26a0 EVACUATE NOW",                  "#FF1744"
        if river_pct > 18 or max_depth_mm > 300:
            return "MANDATORY EVACUATION",                  "#FF6D00"
        if river_pct > 8  or max_depth_mm > 150:
            return "PRE-EVACUATION ALERT",                  "#FFD740"
        if river_pct > 3  or max_depth_mm > 50:
            return "STANDBY \\u2014 prepare to move",       "#69F0AE"
        return             "NORMAL \\u2014 monitoring",      "#B0BEC5"

    sep = "{SEP}"

    def draw(fi: int):
        fi = int(fi) % num_frames
        player["frame"] = fi

        rn = rain_frames[fi] * 1000
        im_rain.set_data(rn)
        im_rain.set_clim(0, min(max(float(rn.max()), 30) * 1.3, 600))
        total_w = rain_frames[fi] + river_frames[fi]
        im_river.set_data(np.clip(np.power(total_w / 1.1, 0.65), 0, 1))

        time_txt.set_text(
            f" Time : {{times_list[fi]}}\\n"
            f" Rain : {{stats['rain_mm'][fi]:.0f}} / {{rainfall_mm:.0f}} mm\\n"
            f" Wind : {{wind_str}}")

        rsk, rsk_col = _risk_str(stats["max_depth_mm"][fi], stats["river_pct"][fi])

        # Baseline comparison lines
        if base_stats and fi < len(base_stats["flooded_pct"]):
            b_pct   = base_stats["flooded_pct"][fi]
            b_depth = base_stats["max_depth_mm"][fi]
            imp_pct  = b_pct   - stats["flooded_pct"][fi]
            imp_dep  = b_depth - stats["max_depth_mm"][fi]
            compare_block = (
                f"\\n"
                f"  EFFECTIVENESS vs BASELINE\\n"
                f"  {{sep}}\\n"
                f"  Baseline Flooded : {{b_pct:.1f}}%\\n"
                f"  With Prevention  : {{stats['flooded_pct'][fi]:.1f}}%\\n"
                f"  Area Reduction   : {{imp_pct:+.1f}}%\\n"
                f"  Depth Reduction  : {{imp_dep:+.0f}} mm\\n"
            )
            if base_depth_line is not None:
                base_depth_line.set_data(
                    time_axis[:fi + 1], base_stats["max_depth_mm"][:fi + 1])
        else:
            compare_block = ""

        stats_txt.set_text(
            f"  SCENARIO\\n"
            f"  {{sep}}\\n"
            f"  {{scenario_name}}\\n"
            f"\\n"
            f"  STORM PARAMETERS\\n"
            f"  {{sep}}\\n"
            f"  Total Rainfall  : {{rainfall_mm:.0f}} mm\\n"
            f"  Avg Rate        : {{rate_mmhr:.1f}} mm/hr\\n"
            f"  Peak Rate       : {{max(rate_frames):.1f}} mm/hr\\n"
            f"  Duration        : {{duration_h:.1f}} hr\\n"
            f"  Pattern         : {{pattern}}\\n"
            f"  Timestep        : {{timestep_min}} min\\n"
            f"  Wind            : {{wind_str}}\\n"
            f"  Soil Saturation : {{soil_sat_pct:.0f}}%\\n"
            f"  Drain Capacity  : {{drain_cap:.1f}} mm/hr\\n"
            f"\\n"
            f"  PREVENTION MEASURES\\n"
            f"  {{sep}}\\n"
            f"  {{prevention_str}}\\n"
            f"\\n"
            f"  HYDROLOGICAL ANALYSIS\\n"
            f"  {{sep}}\\n"
            f"  Return Period   : {{ret_period}}\\n"
            f"  Peak Discharge  : {{peak_Q_m3s:.3f}} m\\u00b3/s\\n"
            f"  Runoff Volume   : {{stats['runoff_vol_m3'][fi]/1000:.1f}} \\u00d7 10\\u00b3 m\\u00b3\\n"
            f"  Watershed Area  : {{total_area_m2/10000:.2f}} ha\\n"
            f"\\n"
            f"  LIVE STATUS  [{{times_list[fi]}}]\\n"
            f"  {{sep}}\\n"
            f"  Elapsed Time    : {{fi * timestep_min}} min\\n"
            f"  Rain Fallen     : {{stats['rain_mm'][fi]:.1f}} mm\\n"
            f"  Max Depth       : {{stats['max_depth_mm'][fi]:.0f}} mm\\n"
            f"  Flooded Area    : {{stats['flooded_pct'][fi]:.1f}}%"
            f" ({{stats['flooded_ha'][fi]:.2f}} ha)\\n"
            f"\\n"
            f"  RIVER OVERFLOW\\n"
            f"  {{sep}}\\n"
            f"  Area Affected   : {{stats['river_pct'][fi]:.1f}}% of grid\\n"
            f"  Max Depth       : {{stats['max_river_mm'][fi]:.0f}} mm\\n"
            f"\\n"
            f"  RISK ASSESSMENT (PAGASA)\\n"
            f"  {{sep}}\\n"
            f"  {{rsk}}"
            f"{{compare_block}}"
        )
        patch = stats_txt.get_bbox_patch()
        if patch is not None:
            patch.set_edgecolor(rsk_col)

        # Update hydrograph
        x_so_far = time_axis[:fi + 1]
        depth_line.set_data(x_so_far, stats["max_depth_mm"][:fi + 1])
        chart_vline.set_xdata([elapsed_list[fi], elapsed_list[fi]])

        if abs(sl_frame.val - fi) > 0.5:
            sl_frame.eventson = False
            sl_frame.set_val(fi)
            sl_frame.eventson = True
        fig.canvas.draw_idle()

    BASE_INTERVAL = 600

    def _anim_step(_):
        if player["playing"]:
            draw(player["frame"] + 1)
        return []

    anim_obj = animation.FuncAnimation(
        fig, _anim_step, interval=BASE_INTERVAL,
        blit=False, cache_frame_data=False)

    def on_frame(val):
        draw(int(val))

    def on_speed(val):
        anim_obj.event_source.interval = max(50, int(BASE_INTERVAL / max(val, 0.01)))

    def on_play_pause(_):
        player["playing"] = not player["playing"]
        if player["playing"]:
            btn_play.label.set_text("\\u23f8 Pause")
            btn_play.ax.set_facecolor("#1B5E20")
        else:
            btn_play.label.set_text("\\u25ba Play")
            btn_play.ax.set_facecolor("#BF360C")
        fig.canvas.draw_idle()

    def on_prev(_):
        player["playing"] = False
        btn_play.label.set_text("\\u25ba Play")
        btn_play.ax.set_facecolor("#BF360C")
        draw(player["frame"] - 1)

    def on_next(_):
        player["playing"] = False
        btn_play.label.set_text("\\u25ba Play")
        btn_play.ax.set_facecolor("#BF360C")
        draw(player["frame"] + 1)

    def _do_export_csv():
        safe = (scenario_name.replace(" ", "_").replace("/", "-")
                             .replace("(", "").replace(")", ""))
        prev_tag = "_prevention" if any_prevention else ""
        p = str(ANIM_DIR / f"flood_{{safe}}{{prev_tag}}_stats.csv")
        _export_sim_csv_50(p, times_list, elapsed_list, rate_frames, stats, timestep_min)
        print(f"  \\u2713 CSV exported  \\u2192  {{p}}")

    def on_save_gif(_):
        safe = (scenario_name.replace(" ", "_").replace("/", "-")
                             .replace("(", "").replace(")", ""))
        prev_tag = "_prevention" if any_prevention else ""
        out = str(ANIM_DIR / f"flood_{{safe}}{{prev_tag}}.gif")
        print(f"\\n  Saving GIF: {{out}}  (may take 30\\u201360 s)\\u2026")
        was = player["playing"]
        player["playing"] = False
        if not PIL_OK:
            print("  [ERROR] pip install Pillow")
            player["playing"] = was
            return

        tmp_fig, (tm, ts) = plt.subplots(1, 2, figsize=(16, 7), facecolor=DARK)
        tm.set_facecolor("black"); ts.set_facecolor(PANEL); ts.axis("off")
        tm.imshow(bg, extent=ext, aspect="auto", zorder=1, interpolation="bilinear")
        tm.imshow(strm_rgba, extent=ext, aspect="auto", zorder=4, interpolation="nearest")
        if wall_mask.any():
            tm.imshow(wall_rgba, extent=ext, aspect="auto", zorder=5, interpolation="nearest")
        if canal_mask.any():
            tm.imshow(canal_rgba, extent=ext, aspect="auto", zorder=5, interpolation="nearest")
        _irn = tm.imshow(rain_frames[0] * 1000, cmap=cmap_rain, vmin=0, vmax=600,
                         extent=ext, aspect="auto", zorder=2, alpha=0.45)
        _irv = tm.imshow((river_frames[0] > 0.005).astype(float), cmap=cmap_river,
                         vmin=0, vmax=1, extent=ext, aspect="auto", zorder=3, alpha=0.45)
        tm.set_xlim(0, W); tm.set_ylim(H, 0)
        _tt = tm.text(0.015, 0.975, "", transform=tm.transAxes,
                      fontsize=11, fontweight="bold", color="white", va="top",
                      bbox=dict(boxstyle="round", facecolor=PANEL, alpha=0.88, edgecolor=ACC))
        _st = ts.text(0.05, 0.97, "", fontsize=8.5, family="monospace",
                      color=TCLR, va="top", transform=ts.transAxes)

        frames_pil = []
        for fi in range(num_frames):
            rn = rain_frames[fi] * 1000
            _irn.set_data(rn)
            _irn.set_clim(0, min(max(float(rn.max()), 30) * 1.3, 600))
            _irv.set_data((river_frames[fi] > 0.005).astype(float))
            rsk_g, _ = _risk_str(stats["max_depth_mm"][fi], stats["river_pct"][fi])
            _tt.set_text(
                f"\\u23f1 {{times_list[fi]}}\\n"
                f"Rain: {{stats['rain_mm'][fi]:.0f}}/{{rainfall_mm:.0f}} mm")
            _st.set_text(
                f"{{scenario_name}}\\nPrevention: {{prevention_str}}\\n\\n"
                f"Return Period : {{ret_period}}\\n"
                f"Peak Discharge: {{peak_Q_m3s:.3f}} m\\u00b3/s\\n"
                f"Watershed Area: {{total_area_m2/10000:.2f}} ha\\n\\n"
                f"Time     : {{times_list[fi]}}\\n"
                f"Rain     : {{stats['rain_mm'][fi]:.1f}} mm\\n"
                f"Flooded  : {{stats['flooded_pct'][fi]:.1f}}%"
                f" ({{stats['flooded_ha'][fi]:.2f}} ha)\\n"
                f"Max Depth: {{stats['max_depth_mm'][fi]:.0f}} mm\\n"
                f"River Fld: {{stats['river_pct'][fi]:.1f}}%\\n"
                f"Risk     : {{rsk_g}}")
            tmp_fig.canvas.draw()
            buf = io.BytesIO()
            tmp_fig.savefig(buf, format="png", dpi=75, bbox_inches="tight", facecolor=DARK)
            buf.seek(0)
            pil_frame = PILImage.open(buf).copy().convert("RGB")  # type: ignore
            frames_pil.append(pil_frame.quantize(method=2))
            print(f"  GIF frame {{fi + 1}}/{{num_frames}} \\u2026", end="\\r", flush=True)

        print()
        plt.close(tmp_fig)
        frames_pil[0].save(out, save_all=True, append_images=frames_pil[1:],
                           loop=0, duration=int(1000 / 5))
        print(f"  \\u2713 Saved GIF  ({{len(frames_pil)}} frames)  \\u2192  {{out}}")
        player["playing"] = was

    def on_save_csv(_):
        _do_export_csv()

    sl_frame.on_changed(on_frame)
    sl_speed.on_changed(on_speed)
    btn_play.on_clicked(on_play_pause)
    btn_prev.on_clicked(on_prev)
    btn_next.on_clicked(on_next)
    btn_gif .on_clicked(on_save_gif)
    btn_csv .on_clicked(on_save_csv)

    draw(0)
    print("\\n  \\u2713 Interactive viewer ready.")
    print("  Controls: Pause/Play | Step | Speed \\u00d7 | GIF | CSV")
    plt.show()
'''

NEW_FIGURE_FINAL = NEW_FIGURE.replace('"{SEP}"', f'"{SEP}"')

result = src[:si] + NEW_FIGURE_FINAL + src[ei:]
SRC.write_text(result, encoding="utf-8")

lines = result.splitlines()
print(f"Done. 50% file now has {len(lines)} lines.")
