#!/usr/bin/env python3
import argparse
import math
import os
import random
import re
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


WIDTH = 1920
HEIGHT = 1080
FPS = 25
DEFAULT_DURATION = 14.0

BG = (5, 8, 12)
PANEL = (9, 14, 20)
PANEL_ALT = (12, 18, 26)
GRID = (18, 28, 38)
TEXT = (226, 236, 245)
MUTED = (120, 138, 152)
GREEN = (116, 242, 93)
CYAN = (92, 214, 255)
BLUE = (72, 98, 255)
AMBER = (255, 188, 84)
RED = (255, 109, 87)
WHITE = (245, 250, 255)

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "Completed"
RNG = random.Random(42)

FONT_CACHE = {}


def parse_args():
    parser = argparse.ArgumentParser(description="Generate full-frame tech B-roll clips.")
    parser.add_argument(
        "--duration",
        type=float,
        default=DEFAULT_DURATION,
        help="Duration for each generated clip.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=FPS,
        help="Frames per second for the generated clips.",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        choices=["code", "training", "sorting", "agent-flow"],
        help="Render only the selected clip ids.",
    )
    return parser.parse_args()


def clamp(value, low=0.0, high=1.0):
    return max(low, min(high, value))


def ease_in_out(value):
    return 0.5 - 0.5 * math.cos(math.pi * clamp(value))


def lerp(a, b, amount):
    return a + (b - a) * amount


def lerp_color(color_a, color_b, amount):
    return tuple(int(lerp(a, b, amount)) for a, b in zip(color_a, color_b))


def alpha_composite(base, overlay, alpha):
    if alpha <= 0:
        return base
    if alpha >= 1:
        return overlay
    return Image.blend(base, overlay, alpha)


def load_font(size, bold=False):
    key = (size, bold)
    if key in FONT_CACHE:
        return FONT_CACHE[key]

    candidates = [
        ("/System/Library/Fonts/Menlo.ttc", 0 if not bold else 1),
        ("/System/Library/Fonts/Supplemental/Courier New.ttf", 0),
        ("/System/Library/Fonts/Courier.ttc", 0 if not bold else 1),
    ]
    for path, index in candidates:
        if not os.path.exists(path):
            continue
        try:
            font = ImageFont.truetype(path, size=size, index=index)
            FONT_CACHE[key] = font
            return font
        except OSError:
            continue

    font = ImageFont.load_default()
    FONT_CACHE[key] = font
    return font


def make_background(accent_a, accent_b):
    y = np.linspace(0.0, 1.0, HEIGHT, dtype=np.float32)[:, None]
    x = np.linspace(0.0, 1.0, WIDTH, dtype=np.float32)[None, :]

    diagonal = (0.65 * x) + (0.35 * y)
    radial = np.sqrt((x - 0.72) ** 2 + (y - 0.22) ** 2)
    radial = 1.0 - clamp_array(radial / 0.95)

    base = np.zeros((HEIGHT, WIDTH, 3), dtype=np.float32)
    for index in range(3):
        base[:, :, index] = BG[index]
        base[:, :, index] += diagonal * accent_a[index] * 0.18
        base[:, :, index] += radial * accent_b[index] * 0.28

    image = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), "RGB")
    draw = ImageDraw.Draw(image)

    for x_pos in range(0, WIDTH, 64):
        color = GRID if x_pos % 256 else lerp_color(GRID, WHITE, 0.08)
        draw.line([(x_pos, 0), (x_pos, HEIGHT)], fill=color, width=1)
    for y_pos in range(0, HEIGHT, 64):
        color = GRID if y_pos % 256 else lerp_color(GRID, WHITE, 0.08)
        draw.line([(0, y_pos), (WIDTH, y_pos)], fill=color, width=1)

    vignette = Image.new("RGB", (WIDTH, HEIGHT), BG)
    mask = Image.new("L", (WIDTH, HEIGHT), 0)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.ellipse(
        (-WIDTH * 0.15, -HEIGHT * 0.20, WIDTH * 1.15, HEIGHT * 1.20),
        fill=110,
    )
    return alpha_composite(image, vignette, 0.18).convert("RGB").point(lambda value: value)


def clamp_array(values, low=0.0, high=1.0):
    return np.minimum(np.maximum(values, low), high)


def draw_panel(draw, box, title=None):
    draw.rounded_rectangle(box, radius=24, fill=PANEL_ALT, outline=lerp_color(GRID, CYAN, 0.20), width=2)
    if title:
        font = load_font(28, bold=True)
        draw.text((box[0] + 22, box[1] + 18), title, fill=TEXT, font=font)


def draw_status_text(draw, x_pos, y_pos, text_value, color=TEXT, size=28, bold=False):
    draw.text((x_pos, y_pos), text_value, fill=color, font=load_font(size, bold=bold))


def build_code_lines():
    templates = [
        "graph = planner.expand(tasks, memory=context.memory, limit=64)",
        "if cache.hit(node.key): return executor.resume(node.key)",
        "emb = encoder(batch_tokens).mean(dim=1)",
        "loss = recon_loss + 0.04 * routing_penalty + aux_loss",
        "optimizer.zero_grad(set_to_none=True)",
        "loss.backward(); clip_grad_norm_(model.parameters(), 1.0)",
        "weights[layer_id] -= lr * gradients[layer_id]",
        "ctx = memory.retrieve(query=prompt, top_k=8, strategy='hybrid')",
        "for tool_call in planner.dispatch(plan, repo_state, runtime):",
        "events.append(trace.emit('tool.exec', name=tool_call.name))",
        "if tests.failed: executor.patch(diff, retries=2)",
        "metrics['acc'] = correct / max(total, 1)",
        "router_score = softmax(gates / temperature, dim=-1)",
        "samples = sampler(step=step_id, top_p=0.92, min_p=0.08)",
        "chunks = chunker.split(document, overlap=64, target_tokens=512)",
        "vector_db.upsert(ids=batch_ids, values=embeddings, namespace='repo')",
        "for epoch in range(epochs): train_loop(epoch, train_loader, model)",
        "print(f'epoch={epoch:03d} loss={loss:.4f} acc={acc:.3f}')",
        "if node.priority > frontier.peek().priority: frontier.push(node)",
        "return merge_artifacts(traces, checkpoints, diagnostics)",
    ]
    lines = []
    for block in range(7):
        lines.append(f"// graph batch {block:02d}")
        for template in templates:
            template = template.replace("epoch", f"epoch_{block}")
            template = template.replace("step_id", str(120 * block + 7))
            template = template.replace("layer_id", str(block % 4))
            lines.append(template)
        lines.append("")
    return lines


def segment_code_line(text_value):
    pattern = re.compile(
        r"//.*$|\"[^\"]*\"|'[^']*'|[A-Za-z_][A-Za-z0-9_]*|\d+\.\d+|\d+|==|!=|<=|>=|->|=>|[{}()\[\],.:;=+\-/*<>%]|\s+|."
    )
    keywords = {
        "for",
        "if",
        "return",
        "in",
        "range",
        "True",
        "False",
        "None",
        "dim",
        "namespace",
        "retries",
        "strategy",
        "print",
    }

    segments = []
    for token in pattern.findall(text_value):
        if token.isspace():
            segments.append((token, TEXT))
        elif token.startswith("//"):
            segments.append((token, MUTED))
        elif token.startswith(("'", '"')):
            segments.append((token, AMBER))
        elif token in keywords:
            segments.append((token, CYAN))
        elif re.fullmatch(r"\d+\.\d+|\d+", token):
            segments.append((token, BLUE))
        elif token in "{}()[],:;=+-/*<>%":
            segments.append((token, WHITE))
        else:
            segments.append((token, GREEN))
    return segments


def build_code_canvas():
    font = load_font(86, bold=False)
    line_no_font = load_font(50, bold=False)
    line_height = 106
    width = 3200
    height = 7200
    image = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(image)

    x_base = 180
    y_base = 120
    for line_index, line in enumerate(build_code_lines()):
        y_pos = y_base + line_index * line_height
        if y_pos > height - 150:
            break
        draw.text((56, y_pos + 16), f"{line_index + 1:02d}", fill=(56, 78, 95), font=line_no_font)
        x_pos = x_base + int(26 * math.sin(line_index * 0.51))
        for token, color in segment_code_line(line):
            draw.text((x_pos, y_pos), token, fill=color, font=font)
            x_pos += draw.textlength(token, font=font)

    return image


def build_training_assets():
    steps = 160
    epochs = np.linspace(0.0, 1.0, steps)
    loss = (1.65 * np.exp(-epochs * 3.2)) + 0.08 + (np.sin(epochs * 18.0) * 0.03)
    acc = 0.46 + (0.48 * (1.0 - np.exp(-epochs * 2.7))) + (np.sin(epochs * 14.0) * 0.01)
    centers = np.array(
        [
            [0.20, 0.30],
            [0.75, 0.25],
            [0.32, 0.78],
            [0.78, 0.72],
        ],
        dtype=np.float32,
    )
    scatter = RNG.random()
    _ = scatter
    points = []
    for cluster_index, center in enumerate(centers):
        cluster_rng = random.Random(100 + cluster_index)
        for _ in range(44):
            start = np.array([cluster_rng.uniform(0.12, 0.88), cluster_rng.uniform(0.15, 0.86)], dtype=np.float32)
            end = center + np.array([cluster_rng.uniform(-0.08, 0.08), cluster_rng.uniform(-0.09, 0.09)], dtype=np.float32)
            points.append((start, end, cluster_index))
    heatmap = np.zeros((12, 12), dtype=np.float32)
    for y_pos in range(12):
        for x_pos in range(12):
            diagonal = 1.0 - abs(y_pos - x_pos) / 11.0
            heatmap[y_pos, x_pos] = clamp(0.16 + diagonal * 0.82 + math.sin((x_pos + y_pos) * 0.55) * 0.08)
    return {
        "loss": loss,
        "acc": acc,
        "points": points,
        "heatmap": heatmap,
    }


def build_sort_assets():
    values = list(range(1, 97))
    RNG.shuffle(values)
    states = []
    comparisons = 0
    moves = 0
    for index in range(1, len(values)):
        key = values[index]
        inner = index - 1
        while inner >= 0 and values[inner] > key:
            comparisons += 1
            values[inner + 1] = values[inner]
            moves += 1
            states.append((list(values), inner, inner + 1, comparisons, moves))
            inner -= 1
        comparisons += 1
        values[inner + 1] = key
        moves += 1
        states.append((list(values), max(inner + 1, 0), index, comparisons, moves))
    return states


def build_agent_assets():
    nodes = {
        "Prompt": (210, 210),
        "Planner": (520, 210),
        "Memory": (820, 130),
        "Repo": (840, 330),
        "Model": (520, 480),
        "Tools": (1180, 520),
        "Terminal": (1180, 210),
        "Tests": (1500, 210),
        "Merge": (1730, 210),
    }
    edges = [
        ("Prompt", "Planner"),
        ("Planner", "Memory"),
        ("Memory", "Planner"),
        ("Planner", "Repo"),
        ("Repo", "Planner"),
        ("Planner", "Model"),
        ("Model", "Planner"),
        ("Planner", "Tools"),
        ("Tools", "Terminal"),
        ("Terminal", "Tests"),
        ("Tests", "Planner"),
        ("Planner", "Merge"),
    ]
    path = [
        ("Prompt", "Planner"),
        ("Planner", "Memory"),
        ("Memory", "Planner"),
        ("Planner", "Repo"),
        ("Repo", "Planner"),
        ("Planner", "Tools"),
        ("Tools", "Terminal"),
        ("Terminal", "Tests"),
        ("Tests", "Planner"),
        ("Planner", "Merge"),
    ]
    logs = [
        "trace.prompt ingest request id=AX7 model=gpt-5.4",
        "trace.memory retrieve top_k=8 miss=2 latency=31ms",
        "trace.plan expand nodes=14 frontier=5 budget=low",
        "trace.repo rg --files matched 128 paths in workspace",
        "trace.tool terminal pytest -q target=auth/session",
        "trace.test failures=1 retry=patch diff=14 lines",
        "trace.memory commit summary shards=3 ttl=2h",
        "trace.plan merge artifacts confidence=0.94",
        "trace.exec publish candidate build=green",
    ]
    return {"nodes": nodes, "edges": edges, "path": path, "logs": logs}


def draw_line_chart(draw, box, values, progress, color, y_min=None, y_max=None):
    x0, y0, x1, y1 = box
    margin = 28
    plot_x0 = x0 + margin
    plot_y0 = y0 + margin
    plot_x1 = x1 - margin
    plot_y1 = y1 - margin

    for step in range(5):
        y_pos = int(lerp(plot_y0, plot_y1, step / 4.0))
        draw.line([(plot_x0, y_pos), (plot_x1, y_pos)], fill=GRID, width=1)
    for step in range(7):
        x_pos = int(lerp(plot_x0, plot_x1, step / 6.0))
        draw.line([(x_pos, plot_y0), (x_pos, plot_y1)], fill=GRID, width=1)

    y_min = float(min(values) if y_min is None else y_min)
    y_max = float(max(values) if y_max is None else y_max)
    y_span = max(y_max - y_min, 1e-5)

    count = max(2, int(progress * len(values)))
    points = []
    for index in range(count):
        x_pos = int(lerp(plot_x0, plot_x1, index / (len(values) - 1)))
        y_norm = (values[index] - y_min) / y_span
        y_pos = int(lerp(plot_y1, plot_y0, y_norm))
        points.append((x_pos, y_pos))

    if len(points) > 1:
        draw.line(points, fill=color, width=4)
    if points:
        px, py = points[-1]
        draw.ellipse((px - 8, py - 8, px + 8, py + 8), fill=color, outline=WHITE, width=2)


def render_code_frame(frame_index, total_frames, assets):
    progress = frame_index / max(total_frames - 1, 1)
    zoom = 1.05 + (0.08 * math.sin(progress * math.pi * 1.6))
    view_width = int(WIDTH / zoom)
    view_height = int(HEIGHT / zoom)
    x_max = max(assets["code_canvas"].width - view_width, 1)
    y_max = max(assets["code_canvas"].height - view_height, 1)
    x_pos = int((0.08 + 0.20 * ease_in_out(progress)) * x_max)
    y_pos = int(progress * 0.72 * y_max)

    viewport = assets["code_canvas"].crop((x_pos, y_pos, x_pos + view_width, y_pos + view_height))
    image = viewport.resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)
    overlay = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    scan_x = int((progress * 1.2 % 1.0) * WIDTH)
    draw.rectangle((scan_x - 52, 0, scan_x + 52, HEIGHT), fill=(20, 255, 150, 18))
    draw.rectangle((WIDTH - 370, 70, WIDTH - 90, 150), fill=(3, 7, 10, 170), outline=(48, 78, 94, 180), width=2)
    draw_status_text(draw, WIDTH - 340, 92, "graph.run", color=WHITE, size=34, bold=True)
    draw_status_text(draw, WIDTH - 340, 122, f"cursor={int(progress * 286):03d}", color=GREEN, size=24)
    return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")


def render_training_frame(frame_index, total_frames, assets):
    progress = frame_index / max(total_frames - 1, 1)
    image = assets["training_bg"].copy()
    draw = ImageDraw.Draw(image)

    loss_panel = (70, 70, 900, 470)
    network_panel = (950, 70, 1850, 470)
    scatter_panel = (70, 530, 640, 980)
    heat_panel = (675, 530, 1220, 980)
    logs_panel = (1250, 530, 1850, 980)

    for box, title in [
        (loss_panel, "train / loss"),
        (network_panel, "model / routing"),
        (scatter_panel, "embeddings"),
        (heat_panel, "attention"),
        (logs_panel, "run log"),
    ]:
        draw_panel(draw, box, title=title)

    draw_line_chart(draw, (loss_panel[0] + 8, loss_panel[1] + 56, 480, 450), assets["loss"], progress, GREEN, y_min=0.0, y_max=1.8)
    draw_line_chart(draw, (490, loss_panel[1] + 56, 880, 450), assets["acc"], progress, CYAN, y_min=0.42, y_max=0.98)
    draw_status_text(draw, 98, 410, f"loss {assets['loss'][max(1, int(progress * (len(assets['loss']) - 1)))]:.4f}", color=GREEN, size=28)
    draw_status_text(draw, 522, 410, f"acc {assets['acc'][max(1, int(progress * (len(assets['acc']) - 1)))]:.3f}", color=CYAN, size=28)

    node_columns = [
        [(1080, 160), (1080, 270), (1080, 380)],
        [(1280, 130), (1280, 220), (1280, 310), (1280, 400)],
        [(1500, 170), (1500, 270), (1500, 370)],
    ]
    for col_index in range(len(node_columns) - 1):
        for src in node_columns[col_index]:
            for dst in node_columns[col_index + 1]:
                if abs(src[1] - dst[1]) > 170:
                    continue
                draw.line((src, dst), fill=lerp_color(GRID, CYAN, 0.24), width=3)
                pulse = (progress * 4.0 + (src[1] + dst[1]) * 0.0018) % 1.0
                pulse_x = int(lerp(src[0], dst[0], pulse))
                pulse_y = int(lerp(src[1], dst[1], pulse))
                draw.ellipse((pulse_x - 7, pulse_y - 7, pulse_x + 7, pulse_y + 7), fill=CYAN)
    for column in node_columns:
        for node_x, node_y in column:
            glow = 0.35 + 0.25 * math.sin(progress * math.pi * 4.0 + node_x * 0.01)
            color = lerp_color(GREEN, CYAN, glow)
            draw.ellipse((node_x - 18, node_y - 18, node_x + 18, node_y + 18), fill=color, outline=WHITE, width=2)

    for start, end, cluster_index in assets["points"]:
        point = start * (1.0 - progress) + end * progress
        px = int(90 + point[0] * 520)
        py = int(570 + point[1] * 370)
        tone = [GREEN, CYAN, AMBER, BLUE][cluster_index]
        draw.ellipse((px - 4, py - 4, px + 4, py + 4), fill=tone)

    heat = assets["heatmap"] * (0.55 + progress * 0.45)
    cell_w = 38
    cell_h = 28
    for y_pos in range(heat.shape[0]):
        for x_pos in range(heat.shape[1]):
            value = float(heat[y_pos, x_pos])
            color = lerp_color((18, 30, 40), GREEN if x_pos >= y_pos else CYAN, value)
            cx = 710 + x_pos * (cell_w + 5)
            cy = 584 + y_pos * (cell_h + 5)
            draw.rounded_rectangle((cx, cy, cx + cell_w, cy + cell_h), radius=8, fill=color)

    visible_logs = []
    log_count = 12
    epoch = int(progress * 420)
    lr = 3.0e-4 - progress * 1.4e-4
    for offset in range(log_count):
        log_epoch = max(0, epoch - (log_count - offset) * 3)
        value_index = clamp(log_epoch / 420.0)
        loss_val = lerp(1.44, 0.12, ease_in_out(value_index))
        acc_val = lerp(0.49, 0.93, ease_in_out(value_index))
        visible_logs.append(
            f"epoch={log_epoch:03d}  loss={loss_val:.4f}  acc={acc_val:.3f}  lr={lr:.1e}"
        )
    for index, line in enumerate(visible_logs):
        color = WHITE if index == len(visible_logs) - 1 else lerp_color(MUTED, CYAN, index / len(visible_logs))
        draw_status_text(draw, 1275, 600 + index * 28, line, color=color, size=22)

    return image


def render_sorting_frame(frame_index, total_frames, assets):
    progress = frame_index / max(total_frames - 1, 1)
    image = assets["sorting_bg"].copy()
    draw = ImageDraw.Draw(image)

    draw_status_text(draw, 84, 72, "insertion sort", color=WHITE, size=38, bold=True)
    draw_status_text(draw, 84, 116, "algorithm trace / comparisons / moves", color=MUTED, size=24)

    q = progress * (len(assets["sort_states"]) - 1)
    index_a = int(q)
    index_b = min(index_a + 1, len(assets["sort_states"]) - 1)
    mix = q - index_a

    values_a, active_left, active_right, comparisons, moves = assets["sort_states"][index_a]
    values_b, _, _, _, _ = assets["sort_states"][index_b]
    values = [lerp(a, b, mix) for a, b in zip(values_a, values_b)]

    chart_box = (90, 170, 1830, 990)
    draw.rounded_rectangle(chart_box, radius=28, fill=PANEL_ALT, outline=lerp_color(GRID, CYAN, 0.18), width=2)

    x0, y0, x1, y1 = chart_box
    count = len(values)
    bar_area_w = x1 - x0 - 80
    bar_w = bar_area_w / count
    floor = y1 - 60
    ceiling = y0 + 80
    max_height = floor - ceiling

    for step in range(6):
        gy = int(lerp(ceiling, floor, step / 5.0))
        draw.line((x0 + 40, gy, x1 - 40, gy), fill=GRID, width=1)

    for index, value in enumerate(values):
        x_pos = x0 + 40 + index * bar_w
        height = max_height * (value / count)
        top = floor - height
        color = lerp_color(CYAN, GREEN, index / max(count - 1, 1))
        if index in {active_left, active_right}:
            color = AMBER if index == active_left else WHITE
        draw.rounded_rectangle(
            (x_pos, top, x_pos + max(bar_w - 2, 4), floor),
            radius=4,
            fill=color,
        )

    code_panel = (1330, 120, 1845, 350)
    draw_panel(draw, code_panel, title="loop")
    code_lines = [
        "for i in range(1, n):",
        "    key = items[i]",
        "    j = i - 1",
        "    while j >= 0 and items[j] > key:",
        "        items[j + 1] = items[j]",
        "        j -= 1",
        "    items[j + 1] = key",
    ]
    for line_index, code_line in enumerate(code_lines):
        draw_status_text(draw, 1358, 170 + line_index * 24, code_line, color=GREEN if line_index >= 3 else CYAN, size=22)

    draw_status_text(draw, 104, 1004, f"comparisons {comparisons}", color=WHITE, size=28)
    draw_status_text(draw, 430, 1004, f"moves {moves}", color=CYAN, size=28)
    draw_status_text(draw, 690, 1004, f"progress {progress * 100:05.1f}%", color=GREEN, size=28)
    return image


def draw_edge_with_pulse(draw, start, end, progress, active=False):
    color = lerp_color(GRID, CYAN if active else MUTED, 0.45 if active else 0.14)
    draw.line((start, end), fill=color, width=4)
    pulse_x = int(lerp(start[0], end[0], progress))
    pulse_y = int(lerp(start[1], end[1], progress))
    radius = 9 if active else 6
    draw.ellipse((pulse_x - radius, pulse_y - radius, pulse_x + radius, pulse_y + radius), fill=GREEN if active else CYAN)


def render_agent_flow_frame(frame_index, total_frames, assets):
    progress = frame_index / max(total_frames - 1, 1)
    image = assets["flow_bg"].copy()
    draw = ImageDraw.Draw(image)

    draw_status_text(draw, 84, 72, "agent trace", color=WHITE, size=38, bold=True)
    draw_status_text(draw, 84, 116, "planner / memory / tools / tests / merge", color=MUTED, size=24)

    active_slot = int(progress * len(assets["path"]) * 1.3) % len(assets["path"])
    pulse_progress = (progress * len(assets["path"]) * 1.3) % 1.0

    for edge_index, edge in enumerate(assets["edges"]):
        active = assets["path"][active_slot] == edge
        draw_edge_with_pulse(
            draw,
            assets["nodes"][edge[0]],
            assets["nodes"][edge[1]],
            pulse_progress if active else (progress * 2.2 + edge_index * 0.07) % 1.0,
            active=active,
        )

    for node_name, (node_x, node_y) in assets["nodes"].items():
        is_active = node_name in assets["path"][active_slot]
        fill = lerp_color(PANEL_ALT, CYAN if is_active else PANEL, 0.40 if is_active else 0.0)
        outline = GREEN if is_active else lerp_color(GRID, CYAN, 0.18)
        box = (node_x - 105, node_y - 34, node_x + 105, node_y + 34)
        draw.rounded_rectangle(box, radius=18, fill=fill, outline=outline, width=3)
        draw_status_text(draw, node_x - 72, node_y - 16, node_name.lower(), color=WHITE, size=24, bold=True)

    right_panel = (1260, 510, 1850, 980)
    bottom_panel = (70, 720, 1180, 980)
    draw_panel(draw, right_panel, title="runtime")
    draw_panel(draw, bottom_panel, title="context / memory")

    visible_logs = []
    step_base = int(progress * 40)
    for offset in range(12):
        visible_logs.append(assets["logs"][(step_base + offset) % len(assets["logs"])])
    for index, line in enumerate(visible_logs):
        color = WHITE if index == 8 else lerp_color(MUTED, CYAN, index / 12.0)
        draw_status_text(draw, 1288, 560 + index * 30, line, color=color, size=22)

    tokens = ["prompt", "summary", "repo diff", "test trace", "tool output", "memory slot"]
    for index, token in enumerate(tokens):
        x0 = 104 + index * 174
        y0 = 790
        width = 150
        draw.rounded_rectangle((x0, y0, x0 + width, y0 + 72), radius=16, fill=PANEL, outline=lerp_color(GRID, CYAN, 0.16), width=2)
        draw_status_text(draw, x0 + 14, y0 + 14, token, color=TEXT, size=22)
        bar = 0.24 + 0.68 * clamp(math.sin(progress * math.pi * 2.0 + index) * 0.5 + 0.5)
        draw.rounded_rectangle((x0 + 14, y0 + 46, x0 + 14 + (width - 28) * bar, y0 + 56), radius=5, fill=GREEN if index % 2 else CYAN)

    metric_names = [("latency", "148ms"), ("tools", "04"), ("tests", "pass"), ("context", "28k")]
    for index, (name, value) in enumerate(metric_names):
        x0 = 100 + index * 250
        draw.rounded_rectangle((x0, 890, x0 + 220, 954), radius=18, fill=PANEL, outline=lerp_color(GRID, CYAN, 0.16), width=2)
        draw_status_text(draw, x0 + 16, 906, name, color=MUTED, size=20)
        draw_status_text(draw, x0 + 108, 902, value, color=WHITE if value == "pass" else CYAN, size=24, bold=True)
    return image


def build_assets():
    return {
        "code_canvas": build_code_canvas(),
        "training_bg": make_background((16, 86, 62), (0, 118, 180)),
        "sorting_bg": make_background((22, 74, 58), (0, 98, 60)),
        "flow_bg": make_background((16, 72, 102), (0, 110, 88)),
        "training": build_training_assets(),
        "sort_states": build_sort_assets(),
        "agent": build_agent_assets(),
    }


def open_ffmpeg_writer(output_path, fps):
    command = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{WIDTH}x{HEIGHT}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "prores_ks",
        "-profile:v",
        "3",
        "-pix_fmt",
        "yuv422p10le",
        "-colorspace",
        "1",
        "-color_primaries",
        "1",
        "-color_trc",
        "1",
        str(output_path),
    ]
    return subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def render_clip(output_path, duration, fps, render_frame):
    total_frames = int(round(duration * fps))
    process = open_ffmpeg_writer(output_path, fps)
    try:
        for frame_index in range(total_frames):
            frame = render_frame(frame_index, total_frames)
            process.stdin.write(frame.tobytes())
            if frame_index % fps == 0:
                print(f"  frame {frame_index:04d}/{total_frames}")
    finally:
        if process.stdin:
            process.stdin.close()
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"ffmpeg failed while rendering {output_path}")


def main():
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    assets = build_assets()

    clip_specs = [
        (
            "code",
            OUTPUT_DIR / "tech_code_surface.mov",
            lambda frame_index, total_frames: render_code_frame(frame_index, total_frames, assets),
        ),
        (
            "training",
            OUTPUT_DIR / "tech_model_training.mov",
            lambda frame_index, total_frames: render_training_frame(
                frame_index,
                total_frames,
                {"training_bg": assets["training_bg"], **assets["training"]},
            ),
        ),
        (
            "sorting",
            OUTPUT_DIR / "tech_sorting_trace.mov",
            lambda frame_index, total_frames: render_sorting_frame(frame_index, total_frames, {"sorting_bg": assets["sorting_bg"], "sort_states": assets["sort_states"]}),
        ),
        (
            "agent-flow",
            OUTPUT_DIR / "tech_agent_flow.mov",
            lambda frame_index, total_frames: render_agent_flow_frame(frame_index, total_frames, {"flow_bg": assets["flow_bg"], **assets["agent"]}),
        ),
    ]

    selected = set(args.only or [spec[0] for spec in clip_specs])
    for clip_id, output_path, renderer in clip_specs:
        if clip_id not in selected:
            continue
        print(f"\nRendering {clip_id} -> {output_path.name}")
        render_clip(output_path, args.duration, args.fps, renderer)

    print("\nDone.")


if __name__ == "__main__":
    main()
