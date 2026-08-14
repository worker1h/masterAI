from __future__ import annotations

import subprocess
from pathlib import Path

import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "submission"
TMP = ROOT / "tmp" / "video_slides"
TEAM = "洪析先知团队"
PROJECT = "洪析先知"
VIDEO = OUT / f"{TEAM}_{PROJECT}_项目视频.mp4"

W, H = 1920, 1080
NAVY = "#17365D"
BLUE = "#2E74B5"
CYAN = "#16A6B6"
PALE = "#EAF2F8"
INK = "#202A35"
MUTED = "#66727F"
WHITE = "#FFFFFF"
ORANGE = "#F28E2B"
FONT = "C:/Windows/Fonts/msyh.ttc"
FONT_BOLD = "C:/Windows/Fonts/msyhbd.ttc"


def f(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_BOLD if bold else FONT, size)


def canvas() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    im = Image.new("RGB", (W, H), WHITE)
    d = ImageDraw.Draw(im)
    d.rectangle((0, 0, 28, H), fill=BLUE)
    d.rectangle((28, 0, 38, H), fill=CYAN)
    return im, d


def footer(d: ImageDraw.ImageDraw, index: int) -> None:
    d.line((96, 1005, W - 96, 1005), fill="#D7E2F0", width=2)
    d.text((96, 1020), f"{TEAM}｜第八届中国研究生人工智能创新大赛", font=f(24), fill=MUTED)
    d.text((W - 130, 1020), f"{index}/7", font=f(24), fill=MUTED, anchor="ra")


def header(d: ImageDraw.ImageDraw, kicker: str, title: str, index: int) -> None:
    d.text((100, 64), kicker, font=f(28, True), fill=BLUE)
    d.text((100, 116), title, font=f(52, True), fill=NAVY)
    footer(d, index)


def wrapped(d: ImageDraw.ImageDraw, text: str, xy: tuple[int, int], width: int, size: int,
            fill: str = INK, spacing: int = 18, bold: bool = False) -> int:
    font = f(size, bold)
    lines: list[str] = []
    line = ""
    for ch in text:
        if ch == "\n":
            lines.append(line)
            line = ""
        elif d.textlength(line + ch, font=font) > width and line:
            lines.append(line)
            line = ch
        else:
            line += ch
    if line:
        lines.append(line)
    y = xy[1]
    for line in lines:
        d.text((xy[0], y), line, font=font, fill=fill)
        y += size + spacing
    return y


def card(d: ImageDraw.ImageDraw, box: tuple[int, int, int, int], title: str, body: str,
         accent: str = BLUE, value: str | None = None) -> None:
    d.rounded_rectangle(box, radius=22, fill="#F7FAFD", outline="#D7E2F0", width=3)
    x1, y1, x2, _ = box
    d.rectangle((x1, y1, x1 + 10, y1 + 80), fill=accent)
    d.text((x1 + 34, y1 + 24), title, font=f(30, True), fill=NAVY)
    if value:
        d.text((x2 - 32, y1 + 24), value, font=f(28, True), fill=accent, anchor="ra")
    wrapped(d, body, (x1 + 34, y1 + 96), x2 - x1 - 68, 27, MUTED, 14)


def fit_image(path: Path, box: tuple[int, int, int, int]) -> Image.Image:
    src = Image.open(path).convert("RGB")
    x1, y1, x2, y2 = box
    src.thumbnail((x2 - x1, y2 - y1), Image.Resampling.LANCZOS)
    return src


def save_slide(im: Image.Image, n: int) -> Path:
    path = TMP / f"slide-{n}.png"
    im.save(path, quality=95)
    return path


def build_slides() -> list[Path]:
    TMP.mkdir(parents=True, exist_ok=True)
    slides: list[Path] = []

    im, d = canvas()
    d.text((100, 84), "第八届中国研究生人工智能创新大赛", font=f(30, True), fill=BLUE)
    d.text((100, 246), PROJECT, font=f(94, True), fill=NAVY)
    d.text((100, 376), "基于多时相 SAR 与地形先验的洪涝智能评估系统", font=f(44), fill=BLUE)
    d.rounded_rectangle((100, 520, 1480, 680), radius=26, fill=PALE)
    wrapped(d, "让模型输出可验证、可解释、可复现的新增洪水空间证据", (142, 566), 1280, 32, NAVY, 18, True)
    d.text((100, 860), "正式实验结果与系统原型演示｜2026.08", font=f(30), fill=MUTED)
    footer(d, 1)
    slides.append(save_slide(im, 1))

    im, d = canvas(); header(d, "01  数据与任务边界", "先核验时相语义，再定义模型任务", 2)
    card(d, (100, 250, 650, 530), "数据规模", "官方 train / val / test 三划分。\n覆盖 206 场洪水事件。", CYAN, "24,578 图块")
    card(d, (685, 250, 1235, 530), "未见事件测试", "由 12 个训练阶段完全未见事件构成。\ntest_holdout 避免同事件泄漏。", BLUE, "562 图块")
    card(d, (1270, 250, 1820, 530), "标签定义", "MASK 中类别 2 为新增洪水；类别 1 永久水体按负类处理。", ORANGE, "Flood = 2")
    d.rounded_rectangle((100, 595, 1820, 930), radius=24, fill="#F7FAFD", outline="#D7E2F0", width=3)
    d.text((140, 630), "散射统计证据", font=f(32, True), fill=NAVY)
    d.text((140, 710), "pre-event", font=f(29, True), fill=MUTED); d.text((680, 710), "event", font=f(29, True), fill=BLUE); d.text((1220, 710), "post-event", font=f(29, True), fill=MUTED)
    d.text((140, 765), "VV/VH  -9.96 / -15.26 dB", font=f(28), fill=INK)
    d.text((680, 765), "-15.18 / -21.17 dB", font=f(28, True), fill=BLUE)
    d.text((1220, 765), "-10.79 / -16.16 dB", font=f(28), fill=INK)
    wrapped(d, "事件期显著下降、事后回升，证明标签对应 event 当前状态；因此本项目是洪涝智能评估，不宣称未来预测。", (140, 840), 1570, 27, INK, 12)
    slides.append(save_slide(im, 2))

    im, d = canvas(); header(d, "02  模型方案", "四组受控消融，定位真正有效的信息源", 3)
    xs = [110, 505, 900, 1295]
    labels = [("E0", "Event SAR", "VV / VH"), ("E1", "Event + DEM", "单时相 + 地形"), ("E2", "Pre + Event", "多时相 SAR"), ("E3", "Pre + Event + DEM", "多时相 + 地形")]
    for i, (tag, title, body) in enumerate(labels):
        x = xs[i]
        d.rounded_rectangle((x, 260, x + 335, 535), radius=24, fill=PALE if tag == "E2" else "#F7FAFD", outline=BLUE if tag == "E2" else "#D7E2F0", width=5 if tag == "E2" else 3)
        d.text((x + 28, 286), tag, font=f(34, True), fill=BLUE)
        wrapped(d, title, (x + 28, 350), 275, 28, NAVY, 12, True)
        wrapped(d, body, (x + 28, 445), 275, 25, MUTED, 10)
    d.rounded_rectangle((230, 650, 1690, 900), radius=28, fill=NAVY)
    items = ["三层 U-Net", "BCE + Dice", "AdamW", "固定 seed 42", "Val 选模 / Holdout 终测"]
    x = 285
    for item in items:
        w = d.textlength(item, font=f(28, True)) + 56
        d.rounded_rectangle((x, 735, x + w, 810), radius=20, fill=WHITE)
        d.text((x + 28, 757), item, font=f(28, True), fill=NAVY)
        x += int(w) + 28
    slides.append(save_slide(im, 3))

    im, d = canvas(); header(d, "03  正式实验", "多时相 SAR 是最稳定的主要增益来源", 4)
    chart_path = ROOT / "outputs" / "formal_summary" / "formal_ablation.png"
    chart = fit_image(chart_path, (100, 240, 1275, 895)); im.paste(chart, (100, 240))
    card(d, (1320, 265, 1820, 510), "最佳方案 E2", "Pre-event + event SAR 在完全未见事件上取得最高总体指标。", BLUE, "IoU 0.4941")
    card(d, (1320, 545, 1820, 790), "相对单时相", "E2 相对 E0 的 IoU 绝对提升；Dice/F1 达到 0.6614。", CYAN, "+0.1324")
    wrapped(d, "DEM 在单时相上有小幅增益，但简单拼接到多时相后未超过 E2；结论按实验证据表述。", (1320, 835), 500, 24, MUTED, 10)
    slides.append(save_slide(im, 4))

    im, d = canvas(); header(d, "04  结果示例", "从雷达影像到概率图与新增洪水掩膜", 5)
    pred_path = next((ROOT / "outputs" / "formal_e2" / "predictions").glob("*.png"))
    pred = fit_image(pred_path, (100, 250, 1320, 850)); im.paste(pred, (100, 250))
    card(d, (1370, 285, 1820, 510), "输出内容", "事件期 VV、真值、像素级概率与阈值化预测可并排复核。", BLUE)
    card(d, (1370, 550, 1820, 775), "泛化设置", "该样本来自训练阶段完全未见的洪水事件，而非随机图块混分。", CYAN)
    wrapped(d, "保留概率图和失败案例，便于人工核查与后续误差分析。", (1370, 830), 440, 25, NAVY, 12, True)
    slides.append(save_slide(im, 5))

    im, d = canvas(); header(d, "05  决策表达", "把像素结果转换为可审计的事件级严重度", 6)
    y = 315
    nodes = [(110, "新增洪水像素", BLUE), (505, "排除永久水体", CYAN), (900, "计算影响比例", BLUE), (1295, "三级严重度", ORANGE)]
    for i, (x, label, color) in enumerate(nodes):
        d.rounded_rectangle((x, y, x + 320, y + 150), radius=24, fill="#F7FAFD", outline=color, width=4)
        d.text((x + 160, y + 75), label, font=f(29, True), fill=NAVY, anchor="mm")
        if i < len(nodes) - 1:
            d.line((x + 330, y + 75, x + 375, y + 75), fill=MUTED, width=6)
            d.polygon([(x + 375, y + 75), (x + 355, y + 62), (x + 355, y + 88)], fill=MUTED)
    d.rounded_rectangle((100, 575, 1820, 875), radius=28, fill=PALE)
    d.text((145, 620), "阈值只由训练事件分布确定", font=f(34, True), fill=NAVY)
    d.text((145, 695), "一般 / 严重 / 非常严重：q33 = 0.01473，q67 = 0.08905", font=f(31), fill=INK)
    wrapped(d, "验证与测试固定复用同一阈值。等级仅表达空间影响比例，不等同于人员伤亡、经济损失或自动告警。", (145, 765), 1570, 27, MUTED, 12)
    slides.append(save_slide(im, 6))

    im, d = canvas(); header(d, "06  阶段成果", "模型、实验与初赛提交材料已形成闭环", 7)
    card(d, (100, 250, 650, 520), "可复现模型", "训练脚本、四组配置、正式权重、数据清单与环境快照。", BLUE)
    card(d, (685, 250, 1235, 520), "证据化实验", "validation 选模、未见事件 holdout、事件宏平均与空事件误报。", CYAN)
    card(d, (1270, 250, 1820, 520), "合规交付", "简介、项目文档、项目视频和辅助材料，按初赛规范命名。", ORANGE)
    d.text((100, 660), "当前最优", font=f(32, True), fill=MUTED)
    d.text((100, 720), "E2 · Pre-event + Event SAR", font=f(50, True), fill=NAVY)
    d.text((100, 800), "Unseen-event IoU 0.4941  ·  Dice/F1 0.6614", font=f(38), fill=BLUE)
    d.text((100, 900), "下一步：多随机种子复验、阈值校准、地形独立编码与轻量 Web Demo", font=f(28), fill=MUTED)
    slides.append(save_slide(im, 7))
    return slides


def encode(slides: list[Path]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    concat = TMP / "slides.txt"
    lines: list[str] = []
    for slide in slides:
        lines.extend([f"file '{slide.as_posix()}'", "duration 8"])
    lines.append(f"file '{slides[-1].as_posix()}'")
    concat.write_text("\n".join(lines) + "\n", encoding="utf-8")
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", str(concat),
           "-vf", "scale=1920:1080:flags=lanczos,format=yuv420p", "-r", "30",
           "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-movflags", "+faststart", str(VIDEO)]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    encode(build_slides())
    print(VIDEO)
