from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import imageio_ffmpeg
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "submission"
TEAM = "洪析先知团队"
PROJECT = "洪析先知"
FILES = {
    "intro": OUT / f"{TEAM}_{PROJECT}_参赛作品简介.pdf",
    "document": OUT / f"{TEAM}_{PROJECT}_项目文档.pdf",
    "video": OUT / f"{TEAM}_{PROJECT}_项目视频.mp4",
    "other": OUT / f"{TEAM}_{PROJECT}_其他.zip",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    for path in FILES.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    intro_reader = PdfReader(FILES["intro"])
    intro_text = "".join(page.extract_text() or "" for page in intro_reader.pages)
    start = intro_text.index("洪析先知面向")
    end = intro_text.index("空间证据。", start) + len("空间证据。")
    body_chars = len("".join(intro_text[start:end].split()))

    document_reader = PdfReader(FILES["document"])
    frames = imageio_ffmpeg.read_frames(str(FILES["video"]))
    video_meta = next(frames)
    frames.close()
    with zipfile.ZipFile(FILES["other"]) as zf:
        names = zf.namelist()
        bad_raw = [n for n in names if n.lower().startswith(("data/", "raw/"))]

    checks = {
        "intro_body_chars_le_300": body_chars <= 300,
        "intro_single_page": len(intro_reader.pages) == 1,
        "document_pdf_readable": len(document_reader.pages) > 0,
        "video_duration_le_300s": video_meta["duration"] <= 300,
        "video_size_le_200mb": FILES["video"].stat().st_size <= 200 * 1024 * 1024,
        "other_zip_size_le_200mb": FILES["other"].stat().st_size <= 200 * 1024 * 1024,
        "other_zip_has_no_raw_data": not bad_raw,
    }
    if not all(checks.values()):
        raise RuntimeError({k: v for k, v in checks.items() if not v})

    report = {
        "status": "PASS",
        "generated_at": "2026-08-12 Asia/Shanghai",
        "team_name_note": "洪析先知团队为当前占位团队名，正式提交前须与报名系统一致。",
        "checks": checks,
        "intro": {"pages": len(intro_reader.pages), "body_chars": body_chars},
        "document": {"pages": len(document_reader.pages)},
        "video": {
            "duration_seconds": video_meta["duration"],
            "fps": video_meta["fps"],
            "source_size": video_meta["source_size"],
        },
        "other_zip": {"entries": len(names), "raw_data_entries": bad_raw},
        "files": {
            key: {
                "name": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for key, path in FILES.items()
        },
    }
    result = OUT / "提交文件校验.json"
    result.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

