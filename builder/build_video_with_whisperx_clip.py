#!/usr/bin/env python3
"""build_video_with_whisperx_clip.py

Builder that:
 - runs WhisperX for word-level alignment
 - groups words into caption segments (with merging of very-short segments)
 - loads precomputed assets/image_index.jsonl + assets/clip_embeddings.npy
 - selects images per segment using CLIP (OpenCLIP), with:
     - reuse penalty
     - soft recency (avoid-window) penalty
     - SBERT fallback (BLIP captions vs segment text)
     - optional face boosting (face_recognition)
 - writes a detailed selection CSV log (assets/selection_log.csv)
 - composes final video (MoviePy) and ensures audio is attached

Default resolution: 1280x720 (720p).
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import torch
from tqdm import tqdm
from PIL import Image as PILImage
from moviepy.editor import (
    ImageClip, AudioFileClip, CompositeVideoClip, ColorClip, vfx
)
import whisperx
import open_clip
from sentence_transformers import SentenceTransformer, util

# optional face_recognition
try:
    import face_recognition
    HAVE_FACE = True
except Exception:
    HAVE_FACE = False

# ---------- Config / defaults ----------
IMAGE_INDEX_JSONL = Path("assets/image_index.jsonl")
CLIP_EMB_NPY = Path("assets/clip_embeddings.npy")
SELECTION_LOG = Path("assets/selection_log.csv")
device = "cuda" if torch.cuda.is_available() else "cpu"

# load CLIP model for text embeddings (OpenCLIP)
clip_model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
clip_model.to(device).eval()
sbert = SentenceTransformer("all-MiniLM-L6-v2", device=device)

# ---------- Helpers ----------
import shutil
def has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None

def parse_res(s: str) -> tuple[int, int]:
    w, h = s.lower().split("x")
    return int(w), int(h)

def list_images(root: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = [p for p in sorted(root.rglob("*")) if p.suffix.lower() in exts]
    if not files:
        raise FileNotFoundError(f"No images found under: {root}")
    return files

def ensure_wav(audio_path: Path) -> Path:
    # Convert to WAV 44.1k stereo for stable timing (if ffmpeg present)
    if not has_ffmpeg():
        return audio_path
    out = audio_path.with_suffix(".wav")
    subprocess.run([
        "ffmpeg", "-y", "-i", str(audio_path),
        "-ar", "44100", "-ac", "2", "-c:a", "pcm_s16le", str(out)
    ], check=True)
    return out

# ---------- Silence detection ----------
def detect_silences(audio_path: Path, noise_db: float = -35.0, min_silence: float = 0.35) -> List[Tuple[float,float]]:
    """Use ffmpeg silencedetect and return list of (start,end) silence windows."""
    if not has_ffmpeg():
        return []
    cmd = [
        "ffmpeg", "-i", str(audio_path),
        "-af", f"silencedetect=noise={noise_db}dB:d={min_silence}",
        "-f", "null", "-"
    ]
    proc = subprocess.run(cmd, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL, text=True)
    silences, start_t = [], None
    for line in (proc.stderr or "").splitlines():
        line = line.strip()
        if "silence_start" in line:
            try: start_t = float(line.split("silence_start:")[1].strip())
            except: start_t = None
        elif "silence_end" in line and start_t is not None:
            try:
                end_t = float(line.split("silence_end:")[1].split("|")[0].strip())
                silences.append((start_t, end_t))
            except: pass
            start_t = None
    return silences

# ---------- Captions / segmentation ----------
import re as _re

def _normalize_spaces(text: str) -> str:
    return _re.sub(r"\s+", " ", text).strip()

def split_into_sentences(text: str) -> List[str]:
    text = _normalize_spaces(text.replace("\n"," ").replace("—","-"))
    protected = {}
    def protect(m):
        key=f"__ABBR_{len(protected)}__"; protected[key]=m.group(0); return key
    text = _re.sub(r"\b(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St)\.", protect, text, flags=_re.I)
    parts = _re.split(r"(?<=[\.\?\!])\s+(?=[\"'\(\[]?[A-Z0-9])", text)
    out=[]
    for p in parts:
        p=p.strip()
        if not p: continue
        for k,v in protected.items(): p=p.replace(k,v)
        out.append(p)
    return out

def chunk_long_sentence(sent: str, max_chars: int = 90) -> List[str]:
    sent = _normalize_spaces(sent)
    if len(sent) <= max_chars: return [sent]
    chunks=[]
    while len(sent) > max_chars:
        cut = (sent.rfind(",",0,max_chars) or sent.rfind(";",0,max_chars)
               or sent.rfind(":",0,max_chars) or sent.rfind(" ",0,max_chars))
        if cut == -1 or cut < max_chars*0.4: cut = max_chars
        chunk = sent[:cut].strip(",;: ").strip()
        if chunk: chunks.append(chunk)
        sent = sent[cut:].strip()
    if sent: chunks.append(sent)
    return chunks

def build_caption_segments(script_text: str, total_duration: float,
                          min_dur: float=1.2, max_dur: float=6.0,
                          silences: Optional[List[Tuple[float,float]]]=None,
                          target_token_chars: int = 90) -> List[Tuple[float,float,str]]:
    sentences = split_into_sentences(script_text)
    chunks=[]
    for s in sentences: chunks.extend(chunk_long_sentence(s, max_chars=target_token_chars))
    if not chunks: return []
    if silences:
        cuts = sorted({max(0.0, min(total_duration, (s+e)/2.0)) for s,e in silences})
        anchors = [0.0] + cuts + [total_duration]
        if len(anchors)-1 > len(chunks):
            interior = anchors[1:-1]
            needed = len(chunks)-1
            if needed <= 0:
                anchors = [0.0, total_duration]
            else:
                idxs = [int(i*(len(interior)-1)/(needed-1)) for i in range(needed)] if needed>1 else [len(interior)//2]
                anchors = [0.0] + [interior[i] for i in sorted(set(idxs))] + [total_duration]
        bins = [(anchors[i], anchors[i+1]) for i in range(len(anchors)-1)]
        segs = assign_chunks_with_bins(chunks, bins)
        segs = [(max(0,s), min(total_duration,e), txt) for s,e,txt in segs if s < total_duration]
        if segs and segs[-1][1] < total_duration:
            segs[-1] = (segs[-1][0], total_duration, segs[-1][2])
        merged = _merge_short_segments(segs, min_img_duration=min_dur)
        return merged
    lens=[max(1,len(c)) for c in chunks]; total_chars=sum(lens)
    raw_durs=[max(min_dur, min(max_dur, (total_duration*(L/total_chars)))) for L in lens]
    scale = total_duration/sum(raw_durs); durs=[d*scale for d in raw_durs]
    segs=[]; t=0.0
    for c,d in zip(chunks,durs):
        s=t; e=min(total_duration, t+d); segs.append((s,e,c)); t=e
        if e>=total_duration: break
    if segs and segs[-1][1] < total_duration:
        segs[-1]=(segs[-1][0], total_duration, segs[-1][2])
    segs = _merge_short_segments(segs, min_img_duration=min_dur)
    return segs

def assign_chunks_with_bins(chunks: List[str], bins: List[Tuple[float,float]]) -> List[Tuple[float,float,str]]:
    if not chunks: return []
    segs=[]; idx=0
    for (start,end) in bins:
        if idx >= len(chunks): break
        remaining_chunks = len(chunks)-idx
        remaining_bins = len(bins)-len(segs)
        n_here = max(1, remaining_chunks // remaining_bins)
        slot = chunks[idx:idx+n_here]; idx += n_here
        lens = [max(1,len(c)) for c in slot]; total=sum(lens); t=start
        for L,c in zip(lens,slot):
            dur = (end-start)*(L/total)
            s,e = t, min(end, t+dur)
            segs.append((s,e,c)); t=e
    while idx < len(chunks):
        s,e,txt = segs[-1]; segs.append((e, e+(e-s), chunks[idx])); idx+=1
    return segs

def _merge_short_segments(segs: List[Tuple[float,float,str]], min_img_duration: float = 1.2) -> List[Tuple[float,float,str]]:
    if min_img_duration <= 0: return segs
    if not segs: return segs
    merged=[]
    i=0; n=len(segs)
    while i < n:
        s,e,txt = segs[i]
        dur = e - s
        if dur >= min_img_duration:
            merged.append((s,e,txt)); i+=1; continue
        if i+1 < n:
            ns, ne, nt = segs[i+1]
            new_seg = (s, ne, (txt + " " + nt).strip())
            segs[i+1] = new_seg
            i += 1
        else:
            if merged:
                ps, pe, pt = merged[-1]
                merged[-1] = (ps, e, (pt + " " + txt).strip())
            else:
                merged.append((s, e, txt))
            i += 1
    final=[]
    for s,e,t in merged:
        if (e - s) < min_img_duration:
            e = s + min_img_duration
        final.append((s,e,t))
    return final

# ---------- Pillow text clip helper ----------
from PIL import ImageDraw, ImageFont, Image as PILImageMod

def _load_font(size: int):
    candidates = ["arial.ttf","Arial.ttf","DejaVuSans.ttf","DejaVuSans-Bold.ttf","/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf","C:\\Windows\\Fonts\\arial.ttf"]
    for p in candidates:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            continue
    return ImageFont.load_default()

def pillow_text_clip(text: str, fontsize: int = 48, color: str = "white", stroke_color: str = "black", stroke_width: int = 2, max_width: int | None = None, align: str = "center", padding: int = 8):
    font = _load_font(fontsize)
    dummy = PILImageMod.new("RGBA",(10,10),(0,0,0,0))
    draw = ImageDraw.Draw(dummy)
    if not max_width or max_width <= 0:
        lines = [text]; w = int(draw.textlength(text, font=font)) + padding*2
    else:
        words = text.split()
        lines=[]; line=""
        for w in words:
            test = (line + " " + w).strip()
            if draw.textlength(test, font=font) <= max_width:
                line = test
            else:
                if line: lines.append(line)
                line = w
        if line: lines.append(line)
        w = max(int(draw.textlength(line, font=font)) for line in lines) + padding*2
    ascent, descent = font.getmetrics()
    line_h = ascent + descent + int(fontsize*0.15)
    h = line_h * len(lines) + padding*2
    img = PILImageMod.new("RGBA", (max(2,w), max(2,h)), (0,0,0,0))
    draw = ImageDraw.Draw(img)
    y = padding
    for line in lines:
        line_w_px = int(draw.textlength(line, font=font))
        if align == "left": x = padding
        elif align == "right": x = w - padding - line_w_px
        else: x = (w - line_w_px)//2
        if stroke_width and stroke_color:
            draw.text((x,y), line, font=font, fill=color, stroke_width=stroke_width, stroke_fill=stroke_color)
        else:
            draw.text((x,y), line, font=font, fill=color)
        y += line_h
    arr = np.array(img)
    from moviepy.editor import ImageClip as _IC
    return _IC(arr, ismask=False)

# ---------- CLIP text & SBERT helpers ----------
def embed_text_clip_np(texts: List[str]):
    toks = open_clip.tokenize(texts).to(device)
    with torch.no_grad():
        t_emb = clip_model.encode_text(toks)
        t_emb = t_emb / t_emb.norm(dim=-1, keepdim=True)
    return t_emb.cpu().numpy()

def sbert_sim(q: str, caps_emb_tensor):
    q_emb = sbert.encode(q, convert_to_tensor=True)
    sims = util.cos_sim(q_emb, caps_emb_tensor).cpu().numpy()[0]
    return sims

# ---------- Selection algorithm ----------
def load_image_index(jsonl_path: Path, emb_path: Path):
    items = [json.loads(l) for l in open(jsonl_path, "r", encoding="utf-8") if l.strip()]
    embs = np.load(emb_path)
    if len(items) != embs.shape[0]:
        raise SystemExit(f"Index length mismatch: {len(items)} json lines vs {embs.shape[0]} embeddings")
    # normalize embeddings to be safe
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embs = embs / norms
    return items, embs

def embed_text_clip_single(text: str):
    toks = open_clip.tokenize([text]).to(device)
    with torch.no_grad():
        t_emb = clip_model.encode_text(toks)
        t_emb = t_emb / t_emb.norm(dim=-1, keepdim=True)
    return t_emb.cpu().numpy()[0]

def choose_best_with_soft_window(seg_text: str, items, clip_embs, used_counts: dict, reuse_penalty: float, max_reuse: Optional[int], last_idx: Optional[int], seg_start_time: float, avoid_window: float, last_used_time: dict, avoid_window_penalty: float = 0.28, sbert_thresh=0.34, clip_thresh=0.16, face_ref_emb=None, face_boost=1.5):
    t_emb = embed_text_clip_single(seg_text)
    sims = clip_embs @ t_emb
    adj = sims.copy()
    N = len(adj)
    reuse_penalties = np.zeros(N, dtype=float)
    for i in range(N):
        used = used_counts.get(i, 0)
        if (max_reuse is not None) and (used >= max_reuse):
            adj[i] = -9999.0
            reuse_penalties[i] = used * reuse_penalty
        else:
            reuse_penalties[i] = used * reuse_penalty
            adj[i] = adj[i] - reuse_penalties[i]
    face_boosts = np.zeros(N, dtype=float)
    if face_ref_emb is not None:
        for i in range(N):
            fe = items[i].get("face_emb")
            if fe:
                try:
                    fe_arr = np.array(fe, dtype=np.float32)
                    d = np.linalg.norm(fe_arr - face_ref_emb)
                    if d < 0.65:
                        face_boosts[i] = face_boost
                        adj[i] += face_boosts[i]
                except Exception:
                    pass
    avoid_penalties = np.zeros(N, dtype=float)
    if avoid_window and avoid_window > 0 and avoid_window_penalty and avoid_window_penalty > 0:
        for i in range(N):
            lu = last_used_time.get(i)
            if lu is None: continue
            delta = seg_start_time - float(lu)
            if delta <= 0:
                pen = avoid_window_penalty
            elif delta < avoid_window:
                frac = max(0.0, min(1.0, (1.0 - (delta / avoid_window))))
                pen = frac * avoid_window_penalty
            else:
                pen = 0.0
            avoid_penalties[i] = pen
            adj[i] = adj[i] - pen
    sorted_idx = np.argsort(-adj)
    chosen = None; chosen_score = None; method="clip"
    for idx in sorted_idx:
        if adj[idx] <= -9000: continue
        chosen = int(idx); chosen_score = float(adj[idx]); break
    if chosen is None:
        chosen = int(np.argmax(sims)); chosen_score = float(sims[chosen]); method="clip"
    if chosen_score < clip_thresh:
        captions = [it.get("caption","") or "" for it in items]
        if any(captions):
            caps_emb = sbert.encode(captions, convert_to_tensor=True)
            sims2 = sbert_sim(seg_text, caps_emb)
            best2 = int(np.argmax(sims2))
            if (max_reuse is None) or (used_counts.get(best2,0) < max_reuse):
                if sims2[best2] > sbert_thresh:
                    rp = used_counts.get(best2, 0) * reuse_penalty
                    lu = last_used_time.get(best2)
                    ap = 0.0
                    if lu is not None:
                        delta = seg_start_time - float(lu)
                        if delta <= 0:
                            ap = avoid_window_penalty
                        elif delta < avoid_window:
                            frac = max(0.0, min(1.0, (1.0 - (delta / avoid_window))))
                            ap = frac * avoid_window_penalty
                    fb = 0.0
                    fe = items[best2].get("face_emb")
                    if face_ref_emb is not None and fe:
                        try:
                            fe_arr = np.array(fe, dtype=np.float32)
                            d = np.linalg.norm(fe_arr - face_ref_emb)
                            if d < 0.65: fb = face_boost
                        except Exception:
                            pass
                    return best2, float(sims2[best2]), "sbert_caption", rp, ap, fb
    reuse_penalty_applied = float(reuse_penalties[chosen]) if chosen is not None else 0.0
    avoid_penalty_applied = float(avoid_penalties[chosen]) if chosen is not None else 0.0
    face_boost_applied = float(face_boosts[chosen]) if chosen is not None else 0.0
    return chosen, chosen_score, method, reuse_penalty_applied, avoid_penalty_applied, face_boost_applied

# ---------- Robust transcription + alignment helper ----------
def transcribe_and_align(
    audio_path: Path,
    device: str,
    model_name: str = "small",
    compute_type: Optional[str] = None,
    try_webrtcvad: bool = True,
    verbose: bool = True
) -> Tuple[Dict[str,Any], list, Any, Dict[str,Any]]:
    """
    Robust transcription + alignment using whisperx.
    Returns (res_wh, words, align_model, metadata).
    """
    if compute_type is None:
        compute_type = "float16" if device == "cuda" else "float32"
    if verbose:
        print(f"[transcribe_and_align] loading whisperx model {model_name} on {device} (compute_type={compute_type})")
    try:
        whisper_model = whisperx.load_model(model_name, device=device, compute_type=compute_type)
    except Exception as e:
        if "float16" in str(e).lower() and compute_type == "float16":
            if verbose: print("[transcribe_and_align] float16 not supported — retrying with float32")
            compute_type = "float32"
            whisper_model = whisperx.load_model(model_name, device=device, compute_type=compute_type)
        else:
            raise

    res_wh = None
    # Attempt 1: default (pyannote if available)
    try:
        if verbose: print("[transcribe_and_align] Attempting default transcription (may use pyannote VAD)...")
        res_wh = whisper_model.transcribe(str(audio_path))
        segs = res_wh.get("segments") if isinstance(res_wh, dict) else None
        if not segs:
            raise RuntimeError("default transcribe returned no segments")
    except Exception as e_default:
        if verbose: print(f"[transcribe_and_align] Default transcribe failed or returned no segments: {repr(e_default)}")
        res_wh = None

    # Attempt 2: webrtcvad (if requested)
    if res_wh is None and try_webrtcvad:
        try:
            if verbose: print("[transcribe_and_align] Retrying with webrtcvad VAD (install webrtcvad via pip if missing)...")
            res_wh = whisper_model.transcribe(str(audio_path), vad_engine="webrtcvad")
            segs = res_wh.get("segments") if isinstance(res_wh, dict) else None
            if not segs:
                raise RuntimeError("webrtcvad returned no segments")
        except TypeError as e_kw:
            try:
                if verbose: print("[transcribe_and_align] vad_engine kwarg unsupported; trying transcribe(..., vad=True)")
                res_wh = whisper_model.transcribe(str(audio_path), vad=True)
                segs = res_wh.get("segments") if isinstance(res_wh, dict) else None
                if not segs:
                    raise RuntimeError("vad=True returned no segments")
            except Exception as e_vad_true:
                if verbose: print("[transcribe_and_align] webrtcvad/vad=True attempt failed:", repr(e_vad_true))
                res_wh = None
        except Exception as e_webrtc:
            if verbose: print("[transcribe_and_align] webrtcvad attempt failed:", repr(e_webrtc))
            res_wh = None

    # Attempt 3: final fallback — no VAD (transcribe whole file)
    if res_wh is None:
        try:
            if verbose: print("[transcribe_and_align] Final fallback: transcribe with VAD disabled (vad=False). This transcribes whole file.")
            res_wh = whisper_model.transcribe(str(audio_path), vad=False)
            if not res_wh:
                raise RuntimeError("transcribe(vad=False) returned nothing")
        except TypeError:
            try:
                if verbose: print("[transcribe_and_align] transcribe(..., vad=False) unsupported; trying plain transcribe again")
                res_wh = whisper_model.transcribe(str(audio_path))
            except Exception as e_plain:
                raise RuntimeError("All transcription attempts failed") from e_plain
        except Exception as e_final:
            raise RuntimeError("All transcription attempts failed") from e_final

    if not isinstance(res_wh, dict):
        raise RuntimeError("Unexpected result from whisperx.transcribe: expected dict")

    if verbose:
        num_segs = len(res_wh.get("segments", []))
        print(f"[transcribe_and_align] Transcription done. segments={num_segs}, detected language={res_wh.get('language')}")

    # Load align model and perform alignment
    try:
        align_model, metadata = whisperx.load_align_model(language_code=res_wh.get("language"), device=device)
    except Exception as e_am:
        if verbose: print("[transcribe_and_align] load_align_model failed with:", repr(e_am))
        align_model, metadata = whisperx.load_align_model(language_code=res_wh.get("language", "en"), device=device)

    try:
        words = whisperx.align(res_wh.get("segments", []), align_model, metadata, str(audio_path), device)
    except Exception as e_align:
        if verbose: print("[transcribe_and_align] Alignment failed:", repr(e_align))
        words = []

    if verbose:
        print(f"[transcribe_and_align] Alignment produced {len(words)} word-level entries (may be 0 if alignment failed).")

    return res_wh, words, align_model, metadata

# ---------- Write final MP4 (Windows friendly) ----------
def write_windows_friendly_mp4(video_clip, audio_clip: AudioFileClip, out_path: Path, fps: int, subtitles: Optional[List[Tuple[float,float,str]]] = None, burn_captions: bool = False, res: Tuple[int,int] = (1280,720), karaoke_groups: Optional[List[Dict[str,Any]]] = None, lower_third: Optional[dict] = None, logo_cfg: Optional[dict] = None):
    base = video_clip.set_audio(audio_clip.set_fps(44100))
    overlays = [base]
    W,H = res
    if burn_captions and subtitles:
        overlays.append(subtitle_bg_clip(duration=base.duration, res=res, bottom_margin=40, height=110, width_ratio=0.96, opacity=0.35))
        for (s,e,t) in subtitles:
            clip = pillow_text_clip(t, fontsize=38, color="white", stroke_color="black", stroke_width=2, max_width=int(W*0.9), align="center", padding=10).set_start(s).set_end(e).set_position(("center", H-100))
            overlays.append(clip)
    if lower_third and lower_third.get("text"):
        lt = make_lower_third(total_duration=base.duration, res=res, text=lower_third["text"], subtext=lower_third.get("subtext"), start=float(lower_third.get("start",0.5)), show_dur=float(lower_third.get("duration",6.0)), theme=str(lower_third.get("theme","dark")))
        if lt is not None: overlays.append(lt)
    if logo_cfg and logo_cfg.get("path"):
        logo = make_logo_bug(logo_path=Path(logo_cfg["path"]), total_duration=base.duration, res=res, position=str(logo_cfg.get("position","top-right")), width=int(logo_cfg.get("width",120)), opacity=float(logo_cfg.get("opacity",0.85)), start=float(logo_cfg.get("start",0.5)), show_dur=(None if logo_cfg.get("duration") in (None,"") else float(logo_cfg.get("duration"))), margin=int(logo_cfg.get("margin",16)), fade=float(logo_cfg.get("fade",0.5)))
        if logo is not None: overlays.append(logo)
    final = CompositeVideoClip(overlays, size=res)
    narr = audio_clip.set_fps(44100)
    final = final.set_audio(narr)
    final.write_videofile(str(out_path), codec="libx264", audio=True, audio_codec="aac", audio_fps=44100, fps=fps, preset="medium", threads=4, temp_audiofile=str(out_path.with_suffix(".m4a")), remove_temp=True, ffmpeg_params=["-movflags","+faststart"])

# small helpers reused from earlier
def subtitle_bg_clip(duration: float, res: tuple[int,int], bottom_margin: int = 40, height: int = 110, width_ratio: float = 0.96, opacity: float = 0.35):
    W,H=res
    bg_w=int(W*width_ratio); bg_h=height
    y = H - bg_h - bottom_margin
    x = int((W - bg_w)/2)
    return ColorClip(size=(bg_w,bg_h), color=(0,0,0)).set_opacity(opacity).set_start(0).set_end(duration).set_position((x,y))

def make_lower_third(total_duration: float, res: tuple[int,int], text: str, subtext: str | None = None, start: float = 0.5, show_dur: float = 6.0, theme: str = "dark"):
    W,H=res
    end = min(total_duration, start+show_dur)
    if end <= start: return None
    if theme == "light":
        box_color=(255,255,255); box_opacity=0.70; title_color="black"; sub_color="black"; accent=(60,120,255)
    elif theme == "brand":
        box_color=(58,0,95); box_opacity=0.65; title_color="white"; sub_color="white"; accent=(0,210,190)
    else:
        box_color=(0,0,0); box_opacity=0.60; title_color="white"; sub_color="white"; accent=(255,210,0)
    box_w=int(W*0.58); box_h=120; margin_bottom=110; y=H-box_h-margin_bottom; x_on=int(W*0.04); x_off=-box_w-40; slide=0.6
    box = ColorClip(size=(box_w,box_h), color=box_color).set_opacity(box_opacity).set_start(start).set_end(end)
    def pos_box(t):
        local = t - start
        def smooth(z): return 3*z*z - 2*z*z*z
        if local <= 0: return (x_off, y)
        if local <= slide:
            k = smooth(max(0.0, min(1.0, local/slide))); x = int(x_off + (x_on - x_off)*k)
        elif local >= (end - start - slide):
            k = smooth(max(0.0, min(1.0, (local - (end - start - slide))/slide))); x = int(x_on + (x_off - x_on)*k)
        else:
            x = x_on
        return (x,y)
    box = box.set_position(pos_box)
    accent_w = 8
    accent = ColorClip(size=(accent_w, box_h), color=accent).set_opacity(1.0).set_start(start).set_end(end).set_position(lambda t: (pos_box(t)[0], pos_box(t)[1]))
    inner_pad = 18
    text_max_w = box_w - accent_w - inner_pad*2
    title_clip = None
    if text:
        title_clip = pillow_text_clip(text, fontsize=42, color=title_color, stroke_color="black", stroke_width=2, max_width=text_max_w, align="left", padding=6).set_start(start).set_end(end).set_position(lambda t:(pos_box(t)[0]+accent_w+inner_pad, pos_box(t)[1]+16))
    sub_clip = None
    if subtext:
        sub_clip = pillow_text_clip(subtext, fontsize=30, color=sub_color, stroke_color="black", stroke_width=2, max_width=text_max_w, align="left", padding=6).set_start(start+0.05).set_end(end).set_position(lambda t:(pos_box(t)[0]+accent_w+inner_pad, pos_box(t)[1]+68))
    elems=[e for e in [box,accent,title_clip,sub_clip] if e is not None]
    return CompositeVideoClip(elems, size=res).set_start(start).set_end(end)

def make_logo_bug(logo_path: Path, total_duration: float, res: tuple[int,int], position: str = "top-right", width: int = 120, opacity: float = 0.85, start: float = 0.5, show_dur: float = None, margin: int = 16, fade: float = 0.5):
    if not logo_path.exists():
        print(f"⚠️ Logo path not found: {logo_path}")
        return None
    end = total_duration if show_dur is None else min(total_duration, start + max(0.1, show_dur))
    if end <= start: return None
    logo = ImageClip(str(logo_path)).resize(width=width).set_start(start).set_end(end).set_opacity(opacity)
    if fade and fade > 0:
        logo = logo.fx(vfx.fadein, fade).fx(vfx.fadeout, fade)
    W,H=res
    def pos():
        if position == "top-left": return (margin, margin)
        if position == "top-right": return (W - logo.w - margin, margin)
        if position == "bottom-left": return (margin, H - logo.h - margin)
        if position == "bottom-right": return (W - logo.w - margin, H - logo.h - margin)
        return (W - logo.w - margin, margin)
    return logo.set_position(pos())

# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser(description="Build slideshow synced to narration (WhisperX + CLIP selection + soft avoid-window + CSV log).")
    parser.add_argument("--images_root", default="assets/images")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--output", default="output.mp4")
    parser.add_argument("--crossfade", type=float, default=0.5)
    parser.add_argument("--zoom", type=float, default=1.05)
    parser.add_argument("--res", default="1280x720")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--script", default=None, help="Plain-text script for auto-captions (SRT created).")
    parser.add_argument("--burn_captions", action="store_true", help="Burn captions into the video (if captions enabled).")
    parser.add_argument("--use_silence", action="store_true", help="Use ffmpeg to detect silences for caption binning.")
    parser.add_argument("--words_json", default=None, help="Word-level timings JSON for karaoke highlight (optional).")
    parser.add_argument("--lower_third_text", default=None)
    parser.add_argument("--logo_path", default=None)
    parser.add_argument("--face_ref", default=None, help="Optional image path to boost person matches (requires face_recognition).")
    parser.add_argument("--max-reuse", type=int, default=4, help="Max times an image may be reused (-1 to disable).")
    parser.add_argument("--reuse-penalty", type=float, default=0.06)
    parser.add_argument("--avoid-repeat-window", type=float, default=8.0, help="Soft window (seconds) for recent-use penalty.")
    parser.add_argument("--avoid-window-penalty", type=float, default=0.28)
    parser.add_argument("--min-img-duration", type=float, default=1.0, help="Minimum image duration (seconds). Short segments will be merged.")
    parser.add_argument("--clip_thresh", type=float, default=0.14, help="CLIP confidence threshold for fallback.")
    parser.add_argument("--sbert_thresh", type=float, default=0.34, help="SBERT threshold for caption fallback.")
    parser.add_argument("--no-captions", action="store_true", help="Disable captions entirely (default: captions enabled).")
    # HQ edge image options
    parser.add_argument("--no-edge-image", action="store_true", help="Disable adding HQ start/end images (default: included).")
    parser.add_argument("--start-image", default=None, help="Optional explicit image path to use at the start of the video.")
    parser.add_argument("--end-image", default=None, help="Optional explicit image path to use at the end of the video.")
    parser.add_argument("--start-duration", type=float, default=3.0, help="Seconds to show the start HQ image (default 3s).")
    parser.add_argument("--end-duration", type=float, default=6.0, help="Seconds to show the end HQ image (default 6s).")
    args = parser.parse_args()

    images_root = Path(args.images_root).resolve()
    audio_path = Path(args.audio).resolve()
    output_path = Path(args.output).resolve()
    res = parse_res(args.res)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio not found: {audio_path}")
    safe_audio = ensure_wav(audio_path)
    audio = AudioFileClip(str(safe_audio))
    audio_dur = float(audio.duration)

    print("Running WhisperX transcription + alignment...")
    device_local = "cuda" if torch.cuda.is_available() else "cpu"
    res_wh, words, align_model, metadata = transcribe_and_align(safe_audio, device_local, model_name="small", compute_type=("float32" if device_local=="cpu" else None), try_webrtcvad=True, verbose=True)

    print(f"WhisperX produced {sum(len(seg.get('words', [])) for seg in res_wh.get('segments', []))} word tokens approx; aligned words: {len(words)}")
    flattened_words = []
    if words and isinstance(words, list) and isinstance(words[0], dict) and "word" in words[0]:
        flattened_words = words
    else:
        flattened_words = []
        for seg in res_wh.get("segments", []):
            for w in seg.get("words", []):
                if "word" in w:
                    flattened_words.append({"word": w["word"], "start": w.get("start", seg.get("start",0.0)), "end": w.get("end", seg.get("end",0.0))})

    silences = None
    if args.use_silence:
        print("Detecting silences via ffmpeg...")
        silences = detect_silences(safe_audio)
        if silences:
            print(f"Detected {len(silences)} silence windows.")

    script_text = ""
    if args.script:
        sp = Path(args.script)
        if sp.exists():
            script_text = sp.read_text(encoding="utf-8", errors="ignore")
        else:
            print("Warning: script path not found; proceeding without script text.")
    else:
        script_text = " ".join([w.get("word","") for w in flattened_words])

    caption_segments = build_caption_segments(script_text, total_duration=audio_dur, min_dur=args.min_img_duration, silences=silences)
    print(f"Generated {len(caption_segments)} caption segments (min-img-duration={args.min_img_duration}).")

    print("Loading image index and CLIP embeddings...")
    if not IMAGE_INDEX_JSONL.exists() or not CLIP_EMB_NPY.exists():
        raise FileNotFoundError("Missing precomputed image index or embeddings. Run precompute_index.py or use fetcher that writes image_index.jsonl and clip_embeddings.npy.")
    items, clip_embs = load_image_index(IMAGE_INDEX_JSONL, CLIP_EMB_NPY)
    clip_embs = clip_embs.astype(np.float32)
    print("Loaded", len(items), "images in index.")

    face_ref_emb = None
    if args.face_ref:
        if HAVE_FACE:
            try:
                im = face_recognition.load_image_file(args.face_ref)
                encs = face_recognition.face_encodings(im)
                if encs:
                    face_ref_emb = np.array(encs[0], dtype=np.float32)
                    print("Loaded face reference embedding.")
                else:
                    print("Warning: no face found in face_ref; continuing without face boost.")
            except Exception as e:
                print("face_ref load error:", e)
        else:
            print("face_recognition not installed; skipping face boost.")

    used_counts = {}
    last_used_time = {}
    ordered_images = []
    last_idx = None
    max_reuse_val = None if args.max_reuse < 0 else args.max_reuse
    SELECTION_LOG.parent.mkdir(parents=True, exist_ok=True)

    with open(SELECTION_LOG, "w", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "segment_idx", "start", "end", "text",
            "chosen_idx", "chosen_path", "score", "method",
            "used_count_before", "used_count_after",
            "reuse_penalty_applied", "avoid_window_penalty_applied", "face_boost_applied",
            "timestamp_selected"
        ])
        for si, (s,e,txt) in enumerate(tqdm(caption_segments, desc="[select]")):
            seg_start = float(s)
            idx, score, method, reuse_penalty_applied, avoid_penalty_applied, face_boost_applied = choose_best_with_soft_window(
                txt,
                items,
                clip_embs,
                used_counts,
                reuse_penalty=float(args.reuse_penalty),
                max_reuse=max_reuse_val,
                last_idx=last_idx,
                seg_start_time=seg_start,
                avoid_window=float(args.avoid_repeat_window),
                last_used_time=last_used_time,
                avoid_window_penalty=float(args.avoid_window_penalty),
                sbert_thresh=float(args.sbert_thresh),
                clip_thresh=float(args.clip_thresh),
                face_ref_emb=face_ref_emb,
                face_boost=1.8,
            )
            chosen = items[idx]
            used_before = used_counts.get(idx, 0)
            used_after = used_before + 1
            writer.writerow([
                si,
                float(s),
                float(e),
                txt.strip(),
                int(idx),
                chosen.get("path",""),
                float(score),
                method,
                int(used_before),
                int(used_after),
                float(reuse_penalty_applied),
                float(avoid_penalty_applied),
                float(face_boost_applied),
                seg_start
            ])
            csvfile.flush()
            used_counts[idx] = used_after
            last_used_time[idx] = seg_start
            last_idx = idx
            ordered_images.append({
                "path": chosen["path"],
                "start": s,
                "end": e,
                "text": txt,
                "score": float(score),
                "method": method
            })

    print("Selection complete. Unique images used:", len(set([o['path'] for o in ordered_images])))
    print("Selection log:", SELECTION_LOG)

    # ---------- START: build clips with absolute start times (with HQ start/end) ----------
    W,H = res
    clips = []
    crossfade = float(args.crossfade)

    # Helper: pick high-quality image path (largest by pixel area)
    def pick_hq_image_path(explicit_path: Optional[str], fallback_ordered_images: list, items_list: list):
        if explicit_path:
            p = Path(explicit_path)
            if p.exists(): return p
        for candidate in fallback_ordered_images:
            pc = Path(candidate)
            if pc.exists():
                try:
                    with PILImage.open(pc) as im:
                        _ = im.size
                    return pc
                except Exception:
                    continue
        best = None
        best_area = 0
        for it in items_list:
            pth = Path(it.get("path",""))
            if not pth.exists(): continue
            try:
                with PILImage.open(pth) as im:
                    w,h = im.size
                area = int(w) * int(h)
                if area > best_area:
                    best_area = area
                    best = pth
            except Exception:
                continue
        if best is None and fallback_ordered_images:
            return Path(fallback_ordered_images[0])
        return best

    ordered_paths = [o["path"] for o in ordered_images]

    start_img_path = pick_hq_image_path(args.start_image, ordered_paths, items)
    end_img_path = pick_hq_image_path(args.end_image, ordered_paths, items)
    start_dur = max(0.1, float(getattr(args, "start_duration", 3.0)))
    end_dur = max(0.1, float(getattr(args, "end_duration", 6.0)))

    # If using start/end HQ images, adjust ordered_images times so they DO NOT overlap with edges.
    # Shift any segment starting before start_dur -> start at start_dur (trim its duration)
    # Trim any segment ending after audio_dur - end_dur -> set end to that boundary
    if not args.no_edge_image:
        # shift/truncate segments to respect the edge images
        new_ordered = []
        for o in ordered_images:
            s = float(o["start"]); e = float(o["end"])
            # trim start region
            if start_img_path is not None:
                if e <= start_dur:
                    # whole segment would be covered by start image -> skip it
                    continue
                if s < start_dur:
                    s = start_dur
            # trim end region
            if end_img_path is not None:
                end_limit = max(0.0, audio_dur - end_dur)
                if s >= audio_dur - end_dur:
                    # fully inside end image region -> skip
                    continue
                if e > audio_dur - end_dur:
                    e = audio_dur - end_dur
                    if e <= s:
                        continue
            # ensure minimum duration
            if e - s < 0.05:
                continue
            new_ordered.append({
                "path": o["path"],
                "start": s,
                "end": e,
                "text": o["text"],
                "score": o["score"],
                "method": o["method"]
            })
        ordered_images = new_ordered

    # Add start HQ clip at t=0 (if requested)
    if not args.no_edge_image and start_img_path is not None and Path(start_img_path).exists():
        sc = ImageClip(str(start_img_path)).set_start(0.0).set_duration(start_dur)
        scale = max(W/sc.w, H/sc.h)
        sc = sc.resize(scale)
        x1 = int(sc.w/2 - W/2); y1 = int(sc.h/2 - H/2)
        sc = sc.crop(x1=x1, y1=y1, x2=x1+W, y2=y1+H).set_position(("center","center"))
        fade_len = min(0.6, start_dur/2.0)
        if fade_len > 0:
            sc = sc.fx(vfx.fadein, fade_len)
        clips.append(sc)

    # Now create ordered segment clips at their absolute timestamps (these were adjusted above)
    for o in ordered_images:
        p = Path(o["path"])
        s = float(o["start"])
        e = float(o["end"])
        dur = max(0.05, e - s)
        clip = ImageClip(str(p)).set_start(s).set_duration(dur)
        scale = max(W/clip.w, H/clip.h)
        clip = clip.resize(scale)
        x1 = int(clip.w/2 - W/2); y1 = int(clip.h/2 - H/2)
        clip = clip.crop(x1=x1, y1=y1, x2=x1+W, y2=y1+H)
        if args.zoom and float(args.zoom) > 1.0:
            per = dur
            zoom_val = float(args.zoom)
            clip = clip.resize(lambda t, dur=per: 1.0 + (zoom_val-1.0)*(t/max(0.001,dur)))
        if crossfade and crossfade > 0:
            fade_len = min(crossfade, dur/2.0)
            if fade_len > 0:
                clip = clip.fx(vfx.fadein, fade_len).fx(vfx.fadeout, fade_len)
        clips.append(clip.set_position(("center","center")))

    # Add end HQ clip finishing exactly at audio end (if requested)
    if not args.no_edge_image and end_img_path is not None and Path(end_img_path).exists():
        end_start = max(0.0, audio_dur - end_dur)
        ec = ImageClip(str(end_img_path)).set_start(end_start).set_duration(end_dur)
        scale = max(W/ec.w, H/ec.h)
        ec = ec.resize(scale)
        x1 = int(ec.w/2 - W/2); y1 = int(ec.h/2 - H/2)
        ec = ec.crop(x1=x1, y1=y1, x2=x1+W, y2=y1+H).set_position(("center","center"))
        fade_len = min(0.8, end_dur/2.0)
        if fade_len > 0:
            ec = ec.fx(vfx.fadeout, fade_len)
        clips.append(ec)

    # Ensure cover full audio duration
    if clips:
        last_end = max((c.start + c.duration) for c in clips)
        if last_end < audio_dur - 1e-3:
            # Use end image if exists, else last ordered image
            filler_path = None
            if end_img_path and Path(end_img_path).exists():
                filler_path = end_img_path
            elif ordered_images:
                filler_path = Path(ordered_images[-1]["path"])
            if filler_path:
                filler_dur = audio_dur - last_end
                filler_clip = ImageClip(str(filler_path)).set_start(last_end).set_duration(filler_dur)
                scale = max(W/filler_clip.w, H/filler_clip.h)
                filler_clip = filler_clip.resize(scale)
                x1 = int(filler_clip.w/2 - W/2); y1 = int(filler_clip.h/2 - H/2)
                filler_clip = filler_clip.crop(x1=x1, y1=y1, x2=x1+W, y2=y1+H)
                clips.append(filler_clip.set_position(("center","center")))
    else:
        clips.append(ColorClip(size=(W,H), color=(0,0,0)).set_start(0).set_duration(audio_dur))

    # Compose as absolute-timed CompositeVideoClip (ensures captions match timestamps exactly)
    final_video = CompositeVideoClip(clips, size=(W,H)).set_duration(audio_dur)
    # ---------- END clip-building ----------

    # Build subtitle_segments using the possibly adjusted ordered_images (ensures captions remain in sync)
    captions_enabled = not args.no_captions
    if captions_enabled:
        # Use the adjusted caption_segments if edge images changed timings — we need subtitle timings
        # We will generate subtitles from the final ordered_images times (which are aligned to images).
        # But user originally expects captions matching script timing; better: take caption_segments and
        # clamp them to visible ranges (avoid start/end regions).
        final_subs = []
        for s,e,t in caption_segments:
            s = float(s); e = float(e)
            # clamp to [0, audio_dur]
            s = max(0.0, min(audio_dur, s))
            e = max(0.0, min(audio_dur, e))
            # if start image present, avoid showing subtitles during start image (shift forward slightly)
            if not args.no_edge_image and start_img_path is not None and s < start_dur:
                if e <= start_dur:
                    continue
                s = start_dur
            # if end image present, avoid showing subtitles during end image
            if not args.no_edge_image and end_img_path is not None and e > audio_dur - end_dur:
                if s >= audio_dur - end_dur:
                    continue
                e = audio_dur - end_dur
            if e - s < 0.05:
                continue
            final_subs.append((s, e, t.strip()))
        subtitle_segments = final_subs
    else:
        subtitle_segments = None

    burn_captions_flag = bool(args.burn_captions) and captions_enabled

    write_windows_friendly_mp4(
        video_clip=final_video,
        audio_clip=audio,
        out_path=output_path,
        fps=int(args.fps),
        subtitles=subtitle_segments,
        burn_captions=burn_captions_flag,
        res=res,
        karaoke_groups=None,
        lower_third=None,
        logo_cfg={"path": args.logo_path, "position":"top-right", "width":120, "opacity":0.85, "start":0.5, "duration":None, "margin":16, "fade":0.5} if args.logo_path else None
    )

    try:
        audio.close()
    except Exception:
        pass
    print("✅ Done. Wrote:", output_path)

if __name__ == "__main__":
    main()
