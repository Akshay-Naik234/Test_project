#!/usr/bin/env python3
""" step1_fetch_from_script_with_serpapi.py - ENHANCED VERSION Major improvements: 1. Scene-aware extraction: Parses script into temporal scenes with context 2. Multi-level semantic matching: CLIP + SBERT + BLIP verification 3. Contextual relevance scoring: Images must match scene narrative 4. Temporal coherence: Ensures image sequence follows script flow 5. Negative filtering: Rejects off-topic/anachronistic content 6. Quality gates: Multiple verification layers before acceptance """
from __future__ import annotations
import os, re, argparse, time, random, json, csv, warnings, logging
from pathlib import Path
from io import BytesIO
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass, asdict
from collections import defaultdict
import requests
from tqdm import tqdm
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
import numpy as np
import imagehash
import urllib.parse

# reduce HF noise on Windows
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
warnings.filterwarnings("ignore", message="Xet Storage is enabled for this repo")
warnings.filterwarnings("ignore", message="QuickGELU mismatch")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("open_clip").setLevel(logging.ERROR)

# ML libs
import torch
import open_clip
from sentence_transformers import SentenceTransformer, util

# BLIP (required for verification)
try:
    from transformers import BlipForConditionalGeneration, BlipProcessor
    BLIP_IMPORT_OK = True
except Exception:
    BlipForConditionalGeneration = None
    BlipProcessor = None
    BLIP_IMPORT_OK = False

# face_recognition
try:
    import face_recognition
    HAVE_FACE = True
except Exception:
    HAVE_FACE = False
BIOGRAPHY_MODE = True
# dotenv optional
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Wikimedia endpoints
COMMONS_API = "https://commons.wikimedia.org/w/api.php"
WIKI_API = "https://en.wikipedia.org/w/api.php"

# outputs
OUT_ROOT = Path("assets")
IMG_DIR = OUT_ROOT / "images"
ATTR_CSV = OUT_ROOT / "attribution.csv"
CREDITS = OUT_ROOT / "CREDITS.txt"
INDEX_JSONL = OUT_ROOT / "image_index.jsonl"
CLIP_NPY = OUT_ROOT / "clip_embeddings.npy"
SCENE_MAP = OUT_ROOT / "scene_mapping.json"

# sizes
MIN_W, MIN_H = 720, 405
TARGET_W, TARGET_H = 1280, 720
TARGET_AR = TARGET_W / TARGET_H
REQUEST_DELAY_SEC = (0.25, 0.9)
HEADERS = {"User-Agent": "ScriptImageFetcher/2.0", "Accept": "application/json"}
ASPECT_RATIO_TOLERANCE = 0.06

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading CLIP model...")
clip_model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
clip_model.to(device).eval()

# BLIP (REQUIRED for verification)
BLIP_AVAILABLE = False
if BLIP_IMPORT_OK:
    try:
        blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", use_fast=True)
        blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device).eval()
        BLIP_AVAILABLE = True
        print("✓ BLIP loaded (caption verification enabled)")
    except Exception as e:
        BLIP_AVAILABLE = False
        blip_processor = None
        blip_model = None
        print(f"⚠️ BLIP failed: {e}")
else:
    blip_processor = None
    blip_model = None
    print("⚠️ BLIP not installed - verification will be limited")

# SBERT
sbert = SentenceTransformer("all-MiniLM-L6-v2", device=device)

# ============================================================================ 
# SCENE EXTRACTION - Parse script into temporal narrative segments
# ============================================================================ 
STOPWORDS = set("""a about above after again against all am an and any are as at be because been before being below between both but by could did do does doing down during each few for from further had has have having he her here hers herself him himself his how i if in into is it its itself just me more most my myself nor not of off on once only or other our ours ourselves out over own same she should so some such than that the their theirs them themselves then there these they this those through to too under until up very was we were what when where which while who whom why with you your yours yourself yourselves""".split())

@dataclass
class SceneSegment:
    """Represents a narrative segment from the script"""
    index: int
    text: str
    keywords: List[str]
    time_period: Optional[str]
    location: Optional[str]
    people: List[str]
    concepts: List[str]
    negative_terms: List[str]
    embedding: Optional[np.ndarray] = None

def extract_time_period(text: str) -> Optional[str]:
    """Extract time period mentions (years, decades, eras)"""
    year_match = re.search(r'\b(1[7-9]\d{2}|20[0-2]\d)\b', text)
    if year_match:
        return year_match.group(1)
    decade_match = re.search(r'\b(eighteen|nineteen|twenty)[\s-]?(twenties|thirties|forties|fifties|sixties|seventies|eighties|nineties)\b', text, re.I)
    if decade_match:
        return decade_match.group(0)
    era_patterns = [
        r'\b(early|late|mid)[\s-]?(life|career|years|childhood|adulthood)\b',
        r'\b(Victorian|Edwardian|Renaissance|Medieval|Ancient|Modern)\s+era\b',
        r'\b(youth|childhood|teenage|young adult|middle age|elderly|old age)\b'
    ]
    for pattern in era_patterns:
        match = re.search(pattern, text, re.I)
        if match:
            return match.group(0)
    return None

def extract_locations(text: str) -> List[str]:
    """Extract location mentions"""
    patterns = [
        r'\bin\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',
        r'\bat\s+(?:the\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',
        r'\b(laboratory|office|university|institute|home|studio|workshop)\b',
    ]
    locations = []
    for pattern in patterns:
        matches = re.findall(pattern, text, re.I)
        locations.extend(matches)
    return list(set(locations))[:3]

def extract_people_mentions(text: str, main_person: str) -> List[str]:
    """Extract people mentioned (beyond main subject)"""
    names = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\b', text)
    names = [n for n in names if n.lower() != main_person.lower()]
    return list(set(names))[:3]

def extract_concepts(text: str) -> List[str]:
    """Extract key scientific/historical concepts"""
    patterns = [
        r'"([^"]+)"',
        r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',
        r'\b(theory|equation|law|principle|discovery|invention|experiment|paper|publication)\s+of\s+(\w+)',
    ]
    concepts = []
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            concepts.extend([m if isinstance(m, str) else ' '.join(m) for m in matches])
    return list(set(concepts))[:5]

def generate_negative_terms(scene: str, person: str) -> List[str]:
    """Generate terms that should NOT appear in images for this scene"""
    negatives = []
    if any(year in scene for year in ['1800', '1900', '19th', 'nineteenth']):
        negatives.extend(['smartphone', 'computer', 'laptop', 'modern car', 'airplane', 'television'])
    if 'he' in scene.lower() or 'his' in scene.lower():
        negatives.extend(['woman', 'female', 'girl', 'she'])
    elif 'she' in scene.lower() or 'her' in scene.lower():
        negatives.extend(['man', 'male', 'boy', 'he'])
    if 'laboratory' in scene.lower() or 'experiment' in scene.lower():
        negatives.extend(['outdoor', 'nature', 'landscape', 'beach', 'mountain'])
    if 'childhood' in scene.lower() or 'young' in scene.lower():
        negatives.extend(['elderly', 'old age', 'gray hair', 'wrinkled'])
    return negatives

def parse_script_into_scenes(script_text: str, person: str, min_words: int = 30) -> List[SceneSegment]:
    """Parse script into narrative scenes with rich context"""
    paragraphs = [p.strip() for p in script_text.split('\n\n') if p.strip()]
    scenes = []
    scene_idx = 0
    for para in paragraphs:
        if len(para.split()) < min_words:
            continue
        time_period = extract_time_period(para)
        locations = extract_locations(para)
        people = extract_people_mentions(para, person)
        concepts = extract_concepts(para)
        negatives = generate_negative_terms(para, person)
        words = re.findall(r'\b[a-z]{4,}\b', para.lower())
        word_freq = defaultdict(int)
        for w in words:
            if w not in STOPWORDS:
                word_freq[w] += 1
        keywords = sorted(word_freq.keys(), key=lambda x: word_freq[x], reverse=True)[:8]
        scene = SceneSegment(
            index=scene_idx,
            text=para,
            keywords=keywords,
            time_period=time_period,
            location=locations[0] if locations else None,
            people=people,
            concepts=concepts,
            negative_terms=negatives
        )
        try:
            scene.embedding = sbert.encode(para, convert_to_tensor=False)
        except:
            scene.embedding = None
        scenes.append(scene)
        scene_idx += 1
    return scenes

# ============================================================================ 
# ENHANCED TOPIC EXTRACTION
# ============================================================================ 
def extract_topics_scene_aware(scenes: List[SceneSegment], person: str, max_topics: int = 15) -> List[Dict]:
    """Extract search topics that are scene-aware and contextually rich"""
    topics = []
    base_queries = [
        f"{person} portrait photograph",
        f"{person} historical photograph",
        f"{person} archival image",
        f"{person} authentic photo",
    ]
    for q in base_queries:
        topics.append({
            'query': q,
            'scene_indices': list(range(len(scenes))),
            'context': 'general portrait',
            'weight': 1.0
        })
    for scene in scenes:
        if scene.time_period:
            topics.append({ 'query': f"{person} {scene.time_period}", 'scene_indices': [scene.index], 'context': f"time: {scene.time_period}", 'weight': 1.2 })
        if scene.location:
            topics.append({ 'query': f"{person} {scene.location}", 'scene_indices': [scene.index], 'context': f"location: {scene.location}", 'weight': 1.1 })
        for concept in scene.concepts[:2]:
            topics.append({ 'query': f"{person} {concept}", 'scene_indices': [scene.index], 'context': f"concept: {concept}", 'weight': 1.3 })
        if len(scene.keywords) >= 2:
            kw_query = f"{person} {' '.join(scene.keywords[:3])}"
            topics.append({ 'query': kw_query, 'scene_indices': [scene.index], 'context': f"keywords: {', '.join(scene.keywords[:3])}", 'weight': 1.0 })
    if len(scenes) > 3:
        scene_groups = []
        used = set()
        for i, scene in enumerate(scenes):
            if i in used:
                continue
            group = [i]
            for j, other in enumerate(scenes[i+1:], start=i+1):
                if j in used:
                    continue
                overlap = set(scene.keywords) & set(other.keywords)
                if len(overlap) >= 2:
                    group.append(j)
                    used.add(j)
            if len(group) >= 2:
                all_keywords = []
                for idx in group:
                    all_keywords.extend(scenes[idx].keywords)
                common = [k for k in all_keywords if all_keywords.count(k) >= 2][:3]
                if common:
                    topics.append({ 'query': f"{person} {' '.join(common)}", 'scene_indices': group, 'context': f"theme: {', '.join(common)}", 'weight': 1.1 })
                used.add(i)
    seen_queries = set()
    unique_topics = []
    for t in topics:
        q_lower = t['query'].lower()
        if q_lower not in seen_queries:
            seen_queries.add(q_lower)
            unique_topics.append(t)
    unique_topics.sort(key=lambda x: x['weight'], reverse=True)
    return unique_topics[:max_topics]

# ============================================================================ 
# UTILITIES
# ============================================================================ 
def _sleep(min_delay=REQUEST_DELAY_SEC[0], max_delay=REQUEST_DELAY_SEC[1]):
    time.sleep(random.uniform(min_delay, max_delay))

def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+","-", s.lower()).strip("-")

def _normalize_url(u: str) -> str:
    if not u: return ""
    u = u.split("?")[0].split("#")[0]
    return u

def _normalize_url_key(u: str) -> str:
    if not u: return ""
    u = _normalize_url(u)
    try:
        p = urllib.parse.urlparse(u)
        return (p.netloc + p.path).rstrip("/")
    except Exception:
        return u

def enhance_clarity(im: Image.Image) -> Image.Image:
    im = im.filter(ImageFilter.UnsharpMask(radius=1.0, percent=120, threshold=2))
    im = ImageEnhance.Sharpness(im).enhance(1.05)
    im = ImageEnhance.Contrast(im).enhance(1.03)
    return im

def _clamp(v, a, b): return max(a, min(b, v))

def _expand_box(box, img_w, img_h):
    """SAFE FACE EXPANSION (No head/body cropping)"""
    top, right, bottom, left = box
    w = max(1, right - left)
    h = max(1, bottom - top)
    cx = left + w / 2.0
    cy = top + h / 2.0
    top_pad = h * 2.1
    bottom_pad = h * 1.4
    side_pad = w * 1.45
    new_left = int(_clamp(cx - (w/2.0 + side_pad), 0, img_w - 1))
    new_right = int(_clamp(cx + (w/2.0 + side_pad), 0, img_w - 1))
    new_top = int(_clamp(top - top_pad, 0, img_h - 1))
    new_bottom = int(_clamp(bottom + bottom_pad, 0, img_h - 1))
    if new_right - new_left < 16:
        extra = (16 - (new_right - new_left)) // 2 + 1
        new_left = int(_clamp(new_left - extra, 0, img_w-1))
        new_right = int(_clamp(new_right + extra, 0, img_w))
    if new_bottom - new_top < 16:
        extra = (16 - (new_bottom - new_top)) // 2 + 1
        new_top = int(_clamp(new_top - extra, 0, img_h-1))
        new_bottom = int(_clamp(new_bottom + extra, 0, img_h))
    return (new_left, new_top, new_right, new_bottom)

def find_best_crop_no_face(im: Image.Image, target_w=TARGET_W, target_h=TARGET_H, downscale_max=480, stride=32):
    """Fast saliency/edge search to find a good 16:9 crop when no face is detected"""
    w0, h0 = im.size
    if h0 > downscale_max:
        scale = downscale_max / float(h0)
        small = im.resize((int(round(w0 * scale)), downscale_max), Image.LANCZOS)
    else:
        scale = 1.0
        small = im.copy()
    edge = small.convert("L").filter(ImageFilter.FIND_EDGES)
    arr = np.asarray(edge, dtype=np.float32)
    arr = arr / (arr.max() + 1e-8)
    sw, sh = arr.shape[1], arr.shape[0]
    targ_ar = TARGET_AR
    scales = [0.9, 0.75, 0.6, 0.45]
    best_score = -1.0
    best_box = (0, 0, w0, h0)
    for frac in scales:
        win_h = max(32, int(round(sh * frac)))
        win_w = int(round(win_h * targ_ar))
        if win_w > sw:
            win_w = sw
            win_h = int(round(win_w / targ_ar))
        if win_h > sh:
            continue
        s = max(8, int(stride * max(0.4, frac)))
        integral = arr.cumsum(axis=0).cumsum(axis=1)
        def rect_sum(x0,y0,x1,y1):
            x0 = max(0, x0); y0 = max(0,y0)
            x1 = min(sw-1, x1); y1 = min(sh-1, y1)
            A = integral[y1, x1]
            B = integral[y0-1, x1] if y0-1 >= 0 else 0.0
            C = integral[y1, x0-1] if x0-1 >= 0 else 0.0
            D = integral[y0-1, x0-1] if (y0-1>=0 and x0-1>=0) else 0.0
            return float(A - B - C + D)
        for y in range(0, sh - win_h + 1, s):
            for x in range(0, sw - win_w + 1, s):
                score = rect_sum(x, y, x + win_w - 1, y + win_h - 1)
                cx = x + win_w / 2.0
                cy = y + win_h / 2.0
                dx = (cx - sw/2.0) / (sw/2.0)
                dy = (cy - sh/2.0) / (sh/2.0)
                center_penalty = (dx*dx + dy*dy) * 0.06
                adj_score = score - center_penalty
                if adj_score > best_score:
                    best_score = adj_score
                    orig_x0 = int(round(x / scale))
                    orig_y0 = int(round(y / scale))
                    orig_x1 = int(round((x + win_w) / scale))
                    orig_y1 = int(round((y + win_h) / scale))
                    orig_x0 = _clamp(orig_x0, 0, w0-1)
                    orig_y0 = _clamp(orig_y0, 0, h0-1)
                    orig_x1 = _clamp(orig_x1, orig_x0+1, w0)
                    orig_y1 = _clamp(orig_y1, orig_y0+1, h0)
                    best_box = (orig_x0, orig_y0, orig_x1, orig_y1)
    return best_box

def save_processed(
    im: Image.Image,
    out_path: Path,
    allow_padding: bool = True,
    face_boxes: Optional[List[Tuple[int,int,int,int]]] = None,
    pad_mode: str = "crop",
    solid_color=(0,0,0)
):
    """
    Improved head/forehead-safe saver:
    - Uses landmarks (if face_recognition present) to include forehead/hairline.
    - Asymmetric padding (more top) and post-crop nudging to avoid head cuts.
    - Preserves "crop", "solid", "blur" behavior.
    """

    w, h = im.size
    target_ar = TARGET_AR

    # Helper: normalize incoming face box formats (face_recognition gives (top,right,bottom,left))
    def _normalize_box(box):
        # Accept formats: (top,right,bottom,left) or (l,t,r,b)
        if not box or len(box) < 4:
            return (0, 0, w, h)
        a, b, c, d = box
        # Heuristic: if first value (a) < second (b) and both small -> likely top,right,bottom,left
        # face_recognition uses (top, right, bottom, left)
        try:
            if a < c and d < b and (0 <= a < h) and (0 <= d < w):
                top, right, bottom, left = int(a), int(b), int(c), int(d)
                left = max(0, left); top = max(0, top)
                right = min(w, right); bottom = min(h, bottom)
                return (left, top, right, bottom)
        except Exception:
            pass
        # assume l,t,r,b
        try:
            left, top, right, bottom = int(a), int(b), int(c), int(d)
            left = max(0, left); top = max(0, top)
            right = min(w, right); bottom = min(h, bottom)
            return (left, top, right, bottom)
        except Exception:
            return (0, 0, w, h)

    # Try to compute landmarks for the chosen face (best-effort)
    def _compute_landmarks_for_box(box_ltrb):
        if not HAVE_FACE:
            return None
        try:
            arr = np.array(im.convert("RGB"))
            all_landmarks = face_recognition.face_landmarks(arr)
            if not all_landmarks:
                return None
            best_idx = None
            best_iou = -1.0
            for i, lm in enumerate(all_landmarks):
                xs = []
                ys = []
                for k, pts in lm.items():
                    for (x,y) in pts:
                        xs.append(x); ys.append(y)
                if not xs or not ys:
                    continue
                l = max(0, min(xs)); r = min(w, max(xs))
                t = max(0, min(ys)); b = min(h, max(ys))
                inter_w = max(0, min(r, box_ltrb[2]) - max(l, box_ltrb[0]))
                inter_h = max(0, min(b, box_ltrb[3]) - max(t, box_ltrb[1]))
                inter_area = inter_w * inter_h
                union_area = (r-l)*(b-t) + (box_ltrb[2]-box_ltrb[0])*(box_ltrb[3]-box_ltrb[1]) - inter_area
                iou = (inter_area / union_area) if union_area > 0 else 0.0
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i
            if best_idx is None:
                return None
            return all_landmarks[best_idx]
        except Exception:
            return None

    # Post-crop validation: ensure face top isn't too close to crop top.
    def _validate_and_fix_crop(crop_ltrb, face_ltrb):
        cl, ct, cr, cb = crop_ltrb
        fl, ft, fr, fb = face_ltrb
        crop_w = cr - cl
        crop_h = cb - ct
        if crop_h <= 0 or crop_w <= 0:
            return crop_ltrb
        rel_face_top = (ft - ct) / float(crop_h)
        rel_face_bottom = (fb - ct) / float(crop_h)
        desired_top_min = 0.28  # Increased from 0.16 for more head space
        desired_top_target = 0.34  # Increased from 0.22
        if rel_face_top < desired_top_min:
            desired_ft = ct + desired_top_target * crop_h
            shift_px = int(desired_ft - ft)
            new_ct = int(_clamp(ct + shift_px, 0, h - crop_h))
            new_cb = new_ct + crop_h
            if new_ct == ct and ct == 0:
                extra = int(max(1, crop_h * 0.12))
                new_cb = int(_clamp(cb + extra, 0, h))
                if new_cb - new_ct > crop_h:
                    new_ct = int(_clamp(new_cb - crop_h, 0, h - 1))
            return (cl, new_ct, cr, new_cb)
        if rel_face_bottom > 0.78:
            shift_px = int((rel_face_bottom - 0.72) * crop_h)
            new_ct = int(_clamp(ct - shift_px, 0, h - crop_h))
            new_cb = new_ct + crop_h
            return (cl, new_ct, cr, new_cb)
        return crop_ltrb

    # -----------------------
    # 1) CROP MODE (face-aware / head-safe)
    # -----------------------
    if pad_mode == "crop":
        if face_boxes:
            norm_boxes = []
            for b in face_boxes:
                l, t, r, btm = _normalize_box(b)
                if (r - l) >= 12 and (btm - t) >= 12:
                    norm_boxes.append((l, t, r, btm))
            if norm_boxes:
                areas = [ (box[2]-box[0])*(box[3]-box[1]) for box in norm_boxes ]
                idx = int(np.argmax(areas))
                face_l, face_t, face_r, face_b = norm_boxes[idx]
                face_w = face_r - face_l
                face_h = face_b - face_t

                landmarks = _compute_landmarks_for_box((face_l, face_t, face_r, face_b))
                if landmarks:
                    eyebrow_ys = []
                    chin_ys = []
                    eye_ys = []
                    for k in ("left_eyebrow","right_eyebrow","chin","left_eye","right_eye"):
                        pts = landmarks.get(k, [])
                        for (x,y) in pts:
                            if k in ("left_eyebrow","right_eyebrow"):
                                eyebrow_ys.append(y)
                            if k == "chin":
                                chin_ys.append(y)
                            if k in ("left_eye","right_eye"):
                                eye_ys.append(y)
                    eyebrow_top = int(min(eyebrow_ys)) if eyebrow_ys else face_t + int(face_h * 0.12)
                    chin_bottom = int(max(chin_ys)) if chin_ys else face_b
                    eyebrow_gap = max(1, face_t - eyebrow_top) if eyebrow_top < face_t else int(face_h * 0.08)
                    top_pad = int(max(face_h * 2.5, eyebrow_gap * 3.0))  # Increased from 1.8 for more head space
                    bottom_pad = int(face_h * 1.05)
                    side_pad = int(face_w * 1.25)
                    # Aggressive adjustment for cropped forehead
                    if eyebrow_top - face_t < int(face_h * 0.05):
                        top_pad += int(face_h * 0.5)  # Extra padding if forehead seems cropped
                else:
                    top_pad = int(face_h * 3.2)
                    bottom_pad = int(face_h * 1.4)
                    side_pad = int(face_w * 1.35)

                new_left = int(_clamp(face_l - side_pad, 0, w - 1))
                new_right = int(_clamp(face_r + side_pad, 0, w))
                new_top = int(_clamp(face_t - top_pad, 0, h - 1))
                new_bottom = int(_clamp(face_b + bottom_pad, 0, h))

                if new_top == 0 and face_t > 8:
                    extra = int(max(1, face_h * 0.9))
                    new_bottom = int(_clamp(new_bottom + extra, 0, h))

                new_left = int(_clamp(new_left, 0, w - 1))
                new_top = int(_clamp(new_top, 0, h - 1))
                new_right = int(_clamp(new_right, new_left + 1, w))
                new_bottom = int(_clamp(new_bottom, new_top + 1, h))

                cw = new_right - new_left
                ch = new_bottom - new_top
                if ch == 0:
                    ch = max(1, int(face_h * 1.5))
                box_ar = (cw / ch) if ch else target_ar

                if box_ar > target_ar:
                    needed_h = int(round(cw / target_ar))
                    cy = (new_top + new_bottom) // 2
                    crop_top = int(_clamp(cy - needed_h // 2, 0, h - 1))
                    crop_bottom = int(_clamp(crop_top + needed_h, 0, h - 1))
                    if crop_top > new_top:
                        crop_top = new_top
                        crop_bottom = int(_clamp(crop_top + needed_h, 0, h - 1))
                    if crop_bottom < new_bottom:
                        crop_bottom = new_bottom
                        crop_top = int(_clamp(crop_bottom - needed_h, 0, h - 1))
                    crop_left = new_left
                    crop_right = new_right
                else:
                    needed_w = int(round(ch * target_ar))
                    cx = (new_left + new_right) // 2
                    crop_left = int(_clamp(cx - needed_w // 2, 0, w - 1))
                    crop_right = int(_clamp(crop_left + needed_w, 0, w - 1))
                    if crop_left > new_left:
                        crop_left = new_left
                        crop_right = int(_clamp(crop_left + needed_w, 0, w - 1))
                    if crop_right < new_right:
                        crop_right = new_right
                        crop_left = int(_clamp(crop_right - needed_w, 0, w - 1))
                    crop_top = new_top
                    crop_bottom = new_bottom

                crop_left = int(_clamp(crop_left, 0, w - 1))
                crop_top = int(_clamp(crop_top, 0, h - 1))
                crop_right = int(_clamp(crop_right, crop_left + 1, w))
                crop_bottom = int(_clamp(crop_bottom, crop_top + 1, h))

                crop_left, crop_top, crop_right, crop_bottom = _validate_and_fix_crop(
                    (crop_left, crop_top, crop_right, crop_bottom),
                    (face_l, face_t, face_r, face_b)
                )

                crop = im.crop((crop_left, crop_top, crop_right, crop_bottom))
                out_img = crop.resize((TARGET_W, TARGET_H), Image.LANCZOS)

                final = enhance_clarity(out_img)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                if face_boxes:
                    for b in face_boxes:
                        try:
                            top, right, bottom, left = b
                            if top <= int(0.12 * h) or (h - bottom) <= int(0.12 * h):
                                return None
                        except Exception:
                            pass
                final.save(out_path, "JPEG", quality=92, optimize=True, progressive=True)
                return final

        try:
            bx0, by0, bx1, by1 = find_best_crop_no_face(im)
        except Exception:
            bx0, by0, bx1, by1 = 0, 0, w, h

        if bx0 <= 0 and by0 <= 0 and bx1 >= w and by1 >= h:
            ar = w / h if h else target_ar
            if ar > target_ar:
                new_w = int(round(h * target_ar))
                x0 = (w - new_w) // 2
                crop = im.crop((x0, 0, x0 + new_w, h))
            else:
                new_h = int(round(w / target_ar))
                y0 = (h - new_h) // 2
                crop = im.crop((0, y0, w, y0 + new_h))
            out_img = crop.resize((TARGET_W, TARGET_H), Image.LANCZOS)
        else:
            left, top, right, bottom = int(bx0), int(by0), int(bx1), int(by1)
            cw = right - left
            ch = bottom - top
            cur_ar = (cw / ch) if ch else target_ar
            if cur_ar > target_ar:
                needed_h = int(round(cw / target_ar))
                cy = (top + bottom) // 2
                top = int(_clamp(cy - needed_h // 2, 0, h - 1))
                bottom = int(_clamp(top + needed_h, 0, h - 1))
            else:
                needed_w = int(round(ch * target_ar))
                cx = (left + right) // 2
                left = int(_clamp(cx - needed_w // 2, 0, w - 1))
                right = int(_clamp(left + needed_w, 0, w - 1))
            left = int(_clamp(left, 0, w - 1))
            top = int(_clamp(top, 0, h - 1))
            right = int(_clamp(right, left + 1, w))
            bottom = int(_clamp(bottom, top + 1, h))
            crop = im.crop((left, top, right, bottom))
            out_img = crop.resize((TARGET_W, TARGET_H), Image.LANCZOS)

        final = enhance_clarity(out_img)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        final.save(out_path, "JPEG", quality=92, optimize=True, progressive=True)
        return final

    # -----------------------
    # 2) SOLID PAD MODE
    # -----------------------
    if pad_mode == "solid":
        scale = min(TARGET_W / w, TARGET_H / h)
        new_w = int(max(1, round(w * scale)))
        new_h = int(max(1, round(h * scale)))

        fg = im.resize((new_w, new_h), Image.LANCZOS)
        bg = Image.new("RGB", (TARGET_W, TARGET_H), solid_color)
        x0 = (TARGET_W - new_w) // 2
        y0 = (TARGET_H - new_h) // 2
        bg.paste(fg, (x0, y0))

        final = enhance_clarity(bg)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        final.save(out_path, "JPEG", quality=92, optimize=True, progressive=True)
        return final

    # -----------------------
    # 3) BLUR PAD MODE (fallback/default)
    # -----------------------
    scale = min(TARGET_W / w, TARGET_H / h)
    new_w = int(max(1, round(w * scale)))
    new_h = int(max(1, round(h * scale)))

    fg = im.resize((new_w, new_h), Image.LANCZOS)
    bg = im.resize((TARGET_W, TARGET_H), Image.LANCZOS).filter(ImageFilter.GaussianBlur(radius=22))
    bg = ImageEnhance.Brightness(bg).enhance(0.92)
    x0 = (TARGET_W - new_w) // 2
    y0 = (TARGET_H - new_h) // 2
    bg.paste(fg, (x0, y0))

    final = enhance_clarity(bg)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    final.save(out_path, "JPEG", quality=92, optimize=True, progressive=True)
    return final


# ============================================================================ 
# NETWORK / SEARCH HELPERS
# ============================================================================ 
def http_get_json(url, params=None, max_retries=3):
    hdrs = dict(HEADERS)
    try:
        for _ in range(max_retries):
            _sleep()
            r = requests.get(url, headers=hdrs, params=params, timeout=25)
            if r.status_code == 200:
                return r.json()
    except Exception:
        return None
    return None

def commons_imageinfo_for_titles(titles: List[str]):
    if not titles:
        return []
    CHUNK=40; results=[]
    for i in range(0, len(titles), CHUNK):
        ch = titles[i:i+CHUNK]
        params = {"action":"query","format":"json","prop":"imageinfo","titles":"|".join(ch), "iiprop":"url|extmetadata|size","iiurlwidth":"2200","iiurlheight":"2200","origin":"*"}
        data = http_get_json(COMMONS_API, params=params)
        if not data:
            continue
        pages = (data.get("query") or {}).get("pages",{})
        for _, p in pages.items():
            ii = (p.get("imageinfo") or [{}])[0]
            if not ii:
                continue
            meta = ii.get("extmetadata") or {}
            full = ii.get("thumburl") or ii.get("url")
            if not full:
                continue
            title = re.sub(r"<.*?>","", str(meta.get("ObjectName",{}).get("value") or p.get("title") or ""))
            author = re.sub(r"<.*?>","", str(meta.get("Artist",{}).get("value") or ""))
            desc = " ".join([str(meta.get(k,{}).get("value","") or "") for k in ("ImageDescription","Credit","Depicts","DepictedPeople")])
            results.append({"filetitle": p.get("title",""), "image_url": full, "title": title, "author": author, "license_name": meta.get("LicenseShortName",{}).get("value",""), "license_url": (meta.get("LicenseUrl",{}).get("value") or ""), "desc": desc, "w": ii.get("width") or 0, "h": ii.get("height") or 0})
    return results

def wikipedia_page_images(title: str, limit:int=200):
    params = {"action":"query","format":"json","titles":title,"prop":"images","imlimit":str(limit),"redirects":1,"origin":"*"}
    data = http_get_json(WIKI_API, params=params)
    if not data:
        return []
    pages = (data.get("query") or {}).get("pages",{})
    files=[]
    for _, p in pages.items():
        for im in p.get("images",[]) or []:
            t = im.get("title","")
            if t.startswith("File:"):
                files.append(t)
    return commons_imageinfo_for_titles(files)

def commons_search_media(query: str, limit: int = 200):
    params = {"action":"query","format":"json","list":"search","srsearch": query,"srnamespace":"6","srlimit":str(limit),"origin":"*"}
    data = http_get_json(COMMONS_API, params=params)
    out=[]
    if not data:
        return out
    hits = (data.get("query") or {}).get("search",[]) or []
    titles = [h.get("title") for h in hits if h.get("title")]
    if titles:
        out = commons_imageinfo_for_titles(titles)
    return out

def pexels_search_images(query: str, per_page: int = 30):
    key = os.getenv("PEXELS_API_KEY")
    if not key:
        return []
    url = "https://api.pexels.com/v1/search"
    headers = {"Authorization": key}
    out=[]
    page = 1
    remaining = per_page
    try:
        while remaining > 0 and page < 10:
            params = {"query": query, "per_page": min(80, remaining), "page": page}
            r = requests.get(url, headers=headers, params=params, timeout=20)
            if r.status_code != 200:
                break
            j = r.json()
            photos = j.get("photos", []) or []
            for p in photos:
                src = p.get("src", {})
                img = src.get("large2x") or src.get("original") or src.get("large") or src.get("medium")
                out.append({"filetitle": p.get("id"), "image_url": _normalize_url(img), "title": p.get("alt") or "", "author": p.get("photographer") or "", "license_name":"Pexels", "license_url":"https://www.pexels.com/", "desc": p.get("alt") or "", "w": p.get("width") or 0, "h": p.get("height") or 0})
            remaining = per_page - len(out)
            if not photos:
                break
            page += 1
            _sleep()
    except Exception:
        return out
    return out

def pixabay_search_images(query: str, per_page: int = 30):
    key = os.getenv("PIXABAY_API_KEY")
    if not key:
        return []
    url = "https://pixabay.com/api/"
    out=[]
    page = 1
    remaining = per_page
    try:
        while remaining > 0 and page < 10:
            params = {"key": key, "q": query, "image_type":"photo", "per_page": min(200, remaining), "page": page}
            r = requests.get(url, params=params, timeout=20)
            if r.status_code != 200:
                break
            j = r.json()
            hits = j.get("hits", []) or []
            for hit in hits:
                out.append({"filetitle": hit.get("id"), "image_url": _normalize_url(hit.get("largeImageURL") or hit.get("webformatURL")), "title": hit.get("tags") or "", "author": hit.get("user") or "", "license_name":"Pixabay", "license_url":"https://pixabay.com/", "desc": hit.get("tags") or "", "w": hit.get("imageWidth") or 0, "h": hit.get("imageHeight") or 0})
            remaining = per_page - len(out)
            if not hits:
                break
            page += 1
            _sleep()
    except Exception:
        return out
    return out

def serpapi_search_images(query: str, per_page: int = 100, max_pages: int = 10, api_key: Optional[str] = None):
    key = api_key or os.getenv("SERPAPI_API_KEY")
    if not key:
        return []
    out = []
    url = "https://serpapi.com/search.json"
    fetched = 0
    page = 0
    page_size = max(10, min(100, int(per_page)))
    try:
        while page < max_pages and fetched < (per_page * max_pages):
            params = {
                "engine": "google_images",
                "q": query,
                "ijn": str(page),
                "api_key": key,
                "num": str(page_size),
            }
            r = requests.get(url, params=params, headers=HEADERS, timeout=30)
            if r.status_code != 200:
                _sleep(0.5, 1.2)
                break
            j = r.json()
            images = j.get("images_results") or []
            if not images:
                break
            for im in images:
                src = im.get("original") or im.get("thumbnail") or im.get("link") or im.get("source")
                if not src:
                    continue
                title = im.get("title") or ""
                author = im.get("source") or ""
                out.append({"filetitle": im.get("position") or title, "image_url": _normalize_url(src), "title": title, "author": author, "license_name": "SerpAPI/GoogleImages", "license_url": "", "desc": im.get("snippet") or "", "w": im.get("width") or 0, "h": im.get("height") or 0})
            fetched += 1
            page += 1
            _sleep(0.2, 0.6)
    except Exception:
        pass
    return out

# ============================================================================ 
# ML HELPERS
# ============================================================================ 
def blip_caption_for_pil(im: Image.Image) -> str:
    if not BLIP_AVAILABLE:
        return ""
    inputs = blip_processor(images=im, return_tensors="pt").to(device)
    with torch.no_grad():
        out = blip_model.generate(**inputs, max_length=40)
    return blip_processor.decode(out[0], skip_special_tokens=True)

def embed_image_clip_np(im: Image.Image):
    img_t = preprocess(im).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = clip_model.encode_image(img_t)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy()[0]

def embed_text_clip_np(texts: List[str]):
    toks = open_clip.tokenize(texts).to(device)
    with torch.no_grad():
        emb = clip_model.encode_text(toks)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy()

# ============================================================================ 
# FACE HELPERS
# ============================================================================ 
def face_embeddings_from_pil(im: Image.Image) -> Tuple[List[np.ndarray], List[Tuple[int,int,int,int]]]:
    if not HAVE_FACE:
        return [], []
    try:
        arr = np.array(im.convert("RGB"))
        boxes = face_recognition.face_locations(arr, model="hog")
        if not boxes:
            return [], []
        encs = face_recognition.face_encodings(arr, known_face_locations=boxes)
        return list(encs), list(boxes)
    except Exception:
        return [], []

def select_medoid_embedding(embs: List[np.ndarray]) -> np.ndarray:
    arr = np.stack(embs, axis=0)
    dists = np.sqrt(((arr[:,None,:] - arr[None,:,:])**2).sum(-1))
    mean_d = dists.mean(axis=1)
    idx = int(np.argmin(mean_d))
    return arr[idx]

# ============================================================================ 
# ENHANCED IMAGE VERIFICATION with Scene Context
# ============================================================================ 
def verify_image_relevance(im: Image.Image, caption: str, scene: SceneSegment, person: str, min_scene_similarity: float = 0.45) -> Tuple[bool, float, str]:
    """ Multi-level verification that image matches the scene context. Returns (is_relevant, score, reason) """
    reasons = []
    scores = []

    # 1. Check negative terms (immediate rejection)
    caption_lower = caption.lower()
    for neg_term in scene.negative_terms:
        if neg_term.lower() in caption_lower:
            return False, 0.0, f"negative_term_detected: {neg_term}"
        # --------------------------------------------------
    # BIOGRAPHY HARD RULE: reject non-photographic faces
    # (wax models, sketches, engravings, album scans)
    # --------------------------------------------------
    if BIOGRAPHY_MODE:
        bad_terms = [
            "wax", "wax model", "statue", "model",
            "sketch", "drawing", "illustration",
            "engraving", "painting",
            "album", "albumen", "print",
            "sculpture", "figure"
        ]

        if any(t in caption_lower for t in bad_terms):
            return False, 0.0, "reject:non_photographic"


    # 2. Scene text similarity (SBERT)
    try:
        if scene.embedding is not None:
            cap_emb = sbert.encode(caption, convert_to_tensor=True)
            scene_emb_tensor = torch.from_numpy(scene.embedding).to(device)
            sim = util.cos_sim(cap_emb, scene_emb_tensor).cpu().item()
            scores.append(sim)
            reasons.append(f"scene_sim={sim:.3f}")
            if sim < min_scene_similarity:
                return False, sim, f"low_scene_similarity: {sim:.3f}"
    except Exception:
        pass

    # 3. Keyword matching (stricter: require at least 2 matches if no person mention)
    keyword_matches = sum(1 for kw in scene.keywords if kw.lower() in caption_lower)
    if scene.keywords:
        kw_score = keyword_matches / len(scene.keywords)
        scores.append(kw_score)
        reasons.append(f"kw_match={keyword_matches}/{len(scene.keywords)}")

    # 4. Time period consistency
    if scene.time_period:
        time_in_caption = scene.time_period.lower() in caption_lower
        if time_in_caption:
            scores.append(1.0)
            reasons.append(f"time_match={scene.time_period}")
        else:
            modern_terms = ['smartphone', 'computer', 'laptop', 'modern', 'contemporary', 'recent']
            if any(term in caption_lower for term in modern_terms):
                if scene.time_period and any(old in scene.time_period for old in ['18', '19', 'early']):
                    return False, 0.0, "anachronism_detected"

    # 5. Location consistency
    if scene.location:
        loc_in_caption = scene.location.lower() in caption_lower
        if loc_in_caption:
            scores.append(1.0)
            reasons.append(f"loc_match={scene.location}")

    # 6. Person mention (boost if present, penalize if absent)
    person_in_caption = person.lower() in caption_lower
    if person_in_caption:
        scores.append(1.0)
        reasons.append("person_mentioned")

    # Calculate final score
    if scores:
        final_score = np.mean(scores)
    else:
        final_score = 0.5

    # Penalize if person not mentioned
    if not person_in_caption:
        final_score -= 0.2
        reasons.append("penalty_no_person_mention")

    # Stricter keyword requirement
    if keyword_matches < 2 and not person_in_caption:
        final_score -= 0.3
        reasons.append("penalty_low_keywords")

    final_score = max(0.0, min(1.0, final_score))  # Clamp score

    is_relevant = (
    final_score >= 0.55 and
    (person_in_caption or keyword_matches >= 2)
)
    return is_relevant, final_score, "; ".join(reasons)


# ============================================================================ 
# MAIN PIPELINE with Scene-Aware Matching
# ============================================================================ 
def run(script_path: Path, person: str, target: int, max_candidates: int, debug: bool, face_ref: Optional[str], ref_candidates: int, face_distance_threshold: float, min_clip_score: float, relax: bool, face_ref_fallback: bool=False, max_ref_fallback: int=60, serpapi_key: Optional[str]=None, serpapi_per_page: int = 100, serpapi_max_pages: int = 10, pad_mode: str = "crop"):
    if not HAVE_FACE:
        # earlier you required face_recognition; keep behavior but allow fallback
        # We'll proceed but warn — half-body rejection and face-boosting use face_recognition
        print("⚠️ Warning: face_recognition is not available. Face matching, landmarks and half-body rejection will be limited.")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    # Parse script into scenes
    script_text = script_path.read_text(encoding="utf-8", errors="ignore")
    scenes = parse_script_into_scenes(script_text, person=person, min_words=30)
    if debug:
        print(f"\n📋 Parsed {len(scenes)} scenes from script")
        for i, scene in enumerate(scenes[:3]):
            print(f" Scene {i}: {scene.text[:100]}...")
            print(f" Keywords: {', '.join(scene.keywords[:5])}")
            print(f" Time: {scene.time_period}, Location: {scene.location}")

    # Extract scene-aware topics
    topics = extract_topics_scene_aware(scenes, person=person, max_topics=20)
    if debug:
        print(f"\n🔍 Generated {len(topics)} search topics:")
        for t in topics[:10]:
            print(f" - {t['query']} (scenes: {t['scene_indices']}, weight: {t['weight']:.2f})")

    # Prepare embeddings for topics
    topic_queries = [t['query'] for t in topics]
    topic_clip_embs = embed_text_clip_np(topic_queries) if topic_queries else np.zeros((0,512), dtype=np.float32)
    topic_sbert_embs = sbert.encode(topic_queries, convert_to_tensor=True) if topic_queries else None

    # Gather image pool
    pool: List[Dict] = []
    person_page_files = set()
    source_counts = {}

    def add_results(src_name: str, items: List[Dict]):
        count_before = len(pool)
        if not items:
            return
        seen_local = set(_normalize_url_key(it.get("image_url") or "") for it in pool)
        for it in items:
            u = _normalize_url(it.get("image_url") or "")
            if not u:
                continue
            key = _normalize_url_key(u)
            if key in seen_local:
                continue
            seen_local.add(key)
            pool.append(it)
        source_counts[src_name] = source_counts.get(src_name, 0) + max(0, len(pool) - count_before)

    # Fetch from Wikipedia person page
    try:
        wiki_person_imgs = wikipedia_page_images(person, limit=250)
        add_results("wikipedia_person", wiki_person_imgs)
        person_page_files.update([it.get("filetitle") for it in wiki_person_imgs if it.get("filetitle")])
    except Exception:
        pass

    serp_key = serpapi_key or os.getenv("SERPAPI_API_KEY")
    pex_key_present = bool(os.getenv("PEXELS_API_KEY"))
    pix_key_present = bool(os.getenv("PIXABAY_API_KEY"))

    def gather_for_query(q: str):
        if not q: return
        if debug:
            print(f"Searching for: {q}")
        try:
            res = commons_search_media(q, limit=200)
            add_results("commons", res)
        except Exception:
            pass
        try:
            res = wikipedia_page_images(q, limit=100)
            add_results("wiki_query", res)
            person_page_files.update([it.get("filetitle") for it in res if it.get("filetitle")])
        except Exception:
            pass
        if pex_key_present:
            try:
                res = pexels_search_images(q, per_page=60)
                add_results("pexels", res)
            except Exception:
                pass
        if pix_key_present:
            try:
                res = pixabay_search_images(q, per_page=60)
                add_results("pixabay", res)
            except Exception:
                pass
        if serp_key:
            try:
                res = serpapi_search_images(q, per_page=serpapi_per_page, max_pages=serpapi_max_pages, api_key=serp_key)
                add_results("serpapi", res)
            except Exception:
                pass

    for topic_dict in topics:
        gather_for_query(topic_dict['query'])
        if len(pool) >= target * 12:
            break

    # Fallback searches
    if len(pool) < max(200, target * 2):
        if debug:
            print(f"Pool small ({len(pool)}), running extra searches...")
        variants = [person, f"{person} portrait", f"{person} photo", f"{person} historical"]
        for v in variants:
            gather_for_query(v)
            if len(pool) >= max(500, target * 6):
                break

    # Final dedupe
    uniq = []
    seen_urls = set()
    for it in pool:
        u = _normalize_url(it.get("image_url") or "")
        key = _normalize_url_key(u)
        if not u:
            continue
        if key in seen_urls:
            continue
        seen_urls.add(key)
        uniq.append(it)
    pool = uniq
    if debug:
        print(f"\n📊 Source counts: {source_counts}")
        print(f"📦 Total candidates after dedup: {len(pool)}")

    # Pre-rank candidates
    def candidate_text(it):
        return ((it.get("title") or "") + " " + (it.get("desc") or "")).strip() or str(it.get("filetitle",""))
    cand_texts = [candidate_text(it) for it in pool]
    cand_embs = None
    try:
        cand_embs = sbert.encode(cand_texts, convert_to_tensor=True)
    except Exception:
        pass

    def candidate_score(it, idx):
        txt_score = 0.0
        try:
            if cand_embs is not None:
                sims = util.cos_sim(cand_embs[idx], topic_sbert_embs).cpu().numpy()[0]
                txt_score = float(np.max(sims))
                txt_score = (txt_score + 1.0)/2.0
            else:
                txt = candidate_text(it)
                if txt:
                    emb = sbert.encode(txt, convert_to_tensor=True)
                    sims = util.cos_sim(emb, topic_sbert_embs).cpu().numpy()[0]
                    txt_score = float(np.max(sims))
                    txt_score = (txt_score + 1.0)/2.0
        except Exception:
            txt_score = 0.0
        w = int(it.get("w") or 0)
        h = int(it.get("h") or 0)
        size_score = 0.0
        if w and h:
            ar = w/h if h else 1.0
            ar_factor = max(0.0, 1.0 - min(1.0, abs(ar - TARGET_AR)/0.5))
            res_factor = min(1.0, min(w / TARGET_W, h / TARGET_H))
            size_score = ar_factor * res_factor
        return 0.75 * txt_score + 0.25 * size_score

    preds = []
    for idx, it in enumerate(pool):
        try:
            sc = candidate_score(it, idx)
        except Exception:
            sc = 0.0
        preds.append((it, sc))
    preds.sort(key=lambda x: x[1], reverse=True)
    pool = [p for p,_ in preds][:max_candidates]
    if debug:
        print(f"🎯 Processing top {len(pool)} candidates (max_candidates={max_candidates})")

    # Build reference embedding
    ref_emb = None
    ref_img_path = None

    def _get_best_face_emb_from_pil(im: Image.Image):
        encs, boxes = face_embeddings_from_pil(im)
        if not encs:
            return None, 0.0, None
        if boxes:
            areas = [(b[2]-b[0])*(b[1]-b[3]) for b in boxes]
            idx = int(np.argmax(areas))
            emb = np.array(encs[idx], dtype=np.float32)
            return emb, float(areas[idx]), emb
        else:
            emb = np.array(encs[0], dtype=np.float32)
            return emb, float(im.width * im.height), emb

    def _try_url_for_face(url: str):
        try:
            _sleep()
            r = requests.get(url, headers=HEADERS, timeout=20)
            r.raise_for_status()
            im = Image.open(BytesIO(r.content)).convert("RGB")
            return _get_best_face_emb_from_pil(im), im
        except Exception:
            return (None, 0.0, None), None

    if face_ref:
        p = Path(face_ref)
        if not p.exists():
            raise SystemExit(f"Provided --face_ref not found: {face_ref}")
        try:
            im = Image.open(p).convert("RGB")
            emb, area, _ = _get_best_face_emb_from_pil(im)
            if emb is not None:
                ref_emb = emb
                ref_img_path = str(p)
                if debug:
                    print("✓ Loaded reference embedding from --face_ref")
            else:
                if debug:
                    print("⚠️ No face in --face_ref; attempting fallback...")
                if face_ref_fallback:
                    fallback_urls = []
                    try:
                        wiki_imgs = wikipedia_page_images(person, limit=80)
                        fallback_urls += [i.get("image_url") for i in wiki_imgs if i.get("image_url")]
                    except Exception:
                        pass
                    fallback_urls += [_normalize_url(it.get("image_url") or "") for it in pool]
                    seen_urls_fb = set()
                    deduped_urls = []
                    for u in fallback_urls:
                        if not u or u in seen_urls_fb:
                            continue
                        seen_urls_fb.add(u)
                        deduped_urls.append(u)
                    max_try = min(max_ref_fallback, len(deduped_urls))
                    if debug:
                        print(f"Trying up to {max_try} online images for reference...")
                    for url in deduped_urls[:max_try]:
                        (emb_tuple, im_fb) = _try_url_for_face(url)
                        emb_candidate, area, _ = emb_tuple if emb_tuple else (None,0.0,None)
                        if emb_candidate is not None:
                            ref_emb = np.array(emb_candidate, dtype=np.float32)
                            try:
                                save_path = OUT_ROOT / "bootstrap_ref.jpg"
                                save_path.parent.mkdir(parents=True, exist_ok=True)
                                im_fb.save(save_path, "JPEG", quality=90)
                                ref_img_path = str(save_path)
                            except Exception:
                                ref_img_path = None
                            if debug:
                                print(f"✓ Bootstrapped reference from: {url}")
                            break
        except Exception as e:
            if debug:
                print(f"Error reading --face_ref: {e}")
    else:
        boot = min(ref_candidates, len(pool))
        collected = []
        if debug:
            print(f"No --face_ref: bootstrapping from top {boot} candidates...")
        for it in pool[:boot]:
            url = _normalize_url(it.get("image_url") or "")
            if not url:
                continue
            try:
                _sleep()
                r = requests.get(url, headers=HEADERS, timeout=20)
                r.raise_for_status()
                im = Image.open(BytesIO(r.content)).convert("RGB")
                encs, boxes = face_embeddings_from_pil(im)
                if encs:
                    if boxes:
                        areas = [(b[2]-b[0])*(b[1]-b[3]) for b in boxes]
                        idx = int(np.argmax(areas))
                        emb = np.array(encs[idx], dtype=np.float32)
                    else:
                        emb = np.array(encs[0], dtype=np.float32)
                    collected.append(emb)
            except Exception:
                continue
        if collected:
            ref_emb = select_medoid_embedding(collected)
            if debug:
                print(f"✓ Bootstrapped medoid from {len(collected)} faces")
        else:
            if debug:
                print("⚠️ Bootstrap failed: no faces found")
            ref_emb = None

    # Evaluate candidates with scene-aware verification
    accepted = []
    clip_emb_list = []
    rows = []
    phash_seen = []
    scene_image_map = defaultdict(list)
    pbar = tqdm(pool, desc="[eval candidates]") if pool else None

    for it in (pool if pbar is None else pbar):
        if len(accepted) >= target:
            break
        url_raw = it.get("image_url") or ""
        url = _normalize_url(url_raw)
        if not url:
            continue
        try:
            _sleep()
            r = requests.get(url, headers=HEADERS, timeout=25)
            r.raise_for_status()
            im = Image.open(BytesIO(r.content)).convert("RGB")
        except Exception:
            continue
        if im.width < MIN_W or im.height < MIN_H:
            if not relax:
                if debug and pbar:
                    pbar.set_postfix_str("reject:small")
                continue

        # Aspect ratio check for solid/blur modes
        if pad_mode in ("solid", "blur"):
            ar = im.width / im.height if im.height else 0.0
            if not (abs(ar - TARGET_AR) <= TARGET_AR * ASPECT_RATIO_TOLERANCE):
                if debug and pbar:
                    pbar.set_postfix_str(f"reject:aspect({ar:.2f})")
                continue

        # -------------------------
        # Face detection + stricter half-body / tiny-face rejection
        # -------------------------
        encs, boxes = face_embeddings_from_pil(im)
        has_face = len(encs) > 0

        # Generate caption for verification (BLIP helps detect "half-body" descriptions)
        caption = ""
        try:
            if BLIP_AVAILABLE:
                caption = blip_caption_for_pil(im)
        except Exception:
            caption = ""

        caption_lower = (caption or "").lower()
        meta_title = (it.get("title") or "").lower()
        meta_desc = (it.get("desc") or "").lower()
        meta_combined = " ".join([caption_lower, meta_title, meta_desc])

        # Terms that typically indicate half-body / torso / full-length person photos
        half_body_terms = [
            "half body", "half-body", "waist-up", "waist up", "upper body", "upper-body",
            "torso", "chest", "3/4", "three-quarter", "three quarter", "full body",
            "full-body", "full-length", "standing", "portrait full-length", "kneeling",
            "waist length", "waist-length"
        ]
        text_says_half_body = any(term in meta_combined for term in half_body_terms)

        # If NO face detected but BLIP/meta describes person as "half body" / "full body", reject
        if not has_face:
            if text_says_half_body:
                if debug and pbar:
                    pbar.set_postfix_str("reject:half_body_no_face")
                continue
            negative_face_terms = ["no face", "face not visible", "face obscured", "face blurred", "profile without face"]
            if any(t in caption_lower for t in negative_face_terms):
                if debug and pbar:
                    pbar.set_postfix_str("reject:no_face_caption")
                continue

        # If a face is detected, ensure the face is sufficiently large and not severely cropped.
        if has_face:
            # convert face_recognition boxes (top, right, bottom, left) -> left,top,right,bottom
            norm_face_boxes = []
            for b in boxes:
                try:
                    top, right, bottom, left = b
                    left, top, right, bottom = int(left), int(top), int(right), int(bottom)
                    left = max(0, min(left, im.width - 1))
                    top = max(0, min(top, im.height - 1))
                    right = max(left + 1, min(right, im.width))
                    bottom = max(top + 1, min(bottom, im.height))
                    norm_face_boxes.append((left, top, right, bottom))
                except Exception:
                    try:
                        l,t,r,b_ = b
                        norm_face_boxes.append((int(l), int(t), int(r), int(b_)))
                    except Exception:
                        continue

            if norm_face_boxes:
                areas = [ (fb[2]-fb[0]) * (fb[3]-fb[1]) for fb in norm_face_boxes ]
                idx_face = int(np.argmax(areas))
                face_l, face_t, face_r, face_b = norm_face_boxes[idx_face]
                face_h = face_b - face_t
                face_w = face_r - face_l
                eye_y = face_t + 0.35 * (face_b - face_t)
                face_area = (face_r - face_l) * (face_b - face_t)
                img_area = im.width * im.height
                aspect_ratio = face_w / max(1, face_h)
                if BIOGRAPHY_MODE and aspect_ratio < 0.65:
                    if debug and pbar:
                        pbar.set_postfix_str(f"reject:side_profile(ar={aspect_ratio:.2f})")
                        continue

                if BIOGRAPHY_MODE and (face_area / img_area) < 0.18:
                    if debug and pbar:
                        pbar.set_postfix_str("reject:face_not_dominant")
                        continue
                if BIOGRAPHY_MODE:
                     # Forehead / hairline cut
                    if face_t <= int(0.08 * im.height):
                        if debug and pbar:
                            pbar.set_postfix_str("reject:forehead_cut")
                            continue

                    # Chin / mouth cut
                    if (im.height - face_b) <= int(0.08 * im.height):
                        if debug and pbar:
                            pbar.set_postfix_str("reject:chin_cut")
                            continue
                    if eye_y <= 0.22 * im.height:
                        if debug and pbar:
                            pbar.set_postfix_str("reject:eyes_too_high")
                            continue


                MIN_FACE_HEIGHT_RATIO = 0.30
                TOP_CLOSE_PX = int(0.04 * im.height)
                BOTTOM_CLOSE_PX = int(0.04 * im.height)
                face_h_ratio = float(face_h) / float(im.height) if im.height else 0.0

                allow_small_face_terms = ["close-up", "close up", "headshot", "head shot", "portrait face", "face close"]
                if face_h_ratio < MIN_FACE_HEIGHT_RATIO:
                    if not any(t in meta_combined for t in allow_small_face_terms):
                        if debug and pbar:
                            pbar.set_postfix_str(f"reject:face_too_small(r={face_h_ratio:.3f})")
                        continue

                if face_t <= TOP_CLOSE_PX:
                    if face_h_ratio < 0.30:
                        if debug and pbar:
                            pbar.set_postfix_str("reject:face_cut_top")
                        continue

                if (im.height - face_b) <= BOTTOM_CLOSE_PX:
                    if face_h_ratio < 0.30:
                        if debug and pbar:
                            pbar.set_postfix_str("reject:face_cut_bottom")
                        continue
            else:
                if debug and pbar:
                    pbar.set_postfix_str("reject:face_detect_failed_after_enc")
                continue

        # -------------------------
        # Continue with rest of verification (scene matching, face matching, CLIP)
        # -------------------------
        # Find best matching scene for this image
        caption_for_scene = caption or ""
        best_scene = None
        best_scene_score = 0.0
        if caption_for_scene and scenes:
            for scene in scenes:
                is_relevant, score, reason = verify_image_relevance(im, caption_for_scene, scene, person)
                if score > best_scene_score:
                    best_scene_score = score
                    best_scene = scene

        # Face matching logic
        if has_face:
            if ref_emb is None:
                if debug and pbar:
                    pbar.set_postfix_str("reject:face_no_ref")
                continue
            if BIOGRAPHY_MODE and face_h_ratio < 0.30:
                if debug and pbar:
                    pbar.set_postfix_str(f"reject:half_or_full_body(r={face_h_ratio:.3f})")
                continue
            # get norm_face_boxes and encs from earlier; we already built norm_face_boxes
            distances = [float(np.linalg.norm(np.array(e) - ref_emb)) for e in encs]
            best_d = float(np.min(distances))
            if best_d <= face_distance_threshold:
                if best_scene and best_scene_score < 0.30:
                    if debug and pbar:
                        pbar.set_postfix_str(f"reject:face_ok_but_scene_mismatch({best_scene_score:.2f})")
                    continue
                try:
                    clip_emb = embed_image_clip_np(im)
                except Exception:
                    if debug and pbar:
                        pbar.set_postfix_str("reject:clip_fail")
                    continue
                sims = (topic_clip_embs @ clip_emb) if topic_clip_embs.size else np.array([])
                best_topic_idx = int(np.argmax(sims)) if sims.size else 0
                best_clip_sim = float(sims[best_topic_idx]) if sims.size else 0.0

                if best_scene:
                    folder_name = f"scene_{best_scene.index:03d}_{slug(best_scene.keywords[0] if best_scene.keywords else 'general')}"
                else:
                    folder_name = slug(topic_queries[best_topic_idx]) if topic_queries else "misc"
                folder = IMG_DIR / folder_name
                folder.mkdir(parents=True, exist_ok=True)
                outp = folder / f"{len(accepted)+1:03d}.jpg"
                proc = save_processed(im, outp, allow_padding=True, face_boxes=boxes if boxes else None, pad_mode=pad_mode)
                try:
                    ph = imagehash.phash(proc)
                    if any(ph - ph2 <= 5 for ph2 in phash_seen):
                        outp.unlink(missing_ok=True)
                        if debug and pbar:
                            pbar.set_postfix_str("reject:duplicate")
                        continue
                    phash_seen.append(ph)
                except Exception:
                    pass
                row_data = {
                    "scene_index": best_scene.index if best_scene else -1,
                    "scene_text": best_scene.text[:100] if best_scene else "",
                    "topic": folder.name,
                    "file": str(outp),
                    "title": it.get("title",""),
                    "author": it.get("author",""),
                    "src": it.get("image_url",""),
                    "license_name": it.get("license_name",""),
                    "license_url": it.get("license_url",""),
                    "width": proc.width,
                    "height": proc.height,
                    "blip_caption": caption,
                    "best_clip_sim": best_clip_sim,
                    "scene_relevance_score": best_scene_score,
                    "face_best_distance": best_d,
                    "face_count": len(encs),
                    "reason": f"face_match(d={best_d:.3f},scene={best_scene_score:.2f})"
                }
                rows.append(row_data)
                accepted.append(row_data)
                clip_emb_list.append(clip_emb)
                if best_scene:
                    scene_image_map[best_scene.index].append(str(outp))
                if pbar:
                    pbar.set_postfix_str(f"accepted={len(accepted)}/{target} face_d={best_d:.3f} scene={best_scene_score:.2f}")
            else:
                if debug and pbar:
                    pbar.set_postfix_str(f"reject:face_no_match(d={best_d:.3f})")
                continue
        else:
            # No face detected - use scene verification
            filetitle = it.get("filetitle") or ""
            accepted_by_person_page = bool(filetitle and filetitle in person_page_files)
            try:
                clip_emb = embed_image_clip_np(im)
            except Exception:
                if debug and pbar:
                    pbar.set_postfix_str("reject:clip_fail")
                continue
            sims = (topic_clip_embs @ clip_emb) if topic_clip_embs.size else np.array([])
            best_topic_idx = int(np.argmax(sims)) if sims.size else 0
            best_clip_sim = float(sims[best_topic_idx]) if sims.size else 0.0

            cap_ok = False
            if caption and best_scene:
                is_relevant, scene_score, reason = verify_image_relevance(im, caption, best_scene, person)
                cap_ok = is_relevant and scene_score >= 0.35

            should_accept = ( accepted_by_person_page or best_clip_sim >= min_clip_score or (cap_ok and best_scene_score >= 0.40) )
            if should_accept:
                if best_scene:
                    folder_name = f"scene_{best_scene.index:03d}_{slug(best_scene.keywords[0] if best_scene.keywords else 'general')}"
                else:
                    folder_name = slug(topic_queries[best_topic_idx]) if topic_queries else "misc"
                folder = IMG_DIR / folder_name
                folder.mkdir(parents=True, exist_ok=True)
                outp = folder / f"{len(accepted)+1:03d}.jpg"
                proc = save_processed(im, outp, allow_padding=True, face_boxes=boxes if boxes else None, pad_mode=pad_mode)
                try:
                    ph = imagehash.phash(proc)
                    if any(ph - ph2 <= 5 for ph2 in phash_seen):
                        outp.unlink(missing_ok=True)
                        if debug and pbar:
                            pbar.set_postfix_str("reject:duplicate")
                        continue
                    phash_seen.append(ph)
                except Exception:
                    pass
                row_data = {
                    "scene_index": best_scene.index if best_scene else -1,
                    "scene_text": best_scene.text[:100] if best_scene else "",
                    "topic": folder.name,
                    "file": str(outp),
                    "title": it.get("title",""),
                    "author": it.get("author",""),
                    "src": it.get("image_url",""),
                    "license_name": it.get("license_name",""),
                    "license_url": it.get("license_url",""),
                    "width": proc.width,
                    "height": proc.height,
                    "blip_caption": caption,
                    "best_clip_sim": best_clip_sim,
                    "scene_relevance_score": best_scene_score,
                    "face_best_distance": None,
                    "face_count": 0,
                    "reason": f"no_face_scene_ok(clip={best_clip_sim:.3f},scene={best_scene_score:.2f},cap_ok={cap_ok},person_page={accepted_by_person_page})"
                }
                rows.append(row_data)
                accepted.append(row_data)
                clip_emb_list.append(clip_emb)
                if best_scene:
                    scene_image_map[best_scene.index].append(str(outp))
                if pbar:
                    pbar.set_postfix_str(f"accepted={len(accepted)}/{target} clip={best_clip_sim:.3f} scene={best_scene_score:.2f}")
            else:
                if debug and pbar:
                    pbar.set_postfix_str(f"reject:no_face_offtopic(clip={best_clip_sim:.3f},scene={best_scene_score:.2f})")
                continue

    # Write outputs
    if rows:
        header_keys = []
        for r in rows:
            for k in r.keys():
                if k not in header_keys:
                    header_keys.append(k)
        with open(ATTR_CSV, "w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=header_keys)
            writer.writeheader()
            for r in rows:
                safe_row = {}
                for k in header_keys:
                    v = r.get(k, "")
                    try:
                        if isinstance(v, (np.integer, np.floating)):
                            v = v.item()
                        elif isinstance(v, np.ndarray):
                            v = v.tolist()
                    except Exception:
                        pass
                    if not isinstance(v, (str, int, float, bool, type(None), list, dict)):
                        try:
                            v = str(v)
                        except Exception:
                            v = ""
                    safe_row[k] = v
                writer.writerow(safe_row)
    else:
        with open(ATTR_CSV, "w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["note"])
            writer.writerow(["no matched assets found"])

    with open(CREDITS, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(f'{r.get("title","")} — {r.get("author","")} — {r.get("src","")} — {r.get("license_name","")} ({r.get("license_url","")})\n')

    with open(INDEX_JSONL, "w", encoding="utf-8") as fh:
        for r, emb in zip(rows, clip_emb_list):
            obj = {
                "path": r.get("file",""),
                "caption": r.get("blip_caption",""),
                "scene_index": r.get("scene_index", -1),
                "scene_text": r.get("scene_text", ""),
                "width": int(r.get("width",0)),
                "height": int(r.get("height",0))
            }
            fh.write(json.dumps(obj, ensure_ascii=False) + "\n")

    if clip_emb_list:
        try:
            np.save(CLIP_NPY, np.stack(clip_emb_list, axis=0))
        except Exception:
            np.save(CLIP_NPY, np.vstack([np.asarray(e) for e in clip_emb_list]))

    scene_map_data = {
        "scenes": [
            {
                "index": s.index,
                "text": s.text,
                "keywords": s.keywords,
                "time_period": s.time_period,
                "location": s.location,
                "images": scene_image_map.get(s.index, [])
            } for s in scenes
        ],
        "total_images": len(rows),
        "images_per_scene": {str(k): len(v) for k, v in scene_image_map.items()}
    }
    with open(SCENE_MAP, "w", encoding="utf-8") as fh:
        json.dump(scene_map_data, fh, indent=2, ensure_ascii=False)

    print(f"\n✅ Done — saved {len(rows)} matched images (target {target})")
    print(f" → images: {IMG_DIR}")
    print(f" → attribution: {ATTR_CSV}")
    print(f" → index: {INDEX_JSONL}")
    print(f" → embeddings: {CLIP_NPY}")
    print(f" → scene mapping: {SCENE_MAP}")
    if debug and scene_image_map:
        print(f"\n📊 Images per scene:")
        for scene_idx in sorted(scene_image_map.keys()):
            print(f" Scene {scene_idx}: {len(scene_image_map[scene_idx])} images")

# ============================================================================ 
# CLI
# ============================================================================ 
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Enhanced scene-aware image fetcher for video scripts")
    ap.add_argument("--script", default="input/script.txt", help="Path to script file")
    ap.add_argument("--person", required=True, help="Main person/subject name")
    ap.add_argument("--target", type=int, default=100, help="Target number of images")
    ap.add_argument("--max-candidates", type=int, default=400, help="Top-ranked candidates to evaluate")
    ap.add_argument("--debug", action="store_true", help="Enable debug output")
    ap.add_argument("--face_ref", default=None, help="Local portrait image path for reference")
    ap.add_argument("--ref-candidates", type=int, default=30, help="Top candidates for bootstrapping reference")
    ap.add_argument("--face-distance-threshold", type=float, default=0.60, help="Face matching threshold (lower = stricter)")
    ap.add_argument("--min-clip-score", type=float, default=0.01, help="Minimum CLIP similarity for non-face images")
    ap.add_argument("--relax", action="store_true", help="Relax some filters for small pools")
    ap.add_argument("--face-ref-fallback", action="store_true", help="Try online fallback if face_ref has no face")
    ap.add_argument("--serpapi-key", default=None, help="SerpAPI API key (or set SERPAPI_API_KEY env var)")
    ap.add_argument("--max-ref-fallback", type=int, default=60, help="Max online images to try for reference fallback")
    ap.add_argument("--serpapi-per-page", type=int, default=100, help="SerpAPI results per page")
    ap.add_argument("--serpapi-max-pages", type=int, default=10, help="SerpAPI max pages per topic")
    ap.add_argument("--pad-mode", choices=["crop","solid","blur"], default="crop", help="Image processing mode: crop (face-aware 16:9), solid (pad with color), blur (pad with blurred bg)")
    args = ap.parse_args()
    run( Path(args.script), args.person, int(args.target), int(args.max_candidates), bool(args.debug), args.face_ref, int(args.ref_candidates), float(args.face_distance_threshold), float(args.min_clip_score), bool(args.relax), bool(args.face_ref_fallback), int(args.max_ref_fallback), serpapi_key=args.serpapi_key, serpapi_per_page=int(args.serpapi_per_page), serpapi_max_pages=int(args.serpapi_max_pages), pad_mode=args.pad_mode )
