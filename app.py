import streamlit as st
import pandas as pd
import numpy as np
import gspread
from google.oauth2.service_account import Credentials
from streamlit_autorefresh import st_autorefresh
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import re
import hashlib
from datetime import datetime
from difflib import SequenceMatcher

st.set_page_config(page_title="Course Decision Dashboard", layout="wide")
st_autorefresh(interval=30_000, key="autorefresh")


# ---- Column names ----
TS_COL = "Timestamp"
COURSE_COL = "Course"
COURSE_NAME_COL = "Course Name"
ASSIGN_COL = "What types of assignments / assessments did the class consist of?"
PROF_COL = "Professor"
SEM_COL = "What semester did you take the class? [Fall/Spring YYYY]"
DIFF_COL = "How would you rate the course's difficulty level? (10 being very difficult)"
USE_COL = "How useful did you find the course material? (10 being very useful)"
LIKED_COL = "Did you like this class?"
COMMENTS_COL = "Please give any other comments. What did you enjoy most about the class? Any tips for success? Any other thoughts?"

# ---- Core vs Elective ----
CORE_COURSE_TITLES = [
    "machine learning for data science",
    "statistical inference",
    "statistical inference & modeling",
    "statistical inference and modeling",
    "computer systems for data science",
    "probability & statistics for data science",
    "probability and statistics for data science",
    "exploratory data analysis & visualization",
    "exploratory data analysis and visualization",
    "algorithms for data science",
]


def classify_course_type(course_value: str) -> str:
    s = (course_value or "")
    s = s.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    s = re.sub(r"\s+", " ", s).strip().lower()
    for core in CORE_COURSE_TITLES:
        if core in s:
            return "Core"
    return "Elective"


# ---- Visual palette ----
TYPE_COLORS = {"Core": "#012169", "Elective": "#75AADB"}          # Columbia blue + lighter
TERM_COLORS = {"Fall": "#C75B12", "Spring": "#2E8B57",
               "Summer": "#E0A800", "Winter": "#4A6FA5"}

def type_badge(course_type: str) -> str:
    color = TYPE_COLORS.get(course_type, "#888")
    return (
        f"<span style='background:{color}; color:#fff; padding:2px 10px; "
        f"border-radius:10px; font-size:0.8rem; font-weight:600;'>{course_type}</span>"
    )

def term_badge(term: str) -> str:
    color = TERM_COLORS.get(term, "#888")
    return (
        f"<span style='background:{color}; color:#fff; padding:2px 8px; "
        f"border-radius:8px; font-size:0.75rem; font-weight:600;'>{term}</span>"
    )

def component_chip(label: str, pct: int | None = None) -> str:
    pct_str = f" · {pct}%" if pct is not None else ""
    return (
        f"<span style='background:#eef2ff; color:#1e3a8a; padding:3px 10px; "
        f"border-radius:12px; font-size:0.78rem; margin:2px 4px 2px 0; "
        f"display:inline-block; border:1px solid #c7d2fe;'>{label}{pct_str}</span>"
    )


# ---- Assignment components ----
KNOWN_COMPONENTS = [
    "Problem Sets", "Projects", "Midterm Exam", "Final Exam",
    "Final Project", "Readings", "Presentations",
]
_COMPONENT_ALIASES = {
    "Problem Sets":   [r"problem\s*sets?", r"\bpsets?\b", r"\bhomework\b"],
    "Projects":       [r"\bprojects?\b"],
    "Midterm Exam":   [r"midterm"],
    "Final Exam":     [r"final\s*exam"],
    "Final Project":  [r"final\s*project"],
    "Readings":       [r"\breadings?\b", r"\bpapers?\b"],
    "Presentations":  [r"presentations?"],
}

# Course-style classifier (derived from components, with comment-text fallback)
STYLE_COLORS = {
    "Exam-driven":      "#c0392b",
    "Project-driven":   "#1e8449",
    "Mixed":            "#7d3c98",
    "Reading-heavy":    "#b9770e",
    "Problem-set-heavy":"#2471a3",
    "Unknown":          "#95a5a6",
}

# Comment-text fallback patterns used when the assignments checkbox is empty
_STYLE_TEXT_HINTS = {
    "exam":    [r"\bmidterm\b", r"\bfinal exam\b", r"\bexams?\b"],
    "project": [r"\bfinal project\b", r"\bprojects?\b", r"\bcapstone\b"],
    "pset":    [r"\bproblem sets?\b", r"\bpsets?\b", r"\bhomework\b", r"\bhw\b"],
    "reading": [r"\breadings?\b", r"\bpapers?\b"],
}

def _infer_style_from_text(comments: list[str]) -> dict[str, int]:
    """Count style hints across raw comment text. Used when checkbox data is empty."""
    counts = {"exam": 0, "project": 0, "pset": 0, "reading": 0}
    for c in comments:
        if not isinstance(c, str):
            continue
        low = c.lower()
        for k, pats in _STYLE_TEXT_HINTS.items():
            if any(re.search(p, low) for p in pats):
                counts[k] += 1
    return counts


def classify_style(comp_freq: dict, n_reviews: int,
                   comment_hints: dict | None = None) -> str:
    """comp_freq: {component: count} from the checkbox field.
    comment_hints: optional {exam/project/pset/reading: count} from comment text fallback.
    Returns a style label."""
    if n_reviews == 0:
        return "Unknown"

    # Primary signal: checkbox data
    if comp_freq:
        pct = {k: v / n_reviews for k, v in comp_freq.items()}
        has_exam = max(pct.get("Midterm Exam", 0), pct.get("Final Exam", 0)) >= 0.4
        has_project = max(pct.get("Projects", 0), pct.get("Final Project", 0)) >= 0.4
        has_psets = pct.get("Problem Sets", 0) >= 0.5
        has_readings = pct.get("Readings", 0) >= 0.5
        if has_exam and has_project:
            return "Mixed"
        if has_exam:
            return "Exam-driven"
        if has_project:
            return "Project-driven"
        if has_psets:
            return "Problem-set-heavy"
        if has_readings:
            return "Reading-heavy"

    # Fallback: scan comment text for hints
    if comment_hints:
        exam_h = comment_hints.get("exam", 0)
        proj_h = comment_hints.get("project", 0)
        pset_h = comment_hints.get("pset", 0)
        read_h = comment_hints.get("reading", 0)
        if exam_h or proj_h or pset_h or read_h:
            top = max([("Exam-driven", exam_h), ("Project-driven", proj_h),
                       ("Problem-set-heavy", pset_h), ("Reading-heavy", read_h)],
                      key=lambda kv: kv[1])
            if exam_h > 0 and proj_h > 0 and abs(exam_h - proj_h) <= 1:
                return "Mixed"
            if top[1] > 0:
                return top[0]

    return "Unknown"

def style_badge(style: str) -> str:
    color = STYLE_COLORS.get(style, "#7f8c8d")
    return (
        f"<span style='background:{color}; color:#fff; padding:2px 10px; "
        f"border-radius:10px; font-size:0.8rem; font-weight:600;'>{style}</span>"
    )


def parse_components(s) -> set:
    if not isinstance(s, str) or not s.strip():
        return set()
    low = s.lower()
    found = set()
    for label, patterns in _COMPONENT_ALIASES.items():
        for pat in patterns:
            if re.search(pat, low):
                found.add(label)
                break
    # Avoid double-counting: if "Final Project" matched, don't also infer "Final Exam"
    # from the bare word "final".
    if "Final Project" in found and "Final Exam" in found and "final exam" not in low:
        found.discard("Final Exam")
    return found


# ---------- Professor name normalization ----------
_PROF_PREFIX_RE = re.compile(r"^\s*(prof\.?|professor|dr\.?|mr\.?|ms\.?|mrs\.?)\s+", re.IGNORECASE)


def _prof_normalize_key(name: str) -> str:
    if not isinstance(name, str):
        return ""
    s = name.strip()
    s = _PROF_PREFIX_RE.sub("", s)
    s = re.sub(r"\s+", " ", s)
    return s.lower()


def _fuzzy_cluster_keys(keys_with_counts: list[tuple[str, int]],
                        threshold: float = 0.82) -> dict:
    """Cluster normalized prof keys by SequenceMatcher similarity.
    Returns a map from each key to its canonical (most-common-spelling) key.
    'eleni drinea' and 'eleni drea' get merged; 'john smith' and 'jane smith' do NOT
    because the threshold is high enough to require shared distinctive substrings."""
    # Sort by frequency desc so most-common spelling becomes the cluster head
    items = sorted(keys_with_counts, key=lambda x: -x[1])
    clusters = []   # list of dicts: {"head": str, "members": set[str]}
    for key, _cnt in items:
        placed = False
        for cl in clusters:
            # Compare against the cluster head (most common spelling)
            if SequenceMatcher(None, key, cl["head"]).ratio() >= threshold:
                cl["members"].add(key)
                placed = True
                break
        if not placed:
            clusters.append({"head": key, "members": {key}})

    mapping = {}
    for cl in clusters:
        for m in cl["members"]:
            mapping[m] = cl["head"]
    return mapping


def build_prof_display_map(prof_series: pd.Series) -> dict:
    """Returns a dict: original raw name -> canonical display name.
    Two passes: (1) exact normalized-key dedupe, (2) fuzzy cluster near-duplicates."""
    s = prof_series.dropna().astype(str).str.strip()
    s = s[s != ""]
    if s.empty:
        return {}
    df = pd.DataFrame({"orig": s.values})
    df["key"] = df["orig"].apply(_prof_normalize_key)

    # Step 1: per-normalized-key, pick the most common original spelling
    per_key_canonical = (
        df.groupby("key")["orig"]
          .agg(lambda x: x.value_counts().idxmax())
          .to_dict()
    )

    # Step 2: fuzzy-cluster the normalized keys to merge near-duplicates
    key_counts = df["key"].value_counts().to_dict()
    key_clusters = _fuzzy_cluster_keys(list(key_counts.items()))

    mapping = {}
    for raw_key, canon_key in key_clusters.items():
        mapping[raw_key] = per_key_canonical.get(canon_key, per_key_canonical.get(raw_key, raw_key))
    return mapping


# ---------- Data loading ----------
@st.cache_data(ttl=30)
def load_data(sheet_id: str, worksheet_name: str) -> pd.DataFrame:
    creds_dict = st.secrets["gcp_service_account"]
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets.readonly",
        "https://www.googleapis.com/auth/drive.readonly",
    ]
    creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
    gc = gspread.authorize(creds)
    sh = gc.open_by_key(sheet_id)
    ws = sh.worksheet(worksheet_name)

    values = ws.get_all_values()
    if not values:
        return pd.DataFrame()

    header = values[0]
    data = values[1:]

    def _norm_header(x: str) -> str:
        x = (x or "")
        x = x.replace("\r", " ").replace("\n", " ").replace("\t", " ")
        x = re.sub(r"\s+", " ", x).strip()
        return x

    clean_headers, seen = [], {}
    for j, h in enumerate(header):
        h = _norm_header(h) or f"Unnamed_{j}"
        if h in seen:
            seen[h] += 1
            h = f"{h}_{seen[h]}"
        else:
            seen[h] = 0
        clean_headers.append(h)

    df = pd.DataFrame(data, columns=clean_headers)

    def _norm_cell(x, preserve_newlines=False):
        if not isinstance(x, str):
            return x
        if preserve_newlines:
            x = re.sub(r"[ \t]+", " ", x).strip()
        else:
            x = x.replace("\r", " ").replace("\n", " ").replace("\t", " ")
            x = re.sub(r"\s+", " ", x).strip()
        return x

    for col in df.columns:
        preserve = (col == COMMENTS_COL)
        df[col] = df[col].apply(lambda v, p=preserve: _norm_cell(v, preserve_newlines=p))

    df = df[df.astype(str).apply(lambda r: any(c.strip() for c in r), axis=1)]

    for col in [DIFF_COL, USE_COL]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if LIKED_COL in df.columns:
        s = df[LIKED_COL].astype(str).str.strip().str.lower()
        df[LIKED_COL] = s.map({"yes": True, "no": False})

    if TS_COL in df.columns:
        df["_ts"] = pd.to_datetime(df[TS_COL], errors="coerce")

    if COURSE_COL in df.columns:
        df = df[df[COURSE_COL].notna() & (df[COURSE_COL].astype(str).str.strip() != "")]

    # If respondent picked "Other" for Course, fall back to the free-text Course Name
    if COURSE_COL in df.columns and COURSE_NAME_COL in df.columns:
        course_str = df[COURSE_COL].astype(str).str.strip()
        name_str = df[COURSE_NAME_COL].astype(str).str.strip()
        is_other = course_str.str.lower().isin(["other", "other:", "others"])
        has_name = name_str.ne("") & name_str.str.lower().ne("nan")
        df.loc[is_other & has_name, COURSE_COL] = name_str[is_other & has_name]

    if COURSE_COL in df.columns:
        df["course_type"] = df[COURSE_COL].astype(str).map(classify_course_type)
    else:
        df["course_type"] = "Elective"

    if PROF_COL in df.columns:
        df["_prof_key"] = df[PROF_COL].apply(_prof_normalize_key)
        # Fuzzy-cluster near-duplicate names (e.g. 'eleni drinea' / 'eleni drea')
        key_counts = df["_prof_key"].value_counts().to_dict()
        cluster_map = _fuzzy_cluster_keys(list(key_counts.items()))
        df["_prof_canon"] = df["_prof_key"].map(cluster_map).fillna(df["_prof_key"])
    else:
        df["_prof_key"] = ""
        df["_prof_canon"] = ""

    # Derive term (Fall/Spring) and year from the semester column
    if SEM_COL in df.columns:
        sem_str = df[SEM_COL].astype(str)
        df["_term"] = sem_str.str.extract(r"(?i)(Fall|Spring|Summer|Winter)", expand=False).str.title()
        df["_year"] = pd.to_numeric(sem_str.str.extract(r"(\d{4})", expand=False), errors="coerce")
    else:
        df["_term"] = pd.NA
        df["_year"] = pd.NA

    # Parse assignment components from the multi-select field
    if ASSIGN_COL in df.columns:
        df["_components"] = df[ASSIGN_COL].apply(parse_components)
    else:
        df["_components"] = [set() for _ in range(len(df))]

    return df


def safe_mean(series: pd.Series):
    series = pd.to_numeric(series, errors="coerce")
    return None if series.dropna().empty else float(series.mean())


def safe_pct_true(series: pd.Series):
    s = series.dropna()
    if s.empty:
        return None
    return float(100.0 * s.mean())


def donut_score(title: str, value, max_value: float = 10.0):
    if value is None or pd.isna(value):
        value = 0.0
    value = max(0.0, min(max_value, float(value)))
    fig = go.Figure(
        data=[go.Pie(values=[value, max_value - value], hole=0.72, sort=False, textinfo="none")]
    )
    fig.update_layout(
        title={"text": title, "x": 0.5},
        showlegend=False,
        height=250,
        margin=dict(l=10, r=10, t=45, b=10),
    )
    return fig


def donut_yesno(title: str, yes_pct):
    if yes_pct is None or pd.isna(yes_pct):
        yes_pct = 0.0
    yes_pct = max(0.0, min(100.0, float(yes_pct)))
    fig = go.Figure(
        data=[go.Pie(values=[yes_pct, 100 - yes_pct], hole=0.72, sort=False, textinfo="none")]
    )
    fig.update_layout(
        title={"text": title, "x": 0.5},
        showlegend=False,
        height=250,
        margin=dict(l=10, r=10, t=45, b=10),
    )
    return fig


def rating_strip(values: pd.Series, title: str, color: str = "#1f77b4"):
    s = pd.to_numeric(values, errors="coerce").dropna()
    fig = go.Figure()

    if s.empty:
        fig.add_annotation(text="No data", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font=dict(color="gray"))
    else:
        rng = np.random.default_rng(42)
        jitter = rng.uniform(-0.15, 0.15, size=len(s))
        fig.add_trace(go.Scatter(
            x=s, y=jitter,
            mode="markers",
            marker=dict(size=14, color=color, opacity=0.65,
                        line=dict(color="white", width=1.5)),
            hovertemplate="Rating: %{x}<extra></extra>",
            showlegend=False,
        ))
        mean_val = s.mean()
        median_val = s.median()
        fig.add_vline(x=mean_val, line_dash="dash", line_color="gray",
                      annotation_text=f"avg {mean_val:.1f}",
                      annotation_position="top")
        # Median line below — robust to outliers
        fig.add_vline(x=median_val, line_dash="dot", line_color="#2c3e50",
                      annotation_text=f"med {median_val:.0f}",
                      annotation_position="bottom")

    fig.update_layout(
        title={"text": title, "x": 0.0, "font": {"size": 14}},
        xaxis=dict(range=[0.5, 10.5], tickmode="linear", tick0=1, dtick=1,
                   title=None, zeroline=False),
        yaxis=dict(range=[-0.5, 0.5], showticklabels=False, zeroline=False,
                   showgrid=False, title=None),
        height=140,
        margin=dict(l=10, r=10, t=35, b=30),
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _sentiment_score(comments: pd.Series) -> float:
    """Returns net sentiment as % (-100..+100) using POS_WORDS/NEG_WORDS.
    Returns NaN when there are no comments with any opinionated words at all
    (i.e. no signal — distinct from neutral consensus)."""
    s = comments.dropna().astype(str)
    s = s[s.str.strip() != ""]
    if s.empty:
        return float("nan")
    pos = neg = 0
    for c in s:
        flags = _classify_comment(c)
        if flags["positive"]:
            pos += 1
        if flags["negative"]:
            neg += 1
    n = pos + neg
    if n == 0:
        return float("nan")
    return 100.0 * (pos - neg) / n


def compute_course_summary(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(COURSE_COL, dropna=False)

    summary = pd.DataFrame({
        "Course": g.size().index,
        "Type": (
            g["course_type"].agg(lambda s: s.mode().iloc[0] if not s.mode().empty else "Elective").values
            if "course_type" in df.columns else "Elective"
        ),
        "n": g.size().values,
        "avg_use": g[USE_COL].mean().values if USE_COL in df.columns else np.nan,
        "med_use": g[USE_COL].median().values if USE_COL in df.columns else np.nan,
        "avg_diff": g[DIFF_COL].mean().values if DIFF_COL in df.columns else np.nan,
        "med_diff": g[DIFF_COL].median().values if DIFF_COL in df.columns else np.nan,
        "liked_pct": (g[LIKED_COL].mean().values * 100.0) if LIKED_COL in df.columns else np.nan,
    })

    # Sentiment from comments (robust extra signal beyond Liked Y/N)
    if COMMENTS_COL in df.columns:
        summary["sentiment"] = g[COMMENTS_COL].apply(_sentiment_score).values
    else:
        summary["sentiment"] = np.nan

    # Course style derived from components (with comment-text fallback)
    if "_components" in df.columns:
        def _style_for_course(course_name):
            sub = df[df[COURSE_COL] == course_name]
            counts = Counter()
            for comps in sub["_components"]:
                for c in comps:
                    counts[c] += 1
            comments = (
                sub[COMMENTS_COL].dropna().astype(str).tolist()
                if COMMENTS_COL in sub.columns else []
            )
            hints = _infer_style_from_text(comments) if comments else None
            return classify_style(dict(counts), len(sub), hints)
        summary["style"] = summary["Course"].apply(_style_for_course).values
    else:
        summary["style"] = "Unknown"

    # Stronger Bayesian shrinkage so n=1 courses don't dominate
    shrink = summary["n"] / (summary["n"] + 8.0)
    summary["value_raw"] = (
        0.45 * summary["avg_use"] +
        0.35 * (summary["liked_pct"] / 10.0) +
        0.20 * (10.0 - summary["avg_diff"])
    )
    summary["value_score"] = summary["value_raw"] * shrink

    def conf(n):
        if n >= 15:
            return "High"
        if n >= 6:
            return "Medium"
        if n >= 3:
            return "Low"
        return "Very low"

    summary["confidence"] = summary["n"].apply(conf)
    return summary


# ========== REVIEW SUMMARIZATION (HF BART) ==========
POS_WORDS = {
    "great", "excellent", "amazing", "loved", "love", "best", "fantastic",
    "wonderful", "enjoyed", "enjoy", "favorite", "useful", "helpful",
    "valuable", "practical", "interesting", "clear", "organized",
    "structured", "engaging", "fun", "solid", "rewarding", "approachable",
    "supportive", "patient", "insightful", "relevant", "applied",
    "fascinating", "thorough", "awesome", "brilliant", "perfect",
}

NEG_WORDS = {
    "bad", "terrible", "awful", "worst", "hated", "hate", "boring", "dry",
    "confusing", "unclear", "disorganized", "messy", "frustrating",
    "unfair", "harsh", "dense", "rushed", "outdated", "repetitive",
    "useless", "pointless", "waste", "overwhelming", "stressful",
    "nightmare", "disappointing", "dull", "unorganized", "vague",
    "tough", "brutal",
}

TIP_PATTERNS = [
    r"\btip\b", r"\badvice\b", r"\brecommend\b", r"\bmake sure\b",
    r"\bstart early\b", r"\bmy suggestion\b", r"\byou should\b",
    r"\bdon'?t (?:skip|miss|wait)\b", r"\boffice hours\b",
    r"\bstay on top\b", r"\bkeep up\b",
]


def _classify_comment(text: str) -> dict:
    low = text.lower()
    words = set(re.findall(r"[a-z][a-z'\-]+", low))
    pos_hits = len(words & POS_WORDS)
    neg_hits = len(words & NEG_WORDS)
    tip_hit = any(re.search(p, low) for p in TIP_PATTERNS)
    return {
        "positive": pos_hits > neg_hits and pos_hits > 0,
        "negative": neg_hits > pos_hits and neg_hits > 0,
        "tip": tip_hit,
    }


def _bucket_comments(comments):
    buckets = {"positive": [], "negative": [], "tip": []}
    for c in comments:
        flags = _classify_comment(c)
        if flags["positive"]:
            buckets["positive"].append(c)
        if flags["negative"]:
            buckets["negative"].append(c)
        if flags["tip"]:
            buckets["tip"].append(c)
    return buckets


_STOPWORDS = set(
    "the a an and or but if then so to of in for on at by with as is was are were "
    "be been being it this that these those i you he she we they me him her us them "
    "my your his hers our their have has had do does did not no yes also very just "
    "from about into over under more most some any all can will would could should "
    "there here what which who whom how when where why".split()
)


def _extractive_summary(text: str, max_sentences: int = 3, max_chars: int = 480) -> str | None:
    """Pick the most representative sentences from `text` using simple word-frequency scoring.
    Deterministic, no network, always works."""
    text = (text or "").strip()
    if not text:
        return None
    sents = re.split(r"(?<=[.!?])\s+", text)
    sents = [s.strip() for s in sents if 18 <= len(s.strip()) <= 320]
    if not sents:
        return text[:max_chars]
    if len(sents) <= max_sentences:
        return " ".join(sents)[:max_chars]

    words = re.findall(r"\b[a-z]{3,}\b", text.lower())
    freq = Counter(w for w in words if w not in _STOPWORDS)
    if not freq:
        return " ".join(sents[:max_sentences])[:max_chars]

    def score(s):
        sw = re.findall(r"\b[a-z]{3,}\b", s.lower())
        if not sw:
            return 0.0
        return sum(freq.get(w, 0) for w in sw if w not in _STOPWORDS) / len(sw)

    ranked = set(sorted(sents, key=score, reverse=True)[:max_sentences])
    # Preserve original order
    chosen = [s for s in sents if s in ranked]
    return " ".join(chosen)[:max_chars]


def _hf_chat_summarize(prompt: str, hf_token: str, timeout: int = 30):
    """Use HF Router chat completion (the current free-tier endpoint).
    Returns (text, status) where status is 'ok' | 'error: ...'."""
    if not hf_token:
        return None, "error: no token"
    try:
        import requests
        # HF Router routes to whichever provider currently serves a given model
        api_url = "https://router.huggingface.co/v1/chat/completions"
        candidate_models = [
            "meta-llama/Llama-3.2-3B-Instruct",
            "mistralai/Mistral-7B-Instruct-v0.3",
            "Qwen/Qwen2.5-7B-Instruct",
        ]
        last_err = None
        for model in candidate_models:
            try:
                resp = requests.post(
                    api_url,
                    headers={"Authorization": f"Bearer {hf_token}"},
                    json={
                        "model": model,
                        "messages": [
                            {
                                "role": "system",
                                "content": (
                                    "You write concise, neutral summaries of student course "
                                    "reviews in the style of an Amazon product review summary. "
                                    "Always start with 'Students find this course' or 'Reviewers say'. "
                                    "Synthesize the overall sentiment — do not quote a single review. "
                                    "Maximum 3 sentences. No bullet points."
                                ),
                            },
                            {"role": "user", "content": prompt[:3500]},
                        ],
                        "max_tokens": 180,
                        "temperature": 0.2,
                    },
                    timeout=timeout,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    msg = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                    if msg and msg.strip():
                        return msg.strip(), "ok"
                    last_err = f"empty response from {model}"
                else:
                    last_err = f"{model} -> HTTP {resp.status_code}: {resp.text[:160]}"
            except Exception as e:
                last_err = f"{model} -> {type(e).__name__}: {e}"
        return None, f"error: {last_err or 'all models failed'}"
    except Exception as e:
        return None, f"error: {type(e).__name__}: {e}"


def _amazon_template(stats: dict, pos_count: int, neg_count: int, tip_count: int) -> str:
    """Amazon-review-style opener built from structured form data. Always renders."""
    n = stats.get("n", 0)
    avg_use = stats.get("avg_use")
    avg_diff = stats.get("avg_diff")
    liked_pct = stats.get("liked_pct")
    sentiment = stats.get("sentiment")
    style = stats.get("style")

    def use_phrase(x):
        if x is None or pd.isna(x): return None
        if x >= 8: return "**highly useful**"
        if x >= 6.5: return "**useful**"
        if x >= 5: return "**moderately useful**"
        if x >= 3: return "of **limited usefulness**"
        return "**not very useful**"

    def diff_phrase(x):
        if x is None or pd.isna(x): return None
        if x >= 8: return "**very challenging**"
        if x >= 6.5: return "**challenging**"
        if x >= 4.5: return "**moderate** in difficulty"
        if x >= 3: return "**manageable**"
        return "**light**"

    parts = []
    use_p = use_phrase(avg_use)
    diff_p = diff_phrase(avg_diff)
    if use_p and diff_p:
        parts.append(f"Students find this course {use_p} and {diff_p}.")
    elif use_p:
        parts.append(f"Students find this course {use_p}.")
    elif diff_p:
        parts.append(f"Students find this course {diff_p}.")

    if liked_pct is not None and not pd.isna(liked_pct):
        if liked_pct >= 85:
            parts.append(f"**{int(liked_pct)}%** would recommend it — a strong consensus.")
        elif liked_pct >= 60:
            parts.append(f"**{int(liked_pct)}%** would recommend it.")
        elif liked_pct >= 35:
            parts.append(f"Only **{int(liked_pct)}%** would recommend it — reactions are mixed.")
        else:
            parts.append(f"Only **{int(liked_pct)}%** would recommend it.")

    if sentiment is not None and not pd.isna(sentiment) and (pos_count + neg_count) >= 3:
        if sentiment >= 40:
            parts.append("Written comments lean positive.")
        elif sentiment <= -40:
            parts.append("Written comments lean critical.")

    if style and style not in ("—", "Mixed"):
        parts.append(f"The course is **{style.lower()}**.")

    if tip_count >= 2:
        parts.append(f"{tip_count} reviewers leave specific tips for doing well.")

    if n == 1:
        parts.append("_Based on 1 review — interpret with caution._")
    elif n <= 3:
        parts.append(f"_Based on {n} reviews — small sample._")

    return " ".join(parts) if parts else f"_Based on {n} review{'s' if n != 1 else ''}._"


def _format_quotes(comments: list[str], max_quotes: int = 2) -> str:
    """Format a few comments as labeled quotes — used when bucket has too few comments to summarize."""
    if not comments:
        return ""
    quoted = []
    for c in comments[:max_quotes]:
        c = c.strip()
        if len(c) > 240:
            c = c[:240].rstrip() + "…"
        quoted.append(f"> {c}")
    return "\n\n".join(quoted)


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def generate_review_summaries(course: str, comments_hash: str,
                              comments_tuple: tuple, stats_tuple: tuple) -> dict:
    """Build an Amazon-style summary block.

    Always returns:
      - 'template': structured-data opener (always present)
      - 'ai':       AI-written 2-3 sentence summary if HF responds, else None
      - 'positive': either an AI-generated paragraph OR a list of raw quote strings
      - 'negative': same
      - 'tips':     same
      - 'positive_is_quotes': bool — tells UI to render as blockquotes
      - 'negative_is_quotes': bool
      - 'tips_is_quotes':     bool
      - 'errors':   list[str] for the expandable details panel
    """
    hf_token = st.secrets.get("HF_API_TOKEN") or ""
    comments = list(comments_tuple)
    stats = dict(stats_tuple)

    buckets = _bucket_comments(comments)
    pos_n, neg_n, tip_n = len(buckets["positive"]), len(buckets["negative"]), len(buckets["tip"])

    result = {
        "template": _amazon_template(stats, pos_n, neg_n, tip_n),
        "ai": None,
        "positive": None, "positive_is_quotes": False,
        "negative": None, "negative_is_quotes": False,
        "tips": None, "tips_is_quotes": False,
        "errors": [],
        "status": "ok",
    }

    # ---- AI overall summary (only attempt if HF token + enough material) ----
    if hf_token and len(comments) >= 2 and sum(len(c) for c in comments) >= 120:
        ai_prompt = (
            "Summarize the following student course reviews in 2-3 sentences, "
            "Amazon-review style. Focus on what students consistently say about the "
            "course's usefulness, difficulty, workload, professor, and what to expect.\n\n"
            "REVIEWS:\n" + "\n---\n".join(c.strip() for c in comments if c.strip())
        )
        ai_text, status = _hf_chat_summarize(ai_prompt, hf_token)
        if status == "ok":
            result["ai"] = ai_text
        else:
            result["errors"].append(f"overall: {status}")

    # ---- Per-bucket: AI-summarize if 3+ comments, else show as quotes ----
    bucket_to_key = {"positive": "positive", "negative": "negative", "tip": "tips"}
    bucket_intro = {
        "positive": "Summarize what students LOVED about this course in 1-2 sentences:",
        "negative": "Summarize the COMMON COMPLAINTS in 1-2 sentences:",
        "tip":      "Summarize the practical TIPS students give in 1-2 sentences:",
    }
    for bkey, out_key in bucket_to_key.items():
        bucket = buckets[bkey]
        if not bucket:
            continue
        if hf_token and len(bucket) >= 3:
            prompt = bucket_intro[bkey] + "\n\n" + "\n---\n".join(bucket)
            text, status = _hf_chat_summarize(prompt, hf_token)
            if status == "ok":
                result[out_key] = text
                continue
            result["errors"].append(f"{bkey}: {status}")
        # Fallback: show comments as labeled quotes (not pretend-summarize)
        result[out_key] = bucket[:2]   # raw list, UI will format
        result[out_key + "_is_quotes"] = True

    return result


def mark_breakdown(assign_text: pd.Series) -> str:
    if assign_text is None or assign_text.dropna().empty:
        return "Not provided"

    combined = " | ".join(assign_text.dropna().astype(str).tolist()).lower()
    keywords = {
        "midterm": ["midterm", "mid-term"],
        "final": ["final exam", "final"],
        "project": ["project", "capstone"],
        "assignment": ["assignment", "homework", "hw"],
        "problem set": ["problem set", "pset", "psets"],
        "quiz": ["quiz", "quizzes"],
        "reading": ["reading", "readings", "paper", "papers"],
        "presentation": ["presentation", "presentations"],
    }

    counts = Counter()
    for k, pats in keywords.items():
        for p in pats:
            counts[k] += len(re.findall(rf"\b{re.escape(p)}\b", combined))

    if sum(counts.values()) == 0:
        raw = " / ".join(assign_text.dropna().astype(str).unique().tolist())
        return raw[:220] + ("..." if len(raw) > 220 else "")

    parts = []
    for k, c in counts.most_common():
        if c <= 0:
            continue
        label = k + ("s" if c != 1 and not k.endswith("s") else "")
        parts.append(f"{c} {label}")
    return ", ".join(parts)


def _render_review_cards(df_slice: pd.DataFrame):
    if df_slice.empty:
        st.caption("_No reviews in this group._")
        return

    if "_ts" in df_slice.columns:
        df_slice = df_slice.sort_values("_ts", ascending=False)

    for _, row in df_slice.iterrows():
        sem = row.get(SEM_COL, "") or "—"
        ts = row.get(TS_COL, "") or ""
        diff = row.get(DIFF_COL, None)
        use = row.get(USE_COL, None)
        liked = row.get(LIKED_COL, None)
        comment = row.get(COMMENTS_COL, "") or ""
        term = row.get("_term", None)

        liked_str = "👍 Yes" if liked is True else ("👎 No" if liked is False else "—")
        diff_str = "—" if pd.isna(diff) else f"{diff:.0f}/10"
        use_str = "—" if pd.isna(use) else f"{use:.0f}/10"
        term_html = term_badge(term) + " " if isinstance(term, str) and term else ""

        # Possible misclick: liked=No but usefulness very high (or vice versa) AND no comment
        inconsistent = False
        if isinstance(comment, str) and not comment.strip():
            if (liked is False and not pd.isna(use) and use >= 8) or \
               (liked is True and not pd.isna(use) and use <= 3):
                inconsistent = True
        flag_html = (
            "<span style='background:#fff3cd; color:#856404; padding:2px 8px; "
            "border-radius:8px; font-size:0.7rem; font-weight:600; margin-left:6px;'>⚠ may be misclick</span>"
            if inconsistent else ""
        )

        with st.container(border=True):
            h1, h2, h3, h4 = st.columns([2.5, 1.3, 1.3, 1.3])
            h1.markdown(
                f"{term_html}<span style='color:#555;'>{sem} · {ts}</span>{flag_html}",
                unsafe_allow_html=True,
            )
            h2.markdown(f"**Useful:** {use_str}")
            h3.markdown(f"**Difficulty:** {diff_str}")
            h4.markdown(f"**Recommend:** {liked_str}")

            if comment.strip():
                st.write(comment)
            else:
                st.caption("_No written comment._")


# ---------- Load ----------
df = load_data(st.secrets["sheet_id"], st.secrets["worksheet_name"])
if df.empty:
    st.info("No responses found in this sheet/tab.")
    st.stop()

# ========== SIDEBAR ==========
st.sidebar.image("assets/dsi_council_logo.png", use_container_width=True)
st.sidebar.markdown(
    "<div style='text-align:center; margin-top:-6px; margin-bottom:14px;'>"
    "<a href='https://docs.google.com/forms/d/e/1FAIpQLScfC3rpdgzNHFrP0D287wZ1N704PdnRB84mr_mqnxiqzkMhJA/viewform' "
    "target='_blank' rel='noopener' "
    "style='display:inline-block; background:#012169; color:#fff; padding:7px 14px; "
    "border-radius:6px; text-decoration:none; font-weight:600; font-size:0.85rem;'>"
    "📝 Submit an Evaluation</a></div>",
    unsafe_allow_html=True,
)
st.sidebar.divider()
st.sidebar.header("Filters")

course_search = st.sidebar.text_input(
    "Search course name",
    placeholder="e.g. machine learning, 4721",
).strip().lower()

_global_prof_map = build_prof_display_map(df[PROF_COL]) if PROF_COL in df.columns else {}
all_profs = sorted(set(_global_prof_map.values()))

prof_filter = st.sidebar.multiselect("Professor", all_profs, default=[])

course_type_filter = st.sidebar.multiselect(
    "Course Type",
    ["Core", "Elective"],
    default=["Core", "Elective"],
)

all_terms = sorted(
    [t for t in df.get("_term", pd.Series(dtype=str)).dropna().unique().tolist() if t]
)
term_filter = st.sidebar.multiselect("Term", all_terms, default=[])

all_years = sorted(
    [int(y) for y in df.get("_year", pd.Series(dtype=float)).dropna().unique().tolist()]
)
year_filter = st.sidebar.multiselect("Year", all_years, default=[])

recent_only = st.sidebar.checkbox(
    "Recent reviews only (last 2 years)",
    value=False,
    help="Hide reviews older than 2 years — useful if a course's professor or curriculum changed.",
)

component_filter = st.sidebar.multiselect(
    "Must have components",
    KNOWN_COMPONENTS,
    default=[],
    help="Show only courses where reviews mention ALL selected components.",
)

st.sidebar.divider()
st.sidebar.subheader("Ranking weights")

# ---- Preset profiles ----
PRESETS = {
    "Easy semester":  {"w_use": 0.10, "w_like": 0.30, "w_ease": 0.60},
    "High-ROI":       {"w_use": 0.70, "w_like": 0.10, "w_ease": 0.20},
    "Just enjoyable": {"w_use": 0.20, "w_like": 0.70, "w_ease": 0.10},
    "Balanced":       {"w_use": 0.45, "w_like": 0.35, "w_ease": 0.20},
}

for _k, _v in PRESETS["Balanced"].items():
    st.session_state.setdefault(_k, _v)

st.sidebar.caption("Quick preset (overrides sliders):")
pcols = st.sidebar.columns(2)
preset_labels = list(PRESETS.keys())
for i, label in enumerate(preset_labels):
    if pcols[i % 2].button(label, key=f"preset_{label}", use_container_width=True):
        for k, v in PRESETS[label].items():
            st.session_state[k] = v
        st.rerun()

w_use = st.sidebar.slider("Usefulness importance", 0.0, 1.0, key="w_use", step=0.05)
w_like = st.sidebar.slider("Liked importance", 0.0, 1.0, key="w_like", step=0.05)
w_ease = st.sidebar.slider("Ease importance (10 - difficulty)", 0.0, 1.0, key="w_ease", step=0.05)
min_reviews = st.sidebar.slider("Minimum # of reviews", 1, int(max(1, df.groupby(COURSE_COL).size().max())), 1)

w_sum = w_use + w_like + w_ease
if w_sum == 0:
    w_use, w_like, w_ease = 0.45, 0.35, 0.20
else:
    w_use, w_like, w_ease = w_use / w_sum, w_like / w_sum, w_ease / w_sum

filtered = df.copy()

if course_search and COURSE_COL in filtered.columns:
    filtered = filtered[filtered[COURSE_COL].astype(str).str.lower().str.contains(course_search, na=False)]

if prof_filter and PROF_COL in filtered.columns:
    # Translate selected display names back to canonical keys via the prof map
    selected_canon = set()
    for p in prof_filter:
        # find any raw key whose display name == p
        for raw_key, display in _global_prof_map.items():
            if display == p:
                # canonical key is whatever raw_key maps to in canon column
                rows = filtered[filtered["_prof_key"] == raw_key]
                if not rows.empty:
                    selected_canon.update(rows["_prof_canon"].unique())
    if selected_canon:
        filtered = filtered[filtered["_prof_canon"].isin(selected_canon)]

if course_type_filter and "course_type" in filtered.columns:
    filtered = filtered[filtered["course_type"].isin(course_type_filter)]
if term_filter and "_term" in filtered.columns:
    filtered = filtered[filtered["_term"].isin(term_filter)]
if year_filter and "_year" in filtered.columns:
    filtered = filtered[filtered["_year"].isin(year_filter)]
if recent_only and "_year" in filtered.columns:
    cutoff = datetime.now().year - 1
    filtered = filtered[filtered["_year"].fillna(0) >= cutoff]
if component_filter and "_components" in filtered.columns and COURSE_COL in filtered.columns:
    needed = set(component_filter)
    course_comp_union = (
        filtered.groupby(COURSE_COL)["_components"]
                .agg(lambda s: set().union(*s) if len(s) else set())
    )
    courses_with_all = course_comp_union[course_comp_union.apply(lambda c: needed.issubset(c))].index
    filtered = filtered[filtered[COURSE_COL].isin(courses_with_all)]

# ========== MAIN ==========
st.title("Course Decision Dashboard")
st.caption("Built from student responses (auto-refreshes). Use filters in the sidebar.")

tab_overview, tab_deep, tab_compare = st.tabs(
    ["📊 Overview (Rankings)", "🔍 Course Deep Dive", "⚖️ Compare Courses"]
)


# ---------- OVERVIEW ----------
with tab_overview:
    summary = compute_course_summary(filtered)
    summary = summary[summary["n"] >= min_reviews].copy()

    if summary.empty:
        st.info("No courses match your current filters.")
    else:
        shrink = summary["n"] / (summary["n"] + 5.0)
        summary["value_raw"] = (
            w_use * summary["avg_use"] +
            w_like * (summary["liked_pct"] / 10.0) +
            w_ease * (10.0 - summary["avg_diff"])
        )
        summary["value_score"] = summary["value_raw"] * shrink
        summary["ease"] = 10.0 - summary["avg_diff"]

        st.subheader("Rankings (personalized)")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            total_courses = summary["Course"].nunique()
            core_courses = int((summary["Type"] == "Core").sum()) if "Type" in summary.columns else 0
            elective_courses = int((summary["Type"] == "Elective").sum()) if "Type" in summary.columns else 0
            st.metric("Courses", total_courses, help=f"Core: {core_courses} | Elective: {elective_courses}")
        with c2:
            st.metric("Total reviews", int(summary["n"].sum()))
        with c3:
            st.metric("Avg usefulness (all)", f"{summary['avg_use'].mean():.2f}" if summary["avg_use"].notna().any() else "—")
        with c4:
            st.metric("Avg difficulty (all)", f"{summary['avg_diff'].mean():.2f}" if summary["avg_diff"].notna().any() else "—")

        st.write(
            f"Ranking weights: **Usefulness {w_use:.2f}**, **Liked {w_like:.2f}**, **Ease {w_ease:.2f}**. "
            f"Minimum reviews: **{min_reviews}**."
        )

        show = summary.sort_values("value_score", ascending=False).copy()
        show["avg_use_disp"] = show.apply(
            lambda r: "—" if pd.isna(r["avg_use"]) else f"{r['avg_use']:.1f} (med {r['med_use']:.0f})",
            axis=1,
        )
        show["avg_diff_disp"] = show.apply(
            lambda r: "—" if pd.isna(r["avg_diff"]) else f"{r['avg_diff']:.1f} (med {r['med_diff']:.0f})",
            axis=1,
        )
        show["liked_disp"] = show["liked_pct"].map(lambda x: "—" if pd.isna(x) else f"{x:.0f}%")
        show["value_disp"] = show["value_score"].map(lambda x: "—" if pd.isna(x) else f"{x:.2f}")
        show["sent_disp"] = show["sentiment"].map(
            lambda x: "—" if pd.isna(x) else (f"+{x:.0f}" if x >= 0 else f"{x:.0f}")
        )

        # Rename for display
        show_disp = show.rename(columns={
            "avg_use_disp": "Useful",
            "avg_diff_disp": "Difficult",
            "liked_disp": "Liked",
            "value_disp": "Score",
            "sent_disp": "Sentiment",
            "style": "Style",
        })

        display_cols = ["Course", "Type", "Style", "n", "Useful", "Difficult", "Liked", "Sentiment", "Score", "confidence"]

        def _row_style(row):
            styles = [""] * len(row)
            cols = list(row.index)
            type_idx = cols.index("Type")
            style_idx = cols.index("Style")
            sent_idx = cols.index("Sentiment")
            conf_idx = cols.index("confidence")

            type_color = TYPE_COLORS.get(row["Type"], "#888")
            styles[type_idx] = f"background-color: {type_color}; color: white; font-weight:600; text-align:center;"

            style_color = STYLE_COLORS.get(row["Style"], "#7f8c8d")
            styles[style_idx] = f"background-color: {style_color}; color: white; font-weight:600; text-align:center;"

            # Sentiment: green if positive, red if negative
            sent_raw = row["Sentiment"]
            if isinstance(sent_raw, str) and sent_raw != "—":
                val = int(sent_raw.replace("+", ""))
                if val >= 20:
                    styles[sent_idx] = "background-color: #d4edda; color: #155724; font-weight:600;"
                elif val <= -20:
                    styles[sent_idx] = "background-color: #f8d7da; color: #721c24; font-weight:600;"

            # Confidence: yellow for Low/Very low
            conf_val = row["confidence"]
            if conf_val == "Very low":
                styles[conf_idx] = "background-color: #fff3cd; color: #856404; font-weight:600;"
            elif conf_val == "Low":
                styles[conf_idx] = "background-color: #fff8e1; color: #8a6d3b;"

            return styles

        styler = (
            show_disp[display_cols]
            .style.apply(_row_style, axis=1)
        )
        st.dataframe(
            styler,
            use_container_width=True,
            column_config={
                "Course": st.column_config.TextColumn(
                    "Course", help="Course code and name as submitted in the form."),
                "Type": st.column_config.TextColumn(
                    "Type", help="Core (required for the program) vs Elective."),
                "Style": st.column_config.TextColumn(
                    "Style",
                    help=(
                        "How the course is assessed, derived from the 'assignments' field "
                        "(plus comment-text fallback when checkboxes are blank). "
                        "Exam-driven / Project-driven / Mixed / Problem-set-heavy / Reading-heavy / Unknown."
                    ),
                ),
                "n": st.column_config.NumberColumn(
                    "n", help="Number of student reviews this course has received."),
                "Useful": st.column_config.TextColumn(
                    "Useful",
                    help=(
                        "Average usefulness rating (1-10). Median in parens — if mean and "
                        "median disagree, a few outliers are pulling the average."
                    ),
                ),
                "Difficult": st.column_config.TextColumn(
                    "Difficult",
                    help="Average difficulty (1-10). Median in parens. 10 = very hard.",
                ),
                "Liked": st.column_config.TextColumn(
                    "Liked",
                    help="% of reviewers who answered 'Yes' to 'Did you like this class?'",
                ),
                "Sentiment": st.column_config.TextColumn(
                    "Sentiment",
                    help=(
                        "Net positive vs negative tone in the written comments, -100 to +100. "
                        "Counts comments with positive words (great, useful, loved...) vs negative "
                        "(confusing, harsh, frustrating...). '—' means no opinionated words were "
                        "found in any comment (no signal — NOT neutral consensus)."
                    ),
                ),
                "Score": st.column_config.TextColumn(
                    "Score",
                    help=(
                        "Personalized value score using the weights in the sidebar. Bayesian-shrunk: "
                        "low-n courses are pulled toward the average so 1-review courses don't dominate."
                    ),
                ),
                "confidence": st.column_config.TextColumn(
                    "Confidence",
                    help=(
                        "How much to trust this row's averages, based purely on review count. "
                        "n≥15: High · n≥6: Medium · n≥3: Low · n<3: Very low. "
                        "Most courses will be 'Very low' until more reviews come in."
                    ),
                ),
            },
        )

        # ---- Bubble chart (replaces the bar plot) ----
        st.subheader("Where each course sits")
        st.caption(
            "Each circle is a course. **Right** = more useful. **Up** = easier. "
            "**Bigger circle** = more reviews. **Greener** = more students liked it."
        )
        bubble_df = summary.dropna(subset=["avg_use", "avg_diff"]).copy()
        if bubble_df.empty:
            st.info("Not enough data to draw the chart.")
        else:
            fig = px.scatter(
                bubble_df,
                x="avg_use",
                y="ease",
                size="n",
                color="liked_pct",
                color_continuous_scale="RdYlGn",
                range_color=[0, 100],
                hover_name="Course",
                hover_data={
                    "avg_use": ":.2f",
                    "ease": ":.2f",
                    "avg_diff": ":.2f",
                    "liked_pct": ":.0f",
                    "n": True,
                    "Type": True,
                },
                size_max=55,
                text="Course",
                labels={
                    "avg_use": "Usefulness (1–10)",
                    "ease": "Ease (10 − difficulty)",
                    "liked_pct": "Liked %",
                    "n": "# reviews",
                },
            )
            fig.update_traces(
                textposition="top center",
                textfont=dict(size=10),
                marker=dict(line=dict(width=1, color="white"), opacity=0.85),
            )
            fig.update_layout(
                height=600,
                margin=dict(l=20, r=20, t=30, b=20),
                xaxis=dict(range=[0, 10.5], dtick=1),
                yaxis=dict(range=[0, 10.5], dtick=1),
            )
            # Add faint "best quadrant" guide line (top-right = useful AND easy)
            fig.add_hline(y=5, line_dash="dot", line_color="lightgray", opacity=0.5)
            fig.add_vline(x=5, line_dash="dot", line_color="lightgray", opacity=0.5)
            st.plotly_chart(fig, use_container_width=True)


# ---------- COURSE DEEP DIVE ----------
with tab_deep:
    course_options = sorted(filtered[COURSE_COL].dropna().unique().tolist())
    if not course_options:
        st.info("No courses match your filters. Try loosening them in the sidebar.")
    else:
        course = st.selectbox("Select a course", course_options)
        _ctype = classify_course_type(course)

        f = filtered[filtered[COURSE_COL] == course].copy()
        n = len(f)

        # Course style chip from components (with comment-text fallback)
        comp_counts = Counter()
        for comps in f.get("_components", []):
            for c in comps:
                comp_counts[c] += 1
        comments_for_hints = (
            f[COMMENTS_COL].dropna().astype(str).tolist()
            if COMMENTS_COL in f.columns else []
        )
        _style_hints = _infer_style_from_text(comments_for_hints) if comments_for_hints else None
        _style = classify_style(dict(comp_counts), n, _style_hints)

        st.markdown(
            f"<div style='margin: 4px 0 10px 0;'>"
            f"Type: {type_badge(_ctype)} &nbsp; "
            f"Style: {style_badge(_style)}"
            f"</div>",
            unsafe_allow_html=True,
        )

        use_series = pd.to_numeric(f.get(USE_COL, pd.Series(dtype=float)), errors="coerce").dropna()
        diff_series = pd.to_numeric(f.get(DIFF_COL, pd.Series(dtype=float)), errors="coerce").dropna()
        avg_use = float(use_series.mean()) if not use_series.empty else None
        med_use = float(use_series.median()) if not use_series.empty else None
        avg_diff = float(diff_series.mean()) if not diff_series.empty else None
        med_diff = float(diff_series.median()) if not diff_series.empty else None
        liked_pct = safe_pct_true(f.get(LIKED_COL, pd.Series(dtype=bool)))

        # Low-n caveat banner
        if n <= 2:
            st.warning(
                f"⚠️ **Only {n} review{'s' if n != 1 else ''}** — a single misclick can swing these "
                f"numbers a lot. Use median (below mean) as a sanity check, and read the comments."
            )
        elif n < 5:
            st.info(f"Only {n} reviews — interpret averages cautiously.")

        k1, k2, k3, k4 = st.columns([1, 1, 1, 2])
        with k1:
            st.metric("Reviews", n)
        with k2:
            st.metric(
                "Usefulness", "—" if avg_use is None else f"{avg_use:.1f} / 10",
                help=f"Median: {med_use:.0f} / 10" if med_use is not None else None,
            )
        with k3:
            st.metric(
                "Difficulty", "—" if avg_diff is None else f"{avg_diff:.1f} / 10",
                help=f"Median: {med_diff:.0f} / 10" if med_diff is not None else None,
            )
        with k4:
            st.metric("Liked", "—" if liked_pct is None else f"{liked_pct:.0f}%")

        d1, d2, d3 = st.columns(3)
        with d1:
            st.plotly_chart(donut_score("Usefulness", avg_use), use_container_width=True)
        with d2:
            st.plotly_chart(donut_score("Difficulty", avg_diff), use_container_width=True)
        with d3:
            st.plotly_chart(donut_yesno("Liked", liked_pct), use_container_width=True)

        st.subheader("Where every review falls")
        st.caption("Each dot is one student's rating. Dashed line is the average.")
        sp1, sp2 = st.columns(2)
        with sp1:
            if USE_COL in f.columns:
                st.plotly_chart(
                    rating_strip(f[USE_COL], "Usefulness (1–10)", color="#2e86de"),
                    use_container_width=True
                )
        with sp2:
            if DIFF_COL in f.columns:
                st.plotly_chart(
                    rating_strip(f[DIFF_COL], "Difficulty (1–10)", color="#e67e22"),
                    use_container_width=True
                )

        # Course component chips (% of reviewers who mentioned each)
        st.markdown("**Course components**")
        if "_components" in f.columns and len(f) > 0:
            comp_counts = Counter()
            for comps in f["_components"]:
                for c in comps:
                    comp_counts[c] += 1
            total = len(f)
            if comp_counts:
                chips_html = "".join(
                    component_chip(label, pct=int(round(100 * cnt / total)))
                    for label, cnt in comp_counts.most_common()
                )
                st.markdown(f"<div>{chips_html}</div>", unsafe_allow_html=True)
            else:
                st.caption("_Not provided._")
        st.caption(
            "Raw: " + mark_breakdown(f.get(ASSIGN_COL, pd.Series(dtype=str)))
        )

        st.divider()

        # ---- REVIEWS SECTION ----
        st.subheader(f"Reviews ({n})")

        # ---- Pinned: most helpful review ----
        _tip_re = re.compile("|".join(TIP_PATTERNS), re.IGNORECASE)
        candidates = f.copy()
        if COMMENTS_COL in candidates.columns:
            candidates["_comment_str"] = candidates[COMMENTS_COL].fillna("").astype(str)
            candidates["_has_tip"] = candidates["_comment_str"].apply(lambda c: bool(_tip_re.search(c)))
            candidates["_clen"] = candidates["_comment_str"].str.len()
            with_tip = candidates[candidates["_has_tip"] & (candidates["_clen"] > 80)]
            pool = with_tip if not with_tip.empty else candidates[candidates["_clen"] > 120]
            if not pool.empty:
                top = pool.sort_values("_clen", ascending=False).iloc[0]
                top_comment = top["_comment_str"]
                top_prof = top.get(PROF_COL, "") or "—"
                top_sem = top.get(SEM_COL, "") or "—"
                top_use = top.get(USE_COL, None)
                top_diff = top.get(DIFF_COL, None)
                use_s = "—" if pd.isna(top_use) else f"{top_use:.0f}/10"
                diff_s = "—" if pd.isna(top_diff) else f"{top_diff:.0f}/10"
                with st.container(border=True):
                    st.markdown(
                        "<span style='background:#27ae60; color:#fff; padding:2px 10px; "
                        "border-radius:10px; font-size:0.75rem; font-weight:600;'>⭐ Most helpful review</span> "
                        f"<span style='color:#666; font-size:0.85rem;'>· Prof. {top_prof} · {top_sem} · "
                        f"Useful {use_s} · Difficulty {diff_s}</span>",
                        unsafe_allow_html=True,
                    )
                    st.write(top_comment)

        with st.container(border=True):
            st.markdown("**Filter reviews**")
            fc1, fc2, fc3, fc4 = st.columns([1.2, 1.5, 1.5, 2])
            with fc1:
                recommend_filter = st.radio(
                    "Recommend", ["All", "Yes", "No"], horizontal=True, key=f"rec_{course}"
                )
            with fc2:
                min_use = st.slider("Min usefulness", 1, 10, 1, key=f"use_{course}")
            with fc3:
                max_diff = st.slider("Max difficulty", 1, 10, 10, key=f"diff_{course}")
            with fc4:
                keyword = st.text_input(
                    "Search in comments", placeholder="e.g. project, exam, interview",
                    key=f"kw_{course}"
                ).strip().lower()

        rf = f.copy()
        if recommend_filter == "Yes" and LIKED_COL in rf.columns:
            rf = rf[rf[LIKED_COL] == True]
        elif recommend_filter == "No" and LIKED_COL in rf.columns:
            rf = rf[rf[LIKED_COL] == False]
        if min_use > 1 and USE_COL in rf.columns:
            rf = rf[rf[USE_COL].fillna(10) >= min_use]
        if max_diff < 10 and DIFF_COL in rf.columns:
            rf = rf[rf[DIFF_COL].fillna(0) <= max_diff]
        if keyword and COMMENTS_COL in rf.columns:
            rf = rf[rf[COMMENTS_COL].fillna("").str.lower().str.contains(keyword, na=False)]

        filtered_n = len(rf)
        if filtered_n != n:
            st.caption(f"Showing **{filtered_n}** of {n} reviews based on filters.")

        # ---- Summary ----
        comments_series = rf.get(COMMENTS_COL, pd.Series(dtype=str)).dropna().astype(str)
        comments_series = comments_series[comments_series.str.strip() != ""]

        # Always render the template summary, even with 0 comments
        rf_use = pd.to_numeric(rf.get(USE_COL, pd.Series(dtype=float)), errors="coerce").dropna()
        rf_diff = pd.to_numeric(rf.get(DIFF_COL, pd.Series(dtype=float)), errors="coerce").dropna()
        rf_liked = rf.get(LIKED_COL, pd.Series(dtype=bool)).dropna()
        stats_for_summary = {
            "n": len(rf),
            "avg_use": float(rf_use.mean()) if not rf_use.empty else None,
            "avg_diff": float(rf_diff.mean()) if not rf_diff.empty else None,
            "liked_pct": float(100.0 * rf_liked.mean()) if not rf_liked.empty else None,
            "sentiment": _sentiment_score(comments_series),
            "style": _style,
        }

        comments_list = comments_series.tolist()
        comments_hash = hashlib.md5(
            ("||".join(sorted(comments_list)) + str(stats_for_summary)).encode("utf-8")
        ).hexdigest()

        with st.spinner("Generating summary..."):
            summaries = generate_review_summaries(
                course, comments_hash,
                tuple(comments_list),
                tuple(stats_for_summary.items()),
            )

        # ---- Render: template always; AI on top if available ----
        with st.container(border=True):
            st.markdown("### 📝 Summary of student reviews")
            if summaries.get("ai"):
                st.write(summaries["ai"])
                st.caption(summaries["template"])
            else:
                st.write(summaries["template"])
                if not st.secrets.get("HF_API_TOKEN"):
                    st.caption("_Add HF_API_TOKEN to Streamlit secrets for AI-written summaries._")
                elif comments_list:
                    st.caption("_AI summary unavailable — based on the structured ratings above._")

        # ---- Per-bucket: AI paragraph OR labeled quotes ----
        has_breakdown = any([summaries.get("positive"),
                             summaries.get("negative"),
                             summaries.get("tips")])
        if has_breakdown:
            s1, s2, s3 = st.columns(3)

            def _render_bucket(col, header, key):
                with col:
                    st.markdown(header)
                    val = summaries.get(key)
                    is_quotes = summaries.get(f"{key}_is_quotes", False)
                    if not val:
                        empty_msg = {
                            "positive": "_Nothing clearly positive flagged._",
                            "negative": "_Nothing clearly negative flagged._",
                            "tips":     "_No explicit tips found._",
                        }[key]
                        st.caption(empty_msg)
                        return
                    if is_quotes:
                        st.caption("_Direct quotes from reviewers:_")
                        for q in val:
                            q_short = q.strip()
                            if len(q_short) > 240:
                                q_short = q_short[:240].rstrip() + "…"
                            st.markdown(f"> {q_short}")
                    else:
                        st.write(val)

            _render_bucket(s1, "#### 👍 What students loved", "positive")
            _render_bucket(s2, "#### 👎 Common complaints", "negative")
            _render_bucket(s3, "#### 💡 Tips for success", "tips")

        # Error details (optional, only if anything failed)
        errs = summaries.get("errors", [])
        if errs:
            with st.expander("AI service notes (debug)"):
                for e in errs:
                    st.code(e)

        st.divider()

        # ---- REVIEW CARDS GROUPED BY PROFESSOR ----
        if filtered_n == 0:
            st.info("No reviews match your filters.")
        else:
            rf = rf.copy()
            # Use the canonical key (fuzzy-merged) so "Drinea" and "Drea" collapse
            rf["_prof_group"] = rf.get("_prof_canon", rf[PROF_COL].apply(_prof_normalize_key))
            rf["_prof_group"] = rf["_prof_group"].replace("", "— unknown —")

            display_map = {}
            for key, grp in rf.groupby("_prof_group"):
                if key == "— unknown —":
                    display_map[key] = "— Unknown —"
                else:
                    names = grp[PROF_COL].dropna().astype(str).str.strip()
                    names = names[names != ""]
                    if len(names) > 0:
                        display_map[key] = names.value_counts().idxmax()
                    else:
                        display_map[key] = "— Unknown —"

            prof_order_keys = (
                rf.groupby("_prof_group").size()
                  .sort_values(ascending=False)
                  .index.tolist()
            )

            # Color-code each prof tab with a deterministic colored circle prefix
            TAB_COLORS = ["🟦", "🟧", "🟩", "🟪", "🟥", "🟨", "🟫", "⬛", "⬜"]
            def _tab_emoji(key: str) -> str:
                if key == "— unknown —":
                    return "⬜"
                idx = int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16) % len(TAB_COLORS)
                return TAB_COLORS[idx]

            tab_labels = [
                f"{_tab_emoji(k)} {display_map[k]} ({rf[rf['_prof_group'] == k].shape[0]})"
                for k in prof_order_keys
            ]
            prof_tabs = st.tabs(tab_labels)
            for prof_tab, key in zip(prof_tabs, prof_order_keys):
                with prof_tab:
                    _render_review_cards(rf[rf["_prof_group"] == key])


# ---------- COMPARE COURSES ----------
with tab_compare:
    st.subheader("Compare courses side-by-side")
    course_options = sorted(filtered[COURSE_COL].dropna().unique().tolist())

    if len(course_options) < 2:
        st.info("Need at least 2 courses matching your filters to compare.")
    else:
        default_sel = course_options[:2] if len(course_options) >= 2 else course_options
        selected = st.multiselect(
            "Pick 2–4 courses to compare", course_options, default=default_sel,
            max_selections=4,
        )
        if len(selected) < 2:
            st.info("Pick at least 2 courses.")
        else:
            comp = filtered[filtered[COURSE_COL].isin(selected)].copy()
            summary = compute_course_summary(comp).set_index("Course").loc[selected].reset_index()

            # ---- Quick-read metric cards at top ----
            st.markdown("#### At a glance")
            cols = st.columns(len(selected))
            for i, course in enumerate(selected):
                row = summary[summary["Course"] == course].iloc[0]
                with cols[i]:
                    with st.container(border=True):
                        st.markdown(f"**{course}**")
                        st.caption(f"{classify_course_type(course)} · {int(row['n'])} reviews · {row['confidence']} confidence")
                        st.metric("Usefulness", f"{row['avg_use']:.1f}/10" if not pd.isna(row['avg_use']) else "—")
                        st.metric("Difficulty", f"{row['avg_diff']:.1f}/10" if not pd.isna(row['avg_diff']) else "—")
                        st.metric("Liked", f"{row['liked_pct']:.0f}%" if not pd.isna(row['liked_pct']) else "—")

            st.divider()

            # ---- Grouped bar comparison (replaces oversized donut grids) ----
            st.markdown("#### 🎯 Head-to-head metrics")
            st.caption(
                "Horizontal bars on the same 0–10 scale. Liked % is shown out of 10 "
                "(e.g. 80% → 8) so all three metrics line up."
            )
            bar_df = summary.copy()
            bar_df["Liked (0–10)"] = bar_df["liked_pct"] / 10.0
            bar_long = bar_df.melt(
                id_vars=["Course"],
                value_vars=["avg_use", "avg_diff", "Liked (0–10)"],
                var_name="Metric",
                value_name="Score",
            )
            metric_label = {
                "avg_use": "Usefulness",
                "avg_diff": "Difficulty",
                "Liked (0–10)": "Liked",
            }
            bar_long["Metric"] = bar_long["Metric"].map(metric_label)
            bar_fig = px.bar(
                bar_long,
                x="Score", y="Course", color="Metric",
                orientation="h", barmode="group",
                range_x=[0, 10],
                color_discrete_map={
                    "Usefulness": "#2E86DE",
                    "Difficulty": "#E67E22",
                    "Liked": "#27AE60",
                },
                text=bar_long["Score"].map(lambda v: "—" if pd.isna(v) else f"{v:.1f}"),
            )
            bar_fig.update_traces(textposition="outside", cliponaxis=False)
            bar_fig.update_layout(
                height=max(220, 90 * len(selected) + 80),
                margin=dict(l=20, r=20, t=20, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(bar_fig, use_container_width=True)

            st.divider()

            # ---- Radar / spider chart (the circle layers one) ----
            st.markdown("#### 🕸️ Overall shape comparison")
            st.caption("Each course is a colored shape. Further from the center = better. Overlap shows similarities.")

            radar_df = summary.copy()
            radar_df["liked_10"] = radar_df["liked_pct"] / 10.0
            radar_df["ease"] = 10.0 - radar_df["avg_diff"]
            cat_names = ["Usefulness", "Liked", "Ease (10-difficulty)"]

            radar_fig = go.Figure()
            for _, r in radar_df.iterrows():
                vals = [r["avg_use"], r["liked_10"], r["ease"]]
                radar_fig.add_trace(go.Scatterpolar(
                    r=vals + [vals[0]],
                    theta=cat_names + [cat_names[0]],
                    fill="toself",
                    name=r["Course"],
                ))

            radar_fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
                showlegend=True,
                height=500,
                margin=dict(l=20, r=20, t=30, b=20),
            )
            st.plotly_chart(radar_fig, use_container_width=True)

            st.divider()

            # ---- Recent comments per course ----
            st.subheader("Recent comments")
            for course in selected:
                fc = comp[comp[COURSE_COL] == course].copy()
                if "_ts" in fc.columns:
                    fc = fc.sort_values("_ts", ascending=False)

                comments = fc.get(COMMENTS_COL, pd.Series(dtype=str)).dropna().astype(str)
                comments = comments[comments.str.strip() != ""]

                st.markdown(f"**{course}**")
                if comments.empty:
                    st.caption("_No comments._")
                else:
                    for c in comments.head(3).tolist():
                        st.write(f"• {c}")
                st.write("")