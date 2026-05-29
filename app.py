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


# ---------- Professor name normalization ----------
_PROF_PREFIX_RE = re.compile(r"^\s*(prof\.?|professor|dr\.?|mr\.?|ms\.?|mrs\.?)\s+", re.IGNORECASE)


def _prof_normalize_key(name: str) -> str:
    if not isinstance(name, str):
        return ""
    s = name.strip()
    s = _PROF_PREFIX_RE.sub("", s)
    s = re.sub(r"\s+", " ", s)
    return s.lower()


def build_prof_display_map(prof_series: pd.Series) -> dict:
    s = prof_series.dropna().astype(str).str.strip()
    s = s[s != ""]
    if s.empty:
        return {}
    df = pd.DataFrame({"orig": s.values})
    df["key"] = df["orig"].apply(_prof_normalize_key)
    mapping = (
        df.groupby("key")["orig"]
          .agg(lambda x: x.value_counts().idxmax())
          .to_dict()
    )
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

    if COURSE_COL in df.columns:
        df["course_type"] = df[COURSE_COL].astype(str).map(classify_course_type)
    else:
        df["course_type"] = "Elective"

    if PROF_COL in df.columns:
        df["_prof_key"] = df[PROF_COL].apply(_prof_normalize_key)
    else:
        df["_prof_key"] = ""

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
        fig.add_vline(x=mean_val, line_dash="dash", line_color="gray",
                      annotation_text=f"avg {mean_val:.1f}",
                      annotation_position="top")

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
        "avg_diff": g[DIFF_COL].mean().values if DIFF_COL in df.columns else np.nan,
        "liked_pct": (g[LIKED_COL].mean().values * 100.0) if LIKED_COL in df.columns else np.nan,
    })

    shrink = summary["n"] / (summary["n"] + 5.0)
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
        return "Low"

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


def _hf_summarize(text: str, hf_token: str,
                  max_length: int = 120, min_length: int = 30,
                  timeout: int = 60):
    if not text.strip():
        return None, "empty"

    if len(text) > 3500:
        text = text[:3500]

    try:
        from huggingface_hub import InferenceClient
    except ImportError:
        print("[HF summarize error] huggingface_hub not installed. Run: pip install huggingface_hub")
        return None, "error"

    try:
        client = InferenceClient(
            provider="hf-inference",
            api_key=hf_token,
            timeout=timeout,
        )
        result = client.summarization(
            text,
            model="facebook/bart-large-cnn",
            generate_parameters={
                "max_length": max_length,
                "min_length": min_length,
                "do_sample": False,
            },
        )
        summary = (
            getattr(result, "generated_text", None)
            or getattr(result, "summary_text", None)
        )
        if not summary and isinstance(result, dict):
            summary = result.get("generated_text") or result.get("summary_text")
        if summary:
            return summary.strip(), "ok"
        print(f"[HF summarize error] Unexpected result shape: {type(result).__name__} -> {result!r}")
        return None, "error"
    except Exception as e:
        print(f"[HF summarize error] {type(e).__name__}: {e}")
        return None, "error"


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def generate_review_summaries(course: str, comments_hash: str, comments_tuple: tuple) -> dict:
    hf_token = st.secrets.get("HF_API_TOKEN")
    if not hf_token:
        return {"status": "no_token", "overall": None,
                "positive": None, "negative": None, "tips": None}

    comments = list(comments_tuple)
    result = {"overall": None, "positive": None,
              "negative": None, "tips": None, "status": "ok"}

    all_text = " ".join(comments)
    overall, status = _hf_summarize(all_text, hf_token, max_length=140, min_length=50)
    if status == "ok":
        result["overall"] = overall

    buckets = _bucket_comments(comments)
    for key in ["positive", "negative", "tip"]:
        bucket = buckets[key]
        if len(bucket) == 0:
            continue
        joined = " ".join(bucket)
        summary, sub_status = _hf_summarize(joined, hf_token)
        if sub_status == "ok":
            out_key = "tips" if key == "tip" else key
            result[out_key] = summary

    if not any([result["overall"], result["positive"],
                result["negative"], result["tips"]]):
        result["status"] = "all_failed"

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

        liked_str = "👍 Yes" if liked is True else ("👎 No" if liked is False else "—")
        diff_str = "—" if pd.isna(diff) else f"{diff:.0f}/10"
        use_str = "—" if pd.isna(use) else f"{use:.0f}/10"

        with st.container(border=True):
            h1, h2, h3, h4 = st.columns([2.5, 1.3, 1.3, 1.3])
            h1.caption(f"**{sem}** · {ts}")
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
st.sidebar.header("Filters")

course_search = st.sidebar.text_input(
    "Search course name",
    placeholder="e.g. machine learning, 4721",
).strip().lower()

_global_prof_map = build_prof_display_map(df[PROF_COL]) if PROF_COL in df.columns else {}
all_profs = sorted(set(_global_prof_map.values()))
all_sems = sorted([s for s in df.get(SEM_COL, pd.Series(dtype=str)).dropna().unique().tolist() if s])

prof_filter = st.sidebar.multiselect("Professor", all_profs, default=[])
sem_filter = st.sidebar.multiselect("Semester", all_sems, default=[])

course_type_filter = st.sidebar.multiselect(
    "Course Type",
    ["Core", "Elective"],
    default=["Core", "Elective"],
)

st.sidebar.divider()
st.sidebar.subheader("Overview sliders (Rankings)")
w_use = st.sidebar.slider("Usefulness importance", 0.0, 1.0, 0.45, 0.05)
w_like = st.sidebar.slider("Liked importance", 0.0, 1.0, 0.35, 0.05)
w_ease = st.sidebar.slider("Ease importance (10 - difficulty)", 0.0, 1.0, 0.20, 0.05)
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
    selected_keys = {_prof_normalize_key(p) for p in prof_filter}
    filtered = filtered[filtered["_prof_key"].isin(selected_keys)]

if sem_filter and SEM_COL in filtered.columns:
    filtered = filtered[filtered[SEM_COL].isin(sem_filter)]
if course_type_filter and "course_type" in filtered.columns:
    filtered = filtered[filtered["course_type"].isin(course_type_filter)]

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
        show["avg_use"] = show["avg_use"].map(lambda x: "—" if pd.isna(x) else f"{x:.2f}")
        show["avg_diff"] = show["avg_diff"].map(lambda x: "—" if pd.isna(x) else f"{x:.2f}")
        show["liked_pct"] = show["liked_pct"].map(lambda x: "—" if pd.isna(x) else f"{x:.0f}%")
        show["value_score"] = show["value_score"].map(lambda x: "—" if pd.isna(x) else f"{x:.2f}")

        st.dataframe(
            show[["Course", "Type", "n", "avg_use", "avg_diff", "liked_pct", "value_score", "confidence"]],
            use_container_width=True
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
        st.caption(f"Type: **{classify_course_type(course)}**")

        f = filtered[filtered[COURSE_COL] == course].copy()
        n = len(f)

        avg_use = safe_mean(f.get(USE_COL, pd.Series(dtype=float)))
        avg_diff = safe_mean(f.get(DIFF_COL, pd.Series(dtype=float)))
        liked_pct = safe_pct_true(f.get(LIKED_COL, pd.Series(dtype=bool)))

        if n < 3:
            st.warning("Only a few reviews for this course. Treat averages as low-confidence.")

        k1, k2, k3, k4 = st.columns([1, 1, 1, 2])
        with k1:
            st.metric("Reviews", n)
        with k2:
            st.metric("Avg usefulness", "—" if avg_use is None else f"{avg_use:.2f} / 10")
        with k3:
            st.metric("Avg difficulty", "—" if avg_diff is None else f"{avg_diff:.2f} / 10")
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

        st.markdown("**Assessment breakdown:** " +
                    mark_breakdown(f.get(ASSIGN_COL, pd.Series(dtype=str))))

        st.divider()

        # ---- REVIEWS SECTION ----
        st.subheader(f"Reviews ({n})")

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

        if not comments_series.empty:
            comments_list = comments_series.tolist()
            comments_hash = hashlib.md5(
                "||".join(sorted(comments_list)).encode("utf-8")
            ).hexdigest()

            with st.spinner("Generating summary from student reviews... (first time takes ~20s)"):
                summaries = generate_review_summaries(
                    course, comments_hash, tuple(comments_list)
                )

            if summaries["status"] == "no_token":
                st.info("Add `HF_API_TOKEN` to your Streamlit secrets to enable AI summaries.")
            elif summaries["status"] == "all_failed":
                st.warning("Summary service is temporarily unavailable. Check the terminal for error details.")
            else:
                if summaries.get("overall"):
                    with st.container(border=True):
                        st.markdown("### 📝 Summary of student reviews")
                        st.write(summaries["overall"])

                has_breakdown = any([summaries.get("positive"),
                                     summaries.get("negative"),
                                     summaries.get("tips")])
                if has_breakdown:
                    s1, s2, s3 = st.columns(3)
                    with s1:
                        st.markdown("#### 👍 What students loved")
                        if summaries.get("positive"):
                            st.write(summaries["positive"])
                        else:
                            st.caption("_Nothing clearly positive flagged._")
                    with s2:
                        st.markdown("#### 👎 Common complaints")
                        if summaries.get("negative"):
                            st.write(summaries["negative"])
                        else:
                            st.caption("_Nothing clearly negative flagged._")
                    with s3:
                        st.markdown("#### 💡 Tips for success")
                        if summaries.get("tips"):
                            st.write(summaries["tips"])
                        else:
                            st.caption("_No explicit tips found._")

                st.divider()

        # ---- REVIEW CARDS GROUPED BY PROFESSOR ----
        if filtered_n == 0:
            st.info("No reviews match your filters.")
        else:
            rf = rf.copy()
            rf["_prof_key_local"] = rf[PROF_COL].apply(_prof_normalize_key).replace("", "— unknown —")

            display_map = {}
            for key, grp in rf.groupby("_prof_key_local"):
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
                rf.groupby("_prof_key_local").size()
                  .sort_values(ascending=False)
                  .index.tolist()
            )

            tab_labels = [
                f"{display_map[k]} ({rf[rf['_prof_key_local'] == k].shape[0]})"
                for k in prof_order_keys
            ]
            prof_tabs = st.tabs(tab_labels)
            for prof_tab, key in zip(prof_tabs, prof_order_keys):
                with prof_tab:
                    _render_review_cards(rf[rf["_prof_key_local"] == key])


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

            # ---- Donut charts per course ----
            st.markdown("#### 🎯 Scores at a glance")
            st.caption("Bigger filled portion = better score on that metric.")

            for course in selected:
                row = summary[summary["Course"] == course].iloc[0]
                st.markdown(f"**{course}**")
                d1, d2, d3 = st.columns(3)
                with d1:
                    st.plotly_chart(
                        donut_score("Usefulness",
                                    None if pd.isna(row["avg_use"]) else float(row["avg_use"])),
                        use_container_width=True,
                        key=f"d_use_{course}",
                    )
                with d2:
                    st.plotly_chart(
                        donut_score("Difficulty",
                                    None if pd.isna(row["avg_diff"]) else float(row["avg_diff"])),
                        use_container_width=True,
                        key=f"d_diff_{course}",
                    )
                with d3:
                    st.plotly_chart(
                        donut_yesno("Liked",
                                    None if pd.isna(row["liked_pct"]) else float(row["liked_pct"])),
                        use_container_width=True,
                        key=f"d_liked_{course}",
                    )

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