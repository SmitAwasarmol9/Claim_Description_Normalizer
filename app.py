import io
import json
import os
import re
from datetime import datetime
from functools import lru_cache
import PyPDF2
import spacy
import streamlit as st
from groq import Groq

# ──────────────────────────────────────────────
# ENV / SECRETS
# ──────────────────────────────────────────────
API_KEY = None

# Try Streamlit secrets (deployment)
if "GROQ_API_KEY" in st.secrets:
    API_KEY = st.secrets["GROQ_API_KEY"]

# Fallback to environment variable (local)
if not API_KEY:
    API_KEY = os.getenv("GROQ_API_KEY")

if not API_KEY:
    st.error("❌ GROQ_API_KEY not found. Set it in Streamlit secrets or environment.")
    st.stop()

# ✅ Create Groq client ONCE (performance fix)
@st.cache_resource
def get_groq_client():
    return Groq(api_key=API_KEY)

client = get_groq_client()

# ──────────────────────────────────────────────
# SPACY — load once
# ──────────────────────────────────────────────
@st.cache_resource
def load_nlp():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        st.error("❌ spaCy model not installed. Check requirements.txt")
        st.stop()

nlp = load_nlp()

# ──────────────────────────────────────────────
# VEHICLE / PRODUCT BLOCKLIST
# ──────────────────────────────────────────────
VEHICLE_BLOCKLIST: frozenset = frozenset({
    "innova", "innova crysta", "fortuner", "corolla", "camry", "glanza",
    "swift", "baleno", "dzire", "alto", "wagon r", "ertiga", "brezza",
    "ciaz", "s-cross", "ignis", "celerio", "vitara",
    "creta", "i20", "i10", "verna", "tucson", "venue", "seltos",
    "sonet", "carnival", "sportage",
    "city", "civic", "jazz", "amaze", "wr-v", "elevate",
    "nexon", "harrier", "safari", "tiago", "altroz", "punch",
    "nexon ev", "tigor", "zest", "bolt",
    "scorpio", "xuv", "bolero", "thar", "duster", "kwid",
    "magnite", "kicks", "terrano",
    "hector", "astor", "gloster",
    "activa", "splendor", "pulsar", "apache", "bullet", "classic 350",
    "duke", "r15", "cbr", "shine", "unicorn", "dio", "jupiter",
    "ntorq", "ray", "pleasure", "fascino",
    "tempo", "traveller", "tata ace",
})

def is_vehicle(text: str) -> bool:
    return text.lower().strip() in VEHICLE_BLOCKLIST

# ──────────────────────────────────────────────
# CLAIM NORMALIZER (GROQ)
# ──────────────────────────────────────────────
def normalize_claim_with_groq(text: str) -> dict:
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "You are an insurance claims expert. Output ONLY valid JSON."
                },
                {
                    "role": "user",
                    "content": _NORMALIZER_PROMPT.format(text=text)
                }
            ],
            temperature=0.2,
            response_format={"type": "json_object"}
        )

        cleaned = response.choices[0].message.content.strip()

        # Clean possible prefix like "json"
        cleaned = re.sub(r"^json\s*", "", cleaned, flags=re.IGNORECASE)

        # ✅ Safe JSON parsing
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return {
                "error": "Invalid JSON from Groq",
                "raw_response": cleaned
            }

    except Exception as e:
        return {
            "error": f"Groq API Error: {str(e)}",
            "raw_response": "Error"
        }

# ──────────────────────────────────────────────
# CLAIM NORMALIZER PROMPT (USED BY GROQ)
# ──────────────────────────────────────────────
_NORMALIZER_PROMPT = """You are an insurance claims expert.

Convert the following raw claim description into structured JSON.

Extract:
- loss_type
- severity (low / medium / high)
- affected_asset
- incident_date (if mentioned, else null)
- location (if mentioned, else null)
- short_summary (1 line)
- confidence_score (0 to 1)

Return ONLY valid JSON. No explanation, no markdown fences.

CLAIM TEXT:
{text}"""

# ⚠️ IMPORTANT:
# DO NOT redefine normalize_claim_with_groq here.
# It is already defined in the previous section using Groq API.
# This avoids overriding the real API function.

# ──────────────────────────────────────────────
# FRAUD ENGINE v4
# ──────────────────────────────────────────────

# Pre-compiled constants — evaluated once at import time
_VAGUE_KEYWORDS: tuple = (
    "don't remember", "not sure", "somewhere", "maybe", "i guess",
    "approximately", "probably", "around somewhere", "i don't know",
    "can't recall", "forgot", "no idea", "not certain", "unclear",
    "i think", "i believe", "roughly", "i suppose", "could be",
)

_FRAUD_PHRASES: tuple = (
    "everything destroyed", "entire car destroyed", "completely destroyed",
    "everything gone", "total loss", "fully destroyed", "nothing left",
    "burned to the ground", "completely wrecked", "all items stolen",
    "entire contents stolen", "wiped out", "totally ruined",
    "complete write-off", "everything was taken",
)

_LOSS_KEYWORDS: tuple = (
    "fire", "theft", "stolen", "flood", "accident", "crash",
    "collision", "vandalism", "explosion", "earthquake",
)

_RECENT_WORDS: tuple = (
    "just now", "right now", "this morning", "tonight",
    "few minutes ago", "an hour ago",
)

_LUXURY_KEYWORDS: tuple = (
    "rolex", "ferrari", "lamborghini", "bentley", "porsche",
    "diamond", "gold jewelry", "luxury watch", "macbook pro",
    "iphone 15", "iphone 16", "designer bag", "louis vuitton",
    "gucci", "yacht", "artwork", "antique",
)

_URGENCY_PHRASES: tuple = (
    "need money urgently", "urgent payment", "need settlement fast",
    "desperate", "immediately", "asap", "need it now",
    "pay immediately", "settle fast", "need funds",
)

_PRIOR_CLAIM_PHRASES: tuple = (
    "last time", "previous claim", "before this", "again",
    "third time", "second time", "another claim",
)

_CONTRADICTION_PAIRS: tuple = (
    ("parked", "driving"),
    ("at home", "on the road"),
    ("no one was there", "witnessed"),
    ("empty", "full"),
)


def calculate_fraud_score(data: dict, raw_text: str) -> tuple[int, str, list[str]]:
    # Safety check
    if not isinstance(data, dict) or "error" in data:
        return 0, "unknown", []

    score = 0
    text_lower = raw_text.lower()
    flags: list[str] = []

    # spaCy processing
    doc = nlp(raw_text)

    detected_dates = [e.text.lower() for e in doc.ents if e.label_ == "DATE"]
    detected_locations = [e.text.lower() for e in doc.ents if e.label_ in ("GPE", "LOC", "FAC")]

    llm_date = str(data.get("incident_date") or "").lower()
    llm_location = str(data.get("location") or "").lower()

    def is_null(v):
        return not v or v.strip() == "" or v == "null"

    # 1. Missing critical fields
    if is_null(llm_date):
        score += 8
        flags.append("⚠️ No incident date provided")

    if is_null(llm_location):
        score += 8
        flags.append("⚠️ No incident location provided")

    # 2. Date not verifiable
    if not is_null(llm_date):
        if not any(d in llm_date or llm_date in d for d in detected_dates):
            score += 10
            flags.append("🔍 Incident date not verifiable from claim text")

    # 3. Location not verifiable
    if not is_null(llm_location):
        if not any(loc in llm_location or llm_location in loc for loc in detected_locations):
            score += 10
            flags.append("🔍 Incident location not verifiable from claim text")

    # 4. Severity
    if str(data.get("severity", "")).lower() == "high":
        score += 5
        flags.append("📈 High severity claim detected")

    # 5. Vague language
    matched_vague = [w for w in _VAGUE_KEYWORDS if w in text_lower]
    if matched_vague:
        score += min(6 * len(matched_vague), 24)
        flags.append(f"🗣️ Vague language detected: {', '.join(matched_vague[:3])}")

    # 6. Exaggeration phrases
    matched_fraud = [p for p in _FRAUD_PHRASES if p in text_lower]
    if matched_fraud:
        score += min(10 * len(matched_fraud), 30)
        flags.append(f"🚨 Exaggeration phrases: {', '.join(matched_fraud[:2])}")

    # 7. Claim length
    word_count = len(text_lower.split())
    if word_count < 7:
        score += 15
        flags.append(f"📏 Claim is very short ({word_count} words) — lacks detail")
    elif word_count < 15:
        score += 5
        flags.append(f"📏 Claim is brief ({word_count} words)")

    # 8. Multiple loss types
    matched_losses = [k for k in _LOSS_KEYWORDS if k in text_lower]
    if len(matched_losses) >= 2:
        score += 12
        flags.append(f"⚡ Multiple loss types in one claim: {', '.join(matched_losses)}")

    # 9. Suspiciously recent with no date
    if any(w in text_lower for w in _RECENT_WORDS) and is_null(llm_date):
        score += 8
        flags.append("⏰ Claim filed immediately with no specific date")

    # 10. High-value / luxury assets
    matched_luxury = [k for k in _LUXURY_KEYWORDS if k in text_lower]
    if matched_luxury:
        score += 8
        flags.append(f"💎 High-value asset keywords: {', '.join(matched_luxury[:2])}")

    # 11. Urgency / pressure language
    if any(p in text_lower for p in _URGENCY_PHRASES):
        score += 10
        flags.append("🔔 Urgency / pressure language detected")

    # 12. Prior claim references
    if any(p in text_lower for p in _PRIOR_CLAIM_PHRASES):
        score += 10
        flags.append("📋 Possible prior claim reference — repeated claimant pattern")

    # 13. Contradictions
    for a, b in _CONTRADICTION_PAIRS:
        if a in text_lower and b in text_lower:
            score += 12
            flags.append(f"❌ Contradictory info: '{a}' and '{b}'")

    # 14. LLM confidence (SAFE FIX)
    try:
        confidence = float(data.get("confidence_score") or 0.5)
    except (ValueError, TypeError):
        confidence = 0.5

    if confidence > 0.9:
        score -= 5
    elif confidence < 0.5:
        score += 8
        flags.append(f"🤖 Low AI confidence ({confidence:.2f}) — unclear claim")
    elif confidence < 0.7:
        score += 3

        # Suspicious crowd presence
    if "people gathered" in text_lower:
        score += 5
    flags.append("👥 Unverified third-party presence at scene")

    # Final normalization
    score = max(0, min(score, 100))
    risk = "high" if score >= 60 else "medium" if score >= 30 else "low"

    return score, risk, flags


# ──────────────────────────────────────────────
# PDF REPORT GENERATOR
# ──────────────────────────────────────────────
def generate_pdf(structured_output, fraud_score, fraud_risk, fraud_flags, entities, claim_text) -> io.BytesIO:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        HRFlowable, Paragraph, SimpleDocTemplate,
        Spacer, Table, TableStyle
    )

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=20 * mm,
        leftMargin=20 * mm,
        topMargin=20 * mm,
        bottomMargin=20 * mm,
    )

    # ── Styles ──
    title_style = ParagraphStyle(
        "Title", fontSize=20, fontName="Helvetica-Bold",
        textColor=colors.HexColor("#1a1a2e"),
        alignment=TA_CENTER, spaceAfter=12
    )

    subtitle_style = ParagraphStyle(
        "Subtitle", fontSize=10, fontName="Helvetica",
        textColor=colors.HexColor("#666666"),
        alignment=TA_CENTER, spaceAfter=16
    )

    section_style = ParagraphStyle(
        "Section", fontSize=13, fontName="Helvetica-Bold",
        textColor=colors.HexColor("#16213e"),
        spaceBefore=14, spaceAfter=6
    )

    normal_style = ParagraphStyle(
        "Normal2", fontSize=10, fontName="Helvetica",
        textColor=colors.HexColor("#333333"),
        spaceAfter=4
    )

    flag_style = ParagraphStyle(
        "Flag", fontSize=9, fontName="Helvetica",
        textColor=colors.HexColor("#b5451b"),
        spaceAfter=3, leftIndent=10
    )

    claim_style = ParagraphStyle(
        "Claim", fontSize=9, fontName="Helvetica-Oblique",
        textColor=colors.HexColor("#444444"),
        spaceAfter=4,
        backColor=colors.HexColor("#f5f5f5"),
        leftIndent=8, rightIndent=8, borderPad=6
    )

    footer_style = ParagraphStyle( 
        "Footer", fontSize=8, fontName="Helvetica",
        textColor=colors.HexColor("#999999"),
        alignment=TA_CENTER, spaceBefore=6
    )

    HR = lambda: HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#dddddd"))

    story = []

    # ── Header ──
    story += [
        Paragraph("Claims Description Normalizer", title_style),
        Paragraph(
            f"Report generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}",
            subtitle_style
        ),
        HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#4cc9f0")),
        Spacer(1, 10),
    ]

    # ── Original claim ──
    safe_claim_text = str(claim_text or "No claim text provided")

    story += [
        Paragraph("Original Claim Text", section_style),
        Paragraph(safe_claim_text, claim_style),
        Spacer(1, 6),
    ]

    # ── Structured data ──
    story += [
        Paragraph("Structured Claim Data", section_style),
        HR(),
        Spacer(1, 4),
    ]

    field_map = {
        "loss_type": "Loss Type",
        "severity": "Severity",
        "affected_asset": "Affected Asset",
        "incident_date": "Incident Date",
        "location": "Location",
        "short_summary": "Short Summary",
        "confidence_score": "AI Confidence Score",
    }

    # SAFE handling if structured_output is broken
    if not isinstance(structured_output, dict):
        structured_output = {}

    table_data = [["Field", "Value"]] + [
        [label, str(structured_output.get(key) or "Not mentioned")]
        for key, label in field_map.items()
    ]

    t = Table(table_data, colWidths=[60 * mm, 110 * mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a1a2e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 10),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 1), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.HexColor("#f9f9f9"), colors.white]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))

    story += [t, Spacer(1, 14)]

    # ── Fraud section ──
    story += [
        Paragraph("Fraud Risk Analysis", section_style),
        HR(),
        Spacer(1, 4)
    ]

    risk_color = {
        "high": colors.HexColor("#c0392b"),
        "medium": colors.HexColor("#e67e22"),
        "low": colors.HexColor("#27ae60")
    }.get(fraud_risk, colors.grey)

    fraud_table = Table(
        [["Fraud Score", "Risk Level"],
         [f"{fraud_score} / 100", str(fraud_risk).upper()]],
        colWidths=[85 * mm, 85 * mm]
    )

    fraud_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a1a2e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 10),
        ("FONTNAME", (0, 1), (-1, 1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 1), (-1, 1), 14),
        ("TEXTCOLOR", (1, 1), (1, 1), risk_color),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("PADDING", (0, 0), (-1, -1), 8),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.HexColor("#f9f9f9")]),
    ]))

    story += [fraud_table, Spacer(1, 8)]

    if fraud_flags:
        story.append(Paragraph("Fraud Detection Flags:", normal_style))
        story += [Paragraph(str(f), flag_style) for f in fraud_flags]
    else:
        story.append(Paragraph("✅ No fraud indicators detected.", normal_style))

    story.append(Spacer(1, 14))

    # ── Entities ──
    if isinstance(entities, dict) and entities:
        story += [
            Paragraph("Extracted Named Entities (spaCy)", section_style),
            HR(),
            Spacer(1, 4)
        ]

        entity_data = [["Entity Type", "Values"]] + [
            [etype, ", ".join(map(str, evals))]
            for etype, evals in entities.items()
        ]

        et = Table(entity_data, colWidths=[60 * mm, 110 * mm])
        et.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a1a2e")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 10),
            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 1), (-1, -1), 9),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.HexColor("#f9f9f9"), colors.white]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("PADDING", (0, 0), (-1, -1), 6),
        ]))

        story.append(et)

    # ── Footer (FIXED: Gemini → Groq) ──
    story += [
        Spacer(1, 20),
        HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cccccc")),
        Paragraph(
            "Generated by Claims Description Normalizer · Groq LLaMA 3 · spaCy · 2026",
            footer_style
        ),
    ]

    doc.build(story)
    buffer.seek(0)

    return buffer

# ──────────────────────────────────────────────
# HELPER FUNCTIONS (PASTE HERE)
# ──────────────────────────────────────────────

def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s.,]", "", text)
    return text.strip()


def extract_text_from_pdf(uploaded_file) -> str:
    try:
        reader = PyPDF2.PdfReader(uploaded_file)
        text = ""

        for page in reader.pages:
            text += page.extract_text() or ""

        if not text.strip():
            raise ValueError("No readable text found in PDF")

        return text

    except Exception as e:
        raise ValueError(f"Error reading PDF: {e}")


def extract_entities(text: str) -> dict:
    doc = nlp(text)

    entities = {}

    for ent in doc.ents:
        label = ent.label_

        if is_vehicle(ent.text):
            label = "PRODUCT"

        if label not in entities:
            entities[label] = []

        if ent.text not in entities[label]:
            entities[label].append(ent.text)

    return entities


# ══════════════════════════════════════════════
# STREAMLIT UI
# ══════════════════════════════════════════════
st.set_page_config(
    page_title="Claims Description Normalizer",
    page_icon="🧾",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700&family=Fira+Code:wght@400;500&display=swap');

* { font-family: 'Outfit', sans-serif; }

:root {
    --primary: #6366f1;
    --primary-light: #818cf8;
    --accent: #ec4899;
    --success: #10b981;
    --warning: #f59e0b;
    --danger: #ef4444;
    --bg-dark: #0f172a;
    --bg-card: #1e293b;
    --text-primary: #f1f5f9;
    --text-secondary: #cbd5e1;
    --border: #334155;
}

.stApp {
    background: linear-gradient(135deg, var(--bg-dark) 0%, #1a2847 100%);
    color: var(--text-primary);
}

::-webkit-scrollbar { width: 8px; }
::-webkit-scrollbar-track { background: var(--bg-dark); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: var(--primary-light); }

.header-container {
    background: linear-gradient(135deg, rgba(99,102,241,.1), rgba(236,72,153,.05));
    border: 1px solid rgba(99,102,241,.2);
    border-radius: 16px;
    padding: 32px 40px;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
}
.header-container::before {
    content: '';
    position: absolute;
    top: -50%; right: -10%;
    width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(99,102,241,.1) 0%, transparent 70%);
    border-radius: 50%;
}
.header-content { position: relative; z-index: 1; }
.header-title {
    font-size: 36px; font-weight: 700;
    background: linear-gradient(135deg, #6366f1, #ec4899);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 8px 0; letter-spacing: -.5px;
}
.header-subtitle { font-size: 15px; color: var(--text-secondary); margin: 0; }
.badge {
    display: inline-block;
    background: rgba(99,102,241,.15);
    border: 1px solid var(--primary); 
    border-radius: 20px;
    padding: 6px 14px; font-size: 12px;
    color: var(--primary-light);
    margin: 12px 8px 0 0; font-weight: 500;
}
/* ── ENTITY STYLING FIXED ── */
.entity-type {
    color: #f59e0b !important;  /* force orange */
    font-weight: 700;
    margin-top: 8px;
    display: block;
}

.entity-value {
    display: inline-block;
    background: rgba(99,102,241,.15);
    border: 1px solid rgba(99,102,241,.3);
    border-radius: 6px;
    padding: 4px 10px;
    font-size: 12px;
    color: #60a5fa !important;  /* force blue */
    margin: 3px 4px;
    font-family: 'Fira Code', monospace;
    font-weight: 500;
}            

/* (rest CSS unchanged) */
</style>
""", unsafe_allow_html=True)

# ── Header ──
st.markdown("""
<div class="header-container">
  <div class="header-content">
    <div class="header-title">🧾 Claims Description Normalizer</div>
    <div class="header-subtitle">Intelligent claim processing with AI-powered fraud detection</div>
    <span class="badge">⚡ Groq LLaMA 3</span>
    <span class="badge">🧠 spaCy NER</span>
    <span class="badge">🔍 Smart Fraud Engine</span>
    <span class="badge">📄 PDF Support</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Input Layout ──
col1, col2 = st.columns([1, 1.3], gap="large")

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">📥 Claim Input</div>', unsafe_allow_html=True)

    input_method = st.radio("Input method:", ["Text", "PDF Upload"], horizontal=True, label_visibility="collapsed")
    claim_text = None

    if input_method == "Text":
        claim_text = st.text_area(
            "",
            height=220,
            placeholder="Describe the claim in detail...",
            label_visibility="collapsed",
        )
    else:
        st.markdown('<div class="upload-box">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Drop your PDF here", type=["pdf"], label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)

        if uploaded_file:
            try:
                with st.spinner("📄 Extracting text from PDF..."):
                    claim_text = extract_text_from_pdf(uploaded_file)
                st.success(f"✅ Extracted {len(claim_text.split())} words from PDF")
            except Exception as e:  # ✅ safer than only ValueError
                st.error(f"❌ PDF Error: {e}")
                claim_text = None

    analyze = st.button("🚀 Analyze Claim")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">📊 Structured Output</div>', unsafe_allow_html=True)
    output_placeholder = st.empty()
    st.markdown('</div>', unsafe_allow_html=True)

# ── Analysis ──
if analyze:
    if not claim_text or not str(claim_text).strip():
        st.warning("⚠️ Please enter or upload a claim description.")
    else:
        with st.spinner("⚙️ Analyzing claim..."):
            text = clean_text(claim_text)

            structured_output = normalize_claim_with_groq(text)

            fraud_score, fraud_risk, fraud_flags = calculate_fraud_score(structured_output, text)

            if isinstance(structured_output, dict):
                structured_output["fraud_score"] = fraud_score
                structured_output["fraud_risk"] = fraud_risk

            # ✅ Safe entity extraction
            try:
                entities = extract_entities(text)
            except Exception:
                entities = {}

        with col2:
            output_placeholder.json(structured_output)

        st.markdown("<hr class='divider'>", unsafe_allow_html=True)

        # ✅ Columns defined properly
        r1, r2, r3 = st.columns([0.9, 1.4, 1.1], gap="large")

        # ── r1: Score ──
        with r1:
            st.markdown('<div class="card">', unsafe_allow_html=True)

            score_color = "#ef4444" if fraud_risk == "high" else "#f59e0b" if fraud_risk == "medium" else "#10b981"

            st.markdown(f"""
            <div class="score-meter">
                <div class="score-number" style="color:{score_color};">{fraud_score}</div>
                <div class="score-label">Fraud Score</div>
            </div>
            """, unsafe_allow_html=True)

            risk_class = {"high": "risk-high", "medium": "risk-medium", "low": "risk-low"}.get(fraud_risk, "risk-low")
            risk_label = {"high": "🚨 HIGH RISK", "medium": "⚠️ MEDIUM RISK", "low": "✅ LOW RISK"}.get(fraud_risk, "")

            st.markdown(f'<div class="risk-banner {risk_class}">{risk_label}</div>', unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

        # ── r2: Flags ──
        with r2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">🔍 Fraud Flags</div>', unsafe_allow_html=True)

            if fraud_flags:
                flags_html = '<div class="flag-box">' + "".join(
                    f'<div class="flag-item">{str(f)}</div>' for f in fraud_flags
                ) + '</div>'
                st.markdown(flags_html, unsafe_allow_html=True)
            else:
                st.markdown('<div class="risk-banner risk-low">✅ No fraud indicators</div>', unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

        # ── r3: Entities ──
        with r3:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">🧠 Entities</div>', unsafe_allow_html=True)

            if entities:
                for etype, evals in entities.items():
                    st.markdown(
                        f'<span class="entity-type">{etype}</span>',
                        unsafe_allow_html=True
                    )

                    values_html = "".join(
                        f'<span class="entity-value">{str(v)}</span>'
                        for v in evals
                    )

                    st.markdown(values_html, unsafe_allow_html=True) 
            else:
                st.markdown(
                    '<span style="color:var(--text-secondary);font-size:13px;">No entities detected.</span>',
                    unsafe_allow_html=True
                )

            st.markdown('</div>', unsafe_allow_html=True)

# ── PDF Download ──
if analyze:
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    try:
        pdf_buffer = generate_pdf(
            structured_output,
            fraud_score,
            fraud_risk,
            fraud_flags,
            entities,
            text
        )

        st.download_button(
            label="📥 Download Full Report as PDF",
            data=pdf_buffer,
            file_name=f"claim_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf",
        )

    except ImportError:
        st.error("📦 Install reportlab: `pip install reportlab`")

# ── Footer ──
st.markdown("<hr class='divider'>", unsafe_allow_html=True)
st.markdown(
    '<p class="footer">⚡ Groq LLaMA 3 · spaCy · Smart Fraud Engine v4 · PDF Support · 2026</p>',
    unsafe_allow_html=True,
)