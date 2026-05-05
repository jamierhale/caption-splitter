import streamlit as st
import spacy
import re
import pandas as pd

st.set_page_config(page_title="Caption Splitter", page_icon="🎭", layout="wide")

# ── Load spaCy model (cached so it only loads once) ───────────────────────────

@st.cache_resource
def load_nlp():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        st.error("spaCy model missing. Run: python -m spacy download en_core_web_sm")
        st.stop()

# ── Text analysis helpers ─────────────────────────────────────────────────────

def is_character_name(line: str) -> bool:
    """
    Detects ALL CAPS character name lines, e.g. HAMLET or OPHELIA:
    Allows letters, digits, spaces, apostrophes, hyphens, periods.
    """
    name = line.strip().rstrip(":").strip()
    if len(name) < 2 or len(name) > 40:
        return False
    return bool(re.match(r"^[A-Z][A-Z\d\s'\-\.]*$", name))

def strip_stage_directions(text: str) -> str:
    """Remove parenthetical ( ) and bracketed [ ] stage directions."""
    text = re.sub(r"\([^)]{1,300}\)", "", text)
    text = re.sub(r"\[[^\]]{1,300}\]", "", text)
    return text

# ── Core chunking logic ───────────────────────────────────────────────────────

def split_text(text: str, nlp, max_chars: int) -> list[str]:
    """
    Recursively split text into chunks of at most max_chars,
    preferring semantically meaningful break points using spaCy's
    dependency parse. Priority order:
      1. After punctuation (, ; : — –)
      2. Before a coordinating conjunction (and, but, or…)
      3. Before a subordinating clause marker (because, when, while…)
      4. At a clause or phrase boundary (advcl, relcl, prep, conj)
      5. At any word boundary (space)
      6. Hard cut at max_chars (last resort)
    Within each tier, prefers the split that fills the first line
    as completely as possible.
    """
    text = text.strip()
    if not text or len(text) <= max_chars:
        return [text] if text else []

    doc = nlp(text)
    tokens = list(doc)
    best_score = -1
    best_idx = -1

    for i in range(1, len(tokens)):
        token = tokens[i]
        prev  = tokens[i - 1]

        # Character position of the potential split (start of token i)
        left = text[: token.idx].rstrip()
        if len(left) > max_chars:
            break
        if not left:
            continue

        score = 0

        # Tier 1 — punctuation in the previous token
        if prev.text in (",", ";", ":", "—", "–", "-"):
            score += 10

        # Tier 2 — coordinating conjunction coming up
        if token.dep_ == "cc" or token.text.lower() in (
            "and", "but", "or", "nor", "yet", "so"
        ):
            score += 8

        # Tier 3 — subordinating clause marker
        if token.dep_ == "mark":
            score += 7

        # Tier 4 — clause / phrase boundary
        if token.dep_ in ("advcl", "relcl", "prep", "conj"):
            score += 5

        # Length bonus: prefer splits that fill the first line
        score += (len(left) / max_chars) * 3

        if score > best_score:
            best_score = score
            best_idx = token.idx

    # Tier 5 — fall back to last space within limit
    if best_idx == -1:
        best_idx = text.rfind(" ", 0, max_chars + 1)

    # Tier 6 — hard cut
    if best_idx <= 0:
        best_idx = max_chars

    left  = text[:best_idx].strip()
    right = text[best_idx:].strip()

    return ([left] if left else []) + split_text(right, nlp, max_chars)


def process_script(
    raw: str,
    remove_dirs: bool,
    char_mode: str,
    max_chars: int,
    nlp,
) -> list[str]:
    """
    Full pipeline:
      • Detect character-name lines → keep or remove
      • Remove stage directions if requested
      • Sentence-split each speech with spaCy
      • Chunk each sentence to max_chars using split_text()
    Returns a flat list of caption-ready strings.
    """
    chunks = []
    current_para: list[str] = []

    def flush_paragraph():
        if not current_para:
            return
        prose = " ".join(current_para)
        doc = nlp(prose)
        for sent in doc.sents:
            for chunk in split_text(sent.text.strip(), nlp, max_chars):
                if chunk:
                    chunks.append(chunk)
        current_para.clear()

    for raw_line in raw.splitlines():
        line = raw_line.strip()

        # Blank line = speech boundary
        if not line:
            flush_paragraph()
            continue

        # Remove inline stage directions first
        if remove_dirs:
            line = strip_stage_directions(line).strip()
            if not line:
                continue  # entire line was a stage direction

        # Character name line?
        if is_character_name(line):
            flush_paragraph()
            if char_mode == "remove":
                continue
            name = line.rstrip(":").strip()
            if char_mode == "title":
                name = name.title()
            chunks.append(name + ":")
            continue

        current_para.append(line)

    flush_paragraph()
    return chunks

# ── Streamlit UI ──────────────────────────────────────────────────────────────

st.title("🎭 Caption Splitter")
st.caption(
    "Paste a script and get caption-ready text split by meaning — "
    "entirely local, no data sent anywhere."
)

# Sidebar settings
with st.sidebar:
    st.header("⚙️ Settings")

    max_chars = st.slider(
        "Max characters per caption line",
        min_value=20, max_value=60, value=42, step=1,
        help="Soft target — the splitter tries to stay under this, "
             "but won't break a word in a bad place to hit it exactly.",
    )

    st.markdown("---")

    remove_dirs = st.checkbox(
        "Remove stage directions",
        value=True,
        help="Strips anything inside ( ) or [ ] from the text.",
    )

    char_mode = st.radio(
        "Character names",
        options=["keep_caps", "title", "remove"],
        format_func=lambda x: {
            "keep_caps": "Keep — ALL CAPS",
            "title":     "Keep — Title Case",
            "remove":    "Remove",
        }[x],
        help="Character names are detected as ALL-CAPS lines "
             "(with or without a trailing colon).",
    )

    st.markdown("---")
    st.caption(
        "Splitting uses spaCy's dependency parser to find natural "
        "break points — clause boundaries, conjunctions, and punctuation "
        "are all preferred over arbitrary character counts."
    )

# Main area
nlp = load_nlp()

script_input = st.text_area(
    "Paste your script here",
    height=300,
    placeholder=(
        "HAMLET:\n"
        "To be, or not to be, that is the question:\n"
        "Whether 'tis nobler in the mind to suffer\n"
        "the slings and arrows of outrageous fortune,\n"
        "or to take arms against a sea of troubles.\n\n"
        "OPHELIA:\n"
        "My lord!\n"
    ),
)

generate = st.button(
    "Generate Captions",
    type="primary",
    disabled=not script_input.strip(),
)

if generate and script_input.strip():
    with st.spinner("Analysing…"):
        all_chunks = process_script(
            script_input, remove_dirs, char_mode, max_chars, nlp
        )

    if not all_chunks:
        st.warning("Nothing to show after processing — try adjusting the settings.")
    else:
        # Pair chunks into two-line caption frames
        frames = []
        for i in range(0, len(all_chunks), 2):
            frames.append({
                "#":              i // 2 + 1,
                "Line 1 (top)":   all_chunks[i],
                "Line 2 (bottom)": all_chunks[i + 1] if i + 1 < len(all_chunks) else "",
            })

        df = pd.DataFrame(frames)

        st.success(f"{len(frames)} caption frames generated from {len(all_chunks)} lines")

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "#":               st.column_config.NumberColumn(width=60),
                "Line 1 (top)":    st.column_config.TextColumn(),
                "Line 2 (bottom)": st.column_config.TextColumn(),
            },
        )

        st.download_button(
            label="⬇ Download as CSV",
            data=df.to_csv(index=False),
            file_name="captions.csv",
            mime="text/csv",
        )
