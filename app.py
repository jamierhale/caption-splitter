import streamlit as st
import anthropic
import json
import pandas as pd

st.set_page_config(page_title="Caption Splitter", page_icon="🎭", layout="wide")

# ── API key ───────────────────────────────────────────────────────────────────

def get_api_key():
    """Check Streamlit secrets first, fall back to sidebar input."""
    try:
        return st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        return None

secrets_key = get_api_key()

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Settings")

    if not secrets_key:
        api_key = st.text_input(
            "Anthropic API key",
            type="password",
            placeholder="sk-ant-...",
            help="Your key is used only for this request and is never stored.",
        )
    else:
        api_key = secrets_key
        st.success("API key loaded from secrets", icon="🔑")

    st.markdown("---")

    max_chars = st.slider(
        "Target characters per line",
        min_value=20, max_value=60, value=42, step=1,
        help="Soft target — Claude will prioritise meaning over hitting this exactly.",
    )

    st.markdown("---")

    remove_dirs = st.checkbox(
        "Remove stage directions",
        value=True,
        help=(
            "Claude will remove text in ( ) or [ ] and any lines that read as "
            "performance instructions rather than speech — even if unbracketed."
        ),
    )

    char_mode = st.radio(
        "Character names",
        options=["keep_caps", "title", "remove"],
        format_func=lambda x: {
            "keep_caps": "Keep — ALL CAPS",
            "title":     "Keep — Title Case",
            "remove":    "Remove",
        }[x],
    )

    st.markdown("---")
    st.caption(
        "Splitting is done entirely by Claude — no data is sent anywhere else. "
        "A typical script scene costs less than 1p to process."
    )

# ── Prompt builder ────────────────────────────────────────────────────────────

def build_prompt(text: str, remove_dirs: bool, char_mode: str, max_chars: int) -> str:
    char_instructions = {
        "remove": (
            "Remove character name labels entirely — do not include them in the output at all."
        ),
        "keep_caps": (
            "Keep character names in ALL CAPS as they appear, followed by a colon "
            "(e.g. HAMLET:). Place each name as its own entry in the array."
        ),
        "title": (
            "Convert character names to Title Case followed by a colon "
            "(e.g. HAMLET → Hamlet:). Place each name as its own entry in the array."
        ),
    }

    dir_instruction = (
        "Remove all stage directions from the output. This includes text in parentheses "
        "or square brackets, and any lines that are clearly performance instructions "
        "rather than speech — even if they are not bracketed. Use your judgement: if a "
        "line is clearly telling a performer what to do rather than being dialogue, remove it."
        if remove_dirs else
        "Keep stage directions in the output, treating them as normal text to be split."
    )

    return f"""You are an expert caption editor for live theatre and performance. \
Take the script text below and split it into individual caption lines ready for display on screen.

CAPTION LINE RULES:
- Target approximately {max_chars} characters per line — treat this as a soft guide, not a hard limit
- Prioritise meaning and natural speech rhythm over hitting the character count exactly
- Never break mid-phrase or mid-thought
- Prefer splitting in this order of priority:
    1. After sentence-ending punctuation (. ! ?)
    2. After a comma, semicolon, colon, or dash
    3. Before a coordinating conjunction (and, but, or, nor, yet, so)
    4. Before a subordinating clause (because, when, while, although, if, that, which, who)
    5. Between natural phrases or breath points
- Keep subjects with their verbs where possible
- Lines will be displayed in pairs on screen — consider how consecutive lines read together
- For verse or poetry, treat each line break as a natural split point

FORMATTING:
- {dir_instruction}
- {char_instructions[char_mode]}

OUTPUT:
Return ONLY a raw JSON array of strings. No explanation, no markdown, no code fences — \
just the JSON array starting with [ and ending with ]. Example:
["To be, or not to be,", "that is the question:"]

SCRIPT:
{text}"""

# ── Main UI ───────────────────────────────────────────────────────────────────

st.title("🎭 Caption Splitter")
st.caption("Paste a script and get caption-ready lines split by meaning.")

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

ready = bool(api_key and script_input.strip())

generate = st.button(
    "Generate Captions",
    type="primary",
    disabled=not ready,
)

if not api_key:
    st.info("Enter your Anthropic API key in the sidebar to get started.", icon="🔑")

# ── Generation ────────────────────────────────────────────────────────────────

if generate and ready:
    prompt = build_prompt(script_input, remove_dirs, char_mode, max_chars)

    with st.spinner("Claude is reading your script…"):
        try:
            client = anthropic.Anthropic(api_key=api_key)
            message = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = message.content[0].text.strip()

            # Robust JSON extraction — strip any accidental markdown fences
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            raw = raw.strip()

            lines = json.loads(raw)

            if not isinstance(lines, list):
                raise ValueError("Response was not a JSON array.")

            lines = [str(l).strip() for l in lines if str(l).strip()]

        except anthropic.AuthenticationError:
            st.error("Invalid API key — please check it in the sidebar.")
            st.stop()
        except anthropic.RateLimitError:
            st.error("Rate limit hit — wait a moment and try again.")
            st.stop()
        except json.JSONDecodeError:
            st.error("Couldn't parse Claude's response. Try again — this is rare.")
            with st.expander("Raw response (for debugging)"):
                st.text(raw)
            st.stop()
        except Exception as e:
            st.error(f"Something went wrong: {e}")
            st.stop()

    # Pair lines into caption frames
    frames = []
    for i in range(0, len(lines), 2):
        frames.append({
            "#":               i // 2 + 1,
            "Line 1 (top)":    lines[i],
            "Line 2 (bottom)": lines[i + 1] if i + 1 < len(lines) else "",
        })

    df = pd.DataFrame(frames)

    st.success(
        f"{len(frames)} caption frames · {len(lines)} lines · "
        f"≈{sum(len(l) for l in lines) // len(lines)} chars avg"
    )

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "#":               st.column_config.NumberColumn(width=55),
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
