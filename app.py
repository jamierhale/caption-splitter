import streamlit as st
import anthropic
import json
import pandas as pd

st.set_page_config(page_title="Caption Splitter", page_icon="🎭", layout="wide")

# ── Password gate ─────────────────────────────────────────────────────────────

def check_password():
    if st.session_state.get("authenticated"):
        return True
    try:
        correct = st.secrets["APP_PASSWORD"]
    except Exception:
        st.error("APP_PASSWORD not set in Streamlit secrets.")
        st.stop()

    st.title("🎭 Caption Splitter")
    entered = st.text_input("Password", type="password")
    if st.button("Enter"):
        if entered == correct:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    st.stop()

check_password()

# ── API key ───────────────────────────────────────────────────────────────────

def get_api_key():
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

    custom_rules = st.text_area(
        "Additional rules for Claude",
        placeholder=(
            "One rule per line, in plain English. For example:\n"
            "- Never split a character's name from their title\n"
            "- Keep 'ladies and gentlemen' on one line\n"
            "- Treat ellipses as sentence endings"
        ),
        height=140,
        help=(
            "These are appended directly to Claude's instructions. "
            "Write them as you would explain them to a person."
        ),
    )

    st.markdown("---")
    st.caption(
        "A typical script scene costs less than 1p to process. "
        "No data is sent anywhere except the Anthropic API."
    )

# ── Prompt builder ────────────────────────────────────────────────────────────

def build_prompt(
    text: str,
    remove_dirs: bool,
    char_mode: str,
    max_chars: int,
    custom_rules: str,
) -> str:

    char_instructions = {
        "remove": (
            "Do not include character name labels in the output at all. "
            "Omit them entirely."
        ),
        "keep_caps": (
            "Keep character names in ALL CAPS as they appear, followed by a colon "
            "(e.g. HAMLET:). Tag these as type 'character'."
        ),
        "title": (
            "Convert character names to Title Case followed by a colon "
            "(e.g. HAMLET → Hamlet:). Tag these as type 'character'."
        ),
    }

    dir_instruction = (
        "Remove all stage directions. This includes text in ( ) or [ ], and any "
        "lines that are clearly performance instructions rather than dialogue — "
        "even if unbracketed. Use your judgement."
        if remove_dirs else
        "Keep stage directions in the output. Tag them as type 'direction'."
    )

    custom_block = ""
    if custom_rules and custom_rules.strip():
        rules_list = "\n".join(
            f"- {r.lstrip('-• ').strip()}"
            for r in custom_rules.strip().splitlines()
            if r.strip()
        )
        custom_block = f"\nADDITIONAL RULES (apply these carefully):\n{rules_list}\n"

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
{custom_block}
FORMATTING:
- {dir_instruction}
- {char_instructions[char_mode]}

OUTPUT FORMAT:
Return ONLY a raw JSON array of objects. Each object must have exactly two fields:
  "type": one of "character", "line", or "direction"
  "text": the caption text

No explanation, no markdown, no code fences — just the raw JSON array.

Example:
[
  {{"type": "character", "text": "Hamlet:"}},
  {{"type": "line", "text": "To be, or not to be,"}},
  {{"type": "line", "text": "that is the question:"}}
]

SCRIPT:
{text}"""

# ── Pairing logic ─────────────────────────────────────────────────────────────

def pair_into_frames(items: list[dict]) -> list[dict]:
    """
    Pairs items into two-column caption frames, applying layout rules:

    RULE 1: Character names always go in Line 1 (left column).
    RULE 2: Line 2 of a character-name frame is the first line of their dialogue
            — the name and their opening words share a frame.
    RULE 3: If a regular line would pair with a character name in L2,
            L2 is left blank instead, and the name starts the next frame.
    """
    frames = []
    i = 0

    while i < len(items):
        item = items[i]

        if item["type"] == "character":
            # L1 = character name
            # L2 = first dialogue line (if next item is not another character name)
            l2 = ""
            if i + 1 < len(items) and items[i + 1]["type"] != "character":
                l2 = items[i + 1]["text"]
                i += 2
            else:
                i += 1
            frames.append({
                "#":               len(frames) + 1,
                "Line 1 (top)":    item["text"],
                "Line 2 (bottom)": l2,
                "_type":           "character",
            })

        else:
            # L1 = this line
            # L2 = next line, but ONLY if next item is not a character name
            # (if it is, leave L2 blank so the name opens its own frame in L1)
            l1 = item["text"]
            l2 = ""
            i += 1
            if i < len(items) and items[i]["type"] != "character":
                l2 = items[i]["text"]
                i += 1
            frames.append({
                "#":               len(frames) + 1,
                "Line 1 (top)":    l1,
                "Line 2 (bottom)": l2,
                "_type":           "line",
            })

    return frames

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
    prompt = build_prompt(
        script_input, remove_dirs, char_mode, max_chars, custom_rules
    )

    with st.spinner("Claude is reading your script…"):
        try:
            client = anthropic.Anthropic(api_key=api_key)
            message = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = message.content[0].text.strip()

            # Strip any accidental markdown fences
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            raw = raw.strip()

            parsed = json.loads(raw)

            if not isinstance(parsed, list):
                raise ValueError("Response was not a JSON array.")

            # Validate and clean each item
            items = []
            for obj in parsed:
                t = obj.get("type", "line")
                text = str(obj.get("text", "")).strip()
                if text:
                    items.append({"type": t, "text": text})

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

    frames = pair_into_frames(items)
    df = pd.DataFrame(frames).drop(columns=["_type"])

    line_count = len(items)
    avg = sum(len(it["text"]) for it in items) // max(line_count, 1)

    st.success(
        f"{len(frames)} caption frames · {line_count} lines · ≈{avg} chars avg"
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
