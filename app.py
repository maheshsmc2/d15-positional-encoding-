import gradio as gr
from PIL import Image
from uuid import uuid4
from datetime import datetime
import time, os, csv, random

APP_TITLE = "Image Caption Generator — Pro Demo"
FAVICON_SVG_DATA_URL = "data:image/svg+xml;utf8," + \
    "%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'%3E%3Crect rx='12' ry='12' width='64' height='64' fill='%232256ff'/%3E%3Ctext x='50%25' y='56%25' font-size='34' text-anchor='middle' fill='white' font-family='Segoe UI, Arial' %3E🖼️%3C/text%3E%3C/svg%3E"

MAX_SIDE_PX = 1280
CONCURRENCY = 2
QUEUE_SIZE = 16
DEFAULT_SEED = 123
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

infer_fn = None
MODEL_NAME = "unknown"
MODEL_VERSION = "unknown"
try:
    from utils import predict as infer_fn
    MODEL_NAME = getattr(infer_fn, "__name__", "predict")
except Exception:
    try:
        from utils import generate_caption as infer_fn
        MODEL_NAME = getattr(infer_fn, "__name__", "generate_caption")
    except Exception:
        pass

try:
    from utils import __version__ as UTIL_VER
    MODEL_VERSION = str(UTIL_VER)
except Exception:
    pass

def _placeholder_infer(image):
    return "🔧 Model not connected — please plug in your caption function."

BLOCKLIST = {"nude", "nsfw", "porn", "gore"}

def sanitize_caption(text: str) -> str:
    if not text:
        return text
    lo = text.lower()
    if any(w in lo for w in BLOCKLIST):
        return "⚠️ Content filtered."
    t = " ".join(text.strip().split())
    if t and t[-1].isalnum():
        t += "."
    return t

def ensure_max_side(img: Image.Image) -> Image.Image:
    w, h = img.size
    if max(w, h) <= MAX_SIDE_PX:
        return img
    scale = MAX_SIDE_PX / float(max(w, h))
    return img.resize((int(w * scale), int(h * scale)))

def maybe_set_seed(seed: int | None):
    if seed is None:
        seed = DEFAULT_SEED
    random.seed(seed)

def write_log_row(path, row_dict):
    header = list(row_dict.keys())
    file_exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        import csv
        w = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            w.writeheader()
        w.writerow(row_dict)

def ui_infer(image: Image.Image, seed: int, session_id: str, log_path: str):
    if image is None:
        return "⚠️ Please upload an image first.", 0.0, "ok"
    start = time.perf_counter()
    status = "ok"
    try:
        maybe_set_seed(seed)
        safe_img = ensure_max_side(image)
        fn = infer_fn if infer_fn is not None else _placeholder_infer
        raw = fn(safe_img)
        out = sanitize_caption(raw)
    except Exception as e:
        status = f"error: {str(e)}"
        out = f"❌ Error: {str(e)}"
    dur = time.perf_counter() - start
    row = {
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "session_id": session_id,
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "seed": seed,
        "runtime_sec": round(dur, 3),
        "caption_len": len(out or ""),
        "status": status,
    }
    write_log_row(log_path, row)
    return out, dur, status

def make_share_text(caption: str, runtime: float):
    if not caption or caption.startswith(("⚠️", "❌")):
        return "No sharable caption yet."
    lines = [
        f"🖼️ Image Caption: {caption}",
        f"🧠 Model: {MODEL_NAME} (v{MODEL_VERSION})",
        f"⏱️ Runtime: {runtime:.2f}s",
        "#AI #Vision #Demo"
    ]
    return "\\n".join(lines)

CUSTOM_CSS = """
.gradio-container { --radius: 14px; }
.dragdrop { border: 2px dashed #9aa0a6; border-radius: 14px; transition: border-color .2s, box-shadow .2s; }
.dragdrop:hover { border-color: #5562ff; box-shadow: 0 6px 24px rgba(0,0,0,.08); }
:root { --bg: #ffffff; --fg: #111827; --muted:#6b7280; }
.dark  { --bg: #0b1020; --fg: #e5e7eb; --muted:#9aa0a6; }
body, .gradio-container { background: var(--bg) !important; color: var(--fg) !important; }
label, .prose, .markdown-body { color: var(--fg) !important; }
"""

def build_ui():
    with gr.Blocks(theme=gr.themes.Soft(), css=CUSTOM_CSS, title=APP_TITLE) as demo:
        gr.HTML(
            f"""
            <script>
              document.title = "{APP_TITLE}";
              (function() {{
                const link = document.createElement('link');
                link.rel = 'icon';
                link.href = "{FAVICON_SVG_DATA_URL}";
                document.head.appendChild(link);
                const meta = document.createElement('meta');
                meta.name = 'description';
                meta.content = 'Caption images with a clean, production-ready Gradio UI.';
                document.head.appendChild(meta);
              }})();
            </script>
            """,
            elem_id="head-inject"
        )

        gr.Markdown("## 🖼️ Image Caption Generator\\nPolished, shareable, and product-ready.")

        session_id = gr.State(str(uuid4()))
        total_requests = gr.State(0)
        ok_requests = gr.State(0)
        last_runtime = gr.State(0.0)
        log_path = gr.State(os.path.join("logs", f"session_{uuid4().hex}.csv"))
        theme_state = gr.State("light")

        with gr.Row():
            toggle_theme = gr.Button("🌓 Toggle Dark/Light")
            gr.Markdown("")

        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                image = gr.Image(
                    label="📤 Upload or drag & drop",
                    type="pil",
                    height=280,
                    format="png",
                    elem_classes="dragdrop"
                )
                with gr.Row():
                    seed = gr.Number(value=DEFAULT_SEED, precision=0, label="Seed (optional)")
                with gr.Row():
                    generate_btn = gr.Button("✨ Generate", variant="primary")
                    clear_btn = gr.ClearButton(value="🗑️ Clear")

                gr.Examples(
                    examples=[
                        ["examples/dog.jpg"],
                        ["examples/food.jpg"],
                        ["examples/city.jpg"],
                    ],
                    inputs=[image],
                    label="Examples"
                )

            with gr.Column(scale=1):
                caption = gr.Textbox(
                    label="📑 Caption",
                    lines=3,
                    placeholder="Your caption will appear here…"
                )
                copy_btn = gr.Button("📋 Copy caption")

                with gr.Accordion("📊 Run Info", open=False):
                    model_info = gr.Markdown(
                        f"**Model:** `{MODEL_NAME}`  |  **Version:** `{MODEL_VERSION}`"
                    )
                    runtime_md = gr.Markdown("**Last runtime:** —")
                    counters_md = gr.Markdown("**Requests:** 0 total, 0 ok")

                with gr.Accordion("🔗 Share Result", open=True):
                    share_text = gr.Textbox(
                        label="Share-ready text",
                        lines=5,
                        placeholder="Run once to generate a shareable snippet…"
                    )
                    with gr.Row():
                        make_share_btn = gr.Button("🧾 Make Share Text")
                        copy_share_btn = gr.Button("📋 Copy Share Text")

                with gr.Row():
                    export_btn = gr.Button("⬇️ Export session logs (CSV)")
                    exported_file = gr.File(label="Download logs", visible=False)

        def update_counters(last_status, total, ok):
            total += 1
            if isinstance(last_status, str) and last_status.startswith("ok"):
                ok += 1
            return total, ok, f"**Requests:** {total} total, {ok} ok"

        gen = generate_btn.click(
            fn=ui_infer,
            inputs=[image, seed, session_id, log_path],
            outputs=[caption, last_runtime, gr.State()],
            show_progress="full",
            api_name="generate",
            concurrency_limit=CONCURRENCY,   # ✅ per-event limit
        )

        gen.then(
            fn=lambda dur: f"**Last runtime:** {dur:.3f}s",
            inputs=last_runtime,
            outputs=runtime_md
        ).then(
            fn=update_counters,
            inputs=[gr.State("ok"), total_requests, ok_requests],
            outputs=[total_requests, ok_requests, counters_md]
        )

        clear_btn.add([image, caption])
        image.change(fn=lambda _: "", inputs=image, outputs=caption)

        copy_btn.click(fn=lambda x: x, inputs=caption, outputs=None, js="navigator.clipboard.writeText")
        copy_share_btn.click(fn=lambda x: x, inputs=share_text, outputs=None, js="navigator.clipboard.writeText")

        make_share_btn.click(
            fn=lambda cap, dur: make_share_text(cap, dur),
            inputs=[caption, last_runtime],
            outputs=share_text
        )

        def finalize_and_expose(log_fp):
            if not os.path.exists(log_fp):
                write_log_row(log_fp, {
                    "timestamp":"", "session_id":"", "model_name":"", "model_version":"",
                    "seed":"", "runtime_sec":"", "caption_len":"", "status":"no-requests"
                })
            return log_fp, gr.update(visible=True)

        export_btn.click(
            fn=finalize_and_expose,
            inputs=[log_path],
            outputs=[exported_file, exported_file]
        )

        toggle_theme.click(
            fn=lambda mode: ("dark" if mode == "light" else "light"),
            inputs=theme_state,
            outputs=theme_state
        ).then(
            fn=None, inputs=None, outputs=None,
            js="""
            () => {
              const root = document.documentElement;
              root.classList.toggle('dark');
              return null;
            }
            """
        )

        gr.Markdown("---")
        gr.Markdown("🏁 **Phase-6 adds:** favicon/title/meta, dark/light toggle, CSS polish, and share-ready text.")

    # ✅ Queue without deprecated args
    demo = demo.queue(
        max_size=QUEUE_SIZE,
        status_update_rate=0.1
    )
    return demo

if __name__ == "__main__":
    demo = build_ui()
    demo.launch(max_threads=CONCURRENCY)
