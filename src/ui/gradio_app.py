"""Gradio UI — agentic legal RAG chat with Sarvam voice & translation."""
from __future__ import annotations

import time
import traceback

import gradio as gr

from src.config import SARVAM_LANGUAGES, UI_TO_BCP47
from src.core.chat_interface import ChatInterface
from src.core.rag_system import RAGSystem
from src import sarvam_client
from src import query_logger


# ── Voice helpers ────────────────────────────────────────────────────────

def _stt(audio_tuple) -> str:
    """Convert Gradio mic audio → text via Sarvam STT."""
    if audio_tuple is None:
        return ""
    sr, samples = audio_tuple
    wav = sarvam_client.numpy_audio_to_wav_bytes(samples, sr)
    resp = sarvam_client.speech_to_text_file(wav, mode="translate")
    return sarvam_client.transcript_from_stt_response(resp)


def _tts(text: str, lang: str):
    """Generate TTS audio (sr, numpy) for Gradio Audio component."""
    if not text.strip():
        return None
    plain = sarvam_client.strip_markdown_for_tts(text)
    bcp = UI_TO_BCP47.get(lang, "en-IN")
    wav_bytes = sarvam_client.text_to_speech_wav_bytes(plain, target_language_code=bcp)
    return sarvam_client.wav_bytes_to_numpy_float32(wav_bytes)


def _translate_if_needed(text: str, lang: str, direction: str = "to_en") -> str:
    """Translate text via Sarvam Mayura if language is not English."""
    if lang == "en" or not sarvam_client.is_configured():
        return text
    bcp = UI_TO_BCP47.get(lang, "en-IN")
    if direction == "to_en":
        return sarvam_client.translate_text(text, source_language_code=bcp, target_language_code="en-IN")
    return sarvam_client.translate_text(text, source_language_code="en-IN", target_language_code=bcp)


# ── App factory ──────────────────────────────────────────────────────────

def create_gradio_ui() -> gr.Blocks:
    rag_system = RAGSystem()
    rag_system.initialize()
    chat_interface = ChatInterface(rag_system)

    # ── event handlers ───────────────────────────────────────────────────

    def voice_to_text(audio, lang):
        if audio is None:
            return ""
        try:
            transcript = _stt(audio)
            if lang != "en" and sarvam_client.is_configured():
                return transcript  # STT mode=translate already gives English
            return transcript
        except Exception:
            return "⚠️ Voice recognition failed"

    def chat_handler(message: str, history: list, lang: str):
        if not message or not message.strip():
            yield history
            return

        t0 = time.time()
        query_en = _translate_if_needed(message, lang, "to_en")

        history = history + [{"role": "user", "content": message}]
        yield history

        final_text = ""
        try:
            for chunk in chat_interface.chat(query_en, history):
                if isinstance(chunk, str):
                    final_text = chunk
                    display = _translate_if_needed(chunk, lang, "from_en") if lang != "en" else chunk
                    yield history + [{"role": "assistant", "content": display}]
                elif isinstance(chunk, list) and chunk:
                    last_plain = next(
                        (m for m in reversed(chunk) if m.get("role") == "assistant" and "metadata" not in m),
                        None,
                    )
                    if last_plain:
                        final_text = last_plain["content"]
                    display_msgs = []
                    for m in chunk:
                        if "metadata" not in m:
                            content = m["content"]
                            if lang != "en" and m.get("role") == "assistant" and len(content) > 20:
                                try:
                                    content = _translate_if_needed(content, lang, "from_en")
                                except Exception:
                                    pass
                            display_msgs.append({"role": m["role"], "content": content})
                        else:
                            display_msgs.append(m)
                    yield history + display_msgs
        except Exception as e:
            traceback.print_exc()
            yield history + [{"role": "assistant", "content": f"❌ Error: {e!s}"}]
            return

        elapsed_ms = int((time.time() - t0) * 1000)
        try:
            query_logger.log_query(
                user_lang=lang, query_text=message, query_en=query_en,
                domain_detected="legal", response_en=final_text,
                response_time_ms=elapsed_ms,
            )
        except Exception:
            pass

    def tts_handler(history, lang):
        if not history:
            return None
        last_assistant = next(
            (m["content"] for m in reversed(history) if m.get("role") == "assistant" and "metadata" not in m),
            None,
        )
        if not last_assistant:
            return None
        try:
            bcp = UI_TO_BCP47.get(lang, "en-IN")
            text_for_tts = last_assistant
            if lang != "en":
                try:
                    text_for_tts = _translate_if_needed(last_assistant, lang, "from_en")
                except Exception:
                    pass
            return _tts(text_for_tts, lang)
        except Exception:
            return None

    def clear_handler():
        chat_interface.clear_session()
        return [], None

    # ── layout ───────────────────────────────────────────────────────────

    with gr.Blocks(title="Nyaya Sahayak — Indian Legal Assistant", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# ⚖️ Nyaya Sahayak\n"
            "*AI-powered Indian Legal Information Assistant*\n\n"
            "Ask about **BNS sections**, **Constitutional articles**, or **Supreme Court judgments**. "
            "Supports voice input and 13 Indian languages via Sarvam AI."
        )

        with gr.Row():
            lang_dd = gr.Dropdown(
                choices=SARVAM_LANGUAGES, value="en", label="Language", scale=1,
            )

        chatbot = gr.Chatbot(
            height=600,
            type="messages",
            placeholder="Ask a legal question…",
            show_label=False,
            layout="bubble",
        )

        with gr.Row():
            txt = gr.Textbox(
                placeholder="Type your legal question here…",
                show_label=False, scale=4, container=False,
            )
            send_btn = gr.Button("Send", variant="primary", scale=1)

        with gr.Row():
            mic = gr.Audio(sources=["microphone"], type="numpy", label="🎤 Voice Input")
            tts_btn = gr.Button("🔊 Read Aloud", scale=1)
            tts_out = gr.Audio(label="Response Audio", autoplay=True)

        clear_btn = gr.Button("🗑️ Clear Chat")

        # ── wiring ───────────────────────────────────────────────────────
        mic.stop_recording(voice_to_text, [mic, lang_dd], [txt])

        send_btn.click(chat_handler, [txt, chatbot, lang_dd], [chatbot]).then(
            lambda: "", None, [txt],
        )
        txt.submit(chat_handler, [txt, chatbot, lang_dd], [chatbot]).then(
            lambda: "", None, [txt],
        )

        tts_btn.click(tts_handler, [chatbot, lang_dd], [tts_out])
        clear_btn.click(clear_handler, None, [chatbot, tts_out])

    return demo
