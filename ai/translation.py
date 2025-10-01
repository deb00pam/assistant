
from googletrans import Translator
_translator = Translator()

def detect_language(text: str, api_key: str = None) -> str:
    """Detect the language of the input text using googletrans."""
    result = _translator.detect(text)
    print(f"[DEBUG] translation_utils.detect_language: '{text}' -> {result.lang}")
    return result.lang

def translate_text(text: str, target: str, source: str = None, api_key: str = None) -> str:
    """Translate text to the target language using googletrans."""
    if source:
        result = _translator.translate(text, src=source, dest=target)
    else:
        result = _translator.translate(text, dest=target)
    print(f"[DEBUG] translation_utils.translate_text: '{text}' ({source or 'auto'}) -> '{result.text}' ({target})")
    return result.text
