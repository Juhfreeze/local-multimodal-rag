from dataclasses import dataclass
from .config import Settings

@dataclass
class ModelList:
    names: list[str]
    default: str
    error: str | None = None

def list_chat_models(settings: Settings, client=None) -> ModelList:
    """Discover installed models without terminal input or model downloads."""
    if client is None:
        import ollama
        client = ollama
    def field(value, name, default=None):
        return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)

    def base_name(name):
        return name.rsplit("/", 1)[-1].split(":", 1)[0]

    try:
        response = client.list()
        names = sorted({
            name for model in field(response, "models", [])
            if isinstance(name := field(model, "model") or field(model, "name"), str) and name
        })
        excluded = {base_name(settings.embed_model), base_name(settings.vision_model)}
        chat_models = []
        for name in names:
            if base_name(name) in excluded:
                continue
            try:
                capabilities = field(client.show(name), "capabilities")
            except Exception:
                capabilities = None  # Older Ollama servers may not expose capabilities.
            if capabilities is not None and "completion" not in capabilities:
                continue
            chat_models.append(name)
    except Exception as exc:
        return ModelList([], settings.llm_model, str(exc))

    default = settings.llm_model if settings.llm_model in chat_models else (
        chat_models[0] if chat_models else settings.llm_model
    )
    return ModelList(chat_models, default)
