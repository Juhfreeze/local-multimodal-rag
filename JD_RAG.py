from jd_rag.config import Settings
from jd_rag.indexing import Indexer
from jd_rag.models import list_chat_models
from jd_rag.chat import ChatSession


def choose_llm_model(settings: Settings) -> str:
    available = list_chat_models(settings)
    if available.error:
        print(f"\nCould not read Ollama model list: {available.error}")
        print(f"Using default model: {available.default}")
        return available.default
    chat_models = available.names
    if not chat_models:
        print(f"\nNo selectable chat models found. Using default: {settings.llm_model}")
        return settings.llm_model

    default_model = available.default
    print("\nAVAILABLE CHAT MODELS")
    for number, model in enumerate(chat_models, 1):
        marker = "  [default]" if model == default_model else ""
        print(f"{number}. {model}{marker}")
    while True:
        try:
            choice = input(f"Choose model number/name [Enter = {default_model}]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\nUsing default model: {default_model}")
            return default_model
        if not choice:
            return default_model
        if choice in chat_models:
            return choice
        if choice.isdecimal() and len(choice) < 10:
            index = int(choice) - 1
            if 0 <= index < len(chat_models):
                return chat_models[index]
        print("Invalid selection. Enter a listed number/name, or press Enter for the default.")


def main() -> None:
    settings = Settings()
    indexer = Indexer(settings, report=print)
    db_exists = indexer.prepare_storage()
    db = indexer.open_database()
    if not db_exists:
        print("\nYour database is empty.")
        choice = input("Index your documents now? (y/n): ").strip().lower()
    else:
        print("\nExisting database loaded.")
        choice = input("Check my_notes for new, modified or deleted files? (y/n): ").strip().lower()
    if choice in {"y", "yes"}:
        indexer.update_and_unload(db)
    else:
        print("Skipping document processing; starting chat with the existing index.")

    selected_model = choose_llm_model(settings)
    session = ChatSession(db, settings, selected_model)
    print(f"\nJD_RAG 0.2.0 ready. Indexed chunks: {db._collection.count()}")
    print(f"Chat model: {selected_model}")
    print(f"Embedding model: {settings.embed_model}")
    print("Ask about your notes. Type 'quit' to exit, or '/update' to rescan.")
    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if question.lower() in {"quit", "exit"}:
            break
        if question.lower() == "/update":
            indexer.update_and_unload(db)
            continue
        if not question:
            continue
        try:
            answer = session.ask(question)
            print(f"\n{answer.model}: {answer.text}")
            if answer.sources:
                print("\nRetrieved sources:")
                for source in answer.sources:
                    print(f"  - {source.label}")
        except Exception as exc:
            print(f"\nError: {exc}")


if __name__ == "__main__":
    main()
