import subprocess
import requests
import httpx
import time
import json

from .backend import LLMBackend, register_backend


@register_backend("ollama")
class OllamaBackend(LLMBackend):

    def __init__(self, model_name: str):
        self.model_name = model_name

    @classmethod
    def from_args(cls, args) -> "LLMBackend":
        return cls(model_name=args.ollama_model)

    async def batch_decide(self, prompt: str) -> dict:

        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.post(
                "http://127.0.0.1:11434/api/chat",
                json={
                    "model": self.model_name,
                    "messages": [
                        {"role": "system",
                         "content": "You are a strategic decision engine."},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.0,
                    "stream": False,
                },
            )

        r.raise_for_status()

        # The JSON that Ollama returns looks like this:
        # 
        # {
        #   "message": {
        #     "role": "assistant",
        #     "content": "{ \"decisions\": [...] }"
        #   }
        # }
        #
        # so we need to get the value of "content" and then treat *it* as a
        # JSON string, which we'll parse with json.loads().
        embedded_content = r.json()['message']['content']
        return json.loads(embedded_content)


def ensure_ollama_running(ollama_model: str):
    """
    Ensure Ollama daemon is running and the required model is available.
    If daemon is not running, start it.
    If model is not present, pull it.
    """

    def daemon_alive() -> bool:
        try:
            r = requests.get(
                "http://127.0.0.1:11434/api/tags",
                timeout=0.5,
            )
            return r.status_code == 200
        except requests.RequestException:
            return False

    # Start daemon (if needed).
    if not daemon_alive():
        print("Starting Ollama daemon...")
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )

        deadline = time.time() + 30
        while time.time() < deadline:
            if daemon_alive():
                print("Ollama daemon is ready.")
                break
            time.sleep(0.25)
        else:
            raise RuntimeError(
                "Ollama daemon did not start within 30 seconds."
            )

    # Ensure model is available.
    r = requests.get(
        "http://127.0.0.1:11434/api/tags",
        timeout=2.0,
    )
    r.raise_for_status()

    available_models = {
        m["name"]
        for m in r.json().get("models", [])
    }

    if ollama_model not in available_models:
        print(f"Model '{ollama_model}' not found locally.")
        print("Pulling model from Ollama registry...")
        subprocess.run(
            ["ollama", "pull", ollama_model],
            check=True,
        )
        print("Model pull complete.")


