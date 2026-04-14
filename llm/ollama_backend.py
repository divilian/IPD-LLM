import json
import requests

class OllamaBackend:
    def __init__(
        self,
        model_name: str,
        host: str = "http://localhost:11434",
        timeout: int = 120,
        seed: int = 123,
        num_ctx: int = 2048,
    ):
        self.model_name = model_name
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.seed = seed
        self.num_ctx = num_ctx

    def ensure_ollama_running(self) -> None:
        r = requests.get(f"{self.host}/api/tags", timeout=5)
        r.raise_for_status()

    def generate_text(
        self,
        prompt: str,
        *,
        system: str | None = None,
        response_format: str | dict | None = None,
        temperature: float = 0.0,
        num_predict: int = 128,
    ) -> str:
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "seed": self.seed,
                "temperature": temperature,
                "num_ctx": self.num_ctx,
                "num_predict": num_predict,
            },
        }

        if system is not None:
            payload["system"] = system

        if response_format is not None:
            payload["format"] = response_format

        r = requests.post(
            f"{self.host}/api/generate",
            json=payload,
            timeout=self.timeout,
        )
        r.raise_for_status()

        data = r.json()

        if "response" not in data:
            raise RuntimeError(f"Unexpected Ollama response payload: {data}")

        if data.get("done") is False:
            raise RuntimeError(f"Ollama did not finish generation: {data}")

        if data.get("done_reason") == "length":
            raise RuntimeError(
                "Ollama output was truncated (done_reason='length'). "
                "Increase num_predict."
            )

        return data["response"].strip()

    def generate_json(
        self,
        prompt: str,
        *,
        schema: dict,
        system: str | None = None,
        temperature: float = 0.0,
        num_predict: int = 128,
    ) -> dict:
        json_guardrail = (
            "Return only valid JSON matching the provided schema. "
            "Do not output any text outside the JSON object."
        )
        effective_system = (
            f"{system}\n\n{json_guardrail}" if system else json_guardrail
        )

        text = self.generate_text(
            prompt,
            system=effective_system,
            response_format=schema,
            temperature=temperature,
            num_predict=num_predict,
        )

        try:
            obj = json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Model returned invalid JSON: {text[:500]!r}"
            ) from e

        if not isinstance(obj, dict):
            raise ValueError(
                f"Expected top-level JSON object, got {type(obj).__name__}"
            )

        return obj
