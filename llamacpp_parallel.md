# How to do parallelism in ABMs with llamacpp.

## Instantiation

This is common to all strategies:

```
from llama_cpp import Llama

llm = Llama(
    model_path="model.gguf",
    n_gpu_layers=-1,
    n_threads=8,
    n_batch=512,
    n_ctx=4096,
)
```

Notes:
* `n_threads` means number of CPU threads used for inference
* `n_batch` does not mean prompt batching. It controls prompt token processing
  chunk size, not multi-prompt batching.
* passing -1 to `n_gpu_layers` means "store _all_ layers in VRAM."

Example of using it:

```
resp = llm("Should the agent cooperate?")
print(resp["choices"][0]["text"])
```

## Strategies for quasi-parallelism

### 1. Strategy A — simplest (sequential)

Each agent calls the LLM one by one:

```
for agent in agents:
    agent.response = llm(agent.prompt)
```

This works but is slowest. (This is essentially what happens if agents simply
call the LLM directly inside `agent.step()` in Mesa.)

### 2. Strategy B — Python parallelism

Agents prepare prompts in parallel, then call an LLM service. (Abandoned this
option; it adds complexity without much benefit.)

### 3. Strategy C - centralized inference loop (prompt aggregation)

The fastest pattern is:

1. agents generate prompts
2. collect them
3. process them in a single centralized inference loop

Example:

```
prompts = [agent.prompt for agent in agents]

responses = [llm(p) for p in prompts]

# Then redistribute:
for agent, resp in zip(agents, responses):
    agent.response = resp

```

This still looks sequential but reuses the loaded model efficiently. And this
avoids loading multiple model copies.

#### Prompt caching

It's probably wise to skip the LLM call if you've already given that identical
prompt (after normalization, such as stripping whitespace) to an LLM
previously. So cache the results in a dict, where keys are prompts and values
are responses.

## Verdict

Strategy C is best for Mesa because it keeps the simulation logic separate from
LLM inference and avoids multiple model instances.
