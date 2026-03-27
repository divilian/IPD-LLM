# Prisoner's Dilemma agenda

## Performance

- [ ] Figure out why LLMs are so slow now! Increased prompt size? (Chat says
  4,363 characters isn't all that long.)

## Bug fixes

- [ ] There's some LLM-related stuff in the IPDAgent base class (`base.py`)
- [ ] Why do LLMtft's do worse than TFT's? Shouldn't they be the same?
- [ ] Add payoffs to global prompt text
- [x] Break into multiple files
- [ ] Investigate `llama_cpp` python package!! (instead of Ollama)
- [ ] Add "outlines" (`from outlines.models.ollama import Ollama`) or Pydantic
  (`from pydantic import BaseModel`) to enforce JSON adherence
- [x] Implement `OllamaClient` class, with `async batch_decide(payloads)`.
- [ ] Backend interface pattern (search Chat)
- [ ] Do I actually tell the LLM what "Prisoner's Dilemma" even *is*??
- [ ] dbl-check OllamaBackend system prompt. it's too generic, right?
- [ ] wire the different backends into command-line args
- [ ] refactor `agent/factory.py` so that all subclasses can be easily
  instantiated in the same way, with args. (See "Code Refactoring Advice" chat
  in IPD sim ChatGPT project)

## Diagnostics

- [x] Add agent query mode, where you can interactively type a node number and
  get its number and types of neighbors
- [ ] Analyze on a per-dyad basis

## Model

- [ ] `AmnesiacAgent` (remembers past plays, but not by whom. tit-for-tats)
- [x] Deterministic AgentFactory
- [x] Give LLM agents the history
- [x] Enable multiple personas

## Independent variables

- [ ] ER edge freq
- [ ] Different graph types
    - [x] SBM
    - [ ] BA

## Questions

- [ ] When enough TFTs interact, can they beat Mean?
    - [ ] Does a stochastic block model, putting TFTs with TFTs and Means with
      Means, tip it so that TFTs win?
    - [ ] In a Barabasi-Albert, what's the effect of putting TFTs on high-freq
      nodes vs. Means?

## Re-code in Concordia and Casevo

