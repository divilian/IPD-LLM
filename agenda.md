# Prisoner's Dilemma agenda

## Bug fixes

- [ ] Why do LLMtft's do worse than TFT's? Shouldn't they be the same?
- [ ] Add payoffs to global prompt text
- [ ] Break into multiple files
- [ ] Investigate `llama_cpp` python package!! (instead of Ollama)
- [ ] Add "outlines" (`from outlines.models.ollama import Ollama`) or Pydantic
  (`from pydantic import BaseModel`) to enforce JSON adherance

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

