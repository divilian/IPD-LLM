# Prisoner's Dilemma agenda

## Student-competition-related

1. Actually charge `inform_foaf()` costs.

## SCC paper related

### Prompt Ablations

How much should I tell it about the game at once?

- give it specific curated metrics (mutual cooperation rate, last n moves, etc.) for use in deciding, or just a raw game history?

- valorizing "mutual cooperation"

Now I can _measure_ how much I'm "steering" towards a particular strategy.

### Misc

- [ ] Capture rewiring decisions with datacollector
- [ ] Lower temperature (this is about policy calls, not creative output. Lower
  temp doesn't mean fancier strategies aren't available, but rather that it'll
  be less stochastic in choosing from its learned distribution.)
- [ ] Spin up a GC instance to run at scale
- [ ] Enable LLMs to rewire their local graph and to request info from peers
    - [ ] Enable agents to lie and to refuse
- [ ] Calculate action metrics (C vs D's; total $ earned; number of rewires;
  etc) based on agent type
- [ ] Experiment with a small number of different prompts
- [ ] Store animation frames for fast replay


## Performance

- [ ] _If_ it turns out to be empirically slow, consider asychronous agents
  with batch decision making (all opponents at once).

## Bug fixes

- [ ] Fix typehint everywhere on `payoff_matrix`: should be `dict[tuple[str, str], tuple[int, int]]`
- [ ] How does `--avg-degree` combine with `--p-same/diff`?
- [x] There's still some LLM-related stuff in the IPDAgent base class
- [ ] Add payoffs to global prompt text
- [x] Break into multiple files
- [x] Implement `OllamaClient` class, with `async batch_decide(payloads)`.
- [x] refactor `agent/factory.py` so that all subclasses can be easily
  instantiated in the same way, with args. (See "Code Refactoring Advice" chat
  in IPD sim ChatGPT project)

## Improvements

* [ ] Make Browser "patience" a command-line arg

### Original-LLM-design-specific

- [ ] Figure out why LLMs are so slow now! Increased prompt size? (Chat says
  4,363 characters isn't all that long.)
- [ ] Do I actually tell the LLM what "Prisoner's Dilemma" even *is*??
- [ ] Rewire the entire original LLM design and delete it. (The `backend`
  hierarchy still seems useful.)
- [ ] Consider OllamaBackend system prompt. it's too generic, right?
- [ ] Add "outlines" (`from outlines.models.ollama import Ollama`) or Pydantic
  (`from pydantic import BaseModel`) to enforce JSON adherence

## Diagnostics

- [ ] Change analyze mode to show all history with any opponent you were _ever_
  connected to, not just who you're connected to at game end
- [x] Add agent query mode, where you can interactively type a node number and
  get its number and types of neighbors
- [ ] Analyze on a per-dyad basis
- [x] Print agent-type-based adjacency matrix, before and after
* [ ] How does the patience parameter impact how well the Browsers do?

## Model

- [ ] Agents which seek to break/form connections with better players
    - [ ] Should agents have to approve friend requests? Interesting!
- [ ] LLMAgents which seek to lose
- [ ] LLMAgents which seek to maximize total combined winnings, not just theirs
- [ ] `AmnesiacAgent` (remembers past plays, but not by whom. tit-for-tats)
- [x] Deterministic AgentFactory
- [x] Give LLM agents the history
- [x] Enable multiple personas

## Independent variables

- [x] ER edge freq
- [x] SBM matrix
- [ ] Different graph types
    - [x] SBM
    - [ ] BA

## Questions

- [ ] When enough TFTs interact, can they beat Mean?
    - [ ] Does a stochastic block model, putting TFTs with TFTs and Means with
      Means, tip it so that TFTs win?
    - [ ] In a Barabasi-Albert, what's the effect of putting TFTs on high-freq
      nodes vs. Means?
- [ ] Why do LLMtft's do worse than TFT's? Shouldn't they be the same?

## Re-code in Concordia and Casevo

