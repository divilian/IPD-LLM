# Prisoner's Dilemma agenda

## Diagnostics

- [ ] Add agent query mode, where you can interactively type a node number and
  get its number and types of neighbors

## Model

- [ ] `AmnesiacAgent` (remembers past plays, but not by whom. tit-for-tats)
- [x] Deterministic AgentFactory

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
