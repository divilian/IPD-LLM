# Perplexity on gossip

Short answer: the idea of “neighbor 1 telling you how neighbor 2 behaves” in a networked PD has been explored under the labels of reputation, indirect reciprocity, and reputation-based rewiring, but it has not (so far) been combined with LLM-enhanced agents.

# 1. Rule‑based agents with neighbor information / reputation

Several papers already do something very close to what you describe:

## Information about how good or bad neighbors are

Tanimoto and coauthors study a spatial PD where players can use information about how cooperative/defective their neighbors are (i.e., indirect reciprocity layered on top of network reciprocity).

The core idea is: a player can condition its behavior not only on direct experience, but on an evaluation of neighbors’ cooperativeness (“good” vs “bad”) that can be informed by observation of others’ interactions.

## Second‑order reputation in spatial PD

Recent work on “second‑order reputation promotes cooperation in the spatial Prisoner’s Dilemma game” studies reputation scores that depend on both what an individual did and who they did it to (the target’s reputation), in a structured population.

This is effectively: “I care what you did to others, and I may learn that from the local social structure,” which is very close to neighbor‑to‑neighbor information about third parties.

## Reputation-based network rewiring

There are PD models on dynamic networks where agents break links to low‑reputation neighbors and reconnect to higher‑reputation ones. For example, “Reputation-based disconnection–reconnection mechanism in the Prisoner’s Dilemma Game within dynamic complex networks” uses a reputation score (based on cooperation frequency) to drive link removal and formation.

These works show:

* Agents on a network.
* IPD or repeated PD interactions with neighbors.
* Information about others’ past cooperative/defective behavior being shared or aggregated as reputation, then used to guide cooperation and/or rewiring.

So: the reputation / neighbor-information mechanism itself is not novel in rule‑based PD-on-network models.

# 2. LLM‑enhanced agents with such information

On the LLM side:

## LLMs in Prisoner’s Dilemma

Recent work like “How do Large Language Models Behave in the Prisoner’s Dilemma?” investigates the cooperative behavior of LLMs (Llama‑2, Llama‑3, GPT‑3.5) in one‑shot and iterated PD, but without an explicit interaction network or multi‑hop reputation flow.

## LLM ABMs and social simulations

LLM‑empowered agent‑based models and social simulations embed LLM agents in social networks, but they rarely (if ever) formalize payoffs as PD and reputation in the sense of “neighbor 1 tells you how neighbor 2 has behaved; you then update strategy and links based on that.”

I do not find any published work that:

1. Puts agents on a network playing IPD with neighbors,
1. Allows them to share third‑party behavioral information (explicit “neighbor 2 has been defecting/cooperating”), and
1. Delegates the decision (cooperate/defect, and possibly rewire) to an LLM that consumes that local reputational information.

So:

With traditional rule‑based agents: your information‑sharing idea has precedents under reputation/indirect reciprocity and reputation‑based network rewiring.

With LLM‑enhanced agents on a networked IPD with such information: this combination appears novel as of the current literature.

If you like, next step could be: I can help you define the minimal reputation signal you would give the LLM (e.g., “fraction of times each neighbor cooperated over last kk rounds”) so that it’s close in spirit to these reputation models but still clearly distinct and LLM‑driven.
