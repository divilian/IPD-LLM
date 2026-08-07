# Experiment Specification: LLMs in Networked Iterated Prisoner’s Dilemma

## 1. Purpose and scope

This document specifies the experimental design for the SSC 2026 project **“LLMs in Networked Iterated Prisoner’s Dilemma.”**

The project studies conventional rule-based agents and large-language-model agents engaged in repeated Prisoner’s Dilemma games on a dynamic social network. Agents may be able to leave existing partners, obtain reputational information about possible replacement partners, and form new ties.

The principal research question is:

> How do LLM agents use reputational information when choosing partners, and how does their behavior change when information may be unavailable or deceptive?

The SSC study is intended to isolate four related mechanisms:

1. the ability to leave an undesirable partner;
2. the ability to use truthful reputation when selecting a replacement;
3. the possibility that reputation is withheld;
4. the possibility that reputation is deliberately corrupted.

This specification governs the SSC implementation and experiments. It does not include the deferred counterfactual-reputation-replay experiment or a large-scale qualitative study of agent rationales.

---

## 2. Research questions

### 2.1 Primary question

How do LLM agents use reputational information when choosing partners, and how does their behavior change when information may be unavailable or deceptive?

### 2.2 Subsidiary questions

1. How do LLM agents’ cooperation rates and payoffs compare with those of conventional rule-based agents?
2. Do agents use rewiring to leave poor partners and obtain better ones?
3. Does truthful reputation improve the quality of newly acquired partners?
4. What is lost when informants may refuse to provide reputation information?
5. What additional harm occurs when informants may deceive?
6. When LLM agents receive reputation requests, how often do they answer truthfully, refuse, or deceive?
7. How do partner choice and reputation affect degree, isolation, network churn, assortative mixing, and exposure to defectors?
8. Are any benefits from selective partnering offset by reduced access to interaction or by information-related costs?

---

## 3. Experimental contrasts

The principal comparisons are defined before any full experimental runs are conducted.

| Contrast | Interpretation |
|---|---|
| Condition 2 − Condition 1 | Effect of allowing exit based on direct experience |
| Condition 3 − Condition 2 | Effect of adding mandatory truthful reputation for replacement choice |
| Condition 4 − Condition 3 | Effect of permitting reputation reports to be withheld |
| Condition 5 − Condition 4 | Incremental effect of permitting deception |
| Condition 5 − Condition 3 | Combined effect of permitting withholding and deception |

The first four adjacent-condition contrasts are the principal contrasts. Condition 5 versus Condition 3 is a secondary summary contrast.

The primary unit of replication is the **initialization seed**. Individual actions, interactions, reports, edges, and agents are not independent experimental replications.

---

## 4. Terminology

### 4.1 Action

An **action** is one agent’s choice of C or D against one opponent during one interaction.

### 4.2 Interaction

An **interaction** is one simultaneous dyadic event in which two adjacent agents each choose an action and receive the corresponding Prisoner’s Dilemma payoff.

### 4.3 Simulation round

A **simulation round** is one complete pass through the phases in Section 9. During the action and interaction phases, every current edge generates one interaction.

### 4.4 Repeated game

A **repeated game** is the complete sequence of interactions between a particular pair of agents during a run, including interactions before and after any period of disconnection and reconnection.

### 4.5 Tie episode

A **tie episode** is one uninterrupted period during which a particular pair of agents remains adjacent.

### 4.6 Direct history

An agent’s **direct history** with another agent is the ordered sequence of interactions that the two agents have personally had with one another.

For an agent $i$, its history with agent $j$ includes:

- $i$'s own actions against $j$;
- $j$'s actions against $i$;
- the resulting interaction payoffs;
- the simulation rounds in which those interactions occurred.

Agents retain direct histories after an edge is removed.

### 4.7 Reputation report

A reputation report is information supplied by an **informant** about the informant’s direct history with a **subject**.

The truthful content of a report is the sequence of actions that the informant personally observed the subject take against the informant.

An informant cannot truthfully report interactions that it did not observe.

### 4.8 Rewiring actor

A rewiring actor is an agent considering the replacement of one existing edge with one new edge.

### 4.9 Eligible candidate

An eligible candidate is a non-neighbor whom the rewiring actor is permitted to select as a replacement partner under the eligibility rules in Section 11.

### 4.10 Successful rewire

A successful rewire is an atomic network update in which:

1. one existing edge incident to the rewiring actor is removed;
2. one new edge incident to the actor is added;
3. the actor’s degree is unchanged;
4. total network edge count is unchanged.

---

## 5. Experimental design

The experiment uses a **seed-blocked matched-initialization design**.

For each initialization seed:

1. generate one initial network;
2. assign agent types to nodes;
3. initialize all agent state that exists before the first simulation round;
4. save the resulting graph, placement, and initial state explicitly;
5. run the saved initialization under all five experimental conditions.

Each initialization seed therefore defines a matched set of five runs.

The implementation must not depend only on reseeding a single global random-number generator. The initial edge list, node-to-agent assignment, agent-specific initialization, and relevant random streams must be saved or independently reproducible.

### 5.1 Random-number streams

Each initialization seed should deterministically generate separate pseudorandom-number generators for distinct simulation mechanisms, including:

- graph generation;
- agent placement;
- stochastic conventional-agent decisions;
- replacement selection in Condition 2;
- candidate sampling, if sampling is used;
- reputation corruption;
- rewiring conflict resolution;
- LLM sampling, when the inference system permits explicit seeding.

These are separate seeded generators, not merely separate calls to one shared generator. A random draw used for reputation corruption must not advance the generator later used for conflict resolution or random replacement.

The purpose is both reproducibility and independence between software mechanisms: adding or removing a random draw in one mechanism should not silently change unrelated random choices elsewhere in the same run.

The generated initial graph, node-to-agent assignment, and initial agent state must also be saved explicitly and reused across the five matched conditions. LLM output should be treated as potentially nondeterministic even when temperature is zero or an inference seed is available.

### 5.2 Identical population composition

All five conditions for a given initialization seed must use:

- the same number of agents;
- the same agent types;
- the same node assignments;
- the same initial graph;
- the same initial direct-history state;
- the same payoff matrix;
- the same number of scheduled simulation rounds.

Only the mechanisms explicitly varied by condition may differ.

---

## 6. Network model

### 6.1 Primary topology

The primary initial network is a connected Watts–Strogatz graph, generated using an implementation equivalent to:

```python
nx.connected_watts_strogatz_graph(n, k, p, seed=...)
```

This topology is used because it supplies local clustering and relatively short path lengths, making friends-of-friends and local reputation meaningful.

The parameters are:

- $n$: number of agents;
- $k$: initial degree;
- $p$: edge-rewiring probability used during graph generation.

The initial graph must be:

- simple;
- undirected;
- connected;
- free of self-loops;
- free of duplicate edges.

No agent may begin the simulation isolated.

### 6.2 Degree evolution

Individual degrees may diverge after rewiring.

A rewiring actor retains its own degree after a successful rewire, but:

- the abandoned agent loses one incident edge;
- the selected replacement gains one incident edge.

Agents may therefore become unusually popular, lose most of their partners, or become isolated.

These changes are substantive experimental outcomes and must not be artificially prevented.

### 6.3 Isolation

An agent may become isolated after the simulation begins.

Isolation is interpreted as endogenous exclusion or ostracism rather than as a simulation error.

An isolated agent:

- participates in no interactions while isolated;
- earns zero interaction payoff in those simulation rounds;
- retains memories of previous partners;
- cannot submit or receive a valid reputation request, because valid requests require current edges;
- cannot initiate a rewire, because it has no current edge to replace;
- may later re-enter the network if an eligible former partner selects it.

An isolated agent who has no eligible former partner cannot re-enter through the local friend-of-a-friend mechanism.

### 6.4 Robustness topology

The required SSC experiment uses only the Watts–Strogatz topology.

A reduced robustness analysis may use one of the following if time permits:

1. a random regular graph with matched $n$ and degree;
2. an Erdős–Rényi graph with matched $n$ and expected degree.

Barabási–Albert networks are excluded from the initial SSC study because their hubs and degree heterogeneity introduce an additional mechanism.

---

## 7. Agent population

### 7.1 Candidate agent types

The proposed research population contains the following nine types:

1. Always Cooperate;
2. Always Defect;
3. Tit-for-Tat;
4. Grim Trigger;
5. Win-Stay, Lose-Shift (Pavlov);
6. BrowserAgent;
7. DeviousAgent;
8. VengefulAgent;
9. LLMAgent.

Classroom tournament agents that do not appear in this list are not automatically included.

All conventional agents must have standardized, research-specific implementations. Their behavior must not depend on student-written code or classroom prompts.

### 7.2 Provisional population size

**Provisional SSC default:**

```text
36 agents: four agents of each of the nine types
```

This gives equal representation to every named type and permits the use of a regular initial degree such as $k=4$ or $k=6$.

The final population size and composition must be frozen after pilot work and before full-condition data collection begins.

### 7.3 Separation of decision policies

The implementation should distinguish, conceptually and preferably in code, between:

- a Prisoner’s Dilemma action policy;
- an information-request policy;
- a reputation-response policy;
- a tie-severing policy;
- a replacement-selection policy.

This avoids treating cooperation behavior, reporting behavior, and network behavior as a single indivisible strategy.

### 7.4 Participation in network decisions

**Provisional design decision:** every non-isolated agent may receive a rewiring opportunity under Conditions 2–5.

Every conventional agent type must have an explicit deterministic or explicitly stochastic algorithm for each decision opportunity it receives, including:

- whether to leave a neighbor;
- whether to request information when requests are available;
- which current neighbor or neighbors to ask;
- how to choose a replacement when candidate choice is available;
- how to answer reputation requests.

A conventional strategy may consistently choose not to use an available action.

LLM agents are not assigned hard-coded behavioral rules for these choices. Their decision procedure is instead defined by the model, frozen prompts, information supplied, available actions, output schema, inference settings, retry policy, and fallback policy. They choose among the permitted actions at runtime.

### 7.5 Conventional reporting policies

The final policies for BrowserAgent, DeviousAgent, and VengefulAgent must be specified as deterministic or explicitly stochastic algorithms before implementation.

Their names alone are not sufficient specifications.

In particular, the final specification must define:

- what information BrowserAgent requests;
- how BrowserAgent scores possible partners;
- whom DeviousAgent attempts to help or harm;
- when DeviousAgent lies rather than tells the truth;
- how VengefulAgent identifies a target of retaliation;
- whether VengefulAgent’s retaliation affects actions, reports, partner choice, or some combination;
- how these agents behave when deception or refusal is forbidden.

Until those policies are written, these three agent types remain an unresolved design dependency.

### 7.6 Behavior when actions are forbidden

Agents must never be offered an action that the current condition prohibits.

For example:

- in Condition 3, a reputation-response policy must return the complete truthful response;
- in Condition 4, it may return truth or refusal but not deception;
- in Condition 5, it may return truth, refusal, or deception.

A conventional policy that would prefer a forbidden action must follow a condition-specific fallback defined in advance. The simulator must not silently reinterpret an illegal action.

---

## 8. LLM agents

### 8.1 Memory model

LLM agents are memoryless across model calls.

Any persistent state used by an LLM agent must be stored by the simulator and supplied again in later prompts.

The LLM must not be assumed to remember:

- earlier prompts;
- previous responses;
- former partners;
- earlier rationales;
- past reports;
- previous simulation rounds.

### 8.2 Information supplied to the LLM

Depending on the decision phase and condition, an LLM prompt may include:

- the agent’s neutral identifier;
- the current simulation round;
- current neighbors;
- direct interaction histories;
- cumulative and recent payoffs;
- previously received reports about a candidate only after that candidate has been independently revealed in the current information phase;
- currently permitted actions;
- current neighbors available to receive an information request;
- former partners currently eligible for reconnection;
- candidate identities revealed by current information responses;
- source-attributed reputation reports received about revealed candidates;
- the request, refusal, and deception costs relevant to the decision;
- the allowed response schema.

Before an information request is answered, the requester must not be shown the identities of unknown friends-of-friends. Those identities are discovered only through responses from current neighbors.

The LLM must not receive:

- hidden strategy labels;
- source-code class names that disclose strategies;
- private histories between third parties except through valid reports;
- candidate identities that have not yet been lawfully revealed;
- information hidden by a refusal;
- true histories underlying deceptive reports;
- information from future simulation rounds;
- actions unavailable in the current condition.

### 8.3 Neutral identifiers

Agents must be identified using semantically neutral labels such as:

```text
A01
A02
A03
```

Identifiers must not encode strategy, condition, degree, or expected behavior.

### 8.4 Decision-call granularity

LLM call granularity should match the scientific structure of each decision rather than follow one universal batching rule.

| Phase | Provisional LLM call granularity |
|---|---|
| Prisoner’s Dilemma action | One call per opponent |
| Information-request selection | One call per requesting agent per information opportunity |
| Reputation response | One call per request |
| Tie severing and replacement | One call per rewiring actor per rewiring opportunity |

Separate action calls for different opponents must receive the same pre-action agent-level snapshot and must not reveal decisions already made elsewhere in the same phase.

Information-request and rewiring decisions remain joint calls because they require comparison among available options and enforcement of limits such as the maximum number of requests or at most one rewire.

Independent calls may be submitted concurrently or batched internally by the inference server. Application-level decision batching and inference-engine batching are distinct choices.

The LLM pilot must compare runtime, prompt-processing time, throughput, memory use, schema-failure rate, and substantive decision quality before the call granularity is finalized. Every response must use a strict machine-readable schema, preferably JSON.

### 8.5 Rationales

The LLM may be asked for a short rationale after each decision.

Rationales are logged for interpretive analysis but do not constitute the primary experimental evidence.

Prompts should require the decision in a dedicated structured field so that a rationale cannot be mistaken for an action.

### 8.6 Model and inference settings

Before the full experiment, the following must be frozen:

- provider;
- exact model identifier;
- model version or checksum where available;
- local quantization and runtime configuration;
- system prompt;
- phase prompts;
- temperature;
- top-$p$;
- maximum output tokens;
- random seed support;
- retry policy;
- invalid-output fallback policy.

The accepted abstract’s proposed local Llama 3.1 8B Instruct model remains a candidate rather than a final commitment.

The main model should be selected according to:

1. ability to follow the structured output format;
2. practical runtime;
3. reproducibility;
4. acceptable invalid-output rate;
5. ability to process the required histories and candidate sets.

A smaller commercial-model validation is optional and must not delay the primary SSC experiment.

---

## 9. Simulation-round structure

Each simulation round uses explicit decision phases.

### 9.1 Phase sequence

For simulation round $t$:

1. **Action decisions**
   - Each non-isolated agent chooses C or D against each current neighbor.
   - Decisions are based only on information available before the current interactions.

2. **Interaction resolution**
   - One Prisoner’s Dilemma interaction is resolved on every existing edge.
   - Both endpoints’ actions and payoffs are recorded.
   - Direct histories are updated.

3. **Information-request decisions**
   - Used only in Conditions 3–5 and only during scheduled rewiring opportunities.
   - Eligible agents decide which current neighbors, if any, to ask for information about those neighbors’ eligible neighbors.
   - The requester pays the request cost for each valid request delivered to an informant, regardless of whether the informant later answers truthfully, refuses, or deceives.

4. **Reputation-response decisions**
   - Informants respond according to the condition.
   - A non-refusal response may reveal eligible candidate identities and source-attributed histories.
   - Refusal and deception costs are charged where applicable; truthful reporting is free to the informant.

5. **Tie-severing and replacement decisions**
   - Agents with a rewiring opportunity decide whether to retain all current ties or replace one.
   - In Condition 2, an agent chooses only whether and whom to leave; the simulator chooses the replacement.
   - In Conditions 3–5, the agent may use former-partner memories and information delivered during the current information phase to select a replacement.

6. **Bulk rewiring resolution**
   - All proposals are validated against the same pre-rewiring graph.
   - A maximal compatible set of valid proposals is selected and applied atomically.
   - Superseded or rejected proposals leave their proposed old edge unchanged.

7. **Recording**
   - Events, costs, graph changes, state summaries, errors, and runtime information are written.

### 9.2 Timing of information

The direct history from simulation round $t$'s interactions is available during the information and rewiring phases of simulation round $t$.

Reports obtained in simulation round $t$ may therefore influence rewiring at the end of simulation round $t$.

New edges formed at the end of simulation round $t$ do not generate a Prisoner’s Dilemma interaction until simulation round $t+1$.

### 9.3 Simultaneity

All decisions in a phase must be collected before the simulator applies any decision from that phase.

An agent must not gain an advantage merely because its decision happened to be processed earlier in an iteration order.

---

## 10. Experimental conditions

### 10.1 Condition 1: Fixed network

Agents remain in repeated Prisoner’s Dilemma games with their initial neighbors.

The condition includes:

- no information requests;
- no reputation responses;
- no tie-severing decisions;
- no replacement-partner decisions;
- no edge additions or removals.

The graph must remain exactly equal to the initial graph for the entire run.

This condition establishes the fixed-network baseline.

### 10.2 Condition 2: Direct-experience exit with random replacement

Agents may replace an existing partner based only on their direct experience with current neighbors.

The agent:

- may decide not to rewire;
- may select one current neighbor to drop;
- is not shown reputational information;
- is not asked to choose among anonymous replacement candidates.

If a valid edge is dropped, the simulator selects one eligible replacement uniformly at random.

The agent chooses whether to exit and whom to leave. The simulator chooses the unknown replacement.

This condition isolates the effect of direct-experience exit without informed partner selection.

### 10.3 Condition 3: Mandatory truthful reputation

An agent may ask one or more current neighbors for information about those neighbors’ eligible neighbors, subject to the request limit.

For every valid request:

- the informant must answer;
- the response must identify all of the informant’s currently eligible neighbors for the requester;
- the response must contain the truthful reportable history for each revealed candidate;
- refusal is forbidden;
- deception is forbidden.

The requester may use the delivered candidate identities and reports when choosing a replacement partner.

This condition estimates the effect of perfectly available truthful local reputation.

### 10.4 Condition 4: Truthful reputation with possible refusal

An agent may ask one or more current neighbors for information about those neighbors’ eligible neighbors, subject to the request limit.

For every valid request, the informant may:

- provide the complete truthful response; or
- refuse.

The informant may not deceive or selectively omit candidates from an answer.

A refusal reveals no candidate identities or histories. The requester is told only that the informant refused. The informant pays the refusal cost.

This condition differs from Condition 3 only in the availability and consequences of refusal.

### 10.5 Condition 5: Corruptible reputation

An agent may ask one or more current neighbors for information about those neighbors’ eligible neighbors, subject to the request limit.

For every valid request, the informant may:

- provide the complete truthful response;
- refuse; or
- deceive.

If deception is selected, the simulator—not the responding agent—constructs the false histories using the corruption mechanism in Section 12. Candidate identities and current edges remain accurate: deception may corrupt reported C/D histories but may not omit real candidates, invent nonexistent candidates, or falsify which agents are currently adjacent.

A refusing informant pays the refusal cost. A deceiving informant pays the deception cost.

This condition differs from Condition 4 only in the availability and consequences of deception.

---

## 11. Rewiring mechanics

### 11.1 Rewiring frequency

An agent may replace at most one edge during a single rewiring phase.

The final experiment must freeze:

- the initial no-rewiring burn-in;
- the interval between rewiring opportunities;
- whether every eligible agent receives an opportunity at each rewiring phase;

**Provisional recommendation:** use a short initial burn-in and periodic rewiring rather than rewiring every simulation round. The exact values should be chosen through rule-based calibration and then held constant across Conditions 2–5.

### 11.2 Candidate eligibility

For an actor $i$, the eligible candidate set is the union of:

1. current friends-of-friends who are not currently adjacent to $i$;
2. former direct partners who are not currently adjacent to $i$ and whose reconnection cooldown has expired.

A current friend-of-a-friend candidate must be connected to $i$ through at least one current mutual neighbor. In Condition 2, the simulator computes this pool internally. In Conditions 3–5, $i$ learns the identities of such candidates only through responses from current neighbors during the current information phase.

An eligible candidate must also:

- not be $i$;
- not already be adjacent to $i$;
- not be the partner proposed for removal in the same rewire;
- not be prohibited by the reconnection cooldown;
- satisfy any candidate-pool cap or sampling rule specified for the experiment.

Merely having appeared in an earlier candidate list or reputation response does not make an agent permanently eligible. Such events may remain in memory and in the logs, but future eligibility still requires a current friend-of-a-friend path or a former direct partnership.

### 11.3 Former partners

Former partners remain in an agent’s memory.

They may become eligible candidates again after the reconnection cooldown expires.

### 11.4 Just-dropped partner

The partner selected for removal cannot be selected as the replacement in the same rewiring proposal.

### 11.5 Reconnection cooldown

**Provisional SSC policy:** after an edge is removed at the end of simulation round $t$, the two former partners may not reconnect during the rewiring phase of simulation round $t+1$. They become eligible to reconnect in simulation round $t+2$.

This one-simulation-round cooldown prevents a nominal rewire from being immediately undone without introducing a large patience parameter.

### 11.6 Re-entry after isolation

An isolated agent cannot be a current friend-of-a-friend because it has no current edges.

It may re-enter the network only if a former direct partner selects it after the reconnection cooldown has expired.

An isolated agent with no eligible former partner cannot re-enter under the local-information mechanism. The simulator must not create a special global rescue mechanism.

### 11.7 Empty candidate pools

If an actor elects to rewire but has no eligible replacement candidate, the proposal is invalid and the graph remains unchanged.

The simulator must not remove the old edge unless it can add a valid replacement edge atomically.

### 11.8 Condition 2 random selection

In Condition 2, the simulator samples uniformly from the actor’s complete eligible candidate pool, consisting of current friends-of-friends and eligible former partners.

The actor is not shown candidate identifiers before deciding whether and whom to leave.

### 11.9 Conditions 3–5 informed selection

At the beginning of an information opportunity, the actor knows its current neighbors, its direct histories, and any eligible former partners. It does not automatically know the identities of current friends-of-friends.

The actor may ask current neighbors for information. A non-refusal response reveals the informant’s currently eligible neighbors and supplies one source-attributed history for each revealed candidate. If multiple informants reveal the same candidate, their reports remain separate.

The actor may then choose among:

- eligible former partners known through direct history;
- current friends-of-friends revealed during the current information phase.

The prompt or conventional policy must distinguish:

- no request made;
- request made but refused;
- truthful or potentially corrupted history received;
- direct history personally observed by the actor.

The actor is not told whether a delivered history is truthful or deceptive.

If the actor requests no information, or if every requested neighbor refuses, the actor discovers no new friend-of-a-friend candidates during that opportunity. Merely learning about a candidate does not preserve that candidate’s eligibility in later simulation rounds.

### 11.10 Duplicate and conflicting rewiring proposals

All rewiring proposals are first generated and validated against the same pre-rewiring graph.

If two agents simultaneously propose adding the same new undirected edge, the edge is treated as a mutual or duplicate match and must be formed. A seeded random tie-break selects which complete proposal is applied:

- the selected proposer’s old edge is removed;
- the shared new edge is added;
- the other proposer’s proposed old edge remains unchanged.

If two proposals attempt to remove the same undirected edge, a seeded random tie-break selects which complete proposal is applied. The selected replacement edge is added, and the other proposed addition is not applied.

After duplicate additions and duplicate removals are resolved, a seeded random priority order is used to select a maximal compatible set from any remaining valid proposals.

For each accepted proposal:

- its old edge is removed;
- its new edge is added.

For each superseded or rejected proposal:

- neither its proposed removal nor its proposed addition is applied.

This preserves atomicity and constant total edge count while preventing fixed node order from determining which proposals succeed. Tie-break priorities and rejection or supersession reasons must be logged.

---

## 12. Reputation mechanism

### 12.1 Neighbor-directed request structure

A reputation request identifies:

- requester $X$;
- informant $Y$.

The requester does not identify a subject, because it does not yet know which eligible friends-of-friends the informant has.

A request is valid only if:

- requests are available in the current condition and simulation phase;
- $X$ and $Y$ are currently adjacent;
- the request does not exceed the per-opportunity request limit.

A valid request may produce an empty truthful response if $Y$ has no currently eligible neighbors for $X$.

For each current neighbor $Z$ of $Y$, $Z$ is included in a non-refusal response only if:

- $Z$ is not $X$;
- $Z$ is not already adjacent to $X$;
- $Z$ is not prohibited by the reconnection cooldown;
- $Z$ satisfies any candidate-pool cap or sampling rule.

Thus every revealed friend-of-a-friend candidate satisfies current edges $X-Y$ and $Y-Z$ at the time of the request.

### 12.2 Reportable history

For each revealed candidate $Z$, the truthful report contains the sequence of actions by $Z$ that $Y$ directly observed during interactions between $Y$ and $Z$.

The report should preserve order and may include simulation-round numbers.

It must not include:

- inferred motives;
- hidden strategy labels;
- interactions reported by someone else;
- global cooperation statistics unavailable to the informant.

When multiple informants report on the same candidate, each report must remain source-attributed and separate.

### 12.3 Truthful response

A truthful response reveals:

- the informant’s identity;
- the complete set of the informant’s currently eligible neighbors for the requester;
- the truthful reportable history for each revealed candidate.

Truthful reporting is the default baseline and carries no cost to the informant.

### 12.4 Refusal

A refusal delivers:

- the identity of the informant;
- a refusal indicator.

It reveals no candidate identities, histories, summaries, or metadata about information that the informant withheld.

The informant pays the refusal cost.

### 12.5 Deception

When the responder chooses deception, the simulator begins with the same complete candidate set and truthful reportable histories that a truthful response would contain.

Candidate identities and current adjacency are transmitted accurately. For each reported action independently:

- C is changed to D with probability $\alpha$;
- D is changed to C with probability $\alpha$;
- otherwise the action is left unchanged.

Deception may not selectively omit candidates, invent candidates, or falsify network ties.

The simulator records both the truthful and delivered histories, but the requester receives only the delivered histories. The informant pays the deception cost.

### 12.6 Corruption parameter

The corruption probability $\alpha$ is global within a run.

The principal SSC experiment will use one value of $\alpha$, selected through pilot work.

Additional values may be used as limited sensitivity analyses but do not form a required factorial dimension.

### 12.7 Zero-flip deception

Because corruption is stochastic, an agent may choose to deceive but the corruption process may change zero reported actions.

Such an event is still classified as an **attempted deception**.

The event log must separately record:

- the responder’s selected action;
- whether any reported actions were actually changed;
- the number and fraction of reported actions changed.

---

## 13. Request, refusal, and deception costs

### 13.1 Net payoff

An agent’s cumulative net payoff is:

$$
\text{net payoff}
=
\text{cumulative interaction payoff}
-
\text{request costs}
-
\text{refusal costs}
-
\text{deception costs}
-
\text{other preregistered costs}.
$$

No unrecorded cost may be introduced after full experiments begin.

### 13.2 Request cost

A request cost is charged to the requester for each valid request delivered to an informant.

The requester pays this cost whether the informant:

- answers truthfully;
- returns an empty truthful response;
- refuses;
- deceives.

The request cost represents the cost of seeking information, not a purchase price paid only for a successful report.

No request cost is charged for a structurally invalid request rejected before delivery. The final retry and fallback policy must specify how model or parsing failures are treated after a valid request has been delivered.

### 13.3 Truthful-reporting cost

Truthful reporting is the default baseline and costs the informant zero payoff units.

### 13.4 Refusal cost

A refusal cost is charged to the informant whenever the informant refuses a valid request.

This cost represents the consequence of withholding reasonably requested local information and prevents refusal from becoming a costless way to disadvantage other agents.

### 13.5 Deception cost

A deception cost is charged to the informant whenever the informant chooses deception.

The cost is charged even if the stochastic corruption process changes zero reported actions.

### 13.6 Cost calibration

The payoff matrix, request cost, refusal cost, and deception cost must be selected together before full runs.

The provisional ordering is:

$$
0 < c_{\text{refusal}} < c_{\text{deception}},
$$

with truthful reporting free. The request cost is separately positive and paid by the requester.

Pilot work should select costs that are:

- nontrivial relative to one interaction payoff;
- not so large that requests, refusals, or deception disappear entirely;
- not so small that requesting, refusing, or deceiving is effectively free.

The final report must express costs relative to the payoff matrix rather than presenting them as scale-free arbitrary numbers. All applicable costs must be stated explicitly in LLM prompts and made available to conventional policies before the corresponding decisions.

---

## 14. Prisoner’s Dilemma interactions

### 14.1 Interactions per edge

Each existing undirected edge generates exactly one simultaneous Prisoner’s Dilemma interaction per simulation round.

Each endpoint chooses one action against the other endpoint.

### 14.2 Payoff matrix

The payoff matrix must satisfy:

$$
T > R > P > S
$$

and

$$
2R > T + S.
$$

The exact numerical values remain to be selected.

All conditions must use the same matrix.

### 14.3 Action history

Action history is stored by ordered actor-opponent pair.

For each directed action, the simulator records:

- actor;
- opponent;
- actor action;
- opponent action;
- actor interaction payoff;
- simulation round.

### 14.4 Rule-based action policies

Always Cooperate, Always Defect, Tit-for-Tat, Grim Trigger, and Win-Stay, Lose-Shift must have concise mathematical or algorithmic definitions.

Win-Stay, Lose-Shift (Pavlov) is initialized with cooperation. Ordered by the previous outcome from the actor’s perspective as $CC$, $CD$, $DC$, and $DD$, its memory-one cooperation vector is:

$$
(1, 0, 0, 1).
$$

Thus it cooperates after $CC$ or $DD$ and defects after $CD$ or $DC$.

---

## 15. Invalid outputs, failures, and fallbacks

### 15.1 Structured validation

Every LLM response must be validated against a phase-specific schema.

Validation must check:

- required fields;
- permitted actions;
- known agent identifiers;
- candidate membership;
- one-action limits;
- valid C/D actions;
- condition-specific restrictions.

### 15.2 Retry policy

The experiment must use a fixed retry and repair policy.

A recommended policy is:

1. attempt the original structured prompt;
2. if invalid, issue one schema-repair prompt containing the invalid output;
3. if still invalid, use the preregistered fallback.

The retry prompt must not supply new substantive information.

### 15.3 Conservative fallbacks

Recommended fallbacks are:

| Phase | Fallback |
|---|---|
| Prisoner’s Dilemma action | Repeat the previous action against that neighbor; use C when there is no previous action |
| Information request | Make no request |
| Mandatory truthful response | Simulator supplies the complete truthful response |
| Optional response | Refuse |
| Rewiring | Keep all current ties |
| Candidate choice | Cancel the rewire |

These fallbacks must be selected before full runs. The cost rules must state whether a fallback refusal caused by an invalid LLM response incurs the refusal cost.

### 15.4 Failure reporting

Invalid outputs, retries, fallbacks, timeouts, and model errors must remain in the data.

Runs must not be silently discarded.

Before full experiments, define:

- the threshold at which a run is classified as failed;
- whether a failed run is rerun;
- whether the original and rerun are both retained;
- which analyses exclude failed runs;
- the sensitivity analysis for runs with elevated fallback rates.

---

## 16. Data collection

The simulator must produce append-only event-level logs sufficient to reconstruct decisions and network states.

JSONL may be used during execution. Completed data may be converted to Parquet.

### 16.1 Run metadata

Record at least:

- run ID;
- condition;
- initialization seed;
- all random-stream seeds;
- graph type and parameters;
- initial node list;
- initial edge list;
- node-to-agent assignment;
- population composition;
- payoff matrix;
- request cost;
- refusal cost;
- deception cost;
- corruption probability;
- burn-in and rewiring schedule;
- request limit;
- candidate eligibility rules;
- reconnection cooldown;
- model provider and identifier;
- model version or local checksum;
- prompt versions or hashes;
- LLM call-granularity rules;
- inference settings;
- output schemas;
- retry and fallback rules;
- code commit;
- dependency or environment record;
- start and completion time;
- runtime;
- completion or failure status.

### 16.2 Action events

For every directed action, record:

- run ID;
- simulation round;
- actor;
- opponent;
- actor type;
- actor action;
- opponent action;
- interaction payoff;
- direct-history length before the decision;
- rationale, where available;
- prompt reference;
- raw-response reference;
- validation status;
- retry count;
- fallback indicator;
- latency, where feasible.

### 16.3 Information-request events

Record:

- requester;
- informant;
- requester type;
- simulation round;
- current neighbors available to receive a request;
- direct information visible at decision time;
- request decision;
- request cost;
- validity;
- invalidity reason;
- the informant’s eligible neighbors as computed by the simulator but not visible to the requester before the response;
- rationale;
- prompt and response references.

### 16.4 Reputation-response events

Record:

- responder;
- requester;
- simulation round;
- permitted response actions;
- selected action: truth, refusal, or deception;
- complete eligible candidate set for that requester-informant pair;
- truthful reportable history for each eligible candidate;
- delivered candidate identities and histories;
- number of reportable actions;
- number and fraction changed;
- corruption probability;
- refusal cost;
- deception cost;
- rationale;
- validation, retry, and fallback fields;
- prompt and response references.

The truthful histories are retained for analysis but must not be exposed to the requester when deception occurs. Hidden candidate identities and histories must not be exposed when refusal occurs.

### 16.5 Rewiring events

Record:

- actor;
- actor type;
- simulation round;
- current neighbors;
- selected neighbor to drop;
- eligible candidate set;
- visible information about each candidate;
- proposed replacement;
- whether replacement was selected by the actor or simulator;
- proposed action;
- validation result;
- conflict-resolution priority;
- applied or rejected status;
- rejection reason;
- rationale;
- prompt and response references.

### 16.6 Network events

Record:

- every edge removal;
- every edge addition;
- the responsible rewiring proposal;
- simulation round;
- graph-level edge count after the bulk update.

Store:

- the complete initial edge list;
- all edge changes;
- the complete final edge list.

The data must be sufficient to reconstruct the graph at every simulation-round boundary.

### 16.7 State summaries

Simulation-round-level summaries may be generated for convenience, but they do not replace event logs.

Useful summaries include:

- total cooperation rate;
- number of interactions;
- number of requests;
- number of refusals;
- number of deceptions;
- number of attempted and successful rewires;
- number of isolated nodes;
- degree distribution;
- cumulative payoff by agent and type.

---

## 17. Outcome measures

Primary outcomes should be few, interpretable, and computed at the seed-condition level.

### 17.1 Quality of acquired partners

The primary measure of acquired-partner quality is the cooperation actually received from a newly acquired partner after the edge is formed.

For each new edge, calculate:

- cooperation received during the first $h$ interactions on that edge;
- interaction payoff earned during those interactions;
- duration of the edge;
- whether the new partner defects on its first interaction.

The value of $h$ must be frozen after pilot work.

A secondary lifetime measure uses all interactions before the edge is removed or the simulation ends.

### 17.2 Exposure to defection

For agent $i$:

$$
\text{defection exposure}_i
=
\frac{\text{interactions in which the opponent chose D}}
{\text{total interactions}}.
$$

This is undefined for an agent with no cumulative interactions. Because the initial graph is connected and interactions begin before endogenous isolation, this should not occur under the planned design unless a run fails before play begins.

### 17.3 Interaction access

Measure separately:

- degree by simulation round;
- interactions per simulation round;
- cumulative interactions;
- number and proportion of simulation rounds isolated;
- first simulation round of isolation;
- duration of isolation episodes;
- probability of ever becoming isolated;
- probability of re-entering after isolation.

### 17.4 Interaction quality

Measure separately:

- interaction payoff per interaction;
- cooperation received per interaction;
- defection exposure;
- quality of newly acquired partners;
- retention duration of new ties.

### 17.5 Overall performance

For each agent, record:

1. cumulative interaction payoff;
2. cumulative net payoff;
3. net payoff per elapsed simulation round;
4. interaction payoff per interaction;
5. cumulative net payoff per interaction;
6. total number of interactions.

The principal overall-payoff measure is net payoff per elapsed simulation round.

### 17.6 Undefined per-interaction outcomes

A simulation round in which an agent has no interactions has no defined round-specific payoff per interaction.

Undefined values must remain missing.

The implementation must not use transformations such as:

```python
payoff / max(interactions, 1)
```

Cumulative payoff per interaction remains defined for agents that interacted before becoming isolated.

### 17.7 Unreliable reputation

Measure:

- frequency of truthful reports;
- frequency of refusals;
- frequency of attempted deception;
- frequency of delivered reports containing at least one changed action;
- candidate rankings under true and delivered histories;
- selection of falsely praised candidates;
- rejection or nonselection of falsely condemned candidates;
- degradation in acquired-partner quality;
- degradation in payoff.

### 17.8 Secondary network outcomes

Secondary outcomes include:

- overall cooperation rate;
- cooperation rate by agent type;
- rewiring attempts and successes;
- information-request frequency;
- edge duration;
- network churn;
- degree trajectories;
- isolation by strategy;
- agent-type edge mixing matrix;
- assortativity by agent type or observed behavior;
- prevalence and duration of LLM–defector ties;
- concentration of incoming edges.

The term **dyad census** should not be used for ordinary counts of undirected agent-type pairings. Use edge composition or mixing matrix instead.

---

## 18. Statistical analysis

### 18.1 Seed-level summaries

For every outcome and condition:

1. calculate agent-level quantities;
2. aggregate agents within the relevant analysis group;
3. produce one summary value per initialization seed and condition.

Separate summaries should be produced for:

- all agents;
- LLMAgents;
- each conventional type;
- selected broader categories, if defined before analysis.

### 18.2 Within-seed contrasts

For each initialization seed, compute the prespecified condition contrasts.

For example:

$$
\Delta_{3-2,s}
=
Y_{s,3} - Y_{s,2},
$$

where $Y_{s,c}$ is the seed-level outcome for seed $s$ under condition $c$.

Report:

- mean contrast across seeds;
- median contrast;
- bootstrap confidence interval across seeds;
- distribution or paired plot of seed-specific contrasts.

### 18.3 Bootstrap procedure

Bootstrap initialization seeds, not individual actions, interactions, or edge events.

When a seed is sampled, all of its condition results must be sampled together.

This preserves the matched design.

### 18.4 Trajectory analysis

For time-varying outcomes, plot condition-level trajectories with uncertainty across seeds.

Potential trajectory measures include:

- cooperation rate;
- net payoff per simulation round;
- mean degree by agent type;
- isolation frequency;
- defection exposure;
- cumulative rewires;
- cumulative information costs.

Smoothing choices must not hide discontinuities caused by rewiring phases.

### 18.5 Action-level models

Action-level models are supplementary.

Possible models include:

- probability of cooperation;
- probability of requesting information;
- probability of leaving a neighbor;
- probability of choosing a candidate;
- probability of truth, refusal, or deception;
- duration of a new edge.

Such models must account for clustering by initialization seed and repeated decisions by agent.

They must not treat thousands of actions as thousands of independent experimental replications.

### 18.6 Planned LLM cooperation model

A supplementary model of LLM cooperation may use predictors such as:

$$
\Pr(\text{cooperate})
\sim
\text{condition}
+
\text{simulation-round fraction}
+
\text{opponent's previous action}
+
\text{opponent's recent cooperation}
+
(1 \mid \text{initialization seed})
+
(1 \mid \text{LLM agent}).
$$

The precise recent-history window and random-effects structure must be selected before fitting the final model.

### 18.7 Multiple outcomes

The study emphasizes effect sizes, uncertainty intervals, and consistency across seeds.

Primary outcomes and contrasts must be identified before inspection of the full results. Secondary and exploratory analyses must be labeled as such.

---

## 19. Pilot and calibration work

Pilot work occurs before the full SSC experiment.

### 19.1 Rule-based verification

Rule-based-only simulations must verify that:

- truthful responses reveal the complete eligible candidate set and exactly matching reportable histories;
- requesters do not see unknown friend-of-a-friend identities before a response reveals them;
- every revealed candidate is connected to the informant by a current edge;
- refusals reveal no candidate identities or histories;
- deception preserves candidate identities and current adjacency while changing only reported actions;
- deception changes reported actions independently with probability $\alpha$;
- attempted deception with zero changes is recorded correctly;
- request, refusal, and deception costs are each applied exactly once when applicable;
- truthful reporting costs the informant nothing;
- a requester pays for every valid delivered request, including one that is refused;
- Condition 1 never changes an edge;
- a successful rewire removes exactly one edge and adds exactly one edge;
- a failed rewire changes no edges;
- mutual proposals for the same new edge form that edge exactly once;
- duplicate removal proposals apply exactly one complete replacement proposal;
- edge count remains constant;
- no self-loops or duplicate edges are produced;
- agents do not see hidden strategy labels;
- each current edge generates exactly one interaction per scheduled simulation round;
- isolated agents do not submit or receive reputation requests;
- initial graphs and placements match across conditions;
- new edges begin interaction only in the following simulation round;
- invalid actions use the specified fallback;
- the graph can be reconstructed from the logs.

### 19.2 Matched-initialization verification

Automated tests should compare the five runs in each seed block before simulation round 1 and confirm equality of:

- node set;
- edge set;
- agent placement;
- initial agent state;
- payoff matrix;
- number of scheduled simulation rounds.

### 19.3 Calibration targets

Pilot runs should select and then fix:

- number of agents;
- initial degree;
- Watts–Strogatz $p$;
- number of simulation rounds;
- burn-in duration;
- rewiring interval;
- information-request limit;
- candidate-pool rule;
- report length or history window;
- request cost;
- refusal cost;
- deception cost;
- corruption probability;
- new-tie evaluation horizon $h$;
- LLM model;
- prompt wording;
- LLM call granularity;
- output schemas;
- retry and fallback rules.

### 19.4 Calibration criteria

Parameters should avoid regimes in which:

- almost no rewiring occurs;
- nearly every agent rewires at every opportunity;
- requests are never made;
- requests are effectively costless;
- nearly every informant refuses;
- refusal is never selected;
- deception is never selected;
- refusal or deception dominates because it is effectively free;
- the network fragments almost immediately;
- the network never changes meaningfully;
- prompts routinely exceed the model’s context;
- invalid outputs are common;
- LLM call granularity creates unacceptable latency or memory use;
- runtime makes the matched five-condition design infeasible.

Calibration is intended to find an informative and computationally feasible regime, not to optimize for a desired treatment effect.

### 19.5 LLM pilot

A small LLM pilot should evaluate:

- schema compliance;
- decision stability;
- prompt comprehension;
- hidden-label leakage;
- context length;
- latency and prompt-processing time;
- throughput and memory use under the proposed call granularity;
- fallback frequency;
- whether rationales correspond to the structured actions;
- whether agents recognize condition-specific restrictions and costs;
- whether separate per-opponent action calls and joint comparative calls represent the intended decision problems.

Pilot runs must not be included in the final confirmatory dataset.

---

## 20. Full SSC experiment

The full SSC experiment begins only after:

- all required policies are specified;
- parameters are frozen;
- prompts are versioned;
- pilot tests pass;
- the code commit is tagged or otherwise preserved;
- the planned outcome calculations have been tested on synthetic logs.

### 20.1 Required experiment

The required experiment consists of:

- one Watts–Strogatz parameterization fixed before full runs;
- one population composition fixed before full runs;
- all five conditions;
- a common set of initialization seeds;
- one primary LLM model;
- one corruption level fixed before full runs;
- one request, refusal, and deception cost structure fixed before full runs.

### 20.2 Number of seeds

The number of initialization seeds must be determined from runtime pilots and Monte Carlo uncertainty.

The final choice should provide enough independent seed blocks to estimate the principal within-seed contrasts with useful precision.

Increasing the number of seeds is preferable to adding unnecessary factorial dimensions.

### 20.3 Sensitivity analyses

Permitted SSC sensitivity analyses include a small number of targeted changes, such as:

- one lower or higher corruption probability;
- one alternative request cost;
- one alternative network topology;
- one smaller commercial model validation.

These analyses are secondary and must not displace the full five-condition primary experiment.

---

## 21. Reproducibility requirements

Every completed run must be traceable to:

- a code commit;
- a saved configuration;
- an initialization record;
- prompt files or hashes;
- exact model settings;
- raw event logs;
- completion and validation status.

The analysis pipeline should operate on immutable raw logs and write derived data separately.

No manual editing of individual event records is permitted.

Any run replacement or rerun must receive a new run ID and retain a link to the original failed or superseded run.

---

## 22. Deferred work

The following are explicitly outside the required SSC implementation:

1. **Counterfactual reputation replay**
   - Replaying identical decision states with alternative reputation presentations is deferred until after SSC.

2. **Large qualitative rationale analysis**
   - Rationales will be preserved.
   - Only a modest descriptive or illustrative analysis will be attempted for SSC if time permits.

3. **Large factorial experiments**
   - The SSC study will not cross many values of topology, corruption, costs, thresholds, population composition, and model choice.

4. **Barabási–Albert topology**
   - Hub formation and initial degree heterogeneity are deferred.

5. **Extensive commercial-model comparison**
   - At most a limited validation run is planned.

---

## 23. Decisions still requiring resolution

The following decisions must be settled before the specification is considered frozen.

| Decision | Current status |
|---|---|
| Final population size and count per type | Provisional: 36, four of each type |
| Formal BrowserAgent policy | Unresolved |
| Formal DeviousAgent policy | Unresolved |
| Formal VengefulAgent policy | Unresolved |
| Whether every agent type may request reports | Provisional: yes |
| Main LLM model | Unresolved pending pilot |
| Commercial validation model | Optional |
| Payoff matrix | Unresolved |
| Watts–Strogatz $k$ and $p$ | Unresolved pending calibration |
| Number of simulation rounds | Unresolved pending calibration |
| Initial burn-in | Unresolved pending calibration |
| Rewiring interval | Unresolved pending calibration |
| Maximum requests per opportunity | Unresolved |
| Candidate-pool cap or sampling | Unresolved |
| Reconnection cooldown | Provisional: one simulation round |
| Report-history length | Unresolved |
| Request cost | Unresolved pending calibration |
| Refusal cost | Unresolved pending calibration |
| Deception cost | Unresolved pending calibration |
| Principal corruption probability $\alpha$ | Unresolved pending calibration |
| New-partner evaluation horizon $h$ | Unresolved pending calibration |
| LLM call granularity | Provisional phase-specific design; finalize after pilot benchmarking |
| Retry and fallback policy | Recommendations given; not frozen |
| Whether fallback refusal incurs refusal cost | Unresolved |
| Run-failure threshold | Unresolved |
| Number of initialization seeds | Unresolved pending runtime pilot |

---

## 24. Implementation acceptance criteria

The research simulator is ready for full SSC runs only when all of the following are true:

- the five conditions correspond exactly to Section 10;
- prohibited actions are absent from agent prompts;
- all conventional policies are formally specified;
- Pavlov is initialized with cooperation and follows the $(1,0,0,1)$ action rule;
- initial states are explicitly saved and reused across conditions;
- distinct random mechanisms use separate reproducible generators;
- decisions are collected before phase-level application;
- unknown friend-of-a-friend identities are revealed only through current-neighbor responses;
- valid requests require a current requester-informant edge, and every revealed candidate has a current informant-candidate edge;
- isolated agents do not submit or receive reputation requests;
- rewiring is atomic;
- successful rewires preserve actor degree and total edge count;
- duplicate and conflicting rewires are resolved without fixed node-order privilege;
- direct and reported histories are distinguishable;
- truthful responses reveal complete candidate sets and truthful histories;
- refusals reveal neither candidate identities nor histories;
- deception is produced only by the controlled history-corruption mechanism and does not falsify candidate identities or network ties;
- request, refusal, and deception costs are applied according to their specified rules;
- truthful reporting is free to the informant;
- all applicable costs are visible to agents before the relevant decision;
- endogenous isolation is allowed;
- undefined per-interaction outcomes remain missing;
- event logs reconstruct every graph and decision;
- all model, prompt, configuration, and code versions are recorded;
- pilot verification tests pass;
- the unresolved-decision table has been replaced by final values or formal policies.

---

## 25. Frozen-design rule

After the full experiment begins, no treatment definition, policy, prompt, parameter, fallback, exclusion rule, or primary outcome may be changed in response to observed experimental results.

Necessary software corrections must be documented. If a correction changes simulated behavior, affected runs must be treated as belonging to a different implementation version and rerun consistently across all five conditions.
