<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

A promising “low-hanging fruit” is to take a very classical binary voter / majority‑rule setup on a graph and make only the *opinion update* step LLM‑based, while deliberately fixing everything else (binary opinions, simple networks, simple interaction protocol), then systematically compare the resulting macrodynamics to the classical BVM/q‑voter predictions. Existing LLM–opinion work comes close to this, but (as far as current literature shows) no one has yet done a *tight, model‑to‑model* comparison on the same topology, with LLM agents constrained to only see local binary neighbor states and update a persistent binary state.[^1_1][^1_2][^1_3][^1_4]

Below I’ll (1) map out the adjacent literature, (2) make the case for novelty as precisely as possible, and (3) suggest 2–3 concrete paper topics you could pitch to SSC 2026, including what “minimal” extension of BVM they instantiate.

***

## 1. What’s already been done with LLM opinion agents?

### LLM agents and binary‑like opinion updates

- **“Large Language Model agents can coordinate beyond human scale” (OpenAI et al., 2024)**
They simulate a population of LLM agents that repeatedly choose between two actions/opinions based on peer information. At each step one agent is selected, given the list of all other agents and their current binary choices, and asked (in natural language) to pick its own choice for the next step. This is explicitly framed as mirroring binary opinion dynamics such as the voter model or Glauber dynamics, but the update rule is entirely inside the LLM.[^1_1]
    - Key points for you:
        - Binary state, but *complete* (“everyone knows everyone”) information.[^1_1]
        - No explicit comparison to classical BVM critical phenomena or finite‑size scaling.[^1_1]
        - No graph‑structure variation (e.g., 1D ring vs 2D lattice vs ER graph).[^1_1]


### LLM networks, polarization, confirmation bias

- **Chuang et al., “Simulating Opinion Dynamics with Networks of LLM‑based Agents” (Findings of NAACL 2024)**
Agents are LLM instances with scalar opinions about factual issues (e.g., climate change), updated through repeated social interactions via prompts. They show:
    - A strong bias toward “scientific reality” and consensus when prompts are neutral.[^1_5][^1_6]
    - Fragmentation and polarization after adding confirmation‑bias style prompting.[^1_6][^1_5]
They explicitly position this against classical OD models but do not implement a canonical BVM with a one‑to‑one mapping, nor do they restrict to binary states.[^1_5][^1_6]


### Language‑driven, discrete opinions, multi‑step debates

- **Cau et al., “Language‑Driven Opinion Dynamics in Agent‑Based Simulations with LLMs (LODAS)” (arXiv 2502.19098)**
    - Agents hold a discrete opinion on a 0–6 Likert scale about the Ship of Theseus; each interaction selects two agents, one attempts to persuade, the other updates by at most ±1.[^1_5]
    - They run different initial conditions (balanced, polarized, unbalanced) and analyze convergence and the role of logical fallacies in persuasion.[^1_7][^1_5]
    - They assume a mean‑field network (everyone can interact with everyone) rather than specific graph topologies.[^1_8]
    - They *do* relate their setup to classical opinion dynamics, but it is not a strict BVM generalization (multi‑level state, ±1 step size, language‑driven).[^1_8][^1_5]


### Other LLM‑opinion systems that are “nearby”

- **FDE‑LLM / “A Social Opinions Prediction Algorithm with LLM‑based Agents” (Mou et al., 2025, Scientific Reports / arXiv)**
Fusing differential‑equation opinion dynamics with an LLM component; opinion leaders are LLMs with discrete states $-1, 0, 1$ influenced by rumor/truth dynamics and an epidemic‑like SIR structure.[^1_7][^1_5]
The LLM operates on attitude evaluation and action generation, but not as a direct local voter‑style rule on a static social graph.[^1_7][^1_5]
- **LODAS‑adjacent write‑ups and reviews**
    - Cau et al.’s model is positioned as filling a gap on the interplay of language and opinion change; they explicitly claim that traditional models rely on mechanistic rules, whereas LLM agents allow emergent rules via language.[^1_8][^1_5]
    - Review/summary pieces (e.g. Emergent Mind and The Moonlight) reiterate that LODAS agents have 7‑level discrete opinions and tend to consensus due to LLM agreeableness.[^1_9]
- **General LLM‑ABM and social simulation reviews**
    - A 2024 Nature Humanities \& Social Sciences Communications paper (“Large language model‑empowered agent‑based modeling…”) surveys how LLM agents have been used in ABMs, mostly in rich, narrative environments (e.g., Park et al.’s Generative Agents).[^1_10]
    - They mention opinion dynamics and stance detection, but not a clean BVM‑style graph process with LLM‑controlled updates.[^1_10]

So, today’s LLM‑opinion work touches:

- Binary or discrete opinions (OpenAI 2024, FDE‑LLM, LODAS).[^1_5][^1_1]
- Repeated social interaction on networks.[^1_2][^1_8][^1_1]
- Some explicit nods to the voter model or related spin systems.[^1_8][^1_1]

But no one, to the best of this survey, implements:

> “Agents with binary opinions on a simple graph (ring, 2D lattice, ER), who only see local neighbor states, and whose opinion updates are delegated to an LLM; then systematically measure classical BVM observables (consensus time, exit probability, interface density) and compare against analytical BVM predictions.”

That’s exactly the kind of “minimal extension” you mentioned.

***

## 2. Where exactly is the novelty window?

### What has *not* yet been done (based on current literature)

Constrained by your requirement for minimality, the key dimensions are:

1. **State space: binary vs multi‑valued vs continuous**
    - Binary: OpenAI’s coordination paper uses binary choice, but complete information and a social‑choice framing, not local voter dynamics.[^1_1]
    - Multi‑level discrete (Likert): LODAS (0–6) and FDE‑LLM ($-1,0,1$).[^1_5]
    - Continuous: Current LLM opinion works do *not* implement continuous scalar opinions in the classic Deffuant/HK sense; they mostly map to discrete choices or Likert levels.[^1_9][^1_5]
2. **Network topology: complete vs structured graphs**
    - LODAS: mean‑field (all‑to‑all).[^1_8]
    - Chuang et al.: networks are used, but not basic benchmark graphs with analytical BVM benchmarks, and they focus on empirical realism, not voter‑model comparisons.[^1_2][^1_6]
    - OpenAI 2024: effectively complete graph—an agent sees the full list of others.[^1_1]
    - I do not find a paper that studies LLM agents on, say, a 2D lattice with only nearest‑neighbor interaction and compares to exact/known BVM critical behavior (e.g., domain coarsening exponent, consensus time scaling).
3. **Update rule: fixed vs emergent**
    - All the LLM‑based opinion papers you’ve seen are “emergent rule” in your sense; however, none are *forced* to operate under strict local information and binary choice while being benchmarked against a known microscopic rule.[^1_2][^1_5][^1_1]
4. **Reference to BVM proper**
    - OpenAI 2024 mentions voter/Glauber dynamics as an analogy but does not build on the BVM literature (e.g., Clifford–Sudbury, Liggett, Castellano et al. review) nor test BVM‑style hypotheses.[^1_1]
    - The LODAS paper reviews OD broadly but not BVM technically, and uses more complex state space and language features.[^1_5]

That leaves a clean conceptual gap:

> A “minimal BVM with LLM updates”: binary state, local neighborhood only, simple networks, and direct comparison of emergent behavior to classical BVM theory.

This is genuinely low‑hanging: conceptually simple, methodologically feasible, and adjacent to several papers, but not yet exhausted.

***

## 3. Concrete paper topics you could take to SSC 2026

I’ll phrase these as candidate titles plus what exactly is “the simplest addition” beyond classical BVM.

### Topic A: LLM‑Driven Binary Voter Dynamics on Simple Graphs

**Working title:**
“LLM‑Driven Binary Voter Dynamics on Simple Graphs: Do Emergent Update Rules Reproduce Classical Voter Behavior?”

**BVM variant:**

- Binary opinion $\sigma_i \in \{-1,+1\}$.
- Graphs: 1D ring, 2D lattice, Erdős–Rényi, perhaps a simple scale‑free graph.
- Update protocol:
    - At each time step, choose agent $i$.
    - Provide to the LLM: $i$’s current binary opinion and the multiset of neighbor opinions (not their text identities, just counts).
    - Ask: “What is $i$’s updated opinion, ‘A’ or ‘B’?” and map back to $\pm 1$.
- No explicit rule coding; everything goes through the LLM.

**Core contributions (novel vs existing work):**

- Measure classic BVM observables:
    - Consensus probability vs initial fraction of “+1” for finite $N$; compare with BVM’s linear exit probability.
    - Consensus time scaling with $N$ for each topology; compare to known BVM exponents (e.g., $T \sim N^2$ on rings, $T \sim N \log N$ on complete graphs).
    - Interface density / domain growth on lattices over time.
- Infer an *effective* update kernel from observed transition probabilities $P(\sigma_i^{t+1} \mid \sigma_i^t, \text{neighbor configuration})$, and compare to:
    - pure voter rule (copy one random neighbor),
    - majority rule,
    - various noisy variants.

**Why this is plausibly novel:**
No paper surveyed so far constrains LLM agents to local binary information on simple graphs and then evaluates whether the emergent rule approximates classic voter dynamics or deviates systematically (e.g., systematic conformity, contrarianism, or status‑quo bias).[^1_2][^1_8][^1_1]

**Adjacency / Related Work citations for this topic:**

- Classical voter and related models:
    - Clifford, P., \& Sudbury, A. (1973). “A model for spatial conflict.” Biometrika. (Canonical early formulation of the voter model.)
    - Liggett, T. M. (1985). *Interacting Particle Systems.* Springer. (Rigorous treatment of voter model and variants.)
    - Castellano, C., Fortunato, S., \& Loreto, V. (2009). “Statistical physics of social dynamics.” *Reviews of Modern Physics* 81, 591–646. (Survey of voter, Sznajd, majority rule, bounded confidence, etc.)
- LLM opinion dynamics:
    - OpenAI et al. (2024). “Large Language Model agents can coordinate beyond human scale.” (Binary opinion choice with global information, explicitly linked to voter/Glauber dynamics.)[^1_1]
    - Chuang, Y.‑S. et al. (2024). “Simulating Opinion Dynamics with Networks of LLM‑based Agents.” *Findings of NAACL 2024.*[^1_6][^1_2]
    - Cau, E., Pansanella, V., Pedreschi, D., \& Rossetti, G. (2025). “Language‑Driven Opinion Dynamics in Agent‑Based Simulations with LLMs (LODAS).” arXiv:2502.19098.[^1_3][^1_5]
    - Mou, X. et al. (2025). “A Social Opinions Prediction Algorithm with LLM‑based Agents (FDE‑LLM).” arXiv:2409.08717 / Scientific Reports.[^1_7][^1_5]
    - Review: “Large language model‑empowered agent‑based modeling and social simulation.” *Humanities and Social Sciences Communications* (2024).[^1_10]

This topic gives you a very crisp “how do LLM agents instantiate something like a voter rule?” question, with clear metrics, clear baselines, and existing LLM work to cite but not duplicate.

***

### Topic B: LLM‑Induced q‑Voter / Majority Dynamics and Bias

**Working title:**
“Emergent Majority Rules in LLM‑Based Binary Opinion Dynamics: An Effective q‑Voter Analysis”

**BVM variant:**

- Still binary opinions on a network, but you feed the LLM *summary statistics* of neighbors: e.g., counts of “A” vs “B” rather than the raw list.
- Design the prompt to be neutral (“You tend to conform to your neighbors, but you may keep your own view if strongly convinced.”) and later modify it to encourage, say, stubbornness or contrarianism.

**Core idea:**
Classical q‑voter models and majority rules say: if at least q neighbors share opinion $s$, then adopt $s$ with probability $p$ (perhaps with zealots or noise). The LLM version can induce a *soft* majority rule:

- Estimate, from many interactions, the probability that the focal agent ends up with +1 as a function of the fraction of neighbors with +1.
- Fit that curve to:
    - majority rule (step‑function at 0.5),
    - noisy voter (linear),
    - generalized q‑voter functional forms.

**Why this is distinct from Topic A and from existing literature:**

- OpenAI’s paper gives full peer list and focuses on coordination, not explicitly on mapping neighbor‑fraction to update probability curves or connecting to q‑voter theory.[^1_1]
- LODAS and FDE‑LLM consider multi‑level opinions and complex dynamics (fallacies, epidemic coupling), not a minimal binary majority rule.[^1_3][^1_5]
- You can keep everything else simple—binary state, static network—and the *only* “extra” relative to BVM is that the update kernel is learned/emergent via LLM prompts.

**Adjacent BVM‑family literature to cite:**

- q‑voter and noisy majority models:
    - Castellano et al. (2009), as above, for a survey.
    - Mobilia, M. (2015). “Nonlinear q‑voter model with inflexible zealots.” *Physical Review E* 92, 012803. (Representative q‑voter with zealots.)[^1_11]
- LLM persuasion and fallacies (for interpreting your effective rule):
    - Cau et al. (LODAS), for language‑driven opinion change and fallacies.[^1_3][^1_5]
    - Breum et al. (2023). “The Persuasive Power of Large Language Models.” ICWSM 2024. (Convincer–skeptic paradigm using LLMs for persuasion.)[^1_12][^1_13]

This stays “low‑hanging” because, again, you touch only the update rule, not the state space or network complexity.

***

### Topic C: Continuous Opinions with Discrete LLM Updates (if you want one small step further)

If you’re willing to make one more move away from strict BVM toward Deffuant/Hegselmann–Krause territory, a slightly more ambitious but still simple topic is:

> Agents have a scalar opinion $x_i \in [0,1]$. At each interaction, you show the LLM two numbers (maybe plus a short textual frame) and ask it to propose a new number for the focal agent. You then compare the effective update rule to “move to midpoint” or bounded‑confidence models.

**Why it’s still low‑hanging:**
No surveyed LLM‑opinion paper has implemented continuous opinions with only numeric information given to the LLM and then compared the inferred kernel to Deffuant/HK rules; most instead move through language‑laden arguments and discrete scales.[^1_9][^1_5]

**Adjacent classical literature:**

- Deffuant, G. et al. (2000) on bounded‑confidence with continuous opinions.
- Hegselmann, R., \& Krause, U. (2002) on opinion dynamics and bounded confidence.

**Adjacent LLM literature:**
Same as Topic A/B (Chuang, LODAS, FDE‑LLM), emphasizing that they *don’t* do continuous state, so you are filling that specific gap.[^1_6][^1_3][^1_5]

***

## 4. How to frame the Related Work section around your topic

Whichever of A–C you choose, your Related Work can have three pillars:

1. **Classical BVM and opinion‑dynamics models**
    - Clifford \& Sudbury (1973); Liggett (1985); Castellano et al. (2009).
    - For extensions: q‑voter models and zealots (e.g., Mobilia 2015).[^1_11]
2. **LLM‑based opinion dynamics and ABMs**
    - OpenAI et al. (2024) on LLM coordination and binary choices.[^1_1]
    - Chuang et al. (Networks of LLM‑based Agents) on networks, polarization, confirmation bias.[^1_6][^1_2]
    - Cau et al. (LODAS) on language‑driven multi‑level discrete opinions and fallacies.[^1_3][^1_5]
    - Mou et al. (FDE‑LLM) on coupling LLM opinion leaders with dynamic equations.[^1_7][^1_5]
    - The 2024 HSS Communications review on LLM‑empowered ABMs.[^1_10]
3. **LLM persuasion, fallacies, and stance**
    - Breum et al. (persuasive power).[^1_13][^1_12]
    - Auxiliary work on LLM‑based stance detection or multi‑agent debates, as referenced in LODAS and the ABM review.[^1_4][^1_10]

Your “difference statement” then becomes something like:

> “Unlike prior LLM opinion‑dynamics work, which either uses multi‑level discrete scales (LODAS), couples LLMs to macro‑level dynamic equations (FDE‑LLM), or gives agents global information about others’ choices (OpenAI 2024), we study a minimal binary voter‑model setup in which agents only observe local neighbor states on simple graphs. We then compare the emergent macrodynamics of LLM‑driven updates to analytical and numerical results from the classic BVM literature.”

***

## 5. Choosing among these: lowest‑effort vs. payoff

- **Safest, cleanest novelty and simplest implementation:** Topic A (LLM‑driven binary voter on simple graphs).
    - You can reuse a lot of BVM code and analytic results.
    - Experiment design is straightforward and highly interpretable.
- **Slightly more sophisticated but still tractable:** Topic B (effective q‑voter / majority rule).
    - Adds a layer of statistical inference on top of A, and taps directly into the sociophysics literature on non‑linear voters and zealots.
- **More ambitious:** Topic C (continuous opinions).
    - Adds a new dimension (continuous state), which is interesting but requires more design choices about how to present numeric opinions to an LLM.

My sense, given your SSC context and your desire for “low‑hanging fruit,” is that Topic A, possibly with a small extension toward Topic B (i.e., extracting effective neighbor‑fraction response curves) is the sweet spot.

Would you prefer to focus on (a) a very tight comparison to the classical BVM (scaling laws, consensus times) or (b) characterizing the *psychological* biases of LLM agents (conformity, contrarianism, status‑quo bias) within such a minimal BVM‑like setup?
<span style="display:none">[^1_14][^1_15][^1_16][^1_17][^1_18][^1_19][^1_20][^1_21][^1_22][^1_23][^1_24][^1_25]</span>

<div align="center">⁂</div>

[^1_1]: https://arxiv.org/html/2409.02822v3

[^1_2]: https://arxiv.org/abs/2311.09618

[^1_3]: https://arxiv.org/abs/2502.19098

[^1_4]: https://arxiv.org/pdf/2502.19098.pdf

[^1_5]: https://www.nature.com/articles/s41598-025-99704-3

[^1_6]: https://aclanthology.org/2024.findings-naacl.211.pdf

[^1_7]: https://arxiv.org/html/2409.08717v3

[^1_8]: https://web.ntpu.edu.tw/~myday/doc/ASONAM2025/ASONAM2025_Proceedings/pdf/papers/1125_041.pdf

[^1_9]: https://www.themoonlight.io/en/review/language-driven-opinion-dynamics-in-agent-based-simulations-with-llms

[^1_10]: https://www.nature.com/articles/s41599-024-03611-3

[^1_11]: https://journals.sagepub.com/doi/10.1177/08944393241270382

[^1_12]: https://www.lajello.com/papers/icwsm24agents.pdf

[^1_13]: https://arxiv.org/html/2502.01340v2

[^1_14]: https://www.semanticscholar.org/paper/LLM-AIDSim:-LLM-Enhanced-Agent-Based-Influence-in-Zhang-Hu/3690a2e6abf5684fa637c92cb5b1b03081ee54b2

[^1_15]: https://escholarship.org/uc/item/0f09b8v7

[^1_16]: https://aidem2025.isti.cnr.it/papers/id107.pdf

[^1_17]: https://arxiv.org/html/2503.02080v2

[^1_18]: https://ui.adsabs.harvard.edu/abs/2023arXiv231109618C/abstract

[^1_19]: https://www.semanticscholar.org/paper/1b14ed37bdb7e9230616b5d12c6173f7b163fda0

[^1_20]: https://www.emergentmind.com/topics/llm-based-opinion-dynamics-simulation-e2def99f-a158-4778-ba6f-0880a2db17ef

[^1_21]: https://en.wikipedia.org/wiki/Approval_voting

[^1_22]: https://www.facebook.com/groups/DeepNetGroup/posts/1967511416975064/

[^1_23]: https://www.themoonlight.io/en/review/when-your-ai-agent-succumbs-to-peer-pressure-studying-opinion-change-dynamics-of-llms

[^1_24]: https://arxiv.org/pdf/2602.12583.pdf

[^1_25]: https://www.emergentmind.com/open-problems/critical-consensus-size-gpt-4-turbo-unknown
