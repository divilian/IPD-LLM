# What looks genuinely novel or thinly explored

Based on those papers, the following aspects of your proposal appear new or at least not systematically treated in one place:

  1. Joint action, network rewiring, and communication decisions per agent Most existing LLM‑IPD work has the model choose only the stage‑game action (cooperate/defect) under a fixed network or simple matching process.

    You propose that each agent in each round decides:

        1. Which action to play with each neighbor.

        2. Which links to sever and which FOAF links to form.

        3. Whether to share or withhold information about FOAFs, and whether to
           lie when sharing.

    Dynamic, endogenous link formation/deletion has been studied with simple strategies, but **I don’t see work where LLM agents simultaneously control game actions, endogenous network rewiring, and gossip/lying decisions in one unified IPD environment. That integrated tri‑decision setting is plausibly novel.**

  2. Mixed population: rule‑based and LLM agents on the same network Networked IPD work typically compares simple strategies (TFT, ALLD, etc.), possibly under evolutionary dynamics. LLM‑focused work tends to use all‑LLM populations (different prompts/models, but all LLMs) or all‑human vs all‑LLM comparisons.
    A heterogeneous population where LLM agents coexist with classic, transparent rule‑based strategies on the same evolving network—and where you study both performance and emergent structure—seems underexplored. I haven’t seen a paper explicitly using LLMs as just another strategy in a classic Axelrod‑style tournament on a graph.

  3. FOAF‑based network formation with mediated, potentially deceptive reports Gossip and reputation mechanisms do appear in the cooperation literature, but your specific mechanism—agents forming links to FOAFs conditional on second‑hand reports, where the reporter can lie strategically—doesn’t show up in current LLM‑IPD work I’m seeing.

    Combined with LLMs, this could give you:

      a. A clean handle on how LLMs use and manipulate reputational information.

      b. A way to study emergent lying vs truth‑telling policies that interact with network formation, not just immediate payoffs.

  4. Systematic analysis of LLM rationales as qualitative data Some multi‑agent LLM papers capture chain‑of‑thought or explanations, but they usually use them for performance or ablation rather than as a primary qualitative object of study.
    Designing the experiment so that:

        Rationales are collected every round for all three decision dimensions (action, rewiring, communication).

        The central contribution is a qualitative/thematic analysis of those rationales (e.g., norms, self‑justifications, “theories” the LLMs deploy about strategic interaction),

    would be a distinctive angle relative to the mostly quantitative, payoff‑based evaluations in the existing LLM‑IPD literature.

    Explicitly varying time horizon (known vs probabilistic length) in a networked, multi‑decision setting IPD work of course considers known vs uncertain horizons, but I’m not seeing that factor combined with:

        Networked interactions,

        LLM agents, and

        Simultaneous choices over actions, links, and communication.

    Showing how LLM rationales and behavior change when you flip from a fixed to a probabilistic horizon in this richer environment could be a neat methodological contribution, even if each piece in isolation is familiar.

How I’d frame the novelty in a paper

You could pitch the contribution along three main axes:

    Environment design
    “We introduce a networked IPD environment where agents jointly choose stage‑game actions, dynamic network rewiring (link creation/deletion), and reputational communication (honest or deceptive gossip about FOAFs).”
    This goes beyond the static or single‑decision setups in prior LLM‑IPD and LLM‑social‑dilemma experiments.

    Agent heterogeneity
    “We study mixed populations of classic rule‑based strategies and LLM‑based agents, allowing direct comparison between hand‑specified decision rules and prompted, language‑based reasoning in the same networked ecosystem.”
    That’s closer in spirit to Axelrod‑style tournaments but with LLMs as strategies, and current LLM work doesn’t quite do this on networks.

    Interpretive focus on LLM rationales
    “Beyond payoffs, we treat LLM self‑reported rationales as qualitative data, analyzing themes such as norm adoption, reciprocity, trust, and explicit trade‑offs between network structure and local payoffs.”
    Existing work measures cooperation rates, exploitability, or deception frequencies, but less work systematically reads the rationales in a rich multi‑decision game.

If you emphasize those three things, I think you can legitimately claim novelty even in a rapidly filling area.
Suggestions to maximize “publishable new ground”

    Make the game tri‑decision structure central and formal: define the three action spaces (C/D with each neighbor, link operations, communication messages), and show how they interact in the payoff and network update rules.

    Be explicit that you are building on but extending network‑IPD and LLM‑IPD baselines: cite static network IPD, LLM IPD on well‑mixed populations, and LLM deceptive agents, then show the missing combination.

    For the qualitative part, pre‑register or at least clearly specify your coding scheme for rationales (e.g., categories like reciprocity, fairness, utilitarianism, rule‑following, self‑inconsistency, explicit lying). This will help reviewers see that the rationales aren’t an afterthought.

    On the IPD side, consider anchoring to a well‑known dynamic‑network
    social‑dilemma model and making “swap simple strategies for LLM agents and
    add rationales” the conceptual step. That will make the methodological
    contribution easier to grasp.
