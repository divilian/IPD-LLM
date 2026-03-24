from typing import List, Tuple, Dict
import textwrap
import json

from agents.personas import PERSONAS

def serialize_summary(history: List[Dict]) -> str:
    if not history:
        return "Summary: This is the first move of the game."

    last = history[-1]
    return (
        "Summary:\n"
        f"Opponent's most recent action: {last['other_action']}"
    )


def serialize_history(history: List[Dict]) -> str:
    if not history:
        return textwrap.dedent("""
            This is the first iteration of the game (neither player has moved
            yet).
        """).strip()

    choices = {"C": "Cooperated", "D": "Defected"}

    lines: List[str] = [
        "Here is the history of your interactions with this opponent so far:"
    ]

    for move in history:
        step = move["step"]
        you = choices[move["self_action"]]
        they = choices[move["other_action"]]
        lines.append(f"    On move {step}, you {you} and they {they}.")

    text = "\n".join(lines)
    return text


def serialize_payoffs(
    payoff_matrix: Dict[Tuple[str, str], Tuple[float, float]],
) -> str:
    """
    The payoff_matrix maps (your_action, opponent_action) -> (your_payoff,
    opponent_payoff) where actions are "C" (Cooperate) or "D" (Defect).
    """
    return textwrap.dedent(f"""
        Payoff matrix:

        Actions: C = Cooperate, D = Defect

        If the agent chooses C and their opponent chooses C, the agent gets
        {payoff_matrix[('C','C')][0]} and their opponent gets
        {payoff_matrix[('C','C')][1]}.
        If the agent chooses C and their opponent chooses D, the agent gets
        {payoff_matrix[('C','D')][0]} and their opponent gets
        {payoff_matrix[('C','D')][1]}.
        If the agent chooses D and their opponent chooses C, the agent gets
        {payoff_matrix[('D','C')][0]} and their opponent gets
        {payoff_matrix[('D','C')][1]}.
        If the agent chooses D and their opponent chooses D, the agent gets
        {payoff_matrix[('D','D')][0]} and their opponent gets
        {payoff_matrix[('D','D')][1]}.

        """
    )


def build_batch_prompt(agent_payloads, payoff_matrix):
    prompt = textwrap.dedent(f"""
    Return JSON only.

    You are generating decisions for multiple independent agents in an
    iterated Prisoner's Dilemma simulation.

    For each listed agent and for each opponent in that agent's "opponents"
    dictionary, return exactly one move:
    C = Cooperate
    D = Defect

    Each agent object below describes one decision-maker.

    """)
    prompt += serialize_payoffs(payoff_matrix)

    prompt += textwrap.dedent(f"""
    Fields:

    id
        The numeric identifier of the agent.

    persona
        The strategy the agent must follow when making decisions.

    opponents
        A dictionary whose keys are opponent IDs. Each entry describes the
        interaction history with that opponent.

        The "history" field contains a list of past interactions with that
        opponent. If the history list is empty, there have been no previous
        interactions with that opponent.

        Example:

            "opponents": {{
                "7": {{
                    "history": [
                        {{"my_move": "C", "opp_move": "D"}},
                        {{"my_move": "D", "opp_move": "D"}}
                    ]
                }},
                "4": {{
                    "history": [
                        {{"my_move": "C", "opp_move": "C"}},
                        {{"my_move": "C", "opp_move": "C"}}
                    ]
                }}
            }}

        This means the agent previously played two rounds against opponents 7
        and 4. Against opponent 7, the agent played C and then D, while
        opponent 7 played D and then D. Against opponent 4, both players
        played C in both rounds.

        Each agent also has a "persona" field. The possible personas and
        their instructions are:
    """
    )
    prompt += textwrap.dedent("\n".join(
        [f"""
    {pn}:
{inst.instructions}
        """ for pn, inst in PERSONAS.items()]
    ))

    prompt += textwrap.dedent(f"""
        Return ONLY valid JSON in this exact format:

        {{
            "decisions": [
              {{"id": 0, "opponent": 3, "move": "C"}},
              {{"id": 0, "opponent": 1, "move": "D"}},
              {{"id": 1, "opponent": 2, "move": "D"}}
            ]
        }}

        Field meanings:
        - id: the agent making the decision
        - opponent: the opponent the move applies to
        - move: C or D

        You must return one decision for every opponent listed for every
        agent shown below.

        The agents listed below are the only agents whose decisions you must
        generate. Opponent IDs may refer to agents not listed below. If a
        listed agent has an opponent whose ID is not listed as one of the
        agents below, you must still return a decision for that interaction.

        {json.dumps(agent_payloads, indent=2)}

        Your response must be a JSON object only. Do not include
        explanations, reasoning, or any text outside the JSON. Do not include
        comments. Do not include // or /* */ anywhere in the output.
    """)
    return prompt

def get_prompt(payoff_matrix, history: List[Dict]) -> str:
    prompt = textwrap.dedent(f"""
        You are a player in an Iterated Prisoner's Dilemma game. In each round,
        you and your opponent will choose to either cooperate or defect. If you
        both cooperate, you'll both be awarded ${payoff_matrix[('C','C')][0]}.
        If you cooperate and your opponent defects, you will get
        ${payoff_matrix[('C','D')][0]} and your opponent will get
        ${payoff_matrix[('C','D')][1]}. If you defect and your opponent
        cooperates, you will get ${payoff_matrix[('D','C')][0]} and your
        opponent will get ${payoff_matrix[('D','C')][1]}. If you both defect,
        you will both be awarded ${payoff_matrix[('D','D')][0]}.
    """).strip()
    if not history:
        prompt += """
            This is the first iteration of the game (neither player has moved
            yet).
        """
    else:
        prompt += serialize_history(history)

    prompt += textwrap.dedent("""
        Do you choose to Cooperate, or Defect?
    """).strip()
    return prompt
