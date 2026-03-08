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
        Payoff matrix:\n
        Actions: C = Cooperate, D = Defect\n\n
        If the agent chooses C and their opponent chooses C: 
        the agent gets {payoff_matrix[('C','C')][0]}, 
        their opponent gets {payoff_matrix[('C','C')][1]}.\n
        If the agent chooses C and their opponent chooses D: 
        the agent gets {payoff_matrix[('C','D')][0]}, 
        their opponent gets {payoff_matrix[('C','D')][1]}.\n
        If the agent chooses D and their opponent chooses C: 
        the agent gets {payoff_matrix[('D','C')][0]}, 
        their opponent gets {payoff_matrix[('D','C')][1]}.\n
        If the agent chooses D and their opponent chooses D: 
        the agent gets {payoff_matrix[('D','D')][0]}, 
        their opponent gets {payoff_matrix[('D','D')][1]}.\n\n
        """
    )


def build_batch_prompt(agent_payloads, payoff_matrix):
    prompt = textwrap.dedent(f"""
    You are making decisions for multiple independent agents in a Prisoner's
    Dilemma simulation.

    For each agent below, decide either:
    C = Cooperate
    D = Defect

    Each agent object describes one decision-maker in the simulation.

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

        The "history" field (if present) contains a list of past interactions
        with that opponent.

        For example:

            "opponents": {{
                "7": {{
                    "history": [
                        {{"my_move": "C", "opp_move": "D"}},
                        {{"my_move": "D", "opp_move": "D"}}
                    ],
                }},
                "4":
                    "history": [
                        {{"my_move": "C", "opp_move": "C"}},
                        {{"my_move": "C", "opp_move": "C"}}
                    ],
                }}
            }}

        means that the agent previously played two rounds against opponents 7
        and 4. In the competition against 7, the agent played C and opponent 7
        responded with D in the first round, and both players played D in the
        second round. Against opponent 4, both this agent and opponent 4
        played C both times.

        If the history field is absent, the agent has no relevant history.

        Each agent also has a "persona" field, which corresponds to specific
        instructions about how it should make its decision. Here are the
        possible personas, and the instructions for each:\n
    """
    )
    prompt += textwrap.dedent("\n".join(
        [f"""
    {pn}:\n{inst.instructions}
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

        Now, here are the actual agents and their opponents. Note that the
        agents listed below are only the agents whose decisions you must
        generate. Opponent IDs may refer to other agents that are not
        listed here. Those agents are controlled by the simulation and you
        should NOT generate decisions for them. However, if an agent that
        *is* listed below has an opponent whose ID is not one of the ids
        in the list, you *should* generate a response for that agent
        against that (non-LLM) opponent.

        {json.dumps(agent_payloads, indent=2)}

        You MUST return a decision for EVERY opponent played by EVERY
        agent listed. If any agent is missing, your response is invalid.
        Do not omit any agent.

        IMPORTANT: your response must be a JSON object only. Do not include
        explanations, reasoning, or text outside the JSON. The JSON must
        not contain comments. Do not include // or /* */ anywhere in the
        output.
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


