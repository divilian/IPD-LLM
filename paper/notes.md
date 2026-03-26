Interestingly, c9397b3 broke when I had exactly one LLM agent. This is
because this prompt:

```
           You are making decisions for multiple independent agents in a
            Prisoner's Dilemma simulation.

            For each agent below, decide either:
            C = Cooperate
            D = Defect

            Return ONLY valid JSON in this exact format:

            {
              "decisions": [
                {"id": 1, "move": "C"},
                {"id": 2, "move": "D"}
              ]
            }

            Agents:

            [
  {
    "id": 0,
    "persona": "vanilla",
    "neighbors": [
      3,
      4,
      5,
      6,
      7,
      8
    ],
    "wealth": 0.0
  }
]
```

produced this output:

```
{
  "decisions": [
    {"id": 0, "move": "D"},
    {"id": 1, "move": "D"},
    {"id": 2, "move": "C"},
    {"id": 3, "move": "C"},
    {"id": 4, "move": "D"},
    {"id": 5, "move": "D"},
    {"id": 6, "move": "C"},
    {"id": 7, "move": "D"},
    {"id": 8, "move": "D"}
  ]
}
```

Presumably it broke because there was only had one agent in the list, yet the
instructions said "multiple independent agents." So the LLM thought, "well, if
I'm supposed to do _multiple_ agents, then it must want outputs for 0 through
8.
