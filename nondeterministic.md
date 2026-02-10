# Explaining failure to fully obey TFT

## Observation

When the LLM is given this system message:

```
You always do what your opponent did on their most recent move. If they
defected last time, you defect this time.If they cooperated last time, you
cooperate this time.If this is the first move of the game, choose
randomly.Respond with exactly one word: "Cooperate" or "Defect".
```

and this user prompt:

```
\n        You are a player in an Iterated Prisoner's Dilemma game. In each
round,\n        you and your opponent will choose to either cooperate or
defect. If you\n        both cooperate, you'll both be awarded $3.0.\n
If you cooperate and your opponent defects, you will get\n        $0.0 and your
opponent will get\n        $5.0. If you defect and your opponent\n
cooperates, you will get $5.0 and your\n        opponent will get $0.0. If you
both defect,\n        you will both be awarded $1.0.\n    Here is the history
of your interactions with this opponent so far:\n    On move 1, you Defected
and they Defected.\n    On move 2, you Defected and they Defected.\n    On move
3, you Defected and they Defected.\n    On move 4, you Defected and they
Defected.\n        Do you choose to Cooperate, or Defect?\n
```

it replies "`Cooperate`", which is clearly wrong.

## ChatGPT's explanations

In prisoner's dilemma, e.g., LLMs exhibit non-deterministic behavior.

“Tit-for-tat” in natural language is often associated with:

- reciprocity
- fairness
- eventual cooperation
- “break the cycle of mutual defection”

So when the model sees:

```
On move 1… Defected / Defected
On move 2… Defected / Defected
On move 3… Defected / Defected
On move 4… Defected / Defected
```

many models implicitly recognize a mutual-defection deadlock and have a strong
learned bias toward “try cooperation now”, even though that is not tit-for-tat
as formally defined.

This is especially common in LLaMA-family models, which are strongly trained on:

- social reasoning
- cooperative narratives
- human-interpretable game explanations

What you’ve discovered is a core result that a lot of recent ABM-with-LLMs work
keeps running into:

LLMs do not reliably implement deterministic policies expressed in natural
language. Even when the policy is trivial.

They are:

- stochastic
- norm-biased
- optimization-suggestible
- context-sensitive in unintended ways

Which means: they will occasionally “defect” from your tit-for-tat, ironically.


Here’s the subtle but important thing you ran into:

- System prompt: “Always mirror the last move”
- User prompt: “Here’s a game with incentives, repeated defection, and a chance
  to choose”

From the model’s perspective, the user prompt reopens the question:

- “Given this situation, what should I do?”
- And once “should” enters the picture, norms flood in.

This is why your model “forgave” after repeated DD rounds. That wasn’t a bug — it was latent social reasoning asserting itself.
