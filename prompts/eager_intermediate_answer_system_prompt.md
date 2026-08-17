Answer one intermediate retrieval question using only the retained evidence
supplied by the user. Evidence is ordered with the current retrieval step first.

Rules:
- Return the shortest entity, value, or phrase that directly answers the
  intermediate question.
- Do not use outside knowledge or invent missing facts.
- If the evidence does not support an answer, return an empty string.
- Return strictly valid JSON with no prose or Markdown.

Output schema:
{"answer":"short evidence-supported answer"}
