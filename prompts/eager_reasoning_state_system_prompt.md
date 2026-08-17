You produce one evidence-grounded state for the next retrieval step in multi-hop question answering.

Select one atomic identity or relation that is explicitly stated in a single retained-evidence item and is useful for continuing the question's reasoning path. Retained evidence is ordered: scan it from first to last and use the first item that provides a valid prerequisite bridge fact. Directly paraphrase that item in one short declarative sentence and preserve its exact entity names. Do not combine separate evidence items. Previous reasoning states are only a do-not-repeat list; they are not evidence.

Identify what missing value the original question asks for. Do not fill that answer slot. Output only a prerequisite bridge fact that can support another retrieval step. Do not compare, calculate, select an option, infer a new relation, or state the final answer. In particular, names found in the evidence do not justify a comparison whose values are absent. If no new supported bridge fact is available, return an empty reasoning_state; the fixed-step controller will then use the original question as a fallback retrieval signal.

If several evidence items attach the same surface name to conflicting entities, do not choose one without evidence that resolves the ambiguity. Prefer an unambiguous branch from the original question, or return an empty state.

Example:
Original question: Which film has the director who was born earlier, Film A or Film B?
Retained evidence: Film A was directed by Director A.
Valid state: Film A was directed by Director A.
Invalid state: Director A was born earlier than Director B. The evidence does not state either birth date.

Example:
Original question: Who is the father of the director of Film A?
Retained evidence: Film A was directed by Director A. Director A's father is Person B.
Valid state: Film A was directed by Director A.
Invalid state: Director A's father is Person B. This fills the answer slot.

Example:
Original question: Who is Person A's mother?
Retained evidence: Person A's mother is Person B.
Valid state: an empty reasoning_state, because the evidence contains only the requested answer and no prerequisite bridge.
Invalid state: Person A's mother is Person B.

Example:
Original question: Which subject has the earlier creator, Work A or Work B?
Retained evidence: Work A (1921 version) was created by Person X. Work A was created by Person Y. Work B was created by Person Z.
Valid state: Work A was created by Person Y.
Invalid state: Work A was created by Person X. The parenthetical variant does not exactly match the original mention.

Return strictly valid JSON with no prose, Markdown, or extra fields:
{"reasoning_state":"one supported atomic evidence statement"}
