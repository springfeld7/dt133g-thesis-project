# Dead Code Insertion — Lexicons

The `dead_code_insertion` rule uses language-specific "lexicons" to produce syntactically valid, typed dead code for multiple languages. Lexicons are small adapters that translate generic generated values into language source snippets.

Where to find them
- `src/transtructiver/mutation/rules/dead_code_insertion/lexicons/` contains concrete lexicon classes such as `cpp_lexicon.py`, `java_lexicon.py`, and `python_lexicon.py`.

What a lexicon does
- Provide `get_assignment_statement(var_name, value)` that returns a language-correct assignment string.
- Provide identity/meaningless modification templates (`IDENTITY_OPS_STR`, `IDENTITY_OPS_NUMERIC`) and lists used by the rule.
- Provide block-formatting helpers (`format_block`) and collections of opaque predicates / unreachable loop headers used to generate control-flow scaffolding.

Extending or adding a lexicon
1. Create a new file in the `lexicons/` directory implementing a subclass of `DeadCodeLexicon` (see `dead_code_lexicon.py` for the base API).
2. Populate the required ClassVars: `OPAQUE_PREDICATES`, `UNREACHABLE_LOOP_HEADERS`, `IDENTITY_OPS_STR`, `IDENTITY_OPS_NUMERIC`.
3. Implement or override `get_assignment_statement`, `format_block`, and any helper used by the rule.
4. Add unit tests under `tests/mutation/rules/dead_code_insertion/lexicons/` to assert deterministic output (use a seeded RNG fixture for reproducibility).

Testing
- Lexicon tests in `tests/mutation/rules/dead_code_insertion/lexicons/` use a seeded `random.Random` instance so outputs are deterministic — follow that convention.

Notes
- Lexicons are intentionally small and deterministic for testability; keep logic simple and put complex templates in the ClassVars where possible.

---