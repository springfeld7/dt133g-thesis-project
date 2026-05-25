# Mutation rules

This package contains the mutation rules that transform Concrete Syntax Trees (CSTs) and emit mutation records for verification and reporting.

## How rules are discovered

The CLI walks `src/transtructiver/mutation/rules/` recursively and registers every subclass of `MutationRule` defined in those modules. A rule is registered under:

- its explicit `rule_name` class attribute, when present
- otherwise a kebab-case name derived from the class name with a trailing `Rule` suffix removed

## Rule contract

Each concrete rule should:

1. Subclass `MutationRule` from [rules/mutation_rule.py](./rules/mutation_rule.py)
2. Implement `apply(self, root: Node, context: MutationContext) -> List[MutationRecord]`
3. Return `MutationRecord` objects with valid metadata for the chosen `MutationAction`

Each `MutationRecord` must include:

- `node_id`: `(row, col)` coordinates for the target node. Synthetic nodes use negative coordinates.
- `action`: a `MutationAction` value from [mutation_types.py](../mutation_types.py)
- `metadata`: action-specific data validated by the verifier

## Adding a new rule

1. Create a new module under `src/transtructiver/mutation/rules/`.
2. Add a rule class whose name ends in `Rule` and, preferably, define a unique `rule_name`.
3. Implement the mutation logic and emit records through the helper methods in `MutationRule` when possible.
4. Add tests under `tests/mutation/rules/`.

## When you add a new action

If a rule needs a new `MutationAction` that is not already handled by the verification layer, update:

- [mutation_types.py](../mutation_types.py)
- the relevant strategy in [../verification/strategies/](../verification/strategies/)
- tests for both the rule and the verifier

---