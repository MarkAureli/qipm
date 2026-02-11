# Transform test fixtures (MPS → presolve → standard form)

Each `.mps` file has a pre-created reference `.std` (standard form c, b, A). Tests compare `transform_instance` output to these references.

| Fixture        | Purpose |
|----------------|--------|
| **min_sum**    | ≤ row (L), two vars with lower bound 0 → slack in standard form |
| **equality**   | = row (E), two vars ≥ 0 → already standard form |
| **three_var**  | Two ≤ rows, three vars → presolve can reduce formulation |
| **bounded_var**| One variable with both LO and UP (bounded column branch) |
| **lower_row**  | ≥ row (G) → row “lower only” branch (slack with -1) |
| **free_var**   | One FR (free) variable → x = x⁺ − x⁻ branch |
| **upper_var**  | One variable with MI + UP (upper-only column branch) |
| **range_row**  | L row + RANGES → row with finite lo < hi (range constraint branch) |

These cover the main branches in `_lp_to_standard_form`: column types (bounded, lower-only, upper-only, free) and row types (equality, ≤, ≥, range).
