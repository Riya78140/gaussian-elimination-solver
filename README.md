# Gaussian Elimination Solver

Solves `Ax = b` and computes matrix rank via partial-pivoting Gaussian elimination. Zero dependencies, ESM-only.

## Usage

```js
import { solve, rank } from "gaussian-elimination-solver";

const { ok, solution } = solve([[2, 1], [1, 3]], [3, 5]);
// ok === true, solution ≈ [0.8, 1.4]

console.log(rank([[1, 2, 3], [2, 4, 6]])); // 1
```

`solve(A, b)` returns `{ ok, singular, solution? }`. When the system is singular it returns `{ ok: false, singular: true }` with no `solution` — it does not throw, so callers branch on `ok` rather than wrapping in try/catch. `rank(A)` returns an integer.

An optional `{ tol }` overrides the default zero-pivot threshold (`1e-12`):

```js
solve(A, b, { tol: 1e-9 });
```

## Why this exists

A small, dependency-free routine for exact-size linear systems where you want to read every line of the solver and not pull in a numerical library. Partial pivoting keeps it stable for moderately scaled inputs; it is not LU-based and does not expose a factorization, which would be overkill for the problem this targets.

## Edge cases

- A zero on the diagonal is handled by row swaps during elimination; you do not need to pre-order the matrix.
- Singular systems (rank-deficient coefficient matrix) are reported, not solved. The library does not attempt least-squares or minimum-norm solutions.
- Inputs are validated: ragged matrices, non-finite entries, and mismatched `b` lengths throw. Empty inputs are treated as a `0x0` system yielding an empty solution.
- Results are floating point; compare with a tolerance, not `===`.

## Design notes

The window stores values eagerly rather than keeping running aggregates. Running
sums drift with floating point over long streams, and recomputing from a small
buffer is cheap enough that the drift is not worth the speed.

