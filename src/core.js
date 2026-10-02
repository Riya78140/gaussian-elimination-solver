/**
 * Core Gaussian elimination engine.
 *
 * Solves Ax = b and computes matrix rank using partial pivoting. All routines
 * work on plain JS arrays of arrays so the library has zero dependencies and
 * stays friendly to environments without Float64Array support.
 */

/**
 * Default absolute tolerance for treating a pivot as zero.
 *
 * Kept generous because linear systems that arise in practice are often
 * poorly scaled; a tighter default would reject legitimately solvable systems
 * whose pivots are small but nonzero relative to the original coefficients.
 * Callers who need a different threshold pass it explicitly.
 */
export const DEFAULT_TOLERANCE = 1e-12;

/**
 * Deep-copy a matrix of numbers.
 * @param {number[][]} m
 * @returns {number[][]}
 */
export function cloneMatrix(m) {
  const rows = m.length;
  const out = new Array(rows);
  for (let i = 0; i < rows; i++) {
    out[i] = m[i].slice();
  }
  return out;
}

/**
 * Validate that a matrix is rectangular and finite.
 * Throws on structural problems (ragged rows, non-array input, NaN/Infinity).
 * @param {unknown} m
 * @param {string} name
 * @returns {{rows: number, cols: number}}
 */
function validateMatrix(m, name) {
  if (!Array.isArray(m)) {
    throw new TypeError(name + " must be an array of arrays");
  }
  const rows = m.length;
  if (rows === 0) {
    return { rows: 0, cols: 0 };
  }
  let cols = -1;
  for (let i = 0; i < rows; i++) {
    const row = m[i];
    if (!Array.isArray(row)) {
      throw new TypeError(name + " row " + i + " is not an array");
    }
    if (cols === -1) {
      cols = row.length;
    } else if (row.length !== cols) {
      throw new Error(name + " is ragged: row 0 has " + cols + " cols, row " + i + " has " + row.length);
    }
    for (let j = 0; j < row.length; j++) {
      const v = row[j];
      if (typeof v !== "number" || !Number.isFinite(v)) {
        throw new TypeError(name + "[" + i + "][" + j + "] is not a finite number");
      }
    }
  }
  return { rows, cols: cols === -1 ? 0 : cols };
}

/**
 * Validate a right-hand-side vector and confirm it has the expected length.
 * @param {unknown} b
 * @param {number} expected
 */
function validateRhs(b, expected) {
  if (!Array.isArray(b)) {
    throw new TypeError("b must be an array");
  }
  if (b.length !== expected) {
    throw new Error("b length " + b.length + " does not match A rows " + expected);
  }
  for (let i = 0; i < b.length; i++) {
    const v = b[i];
    if (typeof v !== "number" || !Number.isFinite(v)) {
      throw new TypeError("b[" + i + "] is not a finite number");
    }
  }
}

/**
 * Row reduce a copy of A with an attached augmented column (if provided) using
 * partial pivoting. Returns the reduced matrix plus a list of pivot column
 * indices, one per pivot row.
 *
 * @param {number[][]} A
 * @param {number[] | null} augment  extra column to carry along, or null
 * @param {number} tol
 * @returns {{rows: number, cols: number, pivots: number[], data: number[][]}}
 */
function reduceWithPivoting(A, augment, tol) {
  const { rows, cols } = validateMatrix(A, "A");
  const acols = augment ? 1 : 0;
  const totalCols = cols + acols;
  const work = new Array(rows);
  for (let i = 0; i < rows; i++) {
    const row = A[i].slice();
    if (augment) row.push(augment[i]);
    work[i] = row;
  }
  const pivots = [];
  let pivotRow = 0;

  for (let pivotCol = 0; pivotCol < cols && pivotRow < rows; pivotCol++) {
    // Partial pivot: pick the row with the largest absolute value in the
    // current column at or below the pivot row. This is more numerically
    // stable than naive forward elimination because it avoids dividing by
    // small pivots, which amplifies rounding error.
    let maxRow = pivotRow;
    let maxVal = Math.abs(work[pivotRow][pivotCol]);
    for (let r = pivotRow + 1; r < rows; r++) {
      const v = Math.abs(work[r][pivotCol]);
      if (v > maxVal) {
        maxVal = v;
        maxRow = r;
      }
    }

    if (maxVal <= tol) {
      // Whole pivot column is effectively zero below the pivot row; skip it.
      continue;
    }

    if (maxRow !== pivotRow) {
      const tmp = work[pivotRow];
      work[pivotRow] = work[maxRow];
      work[maxRow] = tmp;
    }

    const pivot = work[pivotRow][pivotCol];
    for (let r = pivotRow + 1; r < rows; r++) {
      const factor = work[r][pivotCol] / pivot;
      if (factor === 0) continue;
      const rRow = work[r];
      const pRow = work[pivotRow];
      for (let c = pivotCol; c < totalCols; c++) {
        rRow[c] -= factor * pRow[c];
      }
    }

    pivots.push(pivotCol);
    pivotRow++;
  }

  return { rows, cols, pivots, data: work };
}

/**
 * Solve the linear system A x = b.
 *
 * Partial pivoting is used throughout. The solver detects singular systems
 * (no unique solution) and reports them via the returned object rather than
 * throwing, so callers can branch on the outcome.
 *
 * @param {number[][]} A  n-by-n coefficient matrix.
 * @param {number[]} b   right-hand side, length n.
 * @param {{tol?: number}} [options]
 * @returns {{ok: boolean, solution?: number[], singular: boolean}}
 */
export function solve(A, b, options) {
  const tol = (options && typeof options.tol === "number") ? options.tol : DEFAULT_TOLERANCE;
  const { rows, cols } = validateMatrix(A, "A");
  if (rows !== cols) {
    throw new Error("solve requires a square matrix; got " + rows + "x" + cols);
  }
  if (rows === 0) {
    validateRhs(b, 0);
    return { ok: true, singular: false, solution: [] };
  }
  validateRhs(b, rows);

  const { pivots, data } = reduceWithPivoting(A, b, tol);

  if (pivots.length !== rows) {
    return { ok: false, singular: true };
  }

  // Back-substitution on the upper-triangular reduced matrix. The augmented
  // column lives at index `cols` in each row.
  const x = new Array(rows);
  for (let i = rows - 1; i >= 0; i--) {
    const row = data[i];
    const diag = row[i];
    let sum = row[cols];
    for (let j = i + 1; j < rows; j++) {
      sum -= row[j] * x[j];
    }
    x[i] = sum / diag;
  }
  return { ok: true, singular: false, solution: x };
}

/**
 * Compute the rank of a matrix (number of linearly independent rows).
 *
 * @param {number[][]} A
 * @param {{tol?: number}} [options]
 * @returns {number}
 */
export function rank(A, options) {
  const tol = (options && typeof options.tol === "number") ? options.tol : DEFAULT_TOLERANCE;
  const { pivots } = reduceWithPivoting(A, null, tol);
  return pivots.length;
}
