import { test } from "node:test";
import assert from "node:assert/strict";
import { solve, rank, DEFAULT_TOLERANCE } from "../src/index.js";

/**
 * Helper: compare two number arrays element-wise with an absolute tolerance.
 * Floating point arithmetic makes exact equality unreliable, so every numeric
 * assertion goes through here.
 */
function close(a, b, tol) {
  const t = tol || 1e-9;
  assert.equal(a.length, b.length, "length mismatch");
  for (let i = 0; i < a.length; i++) {
    assert.ok(
      Math.abs(a[i] - b[i]) <= t,
      "index " + i + ": " + a[i] + " !~= " + b[i] + " (tol " + t + ")"
    );
  }
}

test("solve returns empty solution for 0x0 system", () => {
  const r = solve([], []);
  assert.equal(r.ok, true);
  assert.equal(r.singular, false);
  assert.deepEqual(r.solution, []);
});

test("solve 1x1 nonzero", () => {
  const r = solve([[3]], [9]);
  assert.equal(r.ok, true);
  assert.equal(r.singular, false);
  close(r.solution, [3]);
});

test("solve 1x1 zero pivot is singular", () => {
  const r = solve([[0]], [5]);
  assert.equal(r.ok, false);
  assert.equal(r.singular, true);
});

test("solve 2x2 well-conditioned", () => {
  const A = [[2, 1], [1, 3]];
  const b = [3, 5];
  const r = solve(A, b);
  assert.equal(r.ok, true);
  close(r.solution, [4 / 5, 7 / 5]);
});

test("solve 3x3", () => {
  const A = [
    [2, 1, -1],
    [-3, -1, 2],
    [-2, 1, 2],
  ];
  const b = [8, -11, -3];
  const r = solve(A, b);
  assert.equal(r.ok, true);
  close(r.solution, [2, 3, -1]);
});

test("solve singular system (dependent rows)", () => {
  const A = [[1, 2], [2, 4]];
  const r = solve(A, [3, 6]);
  assert.equal(r.ok, false);
  assert.equal(r.singular, true);
});

test("solve recovers when zero pivot is avoided by row swap", () => {
  // First-row pivot is zero; partial pivoting must swap rows to proceed.
  const A = [[0, 2], [1, 1]];
  const r = solve(A, [4, 3]);
  assert.equal(r.ok, true);
  close(r.solution, [1, 2]);
});

test("solve throws on non-square matrix", () => {
  assert.throws(() => solve([[1, 2, 3], [4, 5, 6]], [1, 2]), /square/);
});

test("solve throws when b length mismatches A", () => {
  assert.throws(() => solve([[1, 0], [0, 1]], [1]), /length/);
});

test("solve throws on non-finite entry", () => {
  assert.throws(() => solve([[NaN, 0], [0, 1]], [1, 1]), /finite/);
});

test("rank of full-rank matrix", () => {
  assert.equal(rank([[1, 0], [0, 1]]), 2);
});

test("rank of rank-deficient matrix", () => {
  assert.equal(rank([[1, 2], [2, 4]]), 1);
});

test("rank of zero matrix is zero", () => {
  assert.equal(rank([[0, 0], [0, 0]]), 0);
});

test("rank of 3x4 rectangular", () => {
  assert.equal(rank([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]), 2);
});

test("solve does not mutate inputs", () => {
  const A = [[2, 1], [1, 3]];
  const b = [3, 5];
  const A0 = A.map(r => r.slice());
  const b0 = b.slice();
  solve(A, b);
  assert.deepEqual(A, A0);
  assert.deepEqual(b, b0);
});

test("DEFAULT_TOLERANCE is exported and finite", () => {
  assert.equal(typeof DEFAULT_TOLERANCE, "number");
  assert.ok(Number.isFinite(DEFAULT_TOLERANCE));
});
