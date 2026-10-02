/**
 * Public entry point for the Gaussian Elimination Solver.
 *
 * Re-exports the solver and rank functions so consumers can import a single
 * module path. Keeping the barrel thin makes it easy to audit the public API.
 */
export { solve, rank, DEFAULT_TOLERANCE } from "./core.js";
