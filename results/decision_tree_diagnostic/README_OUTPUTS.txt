SUPPLEMENTARY DECISION TREE CASE-LEVEL PROBABILITIES

This directory contains a selected supplementary output from the six-model
analysis: `decision_tree_test_case_probabilities.csv`, with case-level
Decision Tree predicted probabilities and case-level log-loss values.

The analysis script reproduces a broader six-model workflow; it is not a
Decision-Tree-only pipeline. This directory does not contain the complete
output set from that workflow.

For this case-level calculation, probabilities are clipped using machine
epsilon, whereas the primary analysis uses a 1e-15 lower bound. The
case-level log-loss values should therefore be interpreted separately.
