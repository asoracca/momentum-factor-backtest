# Synthetic monthly fixture

Prices are deterministic toy levels, not simulated evidence of market returns.
A, B, C, D compound at -1%, 0%, 1%, 2% per month; E at 3%.
At January 2021 formation, ranks ascending are A, B, C, D; 12-1
scores are respectively `0.99**11-1`, `0`, `1.01**11-1`, `1.02**11-1`.
Long-short holds A=-1, D=+1; long-only D=1; equal-weight each=1/4.
Initial absolute-weight turnover is 2, 1, 1; next month it is 0 for each.
E enters March 2021 with pre-entry price history, then replaces D as winner:
long-short turnover 2, long-only 2, equal-weight 0.4.
B is missing May 2021; it becomes ineligible that month and for the next
12 formation dates. An existing equal-weight holding has unknown May P&L.
D falls 80% in November 2021, then terminates; membership ends December 1.
The November loss is included; no liquidation value is invented for December.
E exits January 2023. Intervals are start-inclusive/end-exclusive, assumed
known at formation time. The synthetic membership does not repair real data.
