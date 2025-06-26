# Option Visualizer
This project is dedicated to the payoff visualization of the simple option portfolios
At the current iteration the project supports visualization of the portfolios with asian, european and american options with the following limitations:
1) No barrier functionality enabled
2) Asian options are not realized path-dependent, thus the averaged value over asianing dates is inputted via the slider

Backlog for the project>
1) Enable barrier functionlity and visualization
2) Create example portfolios, option strategies (digital options) and structured products (Reverse convertibles, etc)
3) Overhaul asianing logic: allow the input of the exact values per asianing dates
4) Ensure vectorization and increased speed
5) Use the portfolios as a mask for the real time-series for the further development (to calculate greeks, risk profiles, etc.)
