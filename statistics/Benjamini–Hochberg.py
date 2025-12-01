import numpy as np
from statsmodels.stats.multitest import multipletests

# your original p-values (example)
p_values = np.array([0.0266, 0.0134, 0.189])

# Benjamini–Hochberg FDR correction
reject, pvals_corrected, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')

print("Original p-values: ", p_values)
print("BH-corrected p-values: ", pvals_corrected)
print("Reject H0 after correction? ", reject)