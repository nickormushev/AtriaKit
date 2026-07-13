"""Package-level constants shared across modules."""

# Leads required by the Kors regression matrix to compute a 3-lead VCG.
VCG_LEADS: list[str] = ["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"]

# Output columns produced by VCG feature computation.
VCG_FEATURE_COLUMNS: list[str] = [
    "vcg_eigenvalues_1", "vcg_eigenvalues_2", "vcg_eigenvalues_3",
    "vcg_roundness", "vcg_flatness", "vcg_area",
    "vcg_axis_elevation", "vcg_axis_azimuth",
    "vcg_sum_fragment_count", "vcg_sum_fragment_width", "vcg_sum_fragment_height",
    "vcg_x_fragment_count", "vcg_x_fragment_width", "vcg_x_fragment_height",
    "vcg_y_fragment_count", "vcg_y_fragment_width", "vcg_y_fragment_height",
    "vcg_z_fragment_count", "vcg_z_fragment_width", "vcg_z_fragment_height",
]

# Leads required to compute the frontal P-wave axis.
AXIS_LEADS: list[str] = ["I", "aVF"]
