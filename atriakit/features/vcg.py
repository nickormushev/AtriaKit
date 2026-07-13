import numpy as np


def vcg_area_calculator(vcg_segment, fs):
    """Compute the VCG loop area as the integral of the 3-D magnitude.

    Args:
        vcg_segment: Array of shape ``(3, n_samples)`` containing the X, Y, Z
            Kors components of the VCG loop.
        fs: Sampling frequency in Hz.

    Returns:
        Loop area in mV·s.
    """
    magnitude = np.linalg.norm(vcg_segment, axis=-2)
    area = np.sum(magnitude) / fs
    return area


def vcg_eigenfeatures_calculator(segment):
    """Compute PCA-based shape features of a VCG loop.

    Centres the loop, runs SVD, and derives eigenvalues (variance per axis),
    roundness (ratio of 2nd to 1st eigenvalue), and flatness (ratio of 3rd
    eigenvalue to the sum of the first two).

    Args:
        segment: Array of shape ``(3, n_samples)`` containing the X, Y, Z
            Kors components of the VCG loop.

    Returns:
        Tuple of ``(eigenvalues, roundness, flatness, eigenvectors)``, where
        ``eigenvalues`` and ``eigenvectors`` are sorted by descending variance.
    """
    X = segment - segment.mean(axis=1, keepdims=True)
    U, S, _ = np.linalg.svd(X, full_matrices=False)

    # Singular values → eigenvalues of the sample covariance matrix
    eigenvalues = (S**2) / (X.shape[1] - 1)

    idx = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[idx]
    eigenevectors = U[:, idx]

    roundness = sorted_eigenvalues[1] / (sorted_eigenvalues[0] + 1e-10)
    flatness = sorted_eigenvalues[2] / (
        sorted_eigenvalues[0] + sorted_eigenvalues[1] + 1e-10
    )

    return (
        sorted_eigenvalues,
        roundness,
        flatness,
        eigenevectors,
    )


def vcg_axis_angles(segment: np.ndarray) -> tuple[float, float]:
    """Return the 3-D axis of a P-wave VCG loop as (elevation, azimuth) in degrees.

    The axis is the net dipole: the time-integral of each Kors XYZ component
    (X = left, Y = inferior, Z = posterior), equivalent to the mean electrical
    axis.

    Args:
        segment: Array of shape ``(3, n_samples)`` containing the X, Y, Z
            Kors components of the VCG loop.

    Returns:
        Tuple of ``(elevation, azimuth)`` in degrees, where elevation is the
        angle above the frontal (XY) plane in ``[-90, 90]`` and azimuth is the
        angle from the X-axis in the frontal plane in ``(-180, 180]``, matching
        the clinical frontal P-wave axis convention.
    """
    # Integrate each component over time → net dipole vector (3,)
    net = np.sum(segment, axis=1)
    elevation = float(
        np.degrees(np.arctan2(net[2], np.sqrt(net[0] ** 2 + net[1] ** 2)))
    )
    azimuth = float(np.degrees(np.arctan2(net[1], net[0])))
    return elevation, azimuth
