import numpy as np

def IEnKS():
    return None


def tsubspaceEnKA(
    theta_mat, y_observed_peturbed_mat, y_predicted_mat, svdt: float = 0.9
):
    """
    tsubspaceEnKA: Implmentation of the Ensemble Kalman Analysis in the
    ensemble subspace. This scheme is more robust in the regime where you have
    a larger number of observations and/or states and/or parameters than
    ensemble members.

    Inputs:
        theta_mat: Parameter ensemble matrix (n x n_ensemble_members array)
        y_observed_peturbed_mat: Perturbed observation ensemble matrix
        (m x n_ensemble_members array)
        y_predicted_mat: Predicted observation ensemble matrix
        (m x n_ensemble_members array)
        svdt: Level of truncation of singular values for pseudoinversion,
        recommended=0.9 (90%)
    Outputs:
        post: Posterior ensemble matrix (n x n_ensemble_members array)
    Dimensions:
        n_ensemble_members is the number of ensemble members, n is the number
        of state variables and/or parameters, and m is the number of
        observations.

    The implementation follows that described in Algorithm 6 in the book of
    Evensen et al. (2022), while adopting the truncated SVD procedure described
    in Emerick (2016) which also adopts the ensemlbe supspace method to the
    ES-MDA.

    Note that the observation error covariance R is defined implicitly here
    through the perturbed observations.
    This matrix (Y) should be perturbed in such a way that it is consistent
    with R in the case of single data
    assimilation (no iterations) or alpha*R in the case of multiple data
    assimilation (iterations). Moreover,
    although we should strictly be perturbing the predicted observations this
    does not make any difference in
    practice (see van Leeuwen, 2020) and simplifies the implmentation of the
    ensemble subspace approach.

    References:
        Evensen et al. 2022: https://doi.org/10.1007/978-3-030-96709-3
        Emerick 2016: https://doi.org/10.1016/j.petrol.2016.01.029
        van Leeuwen 2020: https://doi.org/10.1002/qj.3819

    Code by K. Aalstad (last revised December 2022)
    """

    n_ensemble_members = np.shape(theta_mat)[1]  # Number of ensemble members
    n_observations = np.shape(y_observed_peturbed_mat)[0]
    identity_mat_n_ens_memb = np.eye(n_ensemble_members)

    # Anomaly operator (subtracts ensemble mean)
    anomaly_operator = (
        identity_mat_n_ens_memb
        - np.ones([n_ensemble_members, n_ensemble_members]) / n_ensemble_members
    ) / np.sqrt(n_ensemble_members - 1)

    # Observation anomalies
    y_obs_anomalies = y_observed_peturbed_mat @ anomaly_operator
    # Predicted observation anomalies
    y_predicted_anomalies = y_predicted_mat @ anomaly_operator

    S = y_predicted_anomalies
    # Singular value decomposition
    [U, E, _] = np.linalg.svd(S, full_matrices=False)
    Evr = np.cumsum(E) / np.sum(E)
    N = min(n_ensemble_members, n_observations)
    these = np.arange(N)
    try:
        Nr = min(these[Evr > svdt])  # Truncate small singular values
    except Exception as exception:
        raise exception

    these = np.arange(Nr + 1)  # Exclusive python indexing
    E = E[these]
    U = U[:, these]
    Ei = np.diag(1 / E)
    P = Ei @ (U.T) @ y_obs_anomalies @ (y_obs_anomalies.T) @ U @ (Ei.T)
    [Q, L, _] = np.linalg.svd(P)
    LpI = L + 1
    LpIi = np.diag(1 / LpI)
    UEQ = U @ (Ei.T) @ Q
    # Pseudo-inversion of C=(C_YY+alpha*R) in the ensemble subspace
    Cpinv = UEQ @ LpIi @ UEQ.T
    innovation_mat = y_observed_peturbed_mat - y_predicted_mat  # Innovation
    W = (S.T) @ Cpinv @ innovation_mat
    T = identity_mat_n_ens_memb + W / np.sqrt(n_ensemble_members - 1)
    Xu = theta_mat @ T  # Update

    return Xu