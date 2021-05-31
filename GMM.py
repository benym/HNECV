from sklearn.mixture import GaussianMixture


def initMiu_by_GMM(data, k, target):
    GMM = GaussianMixture(n_components=k, covariance_type='diag', init_params='kmeans', tol=target, reg_covar=1e-5).fit(
        data)
    miu = GMM.means_
    seita = GMM.covariances_
    aerfa = GMM.weights_
    return miu, seita, aerfa
