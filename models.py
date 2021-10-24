from binance.client import Client
import numpy as np
import os
import datetime

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import adfuller

api_key = os.environ.get('binance_key')
api_secret = os.environ.get('binance_secret')
client = Client(api_key, api_secret)

def PCA_risk_model(return_matrix, n_components=10, use_ad_fuller=True, ad_fuller_alpha=0.05):
    '''
    takes a matrix of returns where the first index is time and
        the second is which coin
    returns a list of s-scores, one for each coin
        the scores are the result of decomposing the price movements of a given coin
        into the movements of n_components eigenportfolios found via PCA.
        the residuals from the PCA model are modeled as an Ornstein Uhlenbeck process
        and the final s-score can be interpreted as a z-score for how high or low
        the current cumulative returns are compared to what the model predicts
    '''

    scaler = StandardScaler()
    scaler.fit(return_matrix)

    scaled_returns = scaler.transform(return_matrix)

    pca = PCA(n_components=n_components)

    pca.fit(scaled_returns)

    p_components = pca.components_

    eigen_port_returns = p_components @ np.transpose(scaled_returns)

    coin_eigenport_covar_matrix = eigen_port_returns @ scaled_returns

    for i in range(n_components):
        eigenport_variance = np.var(eigen_port_returns[i])
        coin_eigenport_covar_matrix[i] = coin_eigenport_covar_matrix[i] / eigenport_variance

    expected_returns = np.transpose(eigen_port_returns) @ coin_eigenport_covar_matrix

    residuals = scaled_returns - expected_returns
    cumulative_residuals = np.cumsum(residuals, axis=0)    

    M = cumulative_residuals.shape[1]
    s_scores = np.ones(M)
    for i in range(M):
        if (use_ad_fuller) and (not (adfuller(cumulative_residuals[:, i])[1] < ad_fuller_alpha)):
            s_scores[i] = 0
        else:
            ar_model_results = AutoReg(cumulative_residuals[:, i], lags=1, old_names=False).fit()
            a_hat = ar_model_results.params[0]
            b_hat = ar_model_results.params[1]
            ar_residuals = ar_model_results.resid
            m_OU = a_hat / (1 - b_hat)
            sigma_OU = np.sqrt(np.var(ar_residuals) / (1 - (b_hat**2)))
            if sigma_OU != 0:
                S = -1 * m_OU / sigma_OU
            else:
                S = 0

            #print("Debugging: \n a_hat = {} \n b_hat = {} \n m_OU = {} \n sigma_OU = {} \n S = {} \n".format(a_hat, b_hat, m_OU, sigma_OU, S))
            s_scores[i] = S
            

    return s_scores