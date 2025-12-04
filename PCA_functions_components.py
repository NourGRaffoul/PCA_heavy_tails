import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from sklearn.decomposition import PCA
from scipy.stats import levy_stable
from robpy.pca import ROBPCA
from robpy.pca.spca import PCALocantore
from robpy.preprocessing import DataCleaner, RobustPowerTransformer, RobustScaler
from sklearn.linear_model import HuberRegressor

from parser import custom_dim_list, real_data_list

'''
!!!!!!!!!!!!!!!!!!!!!!!!!!
CATEGORY2: COMPONENTS
!!!!!!!!!!!!!!!!!!!!!!!!!!
'''
def calc_angle_error(vec_estimated, vec_true,mode):
    # Normalize
  if vec_estimated.all==None:
    print("BAD VALUE")
    return -45
  else:
    #print("HERE")
    v_est = vec_estimated / np.linalg.norm(vec_estimated)
    v_true = vec_true / np.linalg.norm(vec_true)
    
    # Dot product
    cos_theta = np.abs(np.dot(v_est, v_true))
    cos_theta = np.clip(cos_theta, 0, 1)
    result= np.degrees(np.arccos(cos_theta))
    #print("result: ",result) #REMOVE
    #if mode=='spectral' and result!=-45:
    #    return min(abs(result-90),result)
    #else: return result
    return result


def calc_angle_error_generous(vec_estimated, v1,v2,v3):
    # Normalize
  if vec_estimated.all==None:
    print("BAD VALUE")
    return -45
  else:
    #print("HERE")
    vec_true=[v1,v2,v3]
    #print(vec_true[0]-v1)
    v_est = vec_estimated / np.linalg.norm(vec_estimated)
    min_result=90
    for i in range(3):
        #print("LOOP ITERATION ",i)
        v_true = vec_true[i] / np.linalg.norm(vec_true[i])
        # Dot product
        cos_theta = np.abs(np.dot(v_est, v_true))
        cos_theta = np.clip(cos_theta, 0, 1)
        result= np.degrees(np.arccos(cos_theta))
        if result<min_result: 
            #print("RESULT: ",result,"    smaller than   ", min_result)
            min_result=result
    return min_result

def get_true_components(cov_matrix,n=0):
    #find eigenvectors and eigenvalues.
    vals, vecs = np.linalg.eigh(cov_matrix)
    # Sort descending
    idx = np.argsort(vals)[::-1]
    return vals[idx], vecs[:, idx]


def get_spectral_true_components(angles, weights,dim,n):
    if n==0: idx = np.argmax(weights) # Find the index of the maximum weight
    else:
        neg_weights=[-1*i for i in weights]
        indices=np.argpartition(neg_weights, n)[:n] #returns the indices of the n smallest elements of neg_weights (so n largest weights)
        idx = indices[-1]

    if dim==2:
        max_angle = angles[idx]
        rad = np.radians(max_angle)
        return np.array([np.cos(rad), np.sin(rad)])
    else:
        return np.array(angles[idx])


def get_ROBPCA(X,n_comp=2,alpha=0.75,spher=False):
        comps_rob=None
        X_robpca = RobustScaler(with_centering=False).fit_transform(X)
        X_robpca = np.clip(X_robpca, -1000, 1000)
        if spher: 
            pca = PCALocantore(n_components=n_comp).fit(X_robpca)
        else: 
            pca = ROBPCA(n_components=n_comp,alpha=alpha).fit(X_robpca)   #PCALocantore (spherical) or ROBPCA
        #for ROBPCA, add alpha between 0.5 and 1 #smal=robust, big=accurate
        comps_rob = pca.components_
        if comps_rob is not None:
            if not (comps_rob.size > 0 and comps_rob.shape[0] > 0):
                    comps_rob = None
        return np.transpose(comps_rob)
        #return comps_rob
    
def get_pc1(covariance,angles,weights,current_mode,dim,n=0,X=None):
    if current_mode=='spectral' : 
            true_pc1 = get_spectral_true_components(angles=angles,weights=weights,dim=dim,n=n)
    elif current_mode in ['real_stock1','real_stock2']:
            true_pc1 = get_spy_beta_truth(X,covariance)
    else:
            true_vals, true_vecs = get_true_components(np.array(covariance))
            true_pc1 = true_vecs[:, n] # The dominant direction
    return true_pc1


#FINANCIAL COMPONENTS
def PCA_trim(X, trim_percentile=10):
    #Remove the top X% most volatile days (largest L2 norms) and run standard PCA on the remaining.
    # Calculate magnitude of market movement each day and find cutoff to trim data
    magnitudes = np.linalg.norm(X, axis=1)
    cutoff = np.percentile(magnitudes, 100 - trim_percentile)

    mask_clean = magnitudes <= cutoff
    X_clean = X[mask_clean]
    # Run Standard PCA on clean data
    pca_clean = PCA(n_components=1)
    pca_clean.fit(X_clean)

    return pca_clean.components_, np.sum(mask_clean)

def get_spy_beta_truth(X, spy_returns):
    """
    Ideally, PC1 should align with the market exposure of each stock.
    We compute this using Robust Linear Regression (Huber) to ignore crash days automatically in the truth definition.
    """
    n_companies = X.shape[1]
    betas = []
    
    # Regress each stock against SPY
    for i in range(n_companies):
        # HuberRegressor is robust to outliers in Y (stock specific crashes)
        # We need to reshape for sklearn
        y = X[:, i]
        x = spy_returns.reshape(-1, 1)
        reg = HuberRegressor().fit(x, y)
        betas.append(reg.coef_[0])
        
    betas = np.array(betas)
    return betas / np.linalg.norm(betas)

