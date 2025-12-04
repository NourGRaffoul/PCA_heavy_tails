import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import levy_stable
from robpy.pca import ROBPCA
from robpy.pca.spca import PCALocantore
from robpy.preprocessing import DataCleaner, RobustPowerTransformer, RobustScaler
from sklearn.linear_model import HuberRegressor
from sklearn.datasets import fetch_olivetti_faces

from parser import custom_dim_list, real_data_list

#CATEGORY1: DATA GENERATION
#CATEGORY2: COMPONENTS
#CATEGORY 3: PLOTTING

'''
!!!!!!!!!!!!!!!!!!!!!!!!!!
CATEGORY1: DATA GENERATION
!!!!!!!!!!!!!!!!!!!!!!!!!!
'''
def generate_random_correlation(n):
    #Generates a valid random correlation matrix.
    #Method: Generate random matrix A, compute C = AA^T, then normalize to correlation.
   
    # A with shape (n, n) generates a Wishart-distributed covariance
    A = np.random.randn(n, n)
    C = np.dot(A, A.T)
    d = np.sqrt(np.diag(C))
    # R_ij = C_ij / (d_i * d_j)
    R = C / np.outer(d, d)
    C2 = C / C[0][0]
    return (C2)

def generate_spectral_measure(dim, n_points):
    if dim==2:
        weights = np.random.uniform(0.1, 1.0, size=n_points)
        angles = np.random.uniform(0, 360, size=n_points)
        return angles.tolist(), weights.tolist()
    else:
        # Generates random points on the unit sphere and associated weights.

        # Gaussian vectors normalized are uniformly distributed on the sphere, se we use np.random.standard_normal then normalize
        raw_points = np.random.standard_normal((n_points, dim))
        norms = np.linalg.norm(raw_points, axis=1, keepdims=True)
        
        # Safety: Handle zero vectors
        norms[norms == 0] = 1.0 
        points = raw_points / norms
        
        # Generate random weights (exponentially distributed)
        weights = np.random.exponential(1.0, size=n_points)

        return points.tolist(), weights.tolist()


## SUBG DATA GENERATION
def gen_subg(n_samples=500, cov_matrix=None, alpha=1.5,center=True): 
    #Covariance
    if cov_matrix is None:
        dim=2
        rho = 0.9
        cov_matrix = np.ones((dim, dim)) * rho
        np.fill_diagonal(cov_matrix, 1.0)
    else:
        cov_matrix = np.array(cov_matrix)
        dim = cov_matrix.shape[0]

    #G ~ N(0, Sigma)
    G = np.random.multivariate_normal(mean=np.zeros(dim), cov=cov_matrix, size=n_samples)
    #A ~ S_{alpha/2}
    if alpha >= 2.0:
        A = np.ones((n_samples, 1)) 
    else:
        a_param = alpha / 2.0
        A = levy_stable.rvs(alpha=a_param, beta=1.0, loc=0, scale=1.0, size=(n_samples, 1))
        A = np.abs(A)

    X = np.sqrt(A) * G
    return X

def gen_student_t(n_samples, cov_matrix, df):
    dim = cov_matrix.shape[0]
    G = np.random.multivariate_normal(np.zeros(dim), cov_matrix, n_samples)
    chisq = np.random.chisquare(df, size=(n_samples, 1))
    scale = np.sqrt(df / chisq)
    return G * scale


def gen_pareto(n_samples, cov_matrix, alpha):
    dim = cov_matrix.shape[0]
    P = np.random.pareto(alpha, size=(n_samples, dim))
    P_centered = P - np.median(P, axis=0)
    L = np.linalg.cholesky(cov_matrix)
    return P_centered @ L.T


def gen_spectral(n_samples, alpha, angles, weights):
    #Generates data from a discrete spectral measure.
    #X = sum(weight_j^(1/alpha) * Z_j * u_j)   where Z_j is symmetric 1D stable, u_j is unit vector at angle_j

    if len(angles) != len(weights):
        raise ValueError("Angles and weights must have the same length")
    
    # Convert angles (degrees) to unit vectors
    rads = np.radians(angles)
    # vectors shape: (n_directions, 2)
    unit_vectors = np.stack([np.cos(rads), np.sin(rads)], axis=1)
    
    n_dirs = len(angles)
    
    # Generate independent symmetric stable variables for each direction
    # Shape: (n_samples, n_directions)
    if alpha >= 2.0:
        Z = np.random.randn(n_samples, n_dirs)
    else:
        # beta=0 means symmetric
        Z = levy_stable.rvs(alpha=alpha, beta=0, loc=0, scale=1.0, size=(n_samples, n_dirs))  #symmetrical here, so its okay if the measure isn't symmetrical
        
    # Scale weights. 
    # To have spectral mass 'w', we need coefficient w^(1/alpha)
    scale_factors = np.array(weights) ** (1/alpha)
    
    # (n_samples, n_dirs) * (n_dirs,) -> (n_samples, n_dirs)
    scaled_Z = Z * scale_factors
    
    # Project onto unit vectors
    # (n_samples, n_dirs) @ (n_dirs, 2) -> (n_samples, 2)
    X = scaled_Z @ unit_vectors
    return X

def gen_spectral_ndim(n_samples, alpha, points, weights):
    # Generates data from a discrete spectral measure in N-dimensions.
    # points (list of lists): coordinates on the unit sphere (cartesian) (not angles in degrees)

    points = np.array(points)
    weights = np.array(weights)
    
    if len(points) != len(weights):
        raise ValueError(f"Points ({len(points)}) and weights ({len(weights)}) must match length")
        
    #Normalize. Norms shape: (n_dirs, 1)
    norms = np.linalg.norm(points, axis=1, keepdims=True)

    norms[norms == 0] = 1.0 
    unit_vectors = points / norms
    
    n_dirs = len(points)
    
    # Generate independent symmetric stable variables. Shape: (n_samples, n_directions)
    if alpha >= 2.0:
        Z = np.random.randn(n_samples, n_dirs)
    else:
        # Beta=0 ensures the distribution is symmetric (S alpha S)
        Z = levy_stable.rvs(alpha=alpha, beta=0, loc=0, scale=1.0, size=(n_samples, n_dirs))
        
    scale_factors = weights ** (1/alpha)
    scaled_Z = Z * scale_factors

    # Project onto unit vectors: (n_samples, n_dirs) @ (n_dirs, n_dim) -> (n_samples, n_dim)
    X = scaled_Z @ unit_vectors
    
    return X


#Generates data that mimics stock returns with volatility clustering.
def gen_financial(n_companies, n_days, rho, alpha,random,covariance):
    # 1. Create a covariance matrix with a strong market factor (Everyone correlates rho with everyone else)
    if random:
        cov=covariance
    else:
        market_vol = rho
        cov = np.full((n_companies, n_companies), market_vol)
        np.fill_diagonal(cov, 1.0)
    
    # 2. Generate Heavy Tailed Data (Student-t with df=3)
    # G ~ N(0, Sigma)
    G = np.random.multivariate_normal(np.zeros(n_companies), cov, n_days)
    # Chi-square scale for tails
    df = alpha
    chisq = np.random.chisquare(df, size=(n_days, 1))
    scale = np.sqrt(df / chisq)
    X = G * scale
    
    # 3. Add "Market Crash" days (Extreme Volatility, 5% of days are 5x more volatile)
    crash_days = np.random.rand(n_days) < 0.05
    X[crash_days] *= 5.0
    return X,cov

def gen_real_stock(CACHE_FILE="real_stock1.csv",n_companies=50, n_days=500):
    BENCHMARK='SPY'
    print(f"Loading data from local cache: {CACHE_FILE}")
    df_prices = pd.read_csv(CACHE_FILE, index_col=0, parse_dates=True)
    df_returns = np.log(df_prices / df_prices.shift(1)).dropna() #Log returns: ln(P_t / P_{t-1})
    spy_returns = df_returns[BENCHMARK].values
    df_stocks = df_returns.drop(columns=[BENCHMARK])
    real_data = df_returns.values
    
    # Handle insufficient days
    if real_data.shape[0] < n_days:
        print(f"Warning: Cache has {real_data.shape[0]} days, but requested {n_days}.")
        real_days = real_data.shape[0]
    else:
        real_days = n_days
        real_data = real_data[:n_days]

    #Handle number of companies
    available_companies = real_data.shape[1]
    
    if n_companies <= available_companies:
        # Randomly select N columns
        indices = np.random.choice(available_companies, n_companies, replace=False)
        X = real_data[:, indices]
        dataset_name = f"Real Market Data ({n_companies} random stocks)"
    else:
        print(f"Requested {n_companies} companies, but only {available_companies} real tickers available.")
        X = real_data
        dataset_name = f"Real Market Data ({available_companies} random stocks)"

    if X.shape[0] > n_days: X = X[:n_days]
    if spy_returns.shape[0] > n_days: spy_returns = spy_returns[:n_days]
    #print("SHAPES: ",X.shape,spy_returns.shape)
    print(dataset_name)
    return X, spy_returns

def get_image_data(n_samples=10, resize=0.5, seed=649):
    try:
        # Load standard face dataset (4096 dimensions = 64x64)
        data = fetch_olivetti_faces(shuffle=True, random_state=seed)
        images = data.images[:n_samples]
        
        # Determine shape
        n_img, h, w = images.shape
        X = data.data[:n_samples]
        
        print(f"Loaded {n_img} images of size {h}x{w} ({h*w} features)")
        return X, (h, w)
        
    except Exception as e:
        print(f"Could not load faces: {e}.")


def corrupt_images(X, image_shape, contamination=0.2):
    #Adds Salt&Pepper noise and Block Occlusions (like the paper).

    X_corr = X.copy()
    n_samples, n_features = X.shape
    h, w = image_shape
    
    # Salt and pepper noise (on random subset of images)
    n_corrupt = int(n_samples * contamination)
    indices = np.random.choice(n_samples, n_corrupt, replace=False)
    
    print(f"Corrupting {n_corrupt} images with noise and occlusions...")
    
    for idx in indices:
        img = X_corr[idx].reshape(h, w)
        
        #10% of pixels
        mask = np.random.rand(h, w)
        img[mask < 0.05] = 0.0 
        img[mask > 0.95] = 1.0
        
        # Occlusion (Black Box)
        # 10x10 black box at random location
        r, c = np.random.randint(0, h-15), np.random.randint(0, w-15)
        img[r:r+15, c:c+15] = 0.0
        
        X_corr[idx] = img.flatten()
        
    return X_corr


def gen_data_wrapper(n_samples=500, cov_matrix=None, param=1.5, mode='subgaussian', 
                     spectral_angles=None, spectral_weights=None,dim=2,rho=0.4,random=False):

    # Setup Covariance (Default Identity)
    if cov_matrix is None:
        cov_matrix = np.eye(dim)
    else:
        cov_matrix = np.array(cov_matrix)

    if mode == 'subgaussian':
        X = gen_subg(n_samples, cov_matrix, alpha=param)
    elif mode == 'student_t':
        df = 100 if param >= 2.0 else param
        X = gen_student_t(n_samples, cov_matrix, df=df)
    elif mode == 'pareto':
        X = gen_pareto(n_samples, cov_matrix, param)
    elif mode == 'financial':
        X,cov_matrix = gen_financial(n_companies=dim,n_days=n_samples,alpha=param, rho=rho,covariance=cov_matrix,random=random)
    elif mode == 'real_stock1':
        X,cov_matrix = gen_real_stock(CACHE_FILE="real_stock1.csv",n_companies=dim,n_days=n_samples)
        # Here, cov_matrix is the SPY returns!!!!! careful
    elif mode== 'real_stock2':
        X,cov_matrix = gen_real_stock(CACHE_FILE="real_stock2.csv",n_companies=dim,n_days=n_samples)
    elif mode == 'spectral':
        if spectral_angles is None or spectral_weights is None:
            raise ValueError("For 'spectral' mode, must provide angles and weights.")
        if dim==2: X = gen_spectral(n_samples, param, spectral_angles, spectral_weights)
        else: X = gen_spectral_ndim(n_samples, param, spectral_angles, spectral_weights)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    return X, cov_matrix