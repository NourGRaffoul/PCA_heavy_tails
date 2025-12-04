import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, TransformerMixin


#PCA ON SUBGAUSSIAN

## RPCA CLASS FROM CANDES PAPER (dganguli/robusty-pca)
#lambda: sparsity penalty, high lambda = super sparse S (L has higher rank)
#mu: shrinkage (small mu = data preserved)
class R_pca:
    def __init__(self, D, mu=None, lmbda=None):
        self.D = D
        self.S = np.zeros(self.D.shape)
        self.Y = np.zeros(self.D.shape)
        if mu:
            self.mu = mu
        else:
            self.mu = np.prod(self.D.shape) / (4 * np.linalg.norm(self.D, ord=1))
        self.mu_inv = 1 / self.mu
        if lmbda:
            self.lmbda = lmbda
        else:
            self.lmbda = 1 / np.sqrt(np.max(self.D.shape))

    @staticmethod
    def frobenius_norm(M):
        return np.linalg.norm(M, ord='fro')

    @staticmethod
    def shrink(M, tau):
        return np.sign(M) * np.maximum((np.abs(M) - tau), np.zeros(M.shape))
    

    def svd_threshold(self, M, tau):
        U, S, V = np.linalg.svd(M, full_matrices=False)
        return np.dot(U, np.dot(np.diag(self.shrink(S, tau)), V))

    def fit(self, tol=None, max_iter=1000):
        iter = 0
        err = np.Inf
        Sk = self.S
        Yk = self.Y
        Lk = np.zeros(self.D.shape)
        if tol:
            _tol = tol
        else:
            _tol = 1E-7 * self.frobenius_norm(self.D)

        while (err > _tol) and iter < max_iter:
            Lk = self.svd_threshold(
                self.D - Sk + self.mu_inv * Yk, self.mu_inv)
            
            residual = self.D - Lk + (self.mu_inv * Yk)
            threshold = self.mu_inv * self.lmbda
            Sk = self.shrink(residual,threshold)

            Yk = Yk + self.mu * (self.D - Lk - Sk)
            err = self.frobenius_norm(self.D - Lk - Sk)
            iter += 1
        self.L = Lk
        self.S = Sk
        s_vals = np.linalg.svd(Lk, compute_uv=False)
        #sparsity = np.mean(np.abs(Sk) > 1e-5)
        #print("sparsity: ",sparsity)
        return Lk, Sk

#TYLER'S ROBUST M-Estimator
class TylerPCA:
    def __init__(self, n_components=2, max_iter=100, tol=1e-5):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.components_ = None
        self.covariance_ = None

    def fit(self, X):
        loc = np.median(X, axis=0)
        X_centered = X - loc
        N, D = X_centered.shape
        Sigma = np.eye(D)
        
        for k in range(self.max_iter):
            try:
                P = np.linalg.inv(Sigma)
                XP = X_centered @ P
                dists_sq = np.sum(XP * X_centered, axis=1)
                dists_sq[dists_sq < 1e-10] = 1e-10
                weights = D / dists_sq
                X_weighted = X_centered * np.sqrt(weights[:, np.newaxis])
                Sigma_new = (X_weighted.T @ X_weighted) / N
                Sigma_new *= (D / np.trace(Sigma_new))
                
                if np.linalg.norm(Sigma_new - Sigma, ord='fro') < self.tol:
                    Sigma = Sigma_new
                    break
                Sigma = Sigma_new
            except np.linalg.LinAlgError:
                break 
        
        self.covariance_ = Sigma
        vals, vecs = np.linalg.eigh(Sigma)
        idx = np.argsort(vals)[::-1]
        self.components_ = vecs[:, idx].T[:self.n_components]
        return self


#PCA on different norms (FLOM)
class LpNormPCA(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=2, p=1):
        self.n_components = n_components
        self.p = p
        self.components_ = None

    def fit(self, X, y=None):
        n_samples, n_features = X.shape
        self.components_ = []
        
        # We work on a copy so we can modify it (deflation)
        X_curr = X.copy()
        
        for i in range(self.n_components):
            # Define Objective: Maximize ||X_curr @ v||_p
            # We minimize the negative because scipy is a minimizer
            def objective(v):
                # Enforce unit norm constraint strictly inside the objective
                norm_v = np.linalg.norm(v)
                if norm_v < 1e-10: return 0
                v_unit = v / norm_v
                
                projection = X_curr @ v_unit
                
                # Calculate L-p norm of the projection
                # Note: For p < 1, we must take abs() before power
                norm_val = np.sum(np.abs(projection) ** self.p)
                return -norm_val

            # 2. Optimization
            # Random initialization
            initial_guess = np.random.randn(n_features)
            
            # SLSQP is robust for bounded/constrained problems
            # We trust the internal normalization in 'objective' to handle constraints,
            # but SLSQP handles the gradient-free nature reasonably well here.
            res = minimize(objective, initial_guess, method='SLSQP')
            
            # Normalize result
            v_opt = res.x / np.linalg.norm(res.x)
            self.components_.append(v_opt)
            
            # 3. Deflation (Remove variance explained by this component)
            # X_new = X_old - (projection_scores * v_opt^T)
            scores = X_curr @ v_opt
            X_curr = X_curr - np.outer(scores, v_opt)
        
        self.components_ = np.array(self.components_)
        return self

    def transform(self, X):
        #Apply dimensionality reduction to X
        if self.components_ is None:
            raise ValueError("Model has not been fitted yet.")
        return X @ self.components_.T

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)


#R_pca (Candes) updated to search for lambda dynamically
class R_pca_dynamic:
    def __init__(self, D, mu=None, lmbda=None):
        self.D = D
        self.S = np.zeros(self.D.shape)
        self.Y = np.zeros(self.D.shape)
        self.L = np.zeros(self.D.shape)
        
        # Heuristic for mu (Step size)
        if mu: self.mu = mu
        else: self.mu = np.prod(self.D.shape) / (4 * np.linalg.norm(self.D, ord=1))

        self.mu_inv = 1 / self.mu
        
        # Theoretical base lambda from paper
        if lmbda: self.base_lambda = lmbda
        else: self.base_lambda = 1 / np.sqrt(np.max(self.D.shape))
             
    @staticmethod
    def frobenius_norm(M):
        return np.linalg.norm(M, ord='fro')

    @staticmethod
    def shrink(M, tau):
        return np.sign(M) * np.maximum((np.abs(M) - tau), 0)

    def svd_threshold(self, M, tau):
        U, S, V = np.linalg.svd(M, full_matrices=False)
        return np.dot(U, np.dot(np.diag(self.shrink(S, tau)), V))

    def _fit_single(self, current_lambda, tol=1E-7, max_iter=1000):
        #Runs RPCA with a specific, fixed lambda.
        #Returns L, S and the rank of L.
        
        D = self.D
        S = np.zeros_like(D)
        Y = np.zeros_like(D)
        L = np.zeros_like(D)
        
        mu_inv = self.mu_inv
        mu = self.mu
        
        iter = 0
        err = np.Inf
        
        while (err > tol) and iter < max_iter:
            L = self.svd_threshold(D - S + mu_inv * Y, mu_inv) #update L
            
            residual = D - L + (mu_inv * Y)
            threshold = mu_inv * current_lambda
            S = self.shrink(residual, threshold) #update S
            
            # Update Lagrange Multiplier
            Y = Y + mu * (D - L - S)
            
            # Convergence check
            err = self.frobenius_norm(D - L - S)
            iter += 1
            
        # Calculate Numerical Rank of L (Singular values > threshold)
        s_vals = np.linalg.svd(L, compute_uv=False)
        # Threshold: small relative to peak energy
        rank = np.sum(s_vals > (s_vals[0] * 1e-4))
        sparsity = np.mean(np.abs(S) > 1e-5)
        return L, S, rank, sparsity

    def fit(self, lfactors=None, target_rank=1, target_sparsity=0.1, rank_mode=False, verbose=False):
        #Searches for the best lambda by trying multiple factors.
        '''
        Args:
            lfactors: List of multipliers to try (e.g. [0.5, 1.0, 2.0]).
                      If None, defaults to a broad range.
            target_rank: Ideally, what rank should L be?
            target_sparsity: Ideally, what sparsity % should S be? (default: 30%)
            rank_mode: are we trying to match the rank (True) or the sparsity (False)
            verbose: Print search progress.
        Returns best_L and best_S
        '''

        if lfactors is None:
            lfactors = np.linspace(1,3,10)
            
        best_error = np.inf
        best_L = None
        best_S = None
        best_fac = None
        
        if verbose:
            if rank_mode:
                print(f"RPCA Dynamic Search | Target Rank: {target_rank}")
            else:
                print(f"RPCA Dynamic Search | Target Sparsity: {target_sparsity}")
            print(f"{'Factor':<10} | {'Rank(L)':<10} | {'Sparsity(S)':<15}")
            print("-" * 45)

        for fac in sorted(lfactors):
            # Try this lambda
            current_lambda = self.base_lambda * fac
            L, S, rank, sparsity = self._fit_single(current_lambda)
            
            if verbose:
                print(f"{fac:<10.2f} | {rank:<10} | {sparsity:.2%}")
        
            rank_dist = abs(rank - target_rank)
            sparsity_dist = abs(sparsity - target_sparsity)
            
            #Penalize wrong rank/sparsity heavily. If they match, penalize deviation from factor 1.0 (theoretical optimum)
            #This favors the "standard" solution if it works, but allows deviation if needed.
            if rank_mode:
                score = rank_dist * 1000 + abs(fac - 1.0)
            else:
                score = sparsity_dist * 1000 + abs(fac - 1.0)
            
            if score < best_error:
                best_error = score
                best_L = L
                best_S = S
                best_fac = fac
        
        if verbose:
            print("-" * 45)
            print(f"Selected Best Factor: {best_fac} (Rank {np.linalg.matrix_rank(best_L)})")
            
        self.L = best_L
        self.S = best_S
        return self.L, self.S



#Implementation of 'Robust Bayesian PCA with Student-t Distribution'.
#Uses MAP-EM (expectation maximization) to robustly estimate components and automatically select dimensionality using ARD.
class BayesianPCA:

    def __init__(self, n_components=None, max_iter=100, tol=1e-4, degrees_of_freedom=4.0):
        self.n_components = n_components #max components to start with
        self.max_iter = max_iter
        self.tol = tol
        self.nu = degrees_of_freedom #'v' in the paper (Control tail heaviness)
        self.components_ = None
        self.mean_ = None
        self.weights_ = None #the 'u' weights (outlier detection)

    def fit(self, X):
        N, D = X.shape
        Q = self.n_components if self.n_components else min(N, D) - 1
        
        ##INITIALIZE

        self.mean_ = np.mean(X, axis=0) 
        W = np.random.randn(D, Q) * 1e-3 #random projection matrix W (D x Q)
        sigma2 = 1.0 #noise variance
        
        # ARD Hyperparameter (one per component, will decide wether to keep component)
        alpha = np.ones(Q) 
        
        ##START ITERATING
        for it in range(self.max_iter):
            W_old = W.copy()
            
            #E-STEP: update latent variables (x) and weights (u) ---
            
            # M = W.T W + sigma2 * I
            M = W.T @ W + sigma2 * np.eye(Q)
            try:
                M_inv = np.linalg.inv(M)
            except np.linalg.LinAlgError:
                break # Stability break
                
            Xc = X - self.mean_
            
            #E[x_n] = M_inv @ W.T @ (t_n - mu)
            E_x = Xc @ W @ M_inv
            
            #Update E[u]
            norm_Xc = np.sum(Xc**2, axis=1)
            proj_Xc = Xc @ W
            term2 = np.sum((proj_Xc @ M_inv) * proj_Xc, axis=1)
            delta_n = (norm_Xc - term2) / sigma2
            
            E_u = (self.nu + D) / (self.nu + delta_n)
            
            #update 2nd moment E[xx^T] approx sum
            # E_xx_sum = sum( u_n * E[x]E[x].T ) + N * Cov(x)
            # Cov(x) = sigma2 * M_inv
            # Note: the true covariance term effectively cancels out bias in the W update
            E_xx_sum = N * sigma2 * M_inv + (E_u[:, None] * E_x).T @ E_x
            
            #M-STEP: Update Parameters ---
            
            #Update W
            S_tx = (X.T * E_u) @ E_x  #weighted cross-covariance
            Reg = sigma2 * np.diag(alpha)
            
            try:
                W = np.linalg.solve(E_xx_sum + Reg, S_tx.T).T
            except np.linalg.LinAlgError:
                pass
                
            #Update alphas
            w_norms = np.sum(W**2, axis=0)
            w_norms[w_norms < 1e-10] = 1e-10
            alpha = D / w_norms
            
            #Update sigma2
            recon = E_x @ W.T
            resid = Xc - recon
            weighted_sse = np.sum(E_u * np.sum(resid**2, axis=1))
            
            #Covariance trace correction: N * sigma2 * Tr(I - sigma2 * M_inv)
            #Actually simpler form: Tr(W^T W * N * sigma2 * M_inv)
            #E[ ||t - Wx||^2 ] approx weighted_sse + term from covariance
            #term = N * Tr(W^T W * Cov(x)) = N * sigma2 * Tr(W^T W M_inv)
            
            trace_correction = N * sigma2 * np.trace((W.T @ W) @ M_inv)
            sigma2 = (weighted_sse + trace_correction) / (N * D)
            
            #Safety floor for sigma2
            sigma2 = max(sigma2, 1e-6)
            
            #Check Convergence
            diff = np.linalg.norm(W - W_old) / (np.linalg.norm(W_old) + 1e-10)
            if diff < self.tol:
                break
        
        #PRUNE COMPONENTS
        relevance = 1.0 / alpha
        threshold = 1e-3 * np.max(relevance)
        active_indices = np.where(relevance > threshold)[0]
        
        if len(active_indices) == 0:
            active_indices = [np.argmax(relevance)]
            
        self.components_ = W[:, active_indices].T
        self.weights_ = E_u 
        
        return self