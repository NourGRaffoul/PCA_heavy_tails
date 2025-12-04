import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

from PCA_classes import *
from PCA_functions_generation import *
from PCA_functions_components import *
from PCA_functions_plotting import *
from parser import get_args, custom_dim_list, real_data_list


def run_image_experiment(args):

    p_value=args.p_value
    n_samples=args.n_samples
    contamination = args.contamination
    n_comp = args.n_comp
    seed=args.seed

    X_clean, shape = get_image_data(n_samples=n_samples,seed=seed)
    X = corrupt_images(X_clean, shape, contamination=contamination) 

    labels={"standard":"Standard \n PCA", "rpca":"RPCA \n(Candes)","robpca":"RobPCA \n(Robpy)",
            "spher":"Spherical \n PCA (Robpy)", "lp":f"Lp Norm PCA (p={p_value[0]})","tyler":"Tyler's M Estimator",
            "dynamic":"RPCA 2 \n(higher lambda)","bay":"Bayesian \nRPCA (Gai)"}
    
    #Run all PCAs on 5 components
    rpca = R_pca(X)
    L, S = rpca.fit()

    rpca_dynamic = R_pca(D=X,lmbda=rpca.lmbda*2)
    L_dynamic,S_dynamic = rpca_dynamic.fit()

    PC={}
    print("Starting ROBPCA...")
    PC["robpca"]=np.transpose((ROBPCA(n_components=n_comp).fit(X)).components_)

    print("Starting Standard PCA...")
    PC["standard"] = (PCA(n_components=n_comp).fit(X)).components_

    print("Starting RPCA...")
    PC["rpca"] = (PCA(n_components=n_comp).fit(L)).components_

    print("Starting RPCA (higher lambda)...")
    PC["dynamic"]=(PCA(n_components=n_comp).fit(L_dynamic)).components_

    print("Starting Robust Bayesian PCA...")
    PC["bay"]=(BayesianPCA(n_components=n_comp).fit(X)).components_

    #print("Starting spherical PCA...") #slow
    #PC["spher"]=np.transpose((PCALocantore(n_components=n_comp).fit(X)).components_)

    #print("Starting Lp Norm PCA...") #suuuuper slow
    #PC["lp"]= (LpNormPCA(n_components=n_comp, p=p_value).fit(X)).components_

    #print("Starting Tyler's M estimator...") #slowww
    #PC["tyler"]= (TylerPCA(n_components=n_comp).fit(X)).components_

    n_methods=len(PC)
    n_col=n_methods+6 #also show X_clean, X, L, S, L_dynamic and S_dynamic
    #pca = (PCA(n_components=n_comp).fit(X))
    n_row=3

    fig, axes = plt.subplots(n_row, n_col, figsize=(2.26 * n_col, 2.0 * n_row))
    fig.suptitle("Principal Component Analysis Results", size=16)

    plot_faces(title=f"Original \n (First {n_row})", images=X_clean, image_shape=shape,col_index=0,n_row=n_row,axes=axes)
    plot_faces(title="Corrupted", images=X, image_shape=shape,col_index=1,n_row=n_row,axes=axes)
    i=2
    for key in PC:
        #plot_faces(labels[key], np.vstack([pca.mean_[None, :], PC[key]]), shape)
        plot_faces(title=labels[key], images=PC[key], image_shape=shape, col_index=i, n_row=n_row, axes=axes)
        if key=="rpca": 
            i+=2
            plot_faces("RPCA \n(L matrix)", L, shape,i-1,n_row,axes)
            plot_faces("RPCA \n(S matrix)", S, shape,i,n_row,axes)
        elif key=="dynamic": 
            i+=2
            plot_faces("RPCA 2\n (L matrix)", L_dynamic, shape,i-1,n_row,axes)
            plot_faces("RPCA 2\n (S matrix)", S_dynamic, shape,i,n_row,axes)
        i+=1

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust to make space for the main title
    plt.show()

if __name__ == "__main__":
    run_image_experiment()