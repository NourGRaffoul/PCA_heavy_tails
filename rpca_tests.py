import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from robpy.pca import ROBPCA
from robpy.pca.spca import PCALocantore
from robpy.preprocessing import DataCleaner, RobustPowerTransformer, RobustScaler
#from rpca_functions import *
import warnings

from PCA_classes import *
from PCA_functions_generation import *
from PCA_functions_components import *
from PCA_functions_plotting import *
from image_test import run_image_experiment
from parser import get_args, custom_dim_list, real_data_list, image_list

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")


def run_experiment(args):
    print(f"Parsed Arguments: {args}")

    #UNPACKING PARSED ARGUMENTS 
    n_samples=args.n_samples
    seed=args.seed
    plot=args.plot
    n_comp = args.n_comp

    current_mode=args.mode
    dim=args.dimension
    alpha_values=args.alphas
    p_value=args.p_value

    weights=args.weights
    angles=args.angles
    rho=args.rho

    random=args.random
    
    plot_error=True #always true except for real data
    generous_flag=False  #if true, it compares the returned component with the top 3 PCs and takes the smallest error

    #CHECK ALL VALUES MAKE SENSE
    if dim>2 and current_mode not in custom_dim_list:
        dim=2
        print("Only 2D is supported here")

    if current_mode=='spectral' and random==False and dim>2:
        random=True
        print("For random vectors in more than 2 dimensions, the spectral measure is always randomly generated.")

    if current_mode=='real_stock_1':
        if n_samples>500:
            n_samples=500
            print("Only 500 days of real stock returns are available in this dataset")
        elif dim>50:
            dim=50
            print("Only 50 companies are available in this dataset")

    if (current_mode!='spectral' and current_mode not in real_data_list):
        if (random or dim>2) :
            covariance=generate_random_correlation(dim)
            cov2 = np.round(covariance[:6, :6], 3)
            print(f'Covariance matrix of dimension {dim} randomly generated. Cov={cov2}')
        else:
            c=args.covariance
            covariance=[[c[0],c[1]],[c[2],c[3]]]
            covariance=np.array(covariance)
            print(f'Covariance matrix: {covariance}')
    else: covariance=np.array([[1,0],[0,1]])

    if current_mode=='spectral':
        if random or dim>2:
            n_points= np.random.randint(2, 5*dim)
            angles,weights=generate_spectral_measure(dim=dim,n_points=n_points)
            print(f"Spectral measure with {n_points} points was randomly generated.")
        #if dim==2: print(f'Angles (in degrees): {angles} \n Weights: {weights}')
        #else: print(f'Points (cartesian coordinates): {angles} \n Weights: {weights}')

    if current_mode in real_data_list: 
        alpha_values = [1] #cannot specify alphas
        plot_error = False #cannot plot error since there is only one value of alpha, it is printed out instead

    if (current_mode in custom_dim_list) and dim>2: plot=False #cannot plot more than 2D currently

    n_comp = min(n_comp,dim) #of course cannot have more components than dimensions
    #DONE

    #Pick seed
    np.random.seed(seed)
    print(f"Running Experiment with Distribution: {current_mode.upper()}")
    

    plot_flags={"standard":True, "rpca":True,"robpca":True,"spher":True,"lp":True,"tyler":True,"dynamic":True, "bay":True}

    labels={"standard":"Standard PCA", "rpca":"Robust PCA (Candes)","robpca":"Robust PCA (Robpy)",
            "spher":"Spherical PCA (Robpy)", "lp":f"Lp Norm PCA (p={p_value[0]})","tyler":"Tyler's M Estimator","dynamic":"RPCA 2 (higher lambda)","bay":"Bayesian RPCA (Gai)"}

    colors={"standard":'red', "rpca":'green',"robpca":'blue',"spher":'cyan',"lp":'violet',"tyler":'lime',"dynamic":'orange',"bay":'purple'}

    errors={}
    errors_generous={}
    for key in labels: 
        errors[key]=[]
        errors_generous[key]=[]

    #get eigenvectors
    if current_mode not in real_data_list:
        true_pc1=get_pc1(covariance=covariance,angles=angles,weights=weights,current_mode=current_mode,dim=dim)
    else: true_pc1 = None #will be filled later

    if generous_flag:
        true_pc2=get_pc1(covariance=covariance,angles=angles,weights=weights,current_mode=current_mode, dim=dim, n=1)
        true_pc3=get_pc1(covariance=covariance,angles=angles,weights=weights,current_mode=current_mode, dim=dim, n=2)

    #get plot title
    title_cov = np.round(covariance[:2, :2], 3)

    if current_mode in custom_dim_list:   
        title_str=f'mode: {current_mode}, \ndimensions: {dim},  seed: {seed},  # samples: {n_samples}'
    else:   
        title_str=f'mode: {current_mode},  seed: {seed},  # samples: {n_samples}'




    ####LOOP STARTS. REPEAT FOR EACH VALUE OF ALPHA.
    for i, a in enumerate(alpha_values):
            print(f"Running PCA comparison for alpha = {a}...")
            
            #Generate Data
            #IMPORTANT: IF THE MODE IS REAL DATA, COVARIANCE WILL CONTAIN THE SPY RETURNS!!!!!!!!!!!!!!!!!!!!!!!!!!
            X, covariance = gen_data_wrapper(n_samples=n_samples, cov_matrix=covariance, param=a, mode=current_mode, 
                                                spectral_angles=angles,spectral_weights=weights,dim=dim,rho=rho,random=random)

            if current_mode=='financial' or (current_mode in real_data_list):
                #re-get true component (true covariance was now returned)
                true_pc1=get_pc1(covariance=covariance,angles=angles,weights=weights,current_mode=current_mode,X=X, dim=dim)
                
            X = np.clip(X, -1000, 1000)

            #Get L and S from RPCA
            rpca = R_pca(X)
            L, S = rpca.fit()

            rpca_dynamic = R_pca(D=X,lmbda=rpca.lmbda*2)
            target_sparsity= 0.5/dim
            L_dynamic,S_dynamic = rpca_dynamic.fit()

            #DO ALL PCAS
            PC={}

            #Do ROBPCA from ROBPY (has its own outlier plotting function)
            #PC["robpca"]=get_ROBPCA(X=X, spher=False)
            #PC["spher"]=get_ROBPCA(X=X, spher=True) 

            PC["robpca"]=np.transpose((ROBPCA(n_components=n_comp).fit(X)).components_)
            PC["spher"]=np.transpose((PCALocantore(n_components=n_comp).fit(X)).components_)
            #score_distances, orthogonal_distances, score_cutoff, od_cutoff = pca.plot_outlier_map(X_robpca, return_distances=True)

            #Fit Standard PCA (python library scikit-learn), custom, and rpca on L
            PC["lp"]= (LpNormPCA(n_components=n_comp, p=p_value).fit(X)).components_
            PC["standard"] = (PCA(n_components=n_comp).fit(X)).components_
            PC["rpca"] = (PCA(n_components=n_comp).fit(L)).components_
            PC["tyler"]= (TylerPCA(n_components=n_comp).fit(X)).components_
            PC["dynamic"]=(PCA(n_components=n_comp).fit(L_dynamic)).components_
            PC["bay"]=(BayesianPCA(n_components=n_comp).fit(X)).components_

            #Calculate error on PC1)
            #Having spectral as a last argument takes the smallest error in angle between PC1 and PC2
            #i.e. it forgives switching them around
            #for key in errors:
            for key in errors:
                #print(key,PC[key].shape,true_pc1.shape)
                errors[key].append(calc_angle_error(PC[key][0], true_pc1, 'spectral'))
                if generous_flag:
                    errors_generous[key].append(calc_angle_error_generous(PC[key][0], true_pc1,true_pc2,true_pc3))

            #Plot alignment 
            if plot: 
                plot_alignment(X=X,S=S,current_mode=current_mode,angles=angles,weights=weights,
                                covariance=covariance,plot_flags=plot_flags,PC=PC,a=a, 
                                colors=colors, labels=labels,title_str=title_str)


    #Plot spectral measure if applicable
    if current_mode=='spectral' and dim==2:
        plt.figure(figsize=(10, 6))
        plot_spectral_measure(plt.gca(), angles, weights, true_pc1)

    #Plot error in angle
    if plot_error:
        fig = plt.figure(figsize=(10, 6))
        for key in errors: 
            plt.plot(alpha_values, errors[key], color=colors[key], linewidth=2, label=f'{labels[key]} Error')
            if generous_flag: plt.plot(alpha_values, errors_generous[key], color=colors[key], linestyle='--')
        
        plt.xlabel('Alpha')
        plt.ylabel('Angle Error (Degrees)')
        fig.suptitle('Error in Principal Direction Estimation')
        plt.title(title_str, loc="left")
        if current_mode!="spectral" and current_mode not in real_data_list:
            plt.title(f"cov: {title_cov}",loc="right")
        plt.grid(True, linestyle='--')
        plt.legend()

    else:
        print_error_table(errors=errors,labels=labels)

    plt.show()

def run_experiment_general(args):
    if args.mode not in image_list:
        run_experiment(args)
    else: run_image_experiment(args)


if __name__ == "__main__":
    
    args = get_args()
    run_experiment_general(args)

