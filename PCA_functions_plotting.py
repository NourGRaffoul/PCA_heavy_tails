
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from sklearn.decomposition import PCA
from scipy.stats import levy_stable
from robpy.pca import ROBPCA
from robpy.pca.spca import PCALocantore
from robpy.preprocessing import DataCleaner, RobustPowerTransformer, RobustScaler
from sklearn.linear_model import HuberRegressor

from PCA_functions_components import *
from parser import custom_dim_list, real_data_list


'''
!!!!!!!!!!!!!!!!!!!!!!!!!!
CATEGORY3: PLOTTING
!!!!!!!!!!!!!!!!!!!!!!!!!!
'''
def print_error_table(errors,labels):
    #print("="*50,"\n",f"{'METHOD':<30} | {'ERROR (Deg)':<10}","\n","-" * 43)
    print("="*50)
    print(f"{'METHOD':<30} | {'ERROR (Deg)':<10}")
    print("-" * 43)

    for key in errors:
        errors[key] = errors[key][0]

    sorted_res = sorted(errors.items(), key=lambda x: x[1] if isinstance(x[1], float) else 999)

    for name, score in sorted_res:
        if isinstance(score, float): print(f"{labels[name]:<30} | {score:.2f}°")
        else: print(f"{labels[name]:<30} | {score[0]}")
    print("="*50)


def get_plot_parameters(X,S):
    plot_radius = np.percentile(np.abs(X), 90) * 1.5
    robust_scale = np.median(np.linalg.norm(X, axis=1))

    s_magnitudes = np.linalg.norm(S, axis=1)
    threshold = 0.5 * robust_scale 
    return plot_radius,robust_scale,s_magnitudes,threshold

def plot_vectors(ax, components, variance, color, style, label_prefix, scale=2.5):
    #Plot both PC1 and PC2 scaled by their standard deviation.
    stds = np.sqrt(variance)
    
    # Plot PC1
    v1 = components[0] * stds[0] * scale
    ax.plot([-v1[0], v1[0]], [-v1[1], v1[1]], 
            color=color, linestyle=style, linewidth=3, label=f'{label_prefix} PC1')
    
    # Plot PC2 
    if len(stds) > 1:
        v2 = components[1] * stds[1] * scale
        ax.plot([-v2[0], v2[0]], [-v2[1], v2[1]], 
                color=color, linestyle=style, linewidth=1.5, alpha=0.7) 

def plot_vector_fixed(ax, components, color, style, label_prefix, ratio, length=1.0):
    #Plots PC1 and PC2 with a fixed visual length to allow angle comparison.
    # Plot PC1
    v1 = components[0]
    v1 = v1 / np.linalg.norm(v1) * length
    ax.plot([-v1[0], v1[0]], [-v1[1], v1[1]], 
            color=color, linestyle=style, linewidth=3, label=f'{label_prefix}')
    
    # Plot PC2 (Perpendicular)
    if len(components) > 1:
        v2 = components[1]
        v2 = v2 / np.linalg.norm(v2) * length * ratio # Make secondary shorter
        ax.plot([-v2[0], v2[0]], [-v2[1], v2[1]], 
                color=color, linestyle=style, linewidth=1.5, alpha=0.7)

def plot_top_directions(ax,angles,weights,plot_radius):
    sorted_indices = np.argsort(weights)[::-1]
    sorted_indices = np.delete(sorted_indices,1)
    top_indices = sorted_indices[:2]
    #print(top_indices)
                    
    for rank, idx in enumerate(top_indices):
        angle = angles[idx]
        weight = weights[idx]
                        
        # Calculate Vector
        rad = np.radians(angle)
        v_true = np.array([np.cos(rad), np.sin(rad)]) * plot_radius
                        
        ls = '-' if rank == 0 else '-.'
        lw = 4 if rank == 0 else 2.5
        lbl = f'True #{rank+1} ({angle}°, w={weight})'
                        
        ax.plot([-v_true[0], v_true[0]], 
                [-v_true[1], v_true[1]], 
                color='black', linestyle=ls, linewidth=lw, label=lbl)

def plot_spectral_measure(ax, angles, weights, true_pc1=None):
    # Draw Unit Circle
    circle = plt.Circle((0, 0), 1, color='k', fill=False, linestyle='--', alpha=0.5)
    ax.add_patch(circle)

    rads = np.radians(angles)
    spec_x = np.cos(rads)
    spec_y = np.sin(rads)
 
    if np.max(weights) > 0:
        scale_factor = 500 / np.max(weights)
    else:
        scale_factor = 1.0
        
    sizes = np.array(weights) * scale_factor
    ax.scatter(spec_x, spec_y, s=sizes, c='black', label='Weights')
    
    #for i, ang in enumerate(angles):
    #    ax.text(spec_x[i]*1.1, spec_y[i]*1.1, f"{ang}°", 
    #               ha='center', va='center', fontsize=8, fontweight='bold')

    if true_pc1 is not None:
        x=true_pc1[0]
        y=true_pc1[1]
        ax.plot([-x, x], [-y, y], 'r-', lw=2, label='True PC1')
    
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.set_title(f'Spectral Measure (Unit Circle)')
    ax.grid(True, linestyle=':')
    ax.legend()


def plot_alignment(X,S,current_mode,angles,weights,covariance,plot_flags,PC,a,colors,labels,title_str):
    fig = plt.figure(figsize=(8, 8))
    plot_radius,robust_scale,s_magnitudes,threshold = get_plot_parameters(X,S)
    outlier_mask = s_magnitudes > threshold
    comp_ratio=0.5

    if current_mode!='spectral': 
            true_vals, true_vecs = get_true_components(np.array(covariance))
        
    plt.scatter(X[outlier_mask, 0], X[outlier_mask, 1], 
            c='grey', s=15, alpha=0.6, label='Outliers (S)')
    plt.scatter(X[~outlier_mask, 0], X[~outlier_mask, 1], 
            c='blue', s=15, alpha=0.3, label='Inliers (L)')

    if current_mode=='spectral': 
        if a<1.4: zoom_factor=17
        else: zoom_factor = 3
        plot_radius=plot_radius*zoom_factor

    # Plot true PCs
    if current_mode == 'spectral': plot_top_directions(plt.gca(),angles,weights,plot_radius)
    else: plot_vector_fixed(plt.gca(), true_vecs.T, 
            'black', '-', 'True', ratio=comp_ratio,length=plot_radius)

    # Plot PCAs
    for key in labels:
        if plot_flags[key]:
            plot_vector_fixed(plt.gca(), PC[key], colors[key], '--', labels[key], ratio=comp_ratio, length=plot_radius)

    fig.suptitle(f'Alpha = {a}')
    plt.title(title_str, loc="left")
    if current_mode!="spectral" and current_mode not in real_data_list:
        title_cov = np.round(covariance[:2, :2], 3)
        plt.title(f"cov: {title_cov}",loc="right")
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.axis('equal') 

    

def plot_faces(title, images, image_shape, col_index, n_row=5, axes=None):
    if axes is None: raise ValueError("Must provide the axes object for plotting.")

    for i, comp in enumerate(images[:n_row]):
        ax = axes[i, col_index] 
        ax.imshow(comp.reshape(image_shape), cmap=plt.cm.gray, interpolation='nearest')
        ax.set_xticks(())
        ax.set_yticks(())
        
        if i == 0: ax.set_title(title, size=12)