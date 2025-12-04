import argparse


custom_dim_list=["financial","real_stock1","real_stock2","subgaussian","spectral","pareto","student_t"] #modes that support custom dimensions
real_data_list=["real_stock1","image1","real_stock2"] #modes with real data, alpha cannot be specified
image_list=["image1"]


def get_args():
    parser = argparse.ArgumentParser(description="Run Robust PCA Experiments with various distributions and parameters.")

    # General Settings
    parser.add_argument('--n_samples', type=int, default=900, 
                        help='Number of samples to generate (default: 300)')

    parser.add_argument('--n_comp', type=int, default=2, 
                        help='Number of components computed by each PCA method (default: 2)')
    
    parser.add_argument('--seed', type=int, default=649, 
                        help='Random seed for reproducibility (default: 42)')
    
    parser.add_argument('--plot', type=bool, default=False, 
                        help='Shows the PCA fitting plot for each alpha (default: False, only metric plot is returned)')

    parser.add_argument('--mode', type=str, default='subgaussian',
                        choices=['subgaussian', 'student_t', 'spectral', 'pareto','financial','real_stock1','real_stock2','image1'],
                        help="Distribution mode for data generation (default: 'subgaussian')")

    parser.add_argument('--random', type=bool, default=False,
                        help="Available on all modes except spectral, overrides the covariance matrix if given and generates it randomly. (default: False)")


    parser.add_argument('--alphas', type=float, nargs='+', default=[1.1, 1.3, 1.5, 1.8, 2.0],
                        help='List of alpha/df values to test (e.g. --alphas 1.1 1.5 2.0)')
    parser.add_argument('--p_value', type=float, nargs=1, default=[0.7],
                        help='Choice of p (<2) for the Lp Norm for FLOM PCA (default: 0.7)')  
    parser.add_argument('--dimension', type=int, default=2,
                        help='Number of dimensions (i.e. number of companies for financial data). (default: 2)')              

    # Spectral Mode
    parser.add_argument('--angles', type=float, nargs='+', default=[45, 135],
                        help="Angles in degrees for 'spectral' mode (default: 45 135). Only available for 2 dimensions, randomly generated otherwise.")
    
    parser.add_argument('--weights', type=float, nargs='+', default=[1.0, 0.3],
                        help="Weights corresponding to angles for 'spectral' mode (default: 1.0 0.3). Only available for 2 dimensions, randomly generated otherwise.")

    # Covariance matric
    parser.add_argument('--covariance', type=float, nargs=4, default=[1,0.4,0.4,1],
                        help="Covariance matrix [[a,b],[c,d]], passed as [a,b,c,d]. (default: [1,0.4,0.4,1]). Only available for 2 dimensions, randomly generated otherwise.")
    parser.add_argument('--rho', type=float, nargs=1, default=0.4,
                        help='Correlation value for financial data (default: 0.4)')  


    # Image mode options
    parser.add_argument('--contamination', type=float, default=0.4,
                        help="Contamination percentage for image mode (default: 0.4)")  
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    # Test the parser
    args = get_args()
    print(f"Parsed Arguments: {args}")