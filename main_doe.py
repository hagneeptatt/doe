import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MaxNLocator
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy import stats
import seaborn as sns
import warnings
import os
from scipy.optimize import differential_evolution

warnings.filterwarnings('ignore')

# =================================================================
# PUBLICATION STYLE SETUP
# =================================================================


def set_publication_style():
    """Applies Matplotlib style and font settings for publication quality."""
    # Use a clean, publication-friendly style
    plt.style.use('seaborn-v0_8-paper')
    
    # Update RC parameters for clearer fonts/sizes
    mpl.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 12,
        'figure.titlesize': 16,
        'figure.dpi': 300, # High resolution for raster exports
    })
    
    # Ensure proper vector output for scaling
    mpl.rcParams['svg.fonttype'] = 'none'


# =================================================================
# DATASET ANALYSIS
# =================================================================
def analyze_dataset(filename):
    """
    General function to analyse any DOE dataset with comprehensive validation.
    """
    print(f"\n{'='*80}")
    print(f"ANALYSING: {filename}")
    print(f"{'='*80}")

    # -----------------------------------------------------------------
    # Load dataset
    # -----------------------------------------------------------------
    if not os.path.exists(filename):
        print(f"ERROR: {filename} not found!")
        return None, None, None, None, None, None, None

    df = pd.read_csv(filename)
    print("Data loaded successfully!")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # -----------------------------------------------------------------
    # Define factors and response
    # -----------------------------------------------------------------
    # Assuming the first column is an index/Run number and the last is the response.
    factor_names = df.columns[1:-1]
    response_name = df.columns[-1]

    print(f"Factor columns: {list(factor_names)}")
    print(f"Response column: {response_name}")

    # Validate existence of columns
    missing_cols = [col for col in list(factor_names) + [response_name] if col not in df.columns]
    if missing_cols:
        print(f"ERROR: Missing columns: {missing_cols}")
        return None, None, None, None, None, None, None

    X = df[factor_names].values
    y = df[response_name].values

    print(f"\nFactors: {list(factor_names)}")
    print(f"Response: {response_name}")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

    # -----------------------------------------------------------------
    # Validate data
    # -----------------------------------------------------------------
    validate_data(df, factor_names, response_name)

    # -----------------------------------------------------------------
    # Fit polynomial regression model
    # -----------------------------------------------------------------
    poly = PolynomialFeatures(degree=2, include_bias=True)
    X_poly = poly.fit_transform(X)
    print(f"\nPolynomial features created: {X_poly.shape[1]}")

    model = LinearRegression()
    model.fit(X_poly, y)

    # -----------------------------------------------------------------
    # Validate model fit
    # -----------------------------------------------------------------
    validate_model(X_poly, y, model, factor_names, response_name)

    return df, X, y, poly, model, factor_names, response_name


# =================================================================
# DATA VALIDATION
# =================================================================
def validate_data(df, factor_names, response_name):
    print(f"\n{'-'*40}")
    print("DATA VALIDATION")
    print(f"{'-'*40}")

    # Missing values
    missing_data = df[list(factor_names) + [response_name]].isnull().sum()
    if missing_data.sum() > 0:
        print("Missing values found:")
        for col, count in missing_data.items():
            if count > 0:
                print(f"   {col}: {count} missing values")
    else:
        print("No missing values found")

    # Duplicate factor combinations
    duplicates = df.duplicated(subset=factor_names).sum()
    print(f"{duplicates} duplicate factor combinations found" if duplicates > 0 else "No duplicate combinations")

    # Summary statistics
    print("\nDescriptive Statistics:")
    print(df[list(factor_names) + [response_name]].describe())

    # Outlier detection
    print("\nOutlier Detection (IQR method):")
    for col in list(factor_names) + [response_name]:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        print(f"   {col}: {outliers} potential outliers" if outliers > 0 else f"   {col}: No outliers detected")

    # Correlation matrix
    corr_matrix = df[list(factor_names) + [response_name]].corr()
    print("\nCorrelation Matrix:")
    print(corr_matrix.round(3))

    # Plot heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, square=True, fmt='.2f')
    plt.title('Factor Correlation Matrix')
    plt.tight_layout()
    plt.savefig(f"correlation_matrix_{response_name}.svg", format="svg", bbox_inches="tight")
    plt.close()


# =================================================================
# MODEL VALIDATION
# =================================================================
def validate_model(X_poly, y, model, factor_names, response_name):
    print(f"\n{'-'*40}")
    print("MODEL VALIDATION")
    print(f"{'-'*40}")

    y_pred = model.predict(X_poly)
    r2 = model.score(X_poly, y)
    mse = mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mse)

    print(f"R²: {r2:.4f}")
    print(f"MSE: {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE: {mae:.6f}")

    n, p = len(y), X_poly.shape[1] - 1
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    print(f"Adjusted R²: {adj_r2:.4f}")

    residuals = y - y_pred

    # Residual normality
    shapiro_stat, shapiro_p = stats.shapiro(residuals)
    print("\nResidual Analysis:")
    print(f"Shapiro-Wilk test p = {shapiro_p:.4f}")
    print("Residuals appear normally distributed" if shapiro_p > 0.05 else "Residuals may not be normal")

    # Autocorrelation
    dw_stat = np.sum(np.diff(residuals)**2) / np.sum(residuals**2)
    print(f"Durbin-Watson statistic: {dw_stat:.4f}")
    print("No strong evidence of autocorrelation" if 1.5 < dw_stat < 2.5 else "Possible autocorrelation detected")

    create_diagnostic_plots(y, y_pred, residuals, response_name)


# =================================================================
# DIAGNOSTIC PLOTS
# =================================================================
def create_diagnostic_plots(y_actual, y_pred, residuals, response_name):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Model Diagnostic Plots', fontsize=16)

    # 1. Actual vs Predicted
    axes[0, 0].scatter(y_actual, y_pred, alpha=0.7)
    axes[0, 0].plot([y_actual.min(), y_actual.max()],
                    [y_actual.min(), y_actual.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel(f'Actual {response_name}')
    axes[0, 0].set_ylabel(f'Predicted {response_name}')
    axes[0, 0].set_title('Actual vs Predicted')
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Residuals vs Predicted
    axes[0, 1].scatter(y_pred, residuals, alpha=0.7)
    axes[0, 1].axhline(y=0, color='r', linestyle='--')
    axes[0, 1].set_xlabel(f'Predicted {response_name}')
    axes[0, 1].set_ylabel('Residuals')
    axes[0, 1].set_title('Residuals vs Predicted')
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Histogram of residuals
    axes[1, 0].hist(residuals, bins=10, alpha=0.7, density=True)
    axes[1, 0].set_xlabel('Residuals')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_title('Distribution of Residuals')
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Q-Q plot
    stats.probplot(residuals, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('Q-Q Plot (Normal)')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"diagnostic_plots_{response_name}.svg", format="svg", bbox_inches="tight")
    plt.close()


# =================================================================
# RESPONSE SURFACE PLOTTING
# =================================================================
def plot_surface(X, poly, model, factor_names, response_name, 
                 factor1_idx, factor2_idx, resolution=30, 
                 optimal_factors_dict=None, optimal_response=None):
    
    print(f"Creating surface plot: {factor_names[factor1_idx]} vs {factor_names[factor2_idx]}")

    f1_min, f1_max = X[:, factor1_idx].min(), X[:, factor1_idx].max()
    f2_min, f2_max = X[:, factor2_idx].min(), X[:, factor2_idx].max()

    f1_range = np.linspace(f1_min, f1_max, resolution)
    f2_range = np.linspace(f2_min, f2_max, resolution)
    F1, F2 = np.meshgrid(f1_range, f2_range)

    grid_size = resolution * resolution
    prediction_grid = np.zeros((grid_size, X.shape[1]))

    prediction_grid[:, factor1_idx] = F1.ravel()
    prediction_grid[:, factor2_idx] = F2.ravel()

    fixed_values = []
    # Set other factors to their mean values
    for i in range(X.shape[1]):
        if i not in [factor1_idx, factor2_idx]:
            mean_val = X[:, i].mean()
            prediction_grid[:, i] = mean_val
            fixed_values.append(f"{factor_names[i]}={mean_val:.3f}")

    grid_poly = poly.transform(prediction_grid)
    predictions = model.predict(grid_poly)
    Z = predictions.reshape(resolution, resolution)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d', computed_zorder=False)
    # Plot the main response surface
    surf = ax.plot_surface(F1, F2, Z, cmap='viridis', alpha=0.8)

    # =================================================
    # PLOTTING OPTIMAL POINT
    # =================================================
    if optimal_factors_dict is not None and optimal_response is not None:
        # Extract the X, Y, Z coordinates for the optimal point
        opt_f1 = optimal_factors_dict[factor_names[factor1_idx]]
        opt_f2 = optimal_factors_dict[factor_names[factor2_idx]]
        
        # NOTE: The optimal Z (response) is the predicted response.
        opt_resp = optimal_response
        
        # Plot the optimal point as a prominent scatter marker (red circle)
        # zorder=100 ensures the point is plotted on top of the surface
        ax.scatter([opt_f1], [opt_f2], [opt_resp], 
             color='red', marker='o', s=50, 
             edgecolor='black', # Good for defining the shape
             alpha=1.0,         # Explicitly set alpha to 1.0 for full opacity
             label='Optimal Point')
        
    # =================================================
    
    ax.set_xlabel(factor_names[factor1_idx])
    ax.set_ylabel(factor_names[factor2_idx])
    ax.set_zlabel(response_name)

    title = f"Response Surface: {factor_names[factor1_idx]} vs {factor_names[factor2_idx]}"
    if fixed_values:
        title += f"\n(Fixed: {', '.join(fixed_values)})"
    if optimal_factors_dict is not None:
        # Ensure the plot title reflects the optimisation
        title += f"\n(Optimal {response_name}: {optimal_response:.4f})"
        ax.legend() # Show the label for the optimal point
        
    ax.set_title(title)

    fig.colorbar(surf, shrink=0.6)
    plt.tight_layout()
    plt.savefig(f"surface_{factor_names[factor1_idx]}_vs_{factor_names[factor2_idx]}_optimised.svg",
                format="svg", bbox_inches="tight")
    plt.close()


def plot_all_combinations(X, poly, model, factor_names, response_name, 
                          optimal_factors_dict=None, optimal_response=None):
    n_factors = len(factor_names)
    n_combinations = n_factors * (n_factors - 1) // 2

    print(f"\n{'-'*40}")
    print(f"CREATING {n_combinations} RESPONSE SURFACE PLOTS")
    print(f"{'-'*40}")

    plot_count = 1
    for i in range(n_factors):
        for j in range(i + 1, n_factors):
            print(f"\nPlot {plot_count}/{n_combinations}:")
            plot_surface(X, poly, model, factor_names, response_name, i, j,
                         optimal_factors_dict=optimal_factors_dict,
                         optimal_response=optimal_response) # Pass optimal data
            plot_count += 1




# =================================================================
# PREDICTING OPTIMAL FACTORS FOR DESIRED RESPONSE
# =================================================================
def prediction_from_model(target, poly, model, X, factor_names):
    """
    Predict factor combination for desired response variable using global optimisation.
    """
    
    # You don't need to define factors array, as you are passing the objective function 
    # as an argument in the optimisation function - the optimisation function is 
    # therefor a 'Higher-Order Function' as it contains a function as an argument. 
    # The optimisation function returns an array of factor values which is iteratively 
    # passed through the objective function.

    def objective(factors):
        # factors is a 1D array from the differential_evolution algorithm
        factors_2d = factors.reshape(1,-1) 
        factors_poly = poly.transform(factors_2d) 
        predicted = model.predict(factors_poly)[0]
        # return objective function which is the difference of squares between the target and predicted response 
        return (predicted - target)**2
    

    # Define bounds (factor min/max from original data)
    bounds = [(X[:, i].min(), X[:, i].max()) for i in range(X.shape[1])]

    # Global optimisation
    result = differential_evolution(objective, bounds)

    # Verify prediction with optimal factors
    optimal_factors_2d = result.x.reshape(1, -1)
    optimal_factors_poly = poly.transform(optimal_factors_2d)
    predicted_response = model.predict(optimal_factors_poly)[0]
    
    # Get the target value used in the calculation for the result dictionary
    target_value = target 

    result_dict = {
        'optimal_factors': dict(zip(factor_names, result.x)),
        'predicted_response': predicted_response,
        'target_response': target_value,
        'error': result.fun,
        'success': result.success
    }

    return result_dict




# =================================================================
# MAIN EXECUTION
# =================================================================
print("=" * 80)
print("STARTING COMPREHENSIVE DOE ANALYSIS")
print("=" * 80)

print("\n" + "=" * 80)
print("LOADING AND ANALYSING DATA")
print("=" * 80)

# NOTE: The file 'bioprint_doe_media_results_large_valve.csv' must exist in the current directory.
df, X, y, poly, model, factor_names, response_name = analyze_dataset(
    filename="bioprint_doe_media_results_large_valve.csv"
)

if model is not None:
    print("\n" + "=" * 80)
    print("CREATING INITIAL RESPONSE SURFACE PLOTS FOR DOE DATA")
    print("=" * 80)
    # Create initial plots (without optimal point)
    plot_all_combinations(X, poly, model, factor_names, response_name)


    # =================================================================
    # OPTIMAL FACTOR PREDICTION SECTION
    # =================================================================
    print("\n" + "=" * 80)
    print("OPTIMISING FACTOR COMBINATION FOR TARGET RESPONSE")
    print("=" * 80)

    try:
        # Ask user for a target response value
        target_value = float(input(f"\nEnter desired target value for {response_name}: "))

        # Run global optimisation
        optimal_result = prediction_from_model(
            target=target_value,
            poly=poly,
            model=model,
            X=X,
            factor_names=factor_names
        )

        # Store the factor dictionary and predicted response for plotting
        opt_factors = optimal_result['optimal_factors']
        opt_response = optimal_result['predicted_response']

        # Display results
        print("\nOptimal Factor Combination Found:")
        for factor, value in opt_factors.items():
            print(f"   {factor}: {value:.4f}")

        print(f"\nPredicted {response_name}: {opt_response:.4f}")
        print(f"Target {response_name}: {optimal_result['target_response']:.4f}")
        print(f"Squared Error: {optimal_result['error']:.6f}")
        print(f"Optimisation Success: {optimal_result['success']}")

        
        # RERUN PLOTTING WITH OPTIMAL VALUES
        print("\n" + "=" * 80)
        print("RECREATING RESPONSE SURFACE PLOTS WITH OPTIMAL POINT 🎯")
        print("=" * 80)
        # Pass the optimization results to the plotting function
        plot_all_combinations(X, poly, model, factor_names, response_name, 
                              optimal_factors_dict=opt_factors, 
                              optimal_response=opt_response)

    except Exception as e:
        print(f"\nERROR during optimisation: {e}")