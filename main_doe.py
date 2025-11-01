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


"""Applies Matplotlib style and font settings for publication quality."""

# Start with a clean, minimal base style
plt.style.use('seaborn-v0_8-white')

mpl.rcParams.update({
    # Font sizes for clarity in print
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 15,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'legend.fontsize': 13,
    'figure.titlesize': 18,

    # Line and marker aesthetics
    'lines.linewidth': 2,
    'lines.markersize': 6,

    # Figure quality and layout
    'figure.dpi': 300,
    'savefig.dpi': 600,
    'figure.autolayout': True,

    # Font embedding (so text stays editable in vector files)
    'svg.fonttype': 'none',
    'pdf.fonttype': 42,

    # Axis and grid appearance
    'axes.linewidth': 1.2,
    'grid.linewidth': 0.8,
    'grid.alpha': 0.3,

    # # Backgrounds
    # 'axes.facecolor': 'none',     # Transparent axes background
    # 'figure.facecolor': 'none',   # Transparent figure background
})

# Optional: use a neutral sans-serif font
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

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

    if not os.path.exists(filename):
        print(f"ERROR: {filename} not found!")
        return None, None, None, None, None, None, None

    df = pd.read_csv(filename)
    print("Data loaded successfully!")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # Define factors and response
    factor_names = df.columns[1:-1]
    response_name = df.columns[-1]

    print(f"Factor columns: {list(factor_names)}")
    print(f"Response column: {response_name}")

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

    validate_data(df, factor_names, response_name)

    # Fit polynomial regression model
    poly = PolynomialFeatures(degree=2, include_bias=True)
    X_poly = poly.fit_transform(X)
    print(f"\nPolynomial features created: {X_poly.shape[1]}")

    model = LinearRegression()
    model.fit(X_poly, y)

    validate_model(X_poly, y, model, factor_names, response_name)

    return df, X, y, poly, model, factor_names, response_name


# =================================================================
# DATA VALIDATION
# =================================================================
def validate_data(df, factor_names, response_name):
    print(f"\n{'-'*40}")
    print("DATA VALIDATION")
    print(f"{'-'*40}")

    missing_data = df[list(factor_names) + [response_name]].isnull().sum()
    if missing_data.sum() > 0:
        print("Missing values found:")
        for col, count in missing_data.items():
            if count > 0:
                print(f"   {col}: {count} missing values")
    else:
        print("No missing values found")

    duplicates = df.duplicated(subset=factor_names).sum()
    print(f"{duplicates} duplicate factor combinations found" if duplicates > 0 else "No duplicate combinations")

    print("\nDescriptive Statistics:")
    print(df[list(factor_names) + [response_name]].describe())

    print("\nOutlier Detection (IQR method):")
    for col in list(factor_names) + [response_name]:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        print(f"   {col}: {outliers} potential outliers" if outliers > 0 else f"   {col}: No outliers detected")

    corr_matrix = df[list(factor_names) + [response_name]].corr()
    print("\nCorrelation Matrix:")
    print(corr_matrix.round(3))

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

    shapiro_stat, shapiro_p = stats.shapiro(residuals)
    print("\nResidual Analysis:")
    print(f"Shapiro-Wilk test p = {shapiro_p:.4f}")
    print("Residuals appear normally distributed" if shapiro_p > 0.05 else "Residuals may not be normal")

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

    # Actual vs Predicted
    axes[0, 0].scatter(y_actual, y_pred, alpha=0.7)
    axes[0, 0].plot([y_actual.min(), y_actual.max()],
                    [y_actual.min(), y_actual.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel(f'Actual {response_name}')
    axes[0, 0].set_ylabel(f'Predicted {response_name}')
    axes[0, 0].set_title('Actual vs Predicted')
    axes[0, 0].grid(True, alpha=0.3)

    # Residuals vs Predicted
    axes[0, 1].scatter(y_pred, residuals, alpha=0.7)
    axes[0, 1].axhline(y=0, color='r', linestyle='--')
    axes[0, 1].set_xlabel(f'Predicted {response_name}')
    axes[0, 1].set_ylabel('Residuals')
    axes[0, 1].set_title('Residuals vs Predicted')
    axes[0, 1].grid(True, alpha=0.3)

    # Histogram of residuals
    axes[1, 0].hist(residuals, bins=10, alpha=0.7, density=True)
    axes[1, 0].set_xlabel('Residuals')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_title('Distribution of Residuals')
    axes[1, 0].grid(True, alpha=0.3)

    # Q-Q plot
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
                 optimal_factors_dict=None, optimal_response=None,
                 vmin=None, vmax=None):
    """
    Plot 3D response surface with optional optimal point and consistent color scale.
    
    Parameters:
    -----------
    vmin, vmax : float, optional
        Color scale limits for consistent coloring across plots
    """
    
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
    for i in range(X.shape[1]):
        if i not in [factor1_idx, factor2_idx]:
            mean_val = X[:, i].mean()
            prediction_grid[:, i] = mean_val
            fixed_values.append(f"{factor_names[i]}={mean_val:.3f}")

    grid_poly = poly.transform(prediction_grid)
    predictions = model.predict(grid_poly)
    Z = predictions.reshape(resolution, resolution)

    # Use provided vmin/vmax or calculate from data
    color_min = vmin if vmin is not None else Z.min()
    color_max = vmax if vmax is not None else Z.max()

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot surface with fixed color scale
    surf = ax.plot_surface(F1, F2, Z, cmap='viridis', alpha=0.8,
                          vmin=color_min, vmax=color_max)

    # Plot contour with matching color scale and specified levels
    contour_levels = np.linspace(color_min, color_max, 15)
    contour = ax.contour(F1, F2, Z, levels=contour_levels,
                        zdir='z', offset=Z.min(), 
                        cmap='viridis', alpha=0.6,
                        vmin=color_min, vmax=color_max)

    # ⭐ FIX: Set Z-axis limits to make contour flush with bottom
    ax.set_zlim(Z.min(), Z.max())

    # Plot optimal point if provided
    if optimal_factors_dict is not None and optimal_response is not None:
        opt_f1 = optimal_factors_dict[factor_names[factor1_idx]]
        opt_f2 = optimal_factors_dict[factor_names[factor2_idx]]
        opt_z = optimal_response
        
        # # On surface (at actual height)
        # ax.scatter([opt_f1], [opt_f2], [opt_z],
        #            color='red', marker='o', s=150, 
        #            edgecolor='black', linewidths=2,
        #            alpha=1.0, zorder=1000,
        #            label=f'Optimal: {opt_z:.2f}')
        
        # On contour plane (projected)
        ax.scatter([opt_f1], [opt_f2], [Z.min()],
                   color='red', marker='x', s=150, 
                   edgecolor='black', linewidths=2,
                   alpha=1.0, zorder=1000,
                   label=f'Optimal: {opt_z:.2f}')
        
        # # Vertical line connecting them
        # ax.plot([opt_f1, opt_f1], 
        #         [opt_f2, opt_f2], 
        #         [Z.min(), opt_z],
        #         color='red', linestyle='--', linewidth=2, alpha=0.7)
    
    ax.set_xlabel(factor_names[factor1_idx])
    ax.set_ylabel(factor_names[factor2_idx])
    ax.set_zlabel(response_name)

    title = f"Response Surface: {factor_names[factor1_idx]} vs {factor_names[factor2_idx]}"
    if fixed_values:
        title += f"\n(Fixed: {', '.join(fixed_values)})"
    if optimal_factors_dict is not None:
        title += f"\n(Target: {optimal_response:.4f})"
        ax.legend()
        
    ax.set_title(title)

    # Colorbar with fixed scale
    fig.colorbar(surf, shrink=0.6, label=response_name)
    plt.tight_layout()
    
    # Create filename
    filename = f"surface_{factor_names[factor1_idx]}_vs_{factor_names[factor2_idx]}"
    if optimal_factors_dict is not None:
        filename += "_optimised"
    plt.savefig(f"{filename}.svg", format="svg", bbox_inches="tight")
    plt.close()


def calculate_global_color_range(X, poly, model, factor_names, resolution=20):
    """
    Calculate global min/max response values across all factor combinations.
    This ensures consistent color scales across all surface plots.
    """
    print("\nCalculating global color scale range...")
    n_factors = len(factor_names)
    all_predictions = []
    
    for i in range(n_factors):
        for j in range(i+1, n_factors):
            # Create prediction grid for this pair
            f1_min, f1_max = X[:, i].min(), X[:, i].max()
            f2_min, f2_max = X[:, j].min(), X[:, j].max()
            
            f1_range = np.linspace(f1_min, f1_max, resolution)
            f2_range = np.linspace(f2_min, f2_max, resolution)
            F1, F2 = np.meshgrid(f1_range, f2_range)
            
            grid_size = resolution * resolution
            prediction_grid = np.zeros((grid_size, X.shape[1]))
            prediction_grid[:, i] = F1.ravel()
            prediction_grid[:, j] = F2.ravel()
            
            # Set other factors to mean
            for k in range(X.shape[1]):
                if k not in [i, j]:
                    prediction_grid[:, k] = X[:, k].mean()
            
            # Make predictions
            grid_poly = poly.transform(prediction_grid)
            predictions = model.predict(grid_poly)
            all_predictions.extend(predictions)
    
    global_vmin = min(all_predictions)
    global_vmax = max(all_predictions)
    
    print(f"Global response range: {global_vmin:.4f} to {global_vmax:.4f}")
    
    return global_vmin, global_vmax


def plot_all_combinations(X, poly, model, factor_names, response_name, 
                          optimal_factors_dict=None, optimal_response=None,
                          vmin=None, vmax=None):
    """
    Plot all 2-factor combinations with consistent color scale.
    """
    n_factors = len(factor_names)
    n_combinations = n_factors * (n_factors - 1) // 2

    print(f"\n{'-'*40}")
    print(f"CREATING {n_combinations} RESPONSE SURFACE PLOTS")
    print(f"{'-'*40}")
    
    # Calculate global color range if not provided
    if vmin is None or vmax is None:
        vmin, vmax = calculate_global_color_range(X, poly, model, factor_names)

    plot_count = 1
    for i in range(n_factors):
        for j in range(i + 1, n_factors):
            print(f"\nPlot {plot_count}/{n_combinations}:")
            plot_surface(X, poly, model, factor_names, response_name, i, j,
                         optimal_factors_dict=optimal_factors_dict,
                         optimal_response=optimal_response,
                         vmin=vmin,
                         vmax=vmax)
            plot_count += 1


# =================================================================
# PREDICTING OPTIMAL FACTORS FOR DESIRED RESPONSE
# =================================================================
def prediction_from_model(target, poly, model, X, factor_names):
    """
    Predict factor combination for desired response variable using global optimisation.
    """
    
    def objective(factors):
        """
        Objective function to minimize the squared difference between 
        predicted and target response.
        """
        factors_2d = factors.reshape(1, -1) 
        factors_poly = poly.transform(factors_2d) 
        predicted = model.predict(factors_poly)[0]
        return (predicted - target)**2

    # Define bounds from original data range
    bounds = [(X[:, i].min(), X[:, i].max()) for i in range(X.shape[1])]

    # Run global optimisation
    result = differential_evolution(objective, bounds)

    # Verify prediction with optimal factors
    optimal_factors_2d = result.x.reshape(1, -1)
    optimal_factors_poly = poly.transform(optimal_factors_2d)
    predicted_response = model.predict(optimal_factors_poly)[0]

    result_dict = {
        'optimal_factors': dict(zip(factor_names, result.x)),
        'predicted_response': predicted_response,
        'target_response': target,
        'error': result.fun,
        'success': result.success
    }

    return result_dict


# =================================================================
# MAIN EXECUTION
# =================================================================
if __name__ == "__main__":
    print("=" * 80)
    print("STARTING COMPREHENSIVE DOE ANALYSIS")
    print("=" * 80)

    print("\n" + "=" * 80)
    print("LOADING AND ANALYSING DATA")
    print("=" * 80)

    df, X, y, poly, model, factor_names, response_name = analyze_dataset(
        filename="mew_doe_results.csv"
    )

    if model is not None:
        # Calculate global color range from actual data
        global_vmin = y.min()
        global_vmax = y.max()
        print(f"\nActual response range in data: {global_vmin:.4f} to {global_vmax:.4f}")
        
        print("\n" + "=" * 80)
        print("CREATING INITIAL RESPONSE SURFACE PLOTS")
        print("=" * 80)
        
        # Create initial plots with consistent color scale
        plot_all_combinations(X, poly, model, factor_names, response_name,
                            vmin=global_vmin, vmax=global_vmax)

        # =================================================================
        # OPTIMAL FACTOR PREDICTION
        # =================================================================
        print("\n" + "=" * 80)
        print("OPTIMISING FACTOR COMBINATION FOR TARGET RESPONSE")
        print("=" * 80)

        try:
            target_value = float(input(f"\nEnter desired target value for {response_name}: "))

            optimal_result = prediction_from_model(
                target=target_value,
                poly=poly,
                model=model,
                X=X,
                factor_names=factor_names
            )

            opt_factors = optimal_result['optimal_factors']
            opt_response = optimal_result['predicted_response']

            print("\nOptimal Factor Combination Found:")
            for factor, value in opt_factors.items():
                print(f"   {factor}: {value:.4f}")

            print(f"\nPredicted {response_name}: {opt_response:.4f}")
            print(f"Target {response_name}: {optimal_result['target_response']:.4f}")
            print(f"Squared Error: {optimal_result['error']:.6f}")
            print(f"Optimisation Success: {optimal_result['success']}")

            print("\n" + "=" * 80)
            print("RECREATING RESPONSE SURFACE PLOTS WITH OPTIMAL POINT")
            print("=" * 80)
            
            # Replot with optimal point and consistent colors
            plot_all_combinations(X, poly, model, factor_names, response_name, 
                                optimal_factors_dict=opt_factors, 
                                optimal_response=opt_response,
                                vmin=global_vmin,
                                vmax=global_vmax)

        except Exception as e:
            print(f"\nERROR during optimisation: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)