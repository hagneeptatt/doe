import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy import stats
import seaborn as sns
import warnings
import os

warnings.filterwarnings('ignore')

def analyze_dataset(filename, factor_columns, response_column):
    """
    General function to analyze any DOE dataset with comprehensive validation
    """
    print(f"\n{'='*80}")
    print(f"ANALYZING: {filename}")
    print(f"{'='*80}")
    
    # Load data
    if not os.path.exists(filename):
        print(f"ERROR: {filename} not found!")
        return None, None, None, None, None
    
    df = pd.read_csv(filename)
    print(f"Data loaded successfully!")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Check if required columns exist
    missing_cols = [col for col in factor_columns + [response_column] if col not in df.columns]
    if missing_cols:
        print(f"ERROR: Missing columns: {missing_cols}")
        return None, None, None, None, None
    
    # Define factors and response
    X = df[factor_columns].values
    y = df[response_column].values
    
    print(f"\nFactors: {factor_columns}")
    print(f"Response: {response_column}")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    
    # Data validation
    validate_data(df, factor_columns, response_column)
    
    # Create polynomial features (degree 2)
    poly = PolynomialFeatures(degree=2, include_bias=True)
    X_poly = poly.fit_transform(X)
    
    print(f"\nPolynomial features created: {X_poly.shape[1]}")
    
    # Fit the model
    model = LinearRegression()
    model.fit(X_poly, y)
    
    # Model validation
    validate_model(X_poly, y, model, factor_columns, response_column)
    
    return df, X, y, poly, model

def validate_data(df, factor_columns, response_column):
    """
    Comprehensive data validation
    """
    print(f"\n{'-'*40}")
    print("DATA VALIDATION")
    print(f"{'-'*40}")
    
    # Check for missing values
    missing_data = df[factor_columns + [response_column]].isnull().sum()
    if missing_data.sum() > 0:
        print("Missing values found:")
        for col, count in missing_data.items():
            if count > 0:
                print(f"   {col}: {count} missing values")
    else:
        print("No missing values")
    
    # Check for duplicates
    duplicates = df.duplicated(subset=factor_columns).sum()
    if duplicates > 0:
        print(f"{duplicates} duplicate factor combinations found")
    else:
        print("No duplicate factor combinations")
    
    # Data summary statistics
    print(f"\nDescriptive Statistics:")
    print(df[factor_columns + [response_column]].describe())
    
    # Check for outliers using IQR method
    print(f"\nOutlier Detection (IQR method):")
    for col in factor_columns + [response_column]:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        if outliers > 0:
            print(f"   {col}: {outliers} potential outliers")
        else:
            print(f"   {col}: No outliers detected")
    
    # Correlation matrix
    print(f"\nCorrelation Matrix:")
    corr_matrix = df[factor_columns + [response_column]].corr()
    print(corr_matrix.round(3))
    
    # Plot correlation heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, fmt='.2f')
    plt.title('Factor Correlation Matrix')
    plt.tight_layout()
    plt.show()

def validate_model(X_poly, y, model, factor_names, response_name):
    """
    Comprehensive model validation
    """
    print(f"\n{'-'*40}")
    print("MODEL VALIDATION")
    print(f"{'-'*40}")
    
    # Basic model metrics
    y_pred = model.predict(X_poly)
    r2 = model.score(X_poly, y)
    mse = mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mse)
    
    print(f"R-squared: {r2:.4f}")
    print(f"Mean Squared Error: {mse:.6f}")
    print(f"Root Mean Squared Error: {rmse:.6f}")
    print(f"Mean Absolute Error: {mae:.6f}")
    
    # Adjusted R-squared
    n = len(y)
    p = X_poly.shape[1] - 1  # exclude intercept
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    print(f"Adjusted R-squared: {adj_r2:.4f}")
    
    # Cross-validation
    if len(y) > 5:  # Only if we have enough data
        cv_scores = cross_val_score(model, X_poly, y, cv=min(5, len(y)//2), 
                                   scoring='r2')
        print(f"Cross-validation R² (mean ± std): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Residual analysis
    residuals = y - y_pred
    
    # Normality test on residuals
    shapiro_stat, shapiro_p = stats.shapiro(residuals)
    print(f"\nResidual Analysis:")
    print(f"Shapiro-Wilk test for normality: p = {shapiro_p:.4f}")
    if shapiro_p > 0.05:
        print("✓ Residuals appear normally distributed")
    else:
        print("Residuals may not be normally distributed")
    
    # Independence test (Durbin-Watson approximation)
    dw_stat = np.sum(np.diff(residuals)**2) / np.sum(residuals**2)
    print(f"Durbin-Watson statistic: {dw_stat:.4f}")
    if 1.5 < dw_stat < 2.5:
        print("✓ No strong evidence of autocorrelation")
    else:
        print("Possible autocorrelation in residuals")
    
    # Create diagnostic plots
    create_diagnostic_plots(y, y_pred, residuals, response_name)

def create_diagnostic_plots(y_actual, y_pred, residuals, response_name):
    """
    Create model diagnostic plots
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Model Diagnostic Plots', fontsize=16)
    
    # 1. Actual vs Predicted
    axes[0,0].scatter(y_actual, y_pred, alpha=0.7)
    axes[0,0].plot([y_actual.min(), y_actual.max()], 
                   [y_actual.min(), y_actual.max()], 'r--', lw=2)
    axes[0,0].set_xlabel(f'Actual {response_name}')
    axes[0,0].set_ylabel(f'Predicted {response_name}')
    axes[0,0].set_title('Actual vs Predicted')
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. Residuals vs Predicted
    axes[0,1].scatter(y_pred, residuals, alpha=0.7)
    axes[0,1].axhline(y=0, color='r', linestyle='--')
    axes[0,1].set_xlabel(f'Predicted {response_name}')
    axes[0,1].set_ylabel('Residuals')
    axes[0,1].set_title('Residuals vs Predicted')
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. Histogram of residuals
    axes[1,0].hist(residuals, bins=10, alpha=0.7, density=True)
    axes[1,0].set_xlabel('Residuals')
    axes[1,0].set_ylabel('Density')
    axes[1,0].set_title('Distribution of Residuals')
    axes[1,0].grid(True, alpha=0.3)
    
    # 4. Q-Q plot
    stats.probplot(residuals, dist="norm", plot=axes[1,1])
    axes[1,1].set_title('Q-Q Plot (Normal)')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_surface(X, poly, model, factor_names, response_name, factor1_idx, factor2_idx, resolution=30):
    """
    Plot 3D response surface for two factors
    """
    print(f"Creating surface plot: {factor_names[factor1_idx]} vs {factor_names[factor2_idx]}")
    
    # Get ranges for the two factors
    f1_min, f1_max = X[:, factor1_idx].min(), X[:, factor1_idx].max()
    f2_min, f2_max = X[:, factor2_idx].min(), X[:, factor2_idx].max()
    
    # Create grid
    f1_range = np.linspace(f1_min, f1_max, resolution)
    f2_range = np.linspace(f2_min, f2_max, resolution)
    F1, F2 = np.meshgrid(f1_range, f2_range)
    
    # Create prediction grid - set other factors to their mean values
    grid_size = resolution * resolution
    prediction_grid = np.zeros((grid_size, X.shape[1]))
    
    # Set the two varying factors
    prediction_grid[:, factor1_idx] = F1.ravel()
    prediction_grid[:, factor2_idx] = F2.ravel()
    
    # Set other factors to their mean values
    fixed_values = []
    for i in range(X.shape[1]):
        if i not in [factor1_idx, factor2_idx]:
            mean_val = X[:, i].mean()
            prediction_grid[:, i] = mean_val
            fixed_values.append(f"{factor_names[i]}={mean_val:.3f}")
    
    # Transform to polynomial and predict
    grid_poly = poly.transform(prediction_grid)
    predictions = model.predict(grid_poly)
    
    # Reshape for plotting
    Z = predictions.reshape(resolution, resolution)
    
    # Create 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    surf = ax.plot_surface(F1, F2, Z, cmap='viridis', alpha=0.8)
    
    ax.set_xlabel(factor_names[factor1_idx])
    ax.set_ylabel(factor_names[factor2_idx])
    ax.set_zlabel(response_name)
    
    title = f'Response Surface: {factor_names[factor1_idx]} vs {factor_names[factor2_idx]}'
    if fixed_values:
        title += f'\n(Fixed: {", ".join(fixed_values)})'
    ax.set_title(title)
    
    fig.colorbar(surf, shrink=0.6)
    plt.tight_layout()
    plt.show()

def plot_all_combinations(X, poly, model, factor_names, response_name):
    """
    Plot all possible 2-factor combinations
    """
    n_factors = len(factor_names)
    n_combinations = n_factors * (n_factors - 1) // 2
    
    print(f"\n{'-'*40}")
    print(f"CREATING {n_combinations} RESPONSE SURFACE PLOTS")
    print(f"{'-'*40}")
    
    plot_count = 1
    for i in range(n_factors):
        for j in range(i+1, n_factors):
            print(f"\nPlot {plot_count}/{n_combinations}:")
            plot_surface(X, poly, model, factor_names, response_name, i, j)
            plot_count += 1

# =================================================================
# MAIN ANALYSIS
# =================================================================

# Dataset 1: Original DOE data
print("="*80)
print("STARTING COMPREHENSIVE DOE ANALYSIS")
print("="*80)

df1, X1, y1, poly1, model1 = analyze_dataset(
    filename="mew_doe_results.csv",
    factor_columns=["pres", "temp", "volt", "dist", "fr"],
    response_column="dia"
)

if model1 is not None:
    print("\n" + "="*80)
    print("CREATING ALL RESPONSE SURFACE PLOTS FOR MEW DOE DATA")
    print("="*80)
    
    # Plot all 10 possible combinations for 5 factors
    plot_all_combinations(X1, poly1, model1, ["pres", "temp", "volt", "dist", "fr"], "dia")

# Dataset 2: Bioprinting DOE data
print("\n" + "="*80)
print("CHECKING FOR BIOPRINTING DATA")
print("="*80)

df2, X2, y2, poly2, model2 = analyze_dataset(
    filename="bioprint_doe_results.csv",
    factor_columns=["press", "vot"],
    response_column="vol"
)

if model2 is not None:
    print("\n" + "="*80)
    print("CREATING RESPONSE SURFACE PLOTS FOR BIOPRINTING DATA")
    print("="*80)
    
    # For 2 factors, only 1 combination
    plot_all_combinations(X2, poly2, model2, ["press", "vot"], "vol")

print(f"\n{'='*80}")
print("COMPREHENSIVE ANALYSIS COMPLETE")
print(f"{'='*80}")