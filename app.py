import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Set random seed for reproducibility
np.random.seed(42)

# =============================================
# Question 1: Calculate Sample Size
# =============================================
def calculate_sample_size():
    """
    Calculate sample size using the formula:
    n0 = (z * σ / E)^2
    n = n0 / (1 + (n0 - 1)/N)  # Finite population correction
    
    For Team 3:
    - Confidence level: 95% (z = 1.96)
    - Margin of error (E): 3
    - Population SD (σ): 10 (given)
    - Population size (N): 979 (from dataset)
    """
    z = 1.96  # For 95% confidence
    sigma = 10  # Given population standard deviation
    E = 3  # Margin of error for Team 3
    N = len(df)  # Population size
    
    # Calculate initial sample size without correction
    n0 = (z * sigma / E)**2
    
    # Apply finite population correction
    n = n0 / (1 + (n0 - 1)/N)
    
    return int(np.ceil(n))

# =============================================
# Load and Prepare Data
# =============================================
# Load the dataset
df = pd.read_csv('Team-3.csv')

# Check data
print("Data Overview:")
print(df.head())
print("\nData Shape:", df.shape)
print("\nColumn Names:", df.columns.tolist())

# Clean column names (remove extra spaces)
df.columns = df.columns.str.strip()

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# =============================================
# Question 1: Calculate Sample Size
# =============================================
sample_size = calculate_sample_size()
print(f"\nQuestion 1: Required sample size (n) = {sample_size}")

# =============================================
# Question 2: Select Sample Using Probability Sampling
# =============================================
def select_sample(data, n, method='random'):
    """
    Select sample using specified probability sampling method.
    Options: 'random', 'systematic', 'stratified'
    """
    if method == 'random':
        # Simple random sampling
        sample = data.sample(n=n, random_state=42)
    elif method == 'systematic':
        # Systematic sampling
        interval = len(data) // n
        start = np.random.randint(0, interval)
        indices = range(start, len(data), interval)
        sample = data.iloc[indices[:n]]
    elif method == 'stratified':
        # Stratified sampling by Gender (for example)
        strata = data['Gender'].value_counts(normalize=True)
        samples_per_stratum = (strata * n).round().astype(int)
        
        # Adjust if total doesn't match due to rounding
        diff = n - samples_per_stratum.sum()
        if diff > 0:
            samples_per_stratum.iloc[0] += diff
        elif diff < 0:
            samples_per_stratum.iloc[-1] += diff
            
        sample = pd.DataFrame()
        for stratum, count in samples_per_stratum.items():
            stratum_sample = data[data['Gender'] == stratum].sample(count, random_state=42)
            sample = pd.concat([sample, stratum_sample])
    
    return sample

# Select sample using random sampling (can change to 'systematic' or 'stratified')
sample = select_sample(df, sample_size, method='random')
print("\nQuestion 2: Selected Sample:")
print(sample[['Name', 'Study Hours (per week)', 'Marks (out of 100)']].head())

# =============================================
# Question 3: Calculate Mean and SD of Sample
# =============================================
sample_mean = sample['Marks (out of 100)'].mean()
sample_std = sample['Marks (out of 100)'].std(ddof=1)  # Sample standard deviation

population_mean = df['Marks (out of 100)'].mean()
population_std = df['Marks (out of 100)'].std(ddof=1)

print("\nQuestion 3:")
print(f"Sample Mean (marks): {sample_mean:.2f}")
print(f"Sample Standard Deviation (marks): {sample_std:.2f}")
print(f"Population Mean (marks): {population_mean:.2f}")
print(f"Population Standard Deviation (marks): {population_std:.2f}")

# =============================================
# Question 4: Sampling Distribution of Sample Means
# =============================================
def create_sampling_distribution(data, n, num_samples=1000):
    """Create sampling distribution by taking multiple samples"""
    sample_means = []
    for _ in range(num_samples):
        sample = data.sample(n=n, random_state=_)
        sample_means.append(sample['Marks (out of 100)'].mean())
    return sample_means

sampling_distribution = create_sampling_distribution(df, sample_size)

# Calculate statistics of sampling distribution
sampling_mean = np.mean(sampling_distribution)
sampling_std = np.std(sampling_distribution, ddof=1)

print("\nQuestion 4: Sampling Distribution of Sample Means")
print(f"Mean of sampling distribution: {sampling_mean:.2f}")
print(f"Standard Error (SE): {sampling_std:.2f}")
print(f"Expected Standard Error (σ/√n): {population_std/np.sqrt(sample_size):.2f}")

# =============================================
# Question 5: Plot Sampling Distribution
# =============================================
plt.figure(figsize=(10, 6))
sns.histplot(sampling_distribution, kde=True, color='skyblue')
plt.axvline(sampling_mean, color='red', linestyle='--', label=f'Mean = {sampling_mean:.2f}')
plt.title('Sampling Distribution of Sample Means', fontsize=14)
plt.xlabel('Sample Means (Marks)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('sampling_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

print("\nQuestion 5: Sampling distribution plot saved as 'sampling_distribution.png'")

# =============================================
# Question 6: Frame Null Hypothesis
# =============================================
print("\nQuestion 6: Hypothesis Formulation")
print("Null Hypothesis (H0): μ = μ0 (population mean and sample mean are equal)")
print("Alternative Hypothesis (H1): μ ≠ μ0 (population mean and sample mean are not equal)")
print(f"Population mean (μ0): {population_mean:.2f}")
print(f"Sample mean (x̄): {sample_mean:.2f}")

# =============================================
# Question 7: Test Hypothesis at 5% and 1% Significance
# =============================================
def hypothesis_test(sample, population_mean, alpha=0.05):
    """Perform one-sample t-test"""
    t_stat, p_value = stats.ttest_1samp(sample['Marks (out of 100)'], population_mean)
    
    print(f"\nSignificance level: {alpha*100}%")
    print(f"t-statistic: {t_stat:.4f}")
    print(f"p-value: {p_value:.4f}")
    
    if p_value < alpha:
        print("Result: Reject the null hypothesis (H0)")
    else:
        print("Result: Fail to reject the null hypothesis (H0)")

print("\nQuestion 7: Hypothesis Testing Results")
hypothesis_test(sample, population_mean, alpha=0.05)  # 5% significance
hypothesis_test(sample, population_mean, alpha=0.01)  # 1% significance

# =============================================
# Question 8: Compute Confidence Intervals
# =============================================
def calculate_confidence_intervals(sample, confidence_levels=[0.90, 0.95, 0.99]):
    """Calculate confidence intervals for different confidence levels"""
    results = {}
    n = len(sample)
    sample_mean = sample['Marks (out of 100)'].mean()
    sample_std = sample['Marks (out of 100)'].std(ddof=1)
    
    for cl in confidence_levels:
        alpha = 1 - cl
        z_critical = stats.norm.ppf(1 - alpha/2)
        margin_of_error = z_critical * (sample_std / np.sqrt(n))
        ci_lower = sample_mean - margin_of_error
        ci_upper = sample_mean + margin_of_error
        
        results[f"{int(cl*100)}%"] = {
            'Lower': ci_lower,
            'Upper': ci_upper,
            'Margin of Error': margin_of_error
        }
    
    return results

confidence_intervals = calculate_confidence_intervals(sample)

print("\nQuestion 8: Confidence Intervals")
for cl, ci in confidence_intervals.items():
    print(f"{cl} CI: ({ci['Lower']:.2f}, {ci['Upper']:.2f})")
    print(f"Margin of Error: ±{ci['Margin of Error']:.2f}")

# =============================================
# Question 9: Decision Based on Analysis
# =============================================
print("\nQuestion 9: Conclusion from Analysis")
print("Based on the hypothesis tests and confidence intervals:")
print("- If the null hypothesis was rejected, there is evidence that the sample mean differs from the population mean.")
print("- If not rejected, there isn't enough evidence to conclude a difference.")
print("- The confidence intervals show the range within which we expect the true population mean to lie.")

# =============================================
# Question 10: Relationship between Study Hours and Marks
# =============================================
plt.figure(figsize=(10, 6))
sns.scatterplot(data=sample, x='Study Hours (per week)', y='Marks (out of 100)', alpha=0.7)
plt.title('Relationship Between Study Hours and Marks Scored', fontsize=14)
plt.xlabel('Study Hours (per week)', fontsize=12)
plt.ylabel('Marks (out of 100)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.savefig('study_hours_vs_marks.png', dpi=300, bbox_inches='tight')
plt.close()

print("\nQuestion 10: Scatter plot of Study Hours vs Marks saved as 'study_hours_vs_marks.png'")

# =============================================
# Question 11: Correlation Coefficient
# =============================================
corr_coef, p_value = stats.pearsonr(sample['Study Hours (per week)'], sample['Marks (out of 100)'])

print("\nQuestion 11: Correlation Analysis")
print(f"Pearson Correlation Coefficient: {corr_coef:.4f}")
print(f"p-value: {p_value:.4f}")

if abs(corr_coef) >= 0.7:
    strength = "strong"
elif abs(corr_coef) >= 0.3:
    strength = "moderate"
else:
    strength = "weak"

if corr_coef > 0:
    direction = "positive"
else:
    direction = "negative"

print(f"There is a {strength} {direction} correlation between study hours and marks scored.")

# =============================================
# Question 12: Regression Lines
# =============================================
# Prepare data
X = sample['Study Hours (per week)'].values.reshape(-1, 1)
y = sample['Marks (out of 100)'].values

# a. Marks on Study Hours
model_a = LinearRegression()
model_a.fit(X, y)
y_pred_a = model_a.predict(X)
slope_a = model_a.coef_[0]
intercept_a = model_a.intercept_
r2_a = r2_score(y, y_pred_a)

# b. Study Hours on Marks
model_b = LinearRegression()
model_b.fit(y.reshape(-1, 1), X)
x_pred_b = model_b.predict(y.reshape(-1, 1))
slope_b = model_b.coef_[0][0]
intercept_b = model_b.intercept_[0]

print("\nQuestion 12: Regression Lines")
print("a. Marks Scored on Study Hours:")
print(f"   Equation: Marks = {intercept_a:.2f} + {slope_a:.2f} * Study_Hours")
print(f"   R-squared: {r2_a:.4f}")

print("\nb. Study Hours on Marks Scored:")
print(f"   Equation: Study_Hours = {intercept_b:.2f} + {slope_b:.2f} * Marks")

# Plot both regression lines
plt.figure(figsize=(12, 6))

# Plot Marks on Study Hours
plt.subplot(1, 2, 1)
sns.scatterplot(data=sample, x='Study Hours (per week)', y='Marks (out of 100)', alpha=0.7)
plt.plot(X, y_pred_a, color='red', label=f'Marks = {intercept_a:.1f} + {slope_a:.1f}*Hours')
plt.title('Marks Scored on Study Hours', fontsize=12)
plt.xlabel('Study Hours (per week)', fontsize=10)
plt.ylabel('Marks (out of 100)', fontsize=10)
plt.legend()
plt.grid(True, alpha=0.3)

# Plot Study Hours on Marks
plt.subplot(1, 2, 2)
sns.scatterplot(data=sample, y='Study Hours (per week)', x='Marks (out of 100)', alpha=0.7)
plt.plot(y, x_pred_b, color='green', label=f'Hours = {intercept_b:.1f} + {slope_b:.1f}*Marks')
plt.title('Study Hours on Marks Scored', fontsize=12)
plt.ylabel('Study Hours (per week)', fontsize=10)
plt.xlabel('Marks (out of 100)', fontsize=10)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('regression_lines.png', dpi=300, bbox_inches='tight')
plt.close()

print("\nRegression plots saved as 'regression_lines.png'")

# =============================================
# Question 13: Predict Marks for Given Study Hours
# =============================================
def predict_marks(hours):
    """Predict marks based on study hours using the regression model"""
    return intercept_a + slope_a * hours

# Example prediction for 15 study hours
study_hours = 15
predicted_marks = predict_marks(study_hours)

print("\nQuestion 13: Prediction Using Regression")
print(f"For a student who studies {study_hours} hours per week:")
print(f"Predicted marks: {predicted_marks:.1f} out of 100")

# =============================================
# Question 14: Test Significance of Correlation Coefficient
# =============================================
def test_correlation_significance(r, n, alpha=0.05):
    """Test if correlation coefficient is significantly different from zero"""
    t_stat = r * np.sqrt(n - 2) / np.sqrt(1 - r**2)
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-2))
    
    print(f"\nSignificance level: {alpha*100}%")
    print(f"t-statistic: {t_stat:.4f}")
    print(f"p-value: {p_value:.4f}")
    
    if p_value < alpha:
        print("Result: Reject the null hypothesis (correlation is significant)")
    else:
        print("Result: Fail to reject the null hypothesis (no significant correlation)")

print("\nQuestion 14: Significance of Correlation Coefficient")
test_correlation_significance(corr_coef, len(sample), alpha=0.05)

# =============================================
# Question 15: Test Significance of Regression Coefficient
# =============================================
def test_regression_significance(X, y):
    """Test significance of regression coefficient using OLS"""
    X_with_const = sm.add_constant(X)
    model = sm.OLS(y, X_with_const).fit()
    
    print("\nQuestion 15: Regression Coefficient Significance")
    print(model.summary())
    
    # Extract p-value for the slope coefficient
    p_value = model.pvalues[1]
    
    print(f"\np-value for slope coefficient: {p_value:.4f}")
    if p_value < 0.05:
        print("Result: Reject the null hypothesis (slope is significant)")
    else:
        print("Result: Fail to reject the null hypothesis (slope is not significant)")

print("\nQuestion 15: Significance of Regression Coefficient")
test_regression_significance(X, y)

# =============================================
# Generate Report with All Results
# =============================================
def generate_report():
    """Generate a text report with all results"""
    report = f"""
    MATHEMATICS & STATISTICS FOR MACHINE LEARNING - ANALYSIS REPORT
    ==============================================================
    
    Team: Team 3
    Sample Size: {sample_size}
    
    1. Sample Size Calculation:
       - Calculated sample size (n): {sample_size}
       - Used formula: n0 = (z * σ / E)^2 with finite population correction
       - Parameters: z=1.96 (95% CI), σ=10, E=3, N={len(df)}
    
    2. Sampling Method:
       - Used simple random sampling
       - Sample size: {len(sample)} records
    
    3. Descriptive Statistics:
       - Sample Mean (marks): {sample_mean:.2f}
       - Sample Standard Deviation: {sample_std:.2f}
       - Population Mean: {population_mean:.2f}
       - Population Standard Deviation: {population_std:.2f}
    
    4. Sampling Distribution:
       - Mean of sampling distribution: {sampling_mean:.2f}
       - Standard Error (SE): {sampling_std:.2f}
       - Expected SE (σ/√n): {population_std/np.sqrt(sample_size):.2f}
    
    5. Hypothesis Testing:
       - Null Hypothesis (H0): μ = {population_mean:.2f}
       - Alternative Hypothesis (H1): μ ≠ {population_mean:.2f}
       - At 5% significance: {'Reject H0' if stats.ttest_1samp(sample['Marks (out of 100)'], population_mean).pvalue < 0.05 else 'Fail to reject H0'}
       - At 1% significance: {'Reject H0' if stats.ttest_1samp(sample['Marks (out of 100)'], population_mean).pvalue < 0.01 else 'Fail to reject H0'}
    
    6. Confidence Intervals:
    """
    
    for cl, ci in confidence_intervals.items():
        report += f"       - {cl} CI: ({ci['Lower']:.2f}, {ci['Upper']:.2f}) ±{ci['Margin of Error']:.2f}\n"
    
    report += f"""
    7. Correlation Analysis:
       - Pearson Correlation Coefficient: {corr_coef:.4f}
       - p-value: {stats.pearsonr(sample['Study Hours (per week)'], sample['Marks (out of 100)'])[1]:.4f}
       - Interpretation: {'Significant' if stats.pearsonr(sample['Study Hours (per week)'], sample['Marks (out of 100)'])[1] < 0.05 else 'Not significant'}
    
    8. Regression Analysis:
       a. Marks on Study Hours:
          - Equation: Marks = {intercept_a:.2f} + {slope_a:.2f} * Study_Hours
          - R-squared: {r2_a:.4f}
       
       b. Study Hours on Marks:
          - Equation: Study_Hours = {intercept_b:.2f} + {slope_b:.2f} * Marks
    
    9. Prediction Example:
       - For 15 study hours: Predicted marks = {predict_marks(15):.1f}
    
    10. Significance Tests:
        - Correlation coefficient is {'significantly' if stats.pearsonr(sample['Study Hours (per week)'], sample['Marks (out of 100)'])[1] < 0.05 else 'not significantly'} different from zero
        - Regression slope is {'significantly' if sm.OLS(y, sm.add_constant(X)).fit().pvalues[1] < 0.05 else 'not significantly'} different from zero
    
    ==============================================================
    Note: All plots have been saved as PNG files in the working directory.
    """
    
    with open("report.txt", "w", encoding="utf-8") as f:
        f.write(report)
    
    return report

# Generate and save the report
report = generate_report()
print("\nFull analysis report saved as 'analysis_report.txt'")
print("\n=== Analysis Complete ===")
