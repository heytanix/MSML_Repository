![Cover Image](/Assets/Cover_Image.jpg)
# Mathematics & Statistics for Machine learning - (Experential Learning) - TEAM 3

This repository contains a statistical analysis of the relationship between students' study hours and their academic performance (marks). The analysis follows a structured approach addressing 15 key statistical concepts and techniques.

## Team Members (Contributors)
- Thanish Chinnappa KC
- Likhith V
- Sahil Patil
- Sudareshwar S
- Samith Shivakumar
- Souharda Mandal

## Table of Contents
- [Libraries Used](#libraries-used)
- [Analysis Questions](#analysis-questions)
- [Output Files](#output-files)

## Libraries Used

The code utilizes several Python libraries for statistical analysis and data visualization:

- **NumPy**: Provides support for numerical computing with arrays and mathematical functions
- **Pandas**: Used for data manipulation and analysis with DataFrames
- **Matplotlib**: Creates static visualizations and plots
- **Seaborn**: Built on Matplotlib, provides enhanced statistical visualizations
- **SciPy**: Implements various statistical functions and tests
- **Statsmodels**: Offers classes and functions for statistical models and hypothesis testing
- **Scikit-learn**: Provides machine learning algorithms for regression analysis and evaluation metrics

<img src="/Assets/NumPyLogo.png" alt="NumPy Logo" width="100"> <img src="/Assets/PandasLogo.png" alt="Pandas Logo" width="100"> <img src="/Assets/MatplotlibLogo.png" alt="Matplotlib Logo" width="100"> <img src="/Assets/SeabornLogo.png" alt="Seaborn Logo" width="100"> <img src="/Assets/SciPyLogo.jpg" alt="SciPy Logo" width="100"> <img src="/Assets/StatsmodelsLogo.jpeg" alt="Statsmodels Logo" width="100"> <img src="/Assets/scikitlearnLogo.jpeg" alt="Scikit-learn Logo" width="100">

## Analysis Questions

### Question 1: Calculate Sample Size
Determines the appropriate sample size using statistical formulas:
- Uses the formula n₀ = (z × σ / E)² where:
  - z = 1.96 (95% confidence level)
  - σ = 10 (population standard deviation)
  - E = 3 (margin of error)
- Applies finite population correction for a population size of 979
- Calculates the required sample size for the study

### Question 2: Select Sample Using Probability Sampling
Implements three different sampling methods:
- **Simple Random Sampling**: Randomly selects elements from the population
- **Systematic Sampling**: Selects elements at regular intervals
- **Stratified Sampling**: Divides the population into subgroups (strata) based on gender and samples proportionally
- The code defaults to random sampling but can use any of these methods

### Question 3: Calculate Mean and Standard Deviation
Computes descriptive statistics for both the sample and population:
- Sample mean and standard deviation of marks
- Population mean and standard deviation for comparison
- Provides insights into how well the sample represents the population

### Question 4: Sampling Distribution of Sample Means
Creates an empirical sampling distribution:
- Takes 1,000 different samples of the calculated sample size
- Calculates the mean of each sample
- Builds a distribution of these sample means
- Computes the mean and standard error of this distribution
- Compares with the theoretical standard error (σ/√n)

### Question 5: Plot Sampling Distribution
Visualizes the sampling distribution:
- Creates a histogram with kernel density estimation
- Marks the mean with a vertical line
- Demonstrates the Central Limit Theorem in action
- Saves the visualization as "sampling_distribution.png"

### Question 6: Frame Null Hypothesis
Formulates statistical hypotheses:
- Null Hypothesis (H₀): The population mean equals the sample mean
- Alternative Hypothesis (H₁): The population mean differs from the sample mean
- States the numerical values for both means

### Question 7: Test Hypothesis at Different Significance Levels
Performs hypothesis testing:
- Uses one-sample t-test to compare sample mean with population mean
- Tests at both 5% and 1% significance levels
- Computes t-statistic and p-value
- Determines whether to reject or fail to reject the null hypothesis

### Question 8: Compute Confidence Intervals
Calculates confidence intervals at multiple confidence levels:
- Generates 90%, 95%, and 99% confidence intervals
- Computes margin of error for each interval
- Shows the range within which the true population mean is expected to fall

### Question 9: Decision Based on Analysis
Interprets results from hypothesis testing and confidence intervals:
- Explains the meaning of rejecting or failing to reject the null hypothesis
- Discusses what the confidence intervals tell us about the population mean
- Provides a statistical conclusion based on the evidence

### Question 10: Relationship between Study Hours and Marks
Explores the correlation between study time and academic performance:
- Creates a scatter plot of study hours vs. marks
- Visualizes the potential relationship between these variables
- Saves the plot as "study_hours_vs_marks.png"

### Question 11: Correlation Coefficient
Quantifies the relationship strength:
- Calculates Pearson's correlation coefficient
- Tests statistical significance with p-value
- Interprets the strength (weak/moderate/strong) and direction (positive/negative) of correlation

### Question 12: Regression Lines
Fits two regression models:
- **Marks on Study Hours**: Predicts marks based on study time
- **Study Hours on Marks**: Predicts study time based on marks
- Calculates slope, intercept, and R² for each model
- Creates side-by-side plots showing both regression lines
- Saves the visualization as "regression_lines.png"

### Question 13: Predict Marks for Given Study Hours
Applies the regression model for prediction:
- Uses the regression equation from Question 12
- Predicts expected marks for a student studying 15 hours per week
- Demonstrates practical application of the regression model

### Question 14: Test Significance of Correlation Coefficient
Determines if the correlation is statistically significant:
- Tests null hypothesis that the correlation equals zero
- Calculates t-statistic and p-value
- Decides whether there is statistically significant evidence of correlation

### Question 15: Test Significance of Regression Coefficient
Evaluates the significance of the regression slope:
- Uses Ordinary Least Squares (OLS) method
- Generates a comprehensive regression summary
- Tests if the slope coefficient differs significantly from zero
- Interprets the p-value for the regression slope

## Present Case outputs
- Regression Lines
![Regression Lines](/OutputImages/regression_lines.png)
- Sampling Distributions
![Sampling Distribution](/OutputImages/sampling_distribution.png)
- Study Hours vs Marks
![Study Hours vs Marks](/OutputImages/study_hours_vs_marks.png)
## Output Files

### 1. sampling_distribution.png
- Visualization of the sampling distribution of sample means
- Shows histogram with density curve
- Demonstrates how sample means distribute approximately normally
- Illustrates the Central Limit Theorem

### 2. study_hours_vs_marks.png
- Scatter plot showing the relationship between study hours and marks
- Each point represents a student's study time and corresponding marks
- Helps visualize any patterns or trends in the data
- Shows the correlation between the two variables

### 3. regression_lines.png
- Two side-by-side plots showing different regression models:
  - Left: Predicting marks based on study hours
  - Right: Predicting study hours based on marks
- Includes regression equations and fitted lines
- Helps understand the predictive relationship between variables

### 4. analysis_report.txt
- Comprehensive summary of all statistical findings
- Includes numerical results from all 15 questions
- Provides parameter values, test statistics, p-values, and interpretations
- Serves as a complete record of the statistical analysis

## Conclusion

This code demonstrates a comprehensive statistical analysis workflow, from sampling theory to hypothesis testing and regression analysis. The analysis reveals important insights about the relationship between study hours and academic performance, supported by appropriate statistical tests and visualizations.
