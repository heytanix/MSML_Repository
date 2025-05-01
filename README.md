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

### Method used in Question 1
- Calculated the initial sample size using the formula for a known population standard deviation: n0=(zα⋅σ/E)^2 , zα=1.96 (95% confidence), σ=10, and E=3.
- Applied Finite Population Correction (FPC) to adjust the sample size: n=(n0)/(1+(n0−1)/N), where N=979.
- Rounded up the results using math.ceil() for practical sample size.

### Method used in Question 2
- Performed Simple Random Sampling using pandas.DataFrame.sample() with n=42, random_state=42, and replace=False to select a sample from the population dataset.
- Visualized population and sample distributions for 'Marks (out of 100)' and 'Study Hours (per week)' using seaborn.histplot() with KDE.
- Conducted statistical comparison by computing and comparing means and standard deviations of population and sample using pandas methods (mean(), std()).
- Generated the sampling distribution of sample means by simulating 1000 samples of size 42 using pandas.sample() and plotting with seaborn.histplot().

### Method used in Question 3
- Loaded the sampled data from 'Team-3_Sample.csv' using pandas.read_csv().
- Calculated the sample mean and standard deviation for 'Marks (out of 100)' and 'Study Hours (per week)' using pandas methods (mean(), std()).
- Presented results in a formatted text table.

### Method used in Question 4
- Simulated 1000 samples of size 42 from the population's 'Marks (out of 100)' using pandas.sample() with random_state=42.
- Computed the mean and standard deviation (standard error) of the sample means using numpy.mean() and numpy.std(ddof=1).
- Visualized the sampling distribution using seaborn.histplot() with KDE, marking population and sampling distribution means.

### Method used in Question 5
- Reused the sample means from Question 4.
- Plotted the empirical sampling distribution using seaborn.histplot() with KDE.
- Overlaid a theoretical normal distribution based on the Central Limit Theorem (CLT) using scipy.stats.norm.pdf() with population mean and standard error (σ/nσ/\(\sqrt{n}\)).
- Added annotations for population and sampling means using matplotlib.pyplot.axvline().

### Method used in Question 6
- Visualized the sampling distribution under the null hypothesis (H0:μ=77.38) using scipy.stats.norm.pdf() with population mean and standard error.
- Shaded critical regions for a two-tailed test at α=0.05α=0.05 using matplotlib.pyplot.fill_between() and critical z-values from scipy.stats.norm.ppf(0.975).
- Marked the sample mean (77.14) on the plot using matplotlib.pyplot.axvline().

### Method used in Question 7
- Conducted a two-tailed z-test for the population mean (H0:μ=77.38) using the sample mean (77.14), population standard deviation (σ=10), and sample size (n=42).
- Calculated the z-score: z=(xˉ−μ0)/(σ/root(n))
- Computed the p-value using scipy.stats.norm.cdf() for a two-tailed test.
- Compared z-score against critical z-values for α=0.05 and α=0.01 using scipy.stats.norm.ppf().
- Visualized the z-distribution with rejection regions using matplotlib.pyplot.plot() and fill_between().

### Method used in Question 8
- Calculated confidence intervals for the population mean at 90%, 95%, and 99% confidence levels using the sample mean (77.14), population standard deviation (σ=10), and standard error.
- Used z-critical values from scipy.stats.norm.ppf() for each confidence level.
- Computed interval bounds: xˉ±z⋅σ/(root(n))
- Visualized the intervals using matplotlib.pyplot.plot() to show the range for each confidence level.

### Method used in Question 9
- Summarized findings from the hypothesis test (Question 7) and confidence intervals (Question 8).
- Concluded that the sample mean is consistent with the population mean, failing to reject H0, based on statistical evidence.
- Presented results in a formatted text output.

### Method used in Question 10
- Loaded the sample data from 'Team-3_Sample.csv'.
- Created a scatter plot with a regression line for 'Study Hours (per week)' vs. 'Marks (out of 100)' using seaborn.regplot().
- Added grid and labels using matplotlib.pyplot for visualization.

### Method used in Question 11
- Calculated the Pearson correlation coefficient between 'Study Hours (per week)' and 'Marks (out of 100)' using scipy.stats.pearsonr().
- Reported the correlation coefficient in a formatted text output.

### Method used in Question 12
- Derived two regression equations:
  -  Marks on Study Hours: Marks=a+b⋅Study Hours
  -  Study Hours on Marks: Study Hours=a′+b′⋅Marks
- Calculated regression coefficients using the Pearson correlation coefficient, sample means, and standard deviations of both variables.
- Used formulas: b=r⋅(sy/sx), a=(yˉ−b⋅xˉ), and similarly for the second regression.
- Presented equations in a formatted text output.

### Method used in Question 13
- Used the regression equation from Question 12 (Marks=a+b⋅Study Hours) to predict marks for a given study hours value (12 hours).
- Computed the predicted value and displayed it in a formatted text output.

### Method used in Question 14
- Recalculated the Pearson correlation coefficient and p-value using scipy.stats.pearsonr().
- Conducted a hypothesis test for the correlation coefficient (H0:ρ=0) at α=0.05 and α=0.01.
- Compared the p-value to significance levels to determine if the correlation is statistically significant.
- Presented results and conclusions in a formatted text output.

### Method used in Question 15
- Tested the significance of the regression coefficient (b) for the regression of Marks on Study Hours.
- Calculated the t-statistic using the correlation coefficient: t=r⋅(root(n)−2/1−r^2).
- Computed the p-value for a two-tailed test using scipy.stats.t.cdf() with df=n−2.
- Compared the p-value to α=0.05 and α=0.01 to determine significance.
- Presented results and conclusions in a formatted text output.

## Outputs
- To be viewed within app.ipynb
- 
## Conclusion

This code demonstrates a comprehensive statistical analysis workflow, from sampling theory to hypothesis testing and regression analysis. The analysis reveals important insights about the relationship between study hours and academic performance, supported by appropriate statistical tests and visualizations.
