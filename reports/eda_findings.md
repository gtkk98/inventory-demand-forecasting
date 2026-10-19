# Inventory & Demand Forecasting System
## Data Quality and Exploratory Data Analysis Report

### 1. Project Objective

The purpose of this analysis is to understand historical store-level product demand, identify data-quality concerns, and discover patterns that can support future daily sales forecasting.

### 2. Dataset Overview

| Metric | Result |
|---|---:|
| Total records | 913,000 |
| Unique dates | 1,826 |
| Unique stores | 10 |
| Unique products | 50 |
| Store-product combinations | 500 |
| Total units sold | 47,704,512 |
| Average sales per record | 52.25 |
| Median sales per record | 47 |
| First date | 2013-01-01 |
| Last date | 2017-12-31 |

The dataset contains daily sales observations from January 2013 through December 2017. The number of records matches a complete grid of 10 stores, 50 products, and 1,826 dates.

### 3. Weekday Demand Analysis

| Day | Average units sold per record |
|---|---:|
| Monday | 41.43 |
| Tuesday | 48.23 |
| Wednesday | 48.37 |
| Thursday | 51.72 |
| Friday | 55.16 |
| Saturday | 58.66 |
| Sunday | 62.14 |

Sunday has the highest average sales per record, while Monday has the lowest. Sunday’s average is approximately 50% higher than Monday’s average.

This pattern suggests that day-of-week features may help improve forecasting accuracy. However, aggregate weekday patterns should not be assumed to apply equally to every product and store.

### 4. Store-Level Analysis

The five stores with the highest cumulative sales are:

| Rank | Store ID | Total units sold |
|---|---:|---:|
| 1 | 2 | 6,120,128 |
| 2 | 8 | 5,856,169 |
| 3 | 3 | 5,435,144 |
| 4 | 10 | 5,360,158 |
| 5 | 9 | 5,025,976 |

Store 2 has the highest cumulative sales in the analysed period. These figures represent totals across the full dataset period and do not necessarily indicate the highest current demand.

### 5. Product-Level Analysis

The five highest-selling products by cumulative sales are:

| Rank | Product ID | Total units sold |
|---|---:|---:|
| 1 | 15 | 1,607,442 |
| 2 | 28 | 1,604,713 |
| 3 | 13 | 1,539,621 |
| 4 | 18 | 1,538,876 |
| 5 | 25 | 1,473,334 |

Product 15 has the highest cumulative sales among the products in the dataset. These totals provide a starting point for identifying high-volume products, but recent trends and store-specific demand should also be considered.

### 6. Data Quality Considerations

The dataset dimensions are consistent with a complete daily grid of all store-product combinations.

The following checks should be verified using the project's validation output before model training:

- Missing values in required columns.
- Invalid dates and data types.
- Duplicate date-store-product keys.
- Negative or non-finite sales values.
- Missing store-product-date combinations.
- Whether sales values are appropriate for the intended forecasting target.

A missing observation must not automatically be replaced with zero without understanding its meaning. Likewise, negative sales values require investigation before deciding how to handle them.

### 7. Potential Forecasting Features

The exploratory analysis suggests investigating these features:

- Calendar features: day of week, day of month, month, and year.
- Weekend indicators.
- Historical sales lags, such as the previous day and previous seven days.
- Rolling averages based only on information available before the forecast date.
- Store and product identifiers.

Time-based train-test splitting should be used rather than randomly splitting individual rows. This helps evaluate performance on future dates without leaking future information into training.

### 8. Generated Analysis Artifacts

The exploration script generates summary reports and visualizations under the `reports/` directory.

Expected artifacts include:

- `dataset_summary.csv`
- `daily_sales_summary.csv`
- `weekday_sales_summary.csv`
- `monthly_sales_summary.csv`
- `figures/daily_sales.png`
- `figures/monthly_sales.png`
- `figures/weekday_sales.png`
- `figures/store_sales.png`

A product-specific chart can also be generated if the optional product-level plotting function has been added and executed.

### 9. Conclusion

The dataset contains five years of daily sales records for 10 stores and 50 products. Aggregate results show meaningful weekday variation, with Sunday having the highest average sales and Monday the lowest. Store and product totals also reveal differences in cumulative sales volume.

The next phase will focus on data preparation and feature engineering. All transformations must preserve the time-series structure and avoid using future information when generating model inputs.
