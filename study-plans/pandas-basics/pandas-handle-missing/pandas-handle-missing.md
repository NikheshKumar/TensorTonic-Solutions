# <span style="font-size: 20px;">Handle Missing Values</span>

<span style="font-size: 14px;">Missing data is the most pervasive data quality problem in real-world datasets. Sensors fail, users skip form fields, databases allow NULL, and file parsers encounter empty cells. Pandas represents missing data with `NaN` (Not a Number) for float columns, `pd.NaT` for datetime columns, and `pd.NA` for nullable extension types. Handling missing data correctly is essential because most statistical operations and machine learning models produce wrong results or crash when given NaN values.</span>

---

## <span style="font-size: 16px;">Detecting Missing Values</span>

<span style="font-size: 14px;">Two complementary methods for detection:</span>

```python
df.isna()     # True where values are missing
df.notna()    # True where values are present
```

<span style="font-size: 14px;">Both return a DataFrame of booleans with the same shape as the original. These are aliases for `isnull()` and `notnull()`, which are identical.</span>

### <span style="font-size: 14px;">Counting Missing Values</span>

```python
df.isna().sum()          # missing count per column
df.isna().sum().sum()    # total missing count
df.isna().mean()         # fraction missing per column
df.isna().any()          # True if column has any missing
```

<span style="font-size: 14px;">The pattern `df.isna().mean()` is particularly useful: a value of 0.15 means 15% of the column is missing. Columns with more than 50% missing data often need to be dropped rather than imputed.</span>

### <span style="font-size: 14px;">Visualizing Missing Patterns</span>

<span style="font-size: 14px;">Missing data is rarely random. Common patterns include:</span>

* <span style="font-size: 14px;">**MCAR** (Missing Completely at Random): Missingness has no relationship to any data. Safe to drop or impute.</span>
* <span style="font-size: 14px;">**MAR** (Missing at Random): Missingness depends on observed data. Imputation should use other columns.</span>
* <span style="font-size: 14px;">**MNAR** (Missing Not at Random): Missingness depends on the missing value itself. Requires domain knowledge to handle.</span>

---

## <span style="font-size: 16px;">Dropping Missing Values</span>

<span style="font-size: 14px;">The `dropna()` method removes rows or columns with missing values:</span>

```python
df.dropna()                    # drop rows with ANY missing value
df.dropna(how='all')           # drop rows where ALL values are missing
df.dropna(subset=['age'])      # drop rows where 'age' is missing
df.dropna(thresh=3)            # keep rows with at least 3 non-null values
df.dropna(axis=1)              # drop columns with any missing value
```

### <span style="font-size: 14px;">When to Drop</span>

* <span style="font-size: 14px;">**Small fraction missing**: If less than 5% of rows have missing values, dropping is often acceptable.</span>
* <span style="font-size: 14px;">**Missing target variable**: In supervised learning, rows missing the target variable are useless and should be dropped.</span>
* <span style="font-size: 14px;">**Columns mostly empty**: A column with 90% missing data carries little information.</span>

### <span style="font-size: 14px;">When NOT to Drop</span>

* <span style="font-size: 14px;">**Systematic missingness**: If missing data correlates with important features, dropping introduces selection bias.</span>
* <span style="font-size: 14px;">**Small dataset**: Every observation matters; imputation is preferable.</span>

---

## <span style="font-size: 16px;">Filling Missing Values</span>

<span style="font-size: 14px;">`fillna()` replaces NaN with a specified value or strategy:</span>

### <span style="font-size: 14px;">Constant Fill</span>

```python
df['age'].fillna(0)              # fill with zero
df['name'].fillna('Unknown')     # fill with placeholder string
df.fillna({'age': 0, 'name': 'Unknown'})  # per-column fill values
```

### <span style="font-size: 14px;">Forward and Backward Fill</span>

```python
df['value'].fillna(method='ffill')  # propagate last valid value forward
df['value'].fillna(method='bfill')  # propagate next valid value backward
```

<span style="font-size: 14px;">Forward fill is appropriate for time series where the last known value is a reasonable estimate (e.g., stock prices, sensor readings). Backward fill is useful for filling initial missing values.</span>

### <span style="font-size: 14px;">Statistical Fill</span>

```python
df['age'].fillna(df['age'].mean())     # fill with column mean
df['age'].fillna(df['age'].median())   # fill with column median
df['city'].fillna(df['city'].mode()[0])  # fill with most frequent value
```

<span style="font-size: 14px;">Mean imputation is simple but biased: it preserves the mean but underestimates variance and destroys correlations. Median imputation is more robust to outliers. Mode imputation is appropriate for categorical data.</span>

### <span style="font-size: 14px;">Group-Based Fill</span>

```python
df['salary'] = df.groupby('department')['salary'].transform(
    lambda x: x.fillna(x.median())
)
```

<span style="font-size: 14px;">This fills missing salaries with the median salary of the same department. Group-based imputation is more accurate than global imputation because it respects the structure of the data.</span>

---

## <span style="font-size: 16px;">Interpolation</span>

<span style="font-size: 14px;">For ordered data (time series, spatial data), interpolation estimates missing values from neighbors:</span>

```python
df['temperature'].interpolate()                  # linear interpolation
df['temperature'].interpolate(method='time')     # time-weighted
df['temperature'].interpolate(method='polynomial', order=2)  # quadratic
```

<span style="font-size: 14px;">Linear interpolation draws a straight line between known points and estimates the missing value at the appropriate position. This is superior to forward fill for smoothly varying quantities.</span>

---

## <span style="font-size: 16px;">NaN Behavior in Aggregations</span>

<span style="font-size: 14px;">Pandas aggregation functions skip NaN by default:</span>

```python
pd.Series([1, 2, np.nan, 4]).mean()  # 2.333 (skips NaN)
pd.Series([1, 2, np.nan, 4]).sum()   # 7.0 (skips NaN)
```

<span style="font-size: 14px;">Pass `skipna=False` to propagate NaN:</span>

```python
pd.Series([1, 2, np.nan, 4]).mean(skipna=False)  # NaN
```

<span style="font-size: 14px;">This default behavior is usually desirable, but be aware that `sum()` silently ignoring missing values can produce misleading totals.</span>

---

## <span style="font-size: 16px;">NaN in Comparisons and Groupby</span>

* <span style="font-size: 14px;">`NaN == NaN` is `False` (IEEE 754 standard). Use `isna()` to check for NaN.</span>
* <span style="font-size: 14px;">`groupby()` excludes NaN keys by default. Pass `dropna=False` to include a group for NaN values.</span>
* <span style="font-size: 14px;">`value_counts()` excludes NaN by default. Pass `dropna=False` to count NaN occurrences.</span>

---

## <span style="font-size: 16px;">Common Pitfalls</span>

* <span style="font-size: 14px;">**Using == to check for NaN**: `x == np.nan` is always False. Use `pd.isna(x)` or `x is np.nan`.</span>
* <span style="font-size: 14px;">**Silent type conversion**: Adding NaN to an integer column converts it to float64. Use nullable integer types (`Int64`) to avoid this.</span>
* <span style="font-size: 14px;">**Filling before analyzing**: Always analyze the pattern of missingness before filling. Filling destroys information about why data was missing.</span>
* <span style="font-size: 14px;">**Mean imputation bias**: Filling with the mean underestimates variance and can distort correlations. For serious analysis, use multiple imputation or model-based approaches.</span>