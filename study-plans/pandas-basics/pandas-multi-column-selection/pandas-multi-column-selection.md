# <span style="font-size: 20px;">Multi-Column Selection</span>

<span style="font-size: 14px;">Selecting multiple columns at once is one of the most common operations in data wrangling. Whether you are preparing features for a model, extracting a subset for reporting, or dropping irrelevant columns, you need to select two or more columns from a DataFrame. Pandas provides several approaches, each suited to different situations.</span>

---

## <span style="font-size: 16px;">List-Based Selection</span>

<span style="font-size: 14px;">The most direct way to select multiple columns is to pass a list of column names inside brackets:</span>

```python
subset = df[['name', 'age', 'salary']]
```

<span style="font-size: 14px;">This always returns a DataFrame, even if the list contains only one name. The columns appear in the order specified in the list, which lets you reorder columns as a side effect:</span>

```python
df[['salary', 'name', 'age']]  # columns reordered
```

### <span style="font-size: 14px;">Dynamic Column Lists</span>

<span style="font-size: 14px;">Because the selection accepts any list, you can build the column list programmatically:</span>

```python
features = [col for col in df.columns if col.startswith('feat_')]
df[features]
```

<span style="font-size: 14px;">This is essential in machine learning pipelines where feature columns are generated dynamically.</span>

---

## <span style="font-size: 16px;">Using loc for Multi-Column Selection</span>

<span style="font-size: 14px;">The `loc` accessor supports simultaneous row and column selection:</span>

```python
df.loc[:, ['name', 'age']]          # all rows, specific columns
df.loc[0:5, ['name', 'age']]        # rows 0-5, specific columns
df.loc[df['age'] > 30, ['name']]    # filtered rows, single column
```

<span style="font-size: 14px;">The syntax is `df.loc[row_selector, column_selector]`. Using `:` for the row selector means "all rows." This is more explicit than bare bracket notation and avoids ambiguity.</span>

### <span style="font-size: 14px;">Column Slicing with loc</span>

<span style="font-size: 14px;">Unlike Python lists, `loc` supports label-based column slicing:</span>

```python
df.loc[:, 'name':'salary']  # all columns from 'name' to 'salary' inclusive
```

<span style="font-size: 14px;">Note that this is inclusive on both ends, unlike Python's usual half-open convention. The column order is determined by the DataFrame's column order.</span>

---

## <span style="font-size: 16px;">Using iloc for Positional Selection</span>

<span style="font-size: 14px;">When you know column positions but not names, use `iloc`:</span>

```python
df.iloc[:, [0, 2, 4]]    # columns at positions 0, 2, 4
df.iloc[:, 1:4]           # columns at positions 1, 2, 3 (half-open)
```

<span style="font-size: 14px;">`iloc` uses integer positions and follows Python's half-open slice convention. This is useful when processing files with no header or when column positions are more stable than names.</span>

---

## <span style="font-size: 16px;">Dropping Columns (Inverse Selection)</span>

<span style="font-size: 14px;">Sometimes it is easier to specify which columns to remove rather than which to keep:</span>

```python
df.drop(columns=['internal_id', 'debug_flag'])
```

<span style="font-size: 14px;">`drop` returns a new DataFrame by default (`inplace=False`). You can also use `df.drop(['col1', 'col2'], axis=1)`, but the `columns=` keyword is more readable.</span>

### <span style="font-size: 14px;">Set-Based Column Selection</span>

<span style="font-size: 14px;">For complex filtering, use set operations on `df.columns`:</span>

```python
keep = set(df.columns) - {'internal_id', 'debug_flag'}
df[list(keep)]
```

---

## <span style="font-size: 16px;">select_dtypes() for Type-Based Selection</span>

<span style="font-size: 14px;">Select columns by data type rather than name:</span>

```python
numeric = df.select_dtypes(include=['int64', 'float64'])
categorical = df.select_dtypes(include=['object', 'category'])
```

<span style="font-size: 14px;">This is the idiomatic way to separate numeric and categorical features for preprocessing. The `include` and `exclude` parameters accept type strings or numpy types.</span>

---

## <span style="font-size: 16px;">filter() for Pattern-Based Selection</span>

<span style="font-size: 14px;">The `filter()` method selects columns by name patterns:</span>

```python
df.filter(like='price')       # contains 'price'
df.filter(regex='^feature_')  # starts with 'feature_'
df.filter(regex='_\d+$')     # ends with one or more digits
```

<span style="font-size: 14px;">Regular expressions make this very powerful for datasets with systematic naming conventions.</span>

---

## <span style="font-size: 16px;">Column Reindexing</span>

<span style="font-size: 14px;">The `reindex()` method selects and reorders columns, optionally filling missing columns with a default value:</span>

```python
df.reindex(columns=['a', 'b', 'c', 'new_col'], fill_value=0)
```

<span style="font-size: 14px;">If 'new_col' does not exist in the original DataFrame, it is created with the fill value. This is useful for ensuring a consistent schema across multiple DataFrames.</span>

---

## <span style="font-size: 16px;">Performance Considerations</span>

<span style="font-size: 14px;">Multi-column selection creates a new DataFrame that references (not copies) the underlying data blocks. This means:</span>

* <span style="font-size: 14px;">**Selection is fast**: Selecting 5 columns from a 100-column DataFrame does not copy the data.</span>
* <span style="font-size: 14px;">**Modification triggers copy-on-write**: If you modify the selected DataFrame, pandas creates a copy of the affected blocks.</span>
* <span style="font-size: 14px;">**Contiguous columns are faster**: Selecting columns that are stored contiguously in memory (e.g., all float64 columns) is slightly faster than selecting scattered columns because fewer blocks need to be referenced.</span>

---

## <span style="font-size: 16px;">Common Pitfalls</span>

* <span style="font-size: 14px;">**Passing a tuple instead of a list**: `df[('a', 'b')]` is interpreted as a single MultiIndex key, not two columns. Always use a list: `df[['a', 'b']]`.</span>
* <span style="font-size: 14px;">**KeyError for missing columns**: If any column in the list does not exist, the entire operation fails with `KeyError`. Validate column names first.</span>
* <span style="font-size: 14px;">**Column order assumptions**: The result follows the order of the list you pass, not the original column order. Be explicit about ordering.</span>
* <span style="font-size: 14px;">**Duplicate column names**: If the DataFrame has duplicate column names (which pandas allows), selecting by name returns all matching columns, which can be surprising.</span>