# <span style="font-size: 20px;">Loc vs iLoc</span>

<span style="font-size: 14px;">The `loc` and `iloc` accessors are the two primary tools for selecting data from a DataFrame by position or label. While bracket notation (`df[...]`) handles simple cases, `loc` and `iloc` provide precise control over both row and column selection simultaneously. Understanding the difference between label-based and position-based indexing is fundamental to avoiding off-by-one errors, alignment bugs, and the dreaded `SettingWithCopyWarning`.</span>

---

## <span style="font-size: 16px;">iloc: Integer Position-Based Indexing</span>

<span style="font-size: 14px;">`iloc` selects data by integer position, exactly like Python list indexing and numpy array indexing:</span>

```python
df.iloc[0]           # first row (Series)
df.iloc[0, 0]        # first row, first column (scalar)
df.iloc[0:3]         # rows 0, 1, 2 (DataFrame)
df.iloc[0:3, 0:2]    # rows 0-2, columns 0-1 (DataFrame)
df.iloc[[0, 3, 5]]   # rows at positions 0, 3, 5
```

### <span style="font-size: 14px;">Key Properties of iloc</span>

* <span style="font-size: 14px;">**Zero-based**: The first row is position 0, regardless of the index labels.</span>
* <span style="font-size: 14px;">**Half-open slices**: `iloc[0:3]` includes positions 0, 1, 2 but NOT 3. This matches Python and numpy conventions.</span>
* <span style="font-size: 14px;">**Negative indexing**: `iloc[-1]` selects the last row, `iloc[-3:]` selects the last three rows.</span>
* <span style="font-size: 14px;">**Ignores labels**: `iloc` does not care about index or column labels. Position 0 is always the first row.</span>

---

## <span style="font-size: 16px;">loc: Label-Based Indexing</span>

<span style="font-size: 14px;">`loc` selects data by index labels and column names:</span>

```python
df.loc['row_a']                  # row with label 'row_a' (Series)
df.loc['row_a', 'col_x']        # specific cell (scalar)
df.loc['row_a':'row_c']         # rows from 'row_a' to 'row_c' inclusive
df.loc[:, 'col_x':'col_z']      # all rows, columns from 'col_x' to 'col_z'
df.loc[['row_a', 'row_c']]      # specific rows by label
```

### <span style="font-size: 14px;">Key Properties of loc</span>

* <span style="font-size: 14px;">**Inclusive slices**: `loc['a':'c']` includes both 'a' and 'c'. This is different from iloc and from Python convention.</span>
* <span style="font-size: 14px;">**Uses labels**: The row selector uses index labels, and the column selector uses column names.</span>
* <span style="font-size: 14px;">**Boolean masks**: `loc` accepts boolean Series for row selection: `df.loc[df['age'] > 30]`.</span>
* <span style="font-size: 14px;">**KeyError on missing labels**: If a label does not exist, `loc` raises `KeyError`.</span>

---

## <span style="font-size: 16px;">The Critical Difference: Inclusive vs. Exclusive Slicing</span>

<span style="font-size: 14px;">This is the most important distinction between loc and iloc:</span>

```python
df = pd.DataFrame({'val': [10, 20, 30, 40]}, index=['a', 'b', 'c', 'd'])

df.iloc[0:2]   # rows at positions 0, 1 (excludes 2)
#     val
# a    10
# b    20

df.loc['a':'c'] # rows labeled 'a' through 'c' (includes 'c')
#     val
# a    10
# b    20
# c    30
```

<span style="font-size: 14px;">The reason for the difference: label slicing must be inclusive because labels are not necessarily ordered or numeric. If you have labels ['mon', 'tue', 'wed'], there is no natural "next label after wed" to use as an exclusive endpoint.</span>

---

## <span style="font-size: 16px;">When Labels Are Integers</span>

<span style="font-size: 14px;">The most confusing case is when the index contains integers that do not start at 0:</span>

```python
df = pd.DataFrame({'val': [100, 200, 300]}, index=[10, 20, 30])

df.iloc[0]    # first row: val=100 (position 0)
df.loc[10]    # row with label 10: val=100

df.iloc[1:3]  # positions 1-2: rows with labels 20, 30
df.loc[10:20] # labels 10 through 20 inclusive: rows with labels 10, 20
```

<span style="font-size: 14px;">With `iloc`, the number is always a position. With `loc`, the number is always a label. After filtering a DataFrame, the integer index may have gaps (e.g., [0, 2, 5]), making `loc[0]` and `iloc[0]` return different rows.</span>

---

## <span style="font-size: 16px;">Assignment with loc and iloc</span>

<span style="font-size: 14px;">Both accessors support assignment, which is the correct way to modify DataFrame values:</span>

```python
df.loc[df['age'] > 30, 'status'] = 'senior'
df.iloc[0, 2] = 999
```

<span style="font-size: 14px;">This is critical because chained indexing (`df[mask]['col'] = val`) may modify a copy instead of the original. Using `loc` or `iloc` for assignment guarantees the modification happens in-place.</span>

---

## <span style="font-size: 16px;">Selecting Single Values</span>

<span style="font-size: 14px;">For accessing a single scalar value, pandas provides optimized accessors:</span>

```python
df.at['row_a', 'col_x']    # label-based, returns scalar
df.iat[0, 0]                # position-based, returns scalar
```

<span style="font-size: 14px;">`at` and `iat` are faster than `loc` and `iloc` for single-value access because they skip the overhead of constructing a Series or DataFrame wrapper.</span>

---

## <span style="font-size: 16px;">Combining Row and Column Selection</span>

<span style="font-size: 14px;">The full power of `loc` and `iloc` comes from selecting rows and columns simultaneously:</span>

```python
# loc: boolean rows + named columns
df.loc[df['age'] > 30, ['name', 'salary']]

# iloc: positional rows + positional columns
df.iloc[0:5, [0, 2, 4]]

# loc: labeled rows + column slice
df.loc['2020-01':'2020-06', 'revenue':'profit']
```

<span style="font-size: 14px;">This two-dimensional selection is something that bare bracket notation `df[...]` cannot do cleanly. Brackets only support row selection (with boolean masks or slices) or column selection (with strings or lists), but not both at once.</span>

---

## <span style="font-size: 16px;">Common Patterns</span>

### <span style="font-size: 14px;">Conditional Update</span>

```python
df.loc[df['score'] < 0, 'score'] = 0  # clip negative scores to 0
```

### <span style="font-size: 14px;">Select Last N Rows</span>

```python
df.iloc[-5:]  # last 5 rows
```

### <span style="font-size: 14px;">Select Every Nth Row</span>

```python
df.iloc[::3]  # every 3rd row
```

---

## <span style="font-size: 16px;">Common Pitfalls</span>

* <span style="font-size: 14px;">**Confusing loc and iloc on integer indices**: When the index contains integers, `loc[1]` means "label 1" and `iloc[1]` means "position 1". These may be different rows.</span>
* <span style="font-size: 14px;">**Inclusive vs. exclusive slicing**: `loc` slices are inclusive on both ends; `iloc` slices are half-open. Mixing these up causes off-by-one errors.</span>
* <span style="font-size: 14px;">**Chained indexing for assignment**: `df[mask]['col'] = val` is unreliable. Always use `df.loc[mask, 'col'] = val`.</span>
* <span style="font-size: 14px;">**Passing a list vs. a scalar**: `df.loc[['a']]` returns a DataFrame; `df.loc['a']` returns a Series. The extra brackets change the return type.</span>