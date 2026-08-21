# <span style="font-size: 20px;">Setting Index</span>

<span style="font-size: 14px;">The index is a fundamental part of a pandas DataFrame. It labels each row and drives alignment in operations between DataFrames. While the default integer index (`RangeIndex`) works for many tasks, setting a meaningful column as the index unlocks faster lookups, natural time-series operations, and automatic alignment in arithmetic. Understanding when and how to set, use, and manipulate the index is essential for efficient pandas work.</span>

---

## <span style="font-size: 16px;">set_index()</span>

<span style="font-size: 14px;">The primary method for setting a column as the index:</span>

```python
df = df.set_index('employee_id')
```

<span style="font-size: 14px;">The specified column is removed from the DataFrame's columns and becomes the row label. By default, this returns a new DataFrame. The original column is consumed: it no longer appears in `df.columns`.</span>

### <span style="font-size: 14px;">Multi-Level Index</span>

```python
df = df.set_index(['department', 'employee_id'])
```

<span style="font-size: 14px;">This creates a MultiIndex (hierarchical index) with two levels, enabling grouped lookups like `df.loc['Engineering']` to select all rows in that department.</span>

### <span style="font-size: 14px;">Preserving the Column</span>

```python
df = df.set_index('date', drop=False)  # keeps 'date' as both index and column
```

### <span style="font-size: 14px;">Appending to Existing Index</span>

```python
df = df.set_index('year', append=True)  # adds 'year' as a second index level
```

---

## <span style="font-size: 16px;">Why Use a Meaningful Index</span>

### <span style="font-size: 14px;">Fast Lookup</span>

<span style="font-size: 14px;">`loc`-based access on an index is $O(1)$ with a hash-based index (the default) versus $O(n)$ for a column scan:</span>

```python
# Fast: O(1) lookup
df = df.set_index('employee_id')
df.loc[12345]

# Slow: O(n) scan
df[df['employee_id'] == 12345]
```

### <span style="font-size: 14px;">Automatic Alignment</span>

<span style="font-size: 14px;">When you perform arithmetic between DataFrames, pandas aligns on the index:</span>

```python
revenue = df.set_index('date')['revenue']
cost = df.set_index('date')['cost']
profit = revenue - cost  # aligned by date automatically
```

### <span style="font-size: 14px;">Time-Series Operations</span>

<span style="font-size: 14px;">Setting a datetime column as the index enables time-based slicing:</span>

```python
df = df.set_index('date')
df.loc['2024-01':'2024-03']  # all rows in Q1 2024
df.resample('ME').mean()       # monthly averages
```

---

## <span style="font-size: 16px;">Index Properties</span>

```python
df.index          # the Index object
df.index.name     # name of the index
df.index.dtype    # data type
df.index.is_unique  # True if no duplicate labels
df.index.is_monotonic_increasing  # True if sorted ascending
```

<span style="font-size: 14px;">A unique, sorted index provides the best performance for lookups and slicing.</span>

---

## <span style="font-size: 16px;">Naming the Index</span>

```python
df.index.name = 'record_id'
# or
df = df.rename_axis('record_id')
```

<span style="font-size: 14px;">The index name appears in output and is preserved through operations. It is especially important for `reset_index()`, which uses the index name as the new column name.</span>

---

## <span style="font-size: 16px;">Sorting the Index</span>

```python
df = df.sort_index()              # ascending
df = df.sort_index(ascending=False)  # descending
```

<span style="font-size: 14px;">A sorted index enables efficient slice operations and is required for `merge_asof()` and some time-series methods.</span>

---

## <span style="font-size: 16px;">Verifying Index Uniqueness</span>

```python
if not df.index.is_unique:
    duplicates = df.index[df.index.duplicated()]
    print(f'Duplicate indices: {duplicates.tolist()}')
```

<span style="font-size: 14px;">Duplicate index values cause `loc` to return multiple rows instead of a single row, which can break downstream code that expects a scalar or Series.</span>

---

## <span style="font-size: 16px;">Common Pitfalls</span>

* <span style="font-size: 14px;">**Forgetting to reassign**: `df.set_index('col')` returns a new DataFrame. Without `df = df.set_index('col')` or `inplace=True`, the operation has no effect.</span>
* <span style="font-size: 14px;">**Duplicate index values**: Setting a non-unique column as the index creates duplicate labels. Use `verify_integrity=True` to catch this.</span>
* <span style="font-size: 14px;">**Lost column**: After `set_index('col')`, the column is no longer in `df.columns`. If you need it later, use `drop=False`.</span>
* <span style="font-size: 14px;">**Integer index confusion**: After setting a string index, `df.loc[0]` looks for the label 0 (which may not exist), not the first row. Use `iloc[0]` for positional access.</span>