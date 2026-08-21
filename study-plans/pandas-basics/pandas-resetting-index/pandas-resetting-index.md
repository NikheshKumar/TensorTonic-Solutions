# <span style="font-size: 20px;">Resetting Index</span>

<span style="font-size: 14px;">`reset_index()` is the inverse of `set_index()`: it moves the current index back into a regular column and replaces it with a default integer index. This operation is essential after GroupBy aggregations (which set the group keys as the index), after filtering (which leaves gaps in the integer index), and when preparing data for export to formats that expect sequential row numbering.</span>

---

## <span style="font-size: 16px;">Basic Usage</span>

```python
df = df.reset_index()
```

<span style="font-size: 14px;">This moves the current index into a new column (using the index's name) and creates a fresh `RangeIndex`. If the index had no name, the column is called "index".</span>

### <span style="font-size: 14px;">Dropping the Index</span>

```python
df = df.reset_index(drop=True)
```

<span style="font-size: 14px;">With `drop=True`, the old index is discarded entirely instead of being converted to a column. This is the most common usage after filtering, where you want a clean sequential index without preserving the old row numbers.</span>

---

## <span style="font-size: 16px;">After GroupBy Operations</span>

<span style="font-size: 14px;">GroupBy aggregations set the group keys as the index by default:</span>

```python
result = df.groupby('dept')['salary'].mean()
# dept
# Engineering    85000
# Marketing      72000
# Name: salary, dtype: float64
```

<span style="font-size: 14px;">To get the department back as a regular column:</span>

```python
result = result.reset_index()
#          dept  salary
# 0  Engineering   85000
# 1    Marketing   72000
```

<span style="font-size: 14px;">Alternatively, use `as_index=False` in the groupby call to avoid the issue entirely.</span>

---

## <span style="font-size: 16px;">Resetting MultiIndex</span>

<span style="font-size: 14px;">With a MultiIndex, you can reset all levels or specific ones:</span>

```python
df.reset_index()              # reset all levels
df.reset_index(level='dept')  # reset only the 'dept' level
df.reset_index(level=0)       # reset the first level
```

<span style="font-size: 14px;">Resetting a specific level is useful when you want to keep one level as the index while converting another back to a column.</span>

---

## <span style="font-size: 16px;">The set_index / reset_index Round-Trip</span>

<span style="font-size: 14px;">A common pattern is to set an index for an operation, then reset it afterward:</span>

```python
# Set date as index for time-based operations
df = df.set_index('date')
monthly = df.resample('ME').mean()

# Reset for further processing
monthly = monthly.reset_index()
```

<span style="font-size: 14px;">This round-trip is lossless: the column that was set as the index is restored exactly as it was (with the same name and type).</span>

---

## <span style="font-size: 16px;">After Filtering</span>

<span style="font-size: 14px;">Boolean filtering preserves the original index, creating gaps:</span>

```python
df = pd.DataFrame({'val': [10, 20, 30, 40, 50]})
filtered = df[df['val'] > 20]
print(filtered.index)  # [2, 3, 4] - gaps at 0, 1
```

<span style="font-size: 14px;">If downstream code assumes contiguous integer indices (e.g., `for i in range(len(df)): df.iloc[i]`), reset the index:</span>

```python
filtered = filtered.reset_index(drop=True)
print(filtered.index)  # [0, 1, 2]
```

---

## <span style="font-size: 16px;">Column Naming</span>

<span style="font-size: 14px;">When the index has a name, `reset_index()` uses that name for the new column:</span>

```python
df.index.name = 'record_id'
df.reset_index()  # creates column 'record_id'
```

<span style="font-size: 14px;">When the index has no name, the column is called "index", which can conflict with the `index` attribute. Rename it immediately:</span>

```python
df = df.reset_index().rename(columns={'index': 'original_position'})
```

---

## <span style="font-size: 16px;">Common Pitfalls</span>

* <span style="font-size: 14px;">**Forgetting to reassign**: Like most pandas methods, `reset_index()` returns a new DataFrame. Without reassignment, the original is unchanged.</span>
* <span style="font-size: 14px;">**Unnamed index conflict**: If the index has no name, the resulting column is called "index", which can cause confusion with the `.index` attribute.</span>
* <span style="font-size: 14px;">**Resetting when you need the index**: If you are about to merge on the index, do not reset it first - use `left_index=True` in the merge instead.</span>
* <span style="font-size: 14px;">**Double reset**: Calling `reset_index()` twice adds an unwanted "index" column on the second call. Check whether the index is already a RangeIndex before resetting.</span>