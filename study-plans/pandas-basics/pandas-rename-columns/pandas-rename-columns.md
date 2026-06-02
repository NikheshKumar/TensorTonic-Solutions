# <span style="font-size: 20px;">Rename Columns</span>

<span style="font-size: 14px;">Column renaming is a routine step in data cleaning. Raw data often arrives with cryptic abbreviations, inconsistent casing, or names that conflict with Python keywords. Pandas provides several methods for renaming, each suited to different scenarios: renaming a few specific columns, applying a transformation to all names, or replacing the entire column index at once.</span>

---

## <span style="font-size: 16px;">The rename() Method</span>

<span style="font-size: 14px;">The most common approach is $\texttt{df.rename(columns=\{...\})}$, which accepts a dictionary mapping old names to new names:</span>

```python
df = df.rename(columns={'old_name': 'new_name', 'emp_id': 'employee_id'})
```

<span style="font-size: 14px;">Only the columns present in the dictionary are renamed; all others remain unchanged. By default, `rename()` returns a new DataFrame. Pass `inplace=True` to modify the original (though this pattern is discouraged in modern pandas style).</span>

### <span style="font-size: 14px;">Using a Function</span>

<span style="font-size: 14px;">When you need to apply the same transformation to all column names, pass a function instead of a dictionary:</span>

```python
df.rename(columns=str.lower)         # all lowercase
df.rename(columns=str.upper)         # all uppercase
df.rename(columns=str.strip)         # remove whitespace
df.rename(columns=lambda x: x.replace(' ', '_'))  # spaces to underscores
```

<span style="font-size: 14px;">The function receives each column name as a string and returns the new name. This is the cleanest way to normalize column naming conventions across an entire dataset.</span>

---

## <span style="font-size: 16px;">Direct Column Assignment</span>

<span style="font-size: 14px;">You can replace the entire column index by assigning a new list:</span>

```python
df.columns = ['id', 'full_name', 'annual_salary']
```

<span style="font-size: 14px;">The list must have exactly the same length as the current number of columns, or pandas raises a `ValueError`. This approach is useful when column names need a complete overhaul, such as when loading headerless files.</span>

---

## <span style="font-size: 16px;">The set_axis() Method</span>

<span style="font-size: 14px;">`set_axis()` is a functional alternative to direct assignment:</span>

```python
df = df.set_axis(['id', 'name', 'salary'], axis=1)
```

<span style="font-size: 14px;">Unlike direct assignment, this returns a new DataFrame and can be chained with other operations.</span>

---

## <span style="font-size: 16px;">add_prefix() and add_suffix()</span>

<span style="font-size: 14px;">These convenience methods prepend or append a string to all column names:</span>

```python
df.add_prefix('raw_')    # 'name' -> 'raw_name'
df.add_suffix('_2024')   # 'name' -> 'name_2024'
```

<span style="font-size: 14px;">This is useful when merging DataFrames from different sources and you need to distinguish identically-named columns.</span>

---

## <span style="font-size: 16px;">Renaming with str Accessor</span>

<span style="font-size: 14px;">Since `df.columns` is an Index object, it supports string operations via the `.str` accessor:</span>

```python
df.columns = df.columns.str.lower()
df.columns = df.columns.str.replace(' ', '_')
df.columns = df.columns.str.strip()
df.columns = df.columns.str.replace(r'[^a-z0-9_]', '', regex=True)
```

<span style="font-size: 14px;">The last example removes all non-alphanumeric characters, which is useful for sanitizing column names for SQL compatibility or machine learning frameworks that require simple names.</span>

---

## <span style="font-size: 16px;">Renaming Index Labels</span>

<span style="font-size: 14px;">`rename()` also supports renaming row index labels:</span>

```python
df.rename(index={0: 'first', 1: 'second'})
df.rename(index=str)  # convert numeric index to string
```

<span style="font-size: 14px;">You can rename both the index name (the label of the index itself) and its values:</span>

```python
df.index.name = 'record_id'
df.rename_axis('record_id')  # same effect, chainable
```

---

## <span style="font-size: 16px;">Handling Duplicate Column Names</span>

<span style="font-size: 14px;">Pandas allows duplicate column names (unlike SQL tables), but they cause problems:</span>

```python
df = pd.DataFrame([[1, 2]], columns=['a', 'a'])
df['a']  # returns BOTH columns as a DataFrame, not a Series
```

<span style="font-size: 14px;">Renaming is essential for resolving duplicates:</span>

```python
cols = df.columns.tolist()
seen = {}
for i, col in enumerate(cols):
    if col in seen:
        seen[col] += 1
        cols[i] = f'{col}_{seen[col]}'
    else:
        seen[col] = 0
df.columns = cols
```

---

## <span style="font-size: 16px;">Method Chaining</span>

<span style="font-size: 14px;">`rename()` integrates cleanly into method chains:</span>

```python
result = (
    df
    .rename(columns={'emp_id': 'id', 'emp_name': 'name'})
    .query('name != "Unknown"')
    .set_index('id')
    .sort_values('salary', ascending=False)
)
```

<span style="font-size: 14px;">This pattern is preferred in modern pandas code because it keeps transformations readable and each step produces a new DataFrame without side effects.</span>

---

## <span style="font-size: 16px;">Naming Conventions</span>

<span style="font-size: 14px;">Good column naming practices for data science:</span>

* <span style="font-size: 14px;">**snake_case**: `annual_salary` instead of `Annual Salary` or `annualSalary`. Snake case is the Python convention and avoids quoting issues.</span>
* <span style="font-size: 14px;">**No spaces**: Spaces require bracket notation (`df['My Col']`) and break dot access.</span>
* <span style="font-size: 14px;">**Lowercase**: Avoids case-sensitivity bugs in joins and groupbys.</span>
* <span style="font-size: 14px;">**Descriptive**: `purchase_date` over `pd` (which shadows the pandas import).</span>

---

## <span style="font-size: 16px;">Common Pitfalls</span>

* <span style="font-size: 14px;">**Forgetting rename returns a new DataFrame**: Without `inplace=True` or reassignment, the rename has no effect.</span>
* <span style="font-size: 14px;">**Misspelling old names in the dict**: If the old name does not match any column, `rename()` silently ignores it. Pass `errors='raise'` to catch typos.</span>
* <span style="font-size: 14px;">**Assigning wrong-length column list**: `df.columns = [...]` requires exactly the right number of names.</span>
* <span style="font-size: 14px;">**Renaming to an existing name**: This creates duplicate column names, which break many pandas operations.</span>