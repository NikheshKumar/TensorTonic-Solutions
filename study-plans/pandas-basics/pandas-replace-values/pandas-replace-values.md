# <span style="font-size: 20px;">Replace Values</span>

<span style="font-size: 14px;">Replacing values is a core data cleaning operation. Raw datasets contain typos, legacy codes, inconsistent formats, and placeholder values that must be standardized before analysis. Pandas provides the `replace()` method with multiple modes for handling everything from simple one-to-one substitutions to complex regex-based transformations.</span>

---

## <span style="font-size: 16px;">Basic Replacement</span>

<span style="font-size: 14px;">The simplest form replaces a single value across the entire DataFrame:</span>

```python
df.replace('N/A', np.nan)         # replace 'N/A' with NaN everywhere
df.replace(0, np.nan)             # replace 0 with NaN everywhere
df['col'].replace('old', 'new')   # replace in a single column
```

<span style="font-size: 14px;">`replace()` returns a new DataFrame by default. The replacement is exact: `'N/A'` matches only the exact string "N/A", not "N/A value" or "not N/A".</span>

---

## <span style="font-size: 16px;">Dictionary-Based Replacement</span>

<span style="font-size: 14px;">For multiple substitutions, pass a dictionary:</span>

```python
df.replace({'M': 'Male', 'F': 'Female', 'U': 'Unknown'})
```

<span style="font-size: 14px;">This scans all columns and replaces any matching values. To restrict replacement to specific columns, either select the column first or pass a nested dictionary:</span>

```python
df.replace({'gender': {'M': 'Male', 'F': 'Female'}})
```

<span style="font-size: 14px;">The outer key is the column name; the inner dictionary specifies the replacements for that column only.</span>

---

## <span style="font-size: 16px;">List-Based Replacement</span>

<span style="font-size: 14px;">Replace multiple values with a single value:</span>

```python
df.replace(['N/A', 'NA', 'missing', '-'], np.nan)
```

<span style="font-size: 14px;">Or replace multiple values with corresponding new values:</span>

```python
df.replace(['Mon', 'Tue', 'Wed'], ['Monday', 'Tuesday', 'Wednesday'])
```

<span style="font-size: 14px;">The lists must be the same length, and replacements are positional: the first element of the first list maps to the first element of the second list.</span>

---

## <span style="font-size: 16px;">Regex Replacement</span>

<span style="font-size: 14px;">For pattern-based replacement, enable regex mode:</span>

```python
df.replace(r'^\s+$', np.nan, regex=True)      # whitespace-only strings -> NaN
df.replace(r'\d+', 'NUM', regex=True)          # any digits -> 'NUM'
df['phone'].replace(r'[^0-9]', '', regex=True) # strip non-digits
```

<span style="font-size: 14px;">Regex replacement uses Python's `re` module. The `regex=True` flag must be set explicitly; without it, the pattern is treated as a literal string.</span>

### <span style="font-size: 14px;">Capture Groups</span>

<span style="font-size: 14px;">You can use capture groups for complex transformations:</span>

```python
# Convert 'YYYY-MM-DD' to 'MM/DD/YYYY'
df['date'].replace(r'(\d{4})-(\d{2})-(\d{2})', r'\2/\3/\1', regex=True)
```

---

## <span style="font-size: 16px;">Method Parameter</span>

<span style="font-size: 14px;">The `method` parameter enables forward-fill or back-fill replacement:</span>

```python
# Replace NaN with the previous valid value
df.replace(np.nan, method='ffill')

# Replace NaN with the next valid value
df.replace(np.nan, method='bfill')
```

<span style="font-size: 14px;">Note: for NaN-specific filling, `df.fillna(method='ffill')` is more idiomatic. The method parameter in `replace()` is more useful for replacing specific sentinel values with surrounding data.</span>

---

## <span style="font-size: 16px;">Replacing with a Function (map/apply)</span>

<span style="font-size: 14px;">For transformations that `replace()` cannot express, use `map()` or `apply()`:</span>

```python
# Conditional replacement with a function
df['grade'] = df['score'].map(lambda x: 'A' if x >= 90 else 'B' if x >= 80 else 'C')

# Dictionary-based mapping (NaN for unmatched)
df['code'] = df['code'].map({'A1': 'Alpha', 'B2': 'Beta'})
```

<span style="font-size: 14px;">`map()` works on Series and replaces values using a function or dictionary. Unlike `replace()`, unmatched values become NaN with dictionary-based `map()`.</span>

---

## <span style="font-size: 16px;">where() and mask()</span>

<span style="font-size: 14px;">For conditional replacement based on boolean conditions:</span>

```python
# Keep values where condition is True, replace others with 0
df['score'].where(df['score'] > 0, 0)

# Replace values where condition is True with -1
df['score'].mask(df['score'] < 0, -1)
```

<span style="font-size: 14px;">`where()` keeps values when the condition is True; `mask()` replaces values when the condition is True. They are inverses of each other.</span>

---

## <span style="font-size: 16px;">clip() for Bounding Values</span>

<span style="font-size: 14px;">To replace extreme values with bounds:</span>

```python
df['score'].clip(lower=0, upper=100)  # clamp to [0, 100]
```

<span style="font-size: 14px;">This is more readable and efficient than using `where()` or `replace()` for range-based clamping.</span>

---

## <span style="font-size: 16px;">Performance Considerations</span>

* <span style="font-size: 14px;">**Simple replace**: Fast, operates on the internal numpy arrays directly.</span>
* <span style="font-size: 14px;">**Regex replace**: Slower, because each value must be matched against the pattern using Python's regex engine.</span>
* <span style="font-size: 14px;">**map with dict**: Very fast for exact matches, uses hash-based lookup.</span>
* <span style="font-size: 14px;">**apply with lambda**: Slowest, because it calls a Python function per element. Use only when vectorized alternatives are not available.</span>

---

## <span style="font-size: 16px;">Common Pitfalls</span>

* <span style="font-size: 14px;">**Forgetting to reassign**: `df.replace(...)` returns a new DataFrame. Without `df = df.replace(...)` or `inplace=True`, the replacement is lost.</span>
* <span style="font-size: 14px;">**Type mismatch**: Replacing integer 0 does not affect the string "0". Ensure the value type matches the column type.</span>
* <span style="font-size: 14px;">**Regex without the flag**: `df.replace(r'\d+', 'X')` looks for the literal string "\d+" unless `regex=True` is set.</span>
* <span style="font-size: 14px;">**Chained replacement order**: When replacing A->B and B->C in the same call, pandas applies all replacements simultaneously (not sequentially), so A becomes B, not C.</span>