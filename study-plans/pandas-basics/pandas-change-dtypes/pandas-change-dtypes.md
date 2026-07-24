# <span style="font-size: 20px;">Change Data Types</span>

<span style="font-size: 14px;">Data type conversion is one of the most critical preprocessing steps in any pandas workflow. Columns loaded from CSV files, databases, or APIs frequently have incorrect types: numeric IDs stored as strings, dates stored as objects, and boolean flags stored as integers. Converting to the correct type unlocks type-specific operations, reduces memory usage, and prevents subtle bugs where string comparison replaces numeric comparison.</span>

---

## <span style="font-size: 16px;">The astype() Method</span>

<span style="font-size: 14px;">`astype()` is the primary method for explicit type conversion:</span>

```python
df['age'] = df['age'].astype(int)
df['salary'] = df['salary'].astype(float)
df['name'] = df['name'].astype(str)
df['flag'] = df['flag'].astype(bool)
df['category'] = df['category'].astype('category')
```

<span style="font-size: 14px;">It also accepts numpy types, pandas extension types, and string aliases:</span>

```python
df['id'].astype('int32')           # 32-bit integer
df['value'].astype('float32')      # 32-bit float
df['flag'].astype('boolean')       # pandas nullable boolean
df['count'].astype('Int64')        # pandas nullable integer (note capital I)
```

### <span style="font-size: 14px;">Converting Multiple Columns</span>

<span style="font-size: 14px;">Pass a dictionary to convert multiple columns at once:</span>

```python
df = df.astype({'age': 'int32', 'salary': 'float32', 'name': 'category'})
```

---

## <span style="font-size: 16px;">Handling Conversion Errors</span>

<span style="font-size: 14px;">`astype()` raises a `ValueError` if any value cannot be converted:</span>

```python
pd.Series(['1', '2', 'three']).astype(int)
# ValueError: invalid literal for int() with base 10: 'three'
```

<span style="font-size: 14px;">For dirty data, use the specialized conversion functions with error handling:</span>

```python
pd.to_numeric(df['col'], errors='coerce')    # unparseable -> NaN
pd.to_numeric(df['col'], errors='ignore')    # unparseable -> keep original
pd.to_datetime(df['date'], errors='coerce')  # unparseable -> NaT
```

<span style="font-size: 14px;">`errors='coerce'` is the most useful option: it converts what it can and marks failures as NaN/NaT, letting you detect and handle bad values after conversion.</span>

---

## <span style="font-size: 16px;">String to Numeric</span>

<span style="font-size: 14px;">Converting strings to numbers is the most common type conversion:</span>

```python
# Clean before converting
df['price'] = df['price'].str.replace('$', '').str.replace(',', '')
df['price'] = pd.to_numeric(df['price'])

# Or in one step with regex
df['price'] = df['price'].replace(r'[\$,]', '', regex=True).astype(float)
```

<span style="font-size: 14px;">Common issues include currency symbols, thousands separators, percentage signs, and whitespace. Always clean the string before converting.</span>

---

## <span style="font-size: 16px;">String to Datetime</span>

<span style="font-size: 14px;">Date conversion is essential for time-based analysis:</span>

```python
df['date'] = pd.to_datetime(df['date'])                    # auto-detect format
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d') # explicit format
df['date'] = pd.to_datetime(df['date'], dayfirst=True)     # DD/MM/YYYY
```

<span style="font-size: 14px;">Specifying the format with `format=` is much faster than auto-detection for large datasets, because the parser does not need to guess the format for each value.</span>

### <span style="font-size: 14px;">Extracting Components</span>

<span style="font-size: 14px;">Once converted to datetime, you can extract components:</span>

```python
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['dayofweek'] = df['date'].dt.dayofweek  # 0=Monday
```

---

## <span style="font-size: 16px;">Numeric to Category</span>

<span style="font-size: 14px;">Converting low-cardinality columns to category type saves memory and speeds up groupby:</span>

```python
df['status'] = df['status'].astype('category')
```

<span style="font-size: 14px;">Memory savings can be dramatic. A column with 10 million rows and 5 unique string values:</span>

* <span style="font-size: 14px;">As object: ~500 MB (each row stores a full Python string reference)</span>
* <span style="font-size: 14px;">As category: ~10 MB (stores 5 unique strings + integer codes)</span>

### <span style="font-size: 14px;">Ordered Categories</span>

<span style="font-size: 14px;">For ordinal data (like size: S, M, L, XL), create an ordered category:</span>

```python
from pandas.api.types import CategoricalDtype
cat_type = CategoricalDtype(categories=['S', 'M', 'L', 'XL'], ordered=True)
df['size'] = df['size'].astype(cat_type)

df[df['size'] > 'M']  # works because category is ordered
```

---

## <span style="font-size: 16px;">Downcasting for Memory Optimization</span>

<span style="font-size: 14px;">`pd.to_numeric()` with `downcast` automatically selects the smallest type:</span>

```python
df['count'] = pd.to_numeric(df['count'], downcast='integer')
# int64 (8 bytes) -> int8 (1 byte) if values fit in [-128, 127]

df['score'] = pd.to_numeric(df['score'], downcast='float')
# float64 (8 bytes) -> float32 (4 bytes)
```

<span style="font-size: 14px;">Be cautious with downcasting: if future data exceeds the smaller type's range, values will overflow silently.</span>

---

## <span style="font-size: 16px;">The convert_dtypes() Method</span>

<span style="font-size: 14px;">Pandas provides an automatic conversion method that selects the best nullable types:</span>

```python
df = df.convert_dtypes()
```

<span style="font-size: 14px;">This converts:</span>
* <span style="font-size: 14px;">Object columns with all integers to `Int64` (nullable)</span>
* <span style="font-size: 14px;">Object columns with all strings to `StringDtype`</span>
* <span style="font-size: 14px;">Object columns with True/False to `BooleanDtype`</span>

<span style="font-size: 14px;">This is a good starting point, but always verify the results - automatic inference is not always correct.</span>

---

## <span style="font-size: 16px;">Common Pitfalls</span>

* <span style="font-size: 14px;">**NaN in integer columns**: Standard int64 cannot hold NaN. Converting a float column with NaN to int raises an error. Use `Int64` (nullable) or fill NaN first.</span>
* <span style="font-size: 14px;">**Boolean conversion surprises**: `astype(bool)` treats 0 as False and all other numbers as True. The string "False" converts to True because it is a non-empty string.</span>
* <span style="font-size: 14px;">**Loss of precision**: Converting float64 to float32 loses precision. Values like 1.0000001 and 1.0 may become identical.</span>
* <span style="font-size: 14px;">**Category with unknown values**: If new data contains values not in the category, they become NaN. Use `cat.add_categories()` first.</span>