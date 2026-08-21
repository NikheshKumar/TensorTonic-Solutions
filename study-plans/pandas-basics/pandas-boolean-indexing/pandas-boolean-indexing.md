# <span style="font-size: 20px;">Boolean Indexing</span>

<span style="font-size: 14px;">Boolean indexing is the primary mechanism for filtering rows in pandas. Instead of writing loops to check each row against a condition, you create a boolean mask - a Series of True/False values - and use it to select only the rows where the condition holds. This is both more concise and dramatically faster than iterative approaches because the comparison runs in vectorized C code under the hood.</span>

---

## <span style="font-size: 16px;">How Boolean Indexing Works</span>

<span style="font-size: 14px;">The process has two steps:</span>

<span style="font-size: 14px;">**Step 1**: Create a boolean mask by comparing a column to a value:</span>

```python
mask = df['age'] > 30
# 0    False
# 1    True
# 2    True
# 3    False
# dtype: bool
```

<span style="font-size: 14px;">**Step 2**: Use the mask to index the DataFrame:</span>

```python
filtered = df[mask]
```

<span style="font-size: 14px;">Pandas keeps every row where the mask is `True` and discards rows where it is `False`. The result is a new DataFrame with the same columns but fewer rows.</span>

### <span style="font-size: 14px;">The Vectorized Comparison</span>

<span style="font-size: 14px;">When you write `df['age'] > 30`, pandas broadcasts the scalar 30 across every element of the 'age' Series and performs the comparison in a single vectorized operation. For a column with $n$ elements, this runs in $O(n)$ time with C-level performance, whereas a Python loop would add interpreter overhead per element.</span>

---

## <span style="font-size: 16px;">Comparison Operators</span>

<span style="font-size: 14px;">All standard comparison operators produce boolean masks:</span>

| Operator | Meaning | Example |
|----------|---------|---------|
| `==` | Equal | `df['city'] == 'NYC'` |
| `!=` | Not equal | `df['status'] != 'inactive'` |
| `>` | Greater than | `df['salary'] > 50000` |
| `>=` | Greater or equal | `df['age'] >= 18` |
| `<` | Less than | `df['score'] < 0.5` |
| `<=` | Less or equal | `df['rank'] <= 10` |

<span style="font-size: 14px;">String comparisons use lexicographic ordering with `==` and `!=`. For pattern matching, use `.str.contains()` or `.str.startswith()`.</span>

---

## <span style="font-size: 16px;">Combining Conditions</span>

<span style="font-size: 14px;">Multiple conditions are combined using bitwise operators, not Python's `and`/`or` keywords:</span>

```python
# AND: both conditions must be true
df[(df['age'] > 30) & (df['salary'] > 50000)]

# OR: either condition must be true
df[(df['age'] > 30) | (df['city'] == 'NYC')]

# NOT: invert the condition
df[~(df['status'] == 'inactive')]
```

### <span style="font-size: 14px;">Why Parentheses Are Required</span>

<span style="font-size: 14px;">Parentheses around each condition are mandatory because Python's operator precedence gives `\&` and `|` higher priority than comparison operators. Without parentheses:</span>

```python
# WRONG: parsed as df['age'] > (30 & df['salary']) > 50000
df[df['age'] > 30 & df['salary'] > 50000]

# CORRECT: each comparison is evaluated first, then combined
df[(df['age'] > 30) & (df['salary'] > 50000)]
```

<span style="font-size: 14px;">Forgetting parentheses is one of the most common pandas bugs. The error message is often confusing ("The truth value of a Series is ambiguous") because it occurs when Python tries to use `and` on two Series instead of `\&`.</span>

---

## <span style="font-size: 16px;">The isin() Method</span>

<span style="font-size: 14px;">For checking membership in a set of values, use `isin()` instead of chaining `==` with `|`:</span>

```python
# Instead of:
df[(df['city'] == 'NYC') | (df['city'] == 'LA') | (df['city'] == 'Chicago')]

# Use:
df[df['city'].isin(['NYC', 'LA', 'Chicago'])]
```

<span style="font-size: 14px;">`isin()` is more readable and slightly faster because it uses hash-based lookup internally.</span>

---

## <span style="font-size: 16px;">The between() Method</span>

<span style="font-size: 14px;">For range-based filtering:</span>

```python
# Instead of:
df[(df['age'] >= 25) & (df['age'] <= 35)]

# Use:
df[df['age'].between(25, 35)]  # inclusive on both ends by default
```

<span style="font-size: 14px;">The `inclusive` parameter controls which bounds are included: `'both'` (default), `'left'`, `'right'`, or `'neither'`.</span>

---

## <span style="font-size: 16px;">The query() Method</span>

<span style="font-size: 14px;">For complex boolean expressions, `query()` provides a string-based syntax that avoids repetitive `df['col']` references:</span>

```python
# Standard boolean indexing:
df[(df['age'] > 30) & (df['salary'] > 50000) & (df['city'] != 'LA')]

# Equivalent query:
df.query('age > 30 and salary > 50000 and city != "LA"')
```

<span style="font-size: 14px;">Query uses Python-like syntax with `and`, `or`, `not` instead of `\&`, `|`, `\~`. It can also reference local variables with the `@` prefix:</span>

```python
threshold = 50000
df.query('salary > @threshold')
```

---

## <span style="font-size: 16px;">Handling NaN in Boolean Indexing</span>

<span style="font-size: 14px;">NaN values in comparisons always produce `False`:</span>

```python
s = pd.Series([1, 2, None, 4])
s > 1  # [False, True, False, True] - NaN row is False
```

<span style="font-size: 14px;">This means NaN rows are silently excluded by any comparison filter. To explicitly include or handle NaN, use `isna()` and `notna()`:</span>

```python
df[df['age'].isna()]     # rows where age is missing
df[df['age'].notna()]    # rows where age is not missing
df[(df['age'] > 30) | df['age'].isna()]  # include NaN rows
```

---

## <span style="font-size: 16px;">Boolean Indexing with loc</span>

<span style="font-size: 14px;">Combining row filtering with column selection:</span>

```python
df.loc[df['age'] > 30, 'salary']            # Series of salaries where age > 30
df.loc[df['age'] > 30, ['name', 'salary']]  # DataFrame with name and salary
```

<span style="font-size: 14px;">This is the most precise way to select a subset: specific rows and specific columns in one expression.</span>

---

## <span style="font-size: 16px;">Performance</span>

<span style="font-size: 14px;">Boolean indexing is highly optimized in pandas:</span>

* <span style="font-size: 14px;">**Vectorized comparisons**: Run at C speed, not Python interpreter speed</span>
* <span style="font-size: 14px;">**Memory**: The boolean mask uses 1 byte per element. For 10 million rows, the mask is only 10 MB.</span>
* <span style="font-size: 14px;">**query() with numexpr**: If the `numexpr` library is installed, `query()` compiles expressions to native code for an additional speedup on large DataFrames.</span>

---

## <span style="font-size: 16px;">Common Pitfalls</span>

* <span style="font-size: 14px;">**Using `and`/`or` instead of `&`/`|`**: Python's logical operators do not work element-wise on Series. Use bitwise operators for combining masks.</span>
* <span style="font-size: 14px;">**Missing parentheses**: Operator precedence of `\&` is higher than `>`, so parentheses are mandatory around each condition.</span>
* <span style="font-size: 14px;">**Silent NaN exclusion**: Comparison with NaN always returns False, so filtered results silently drop missing rows.</span>
* <span style="font-size: 14px;">**Chained indexing**: `df[mask]['col'] = val` modifies a copy, not the original. Use `df.loc[mask, 'col'] = val` instead.</span>