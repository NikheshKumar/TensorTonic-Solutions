# <span style="font-size: 20px;">Drop Duplicates</span>

<span style="font-size: 14px;">Duplicate rows appear in datasets for many reasons: database replication delays, overlapping data extracts, retry logic in data pipelines, user-submitted data with accidental resubmissions, or merge operations that produce cartesian products. Identifying and removing duplicates is a fundamental data cleaning step. Pandas provides `duplicated()` for detection and `drop_duplicates()` for removal, with parameters that give precise control over what counts as a "duplicate."</span>

---

## <span style="font-size: 16px;">Detecting Duplicates</span>

<span style="font-size: 14px;">The `duplicated()` method returns a boolean Series marking duplicate rows:</span>

```python
df.duplicated()           # True for rows that are duplicates of earlier rows
df.duplicated(keep='last')  # True for rows that are duplicates of later rows
df.duplicated(keep=False)   # True for ALL duplicated rows (including first occurrence)
```

<span style="font-size: 14px;">The `keep` parameter controls which occurrence is considered "original":</span>

* <span style="font-size: 14px;">`keep='first'` (default): The first occurrence is not a duplicate; subsequent ones are.</span>
* <span style="font-size: 14px;">`keep='last'`: The last occurrence is not a duplicate; earlier ones are.</span>
* <span style="font-size: 14px;">`keep=False`: All occurrences of duplicated rows are marked as duplicates.</span>

### <span style="font-size: 14px;">Counting Duplicates</span>

```python
df.duplicated().sum()           # number of duplicate rows
df.duplicated(keep=False).sum() # number of rows involved in duplication
```

---

## <span style="font-size: 16px;">Removing Duplicates</span>

<span style="font-size: 14px;">`drop_duplicates()` removes duplicate rows and returns a new DataFrame:</span>

```python
df_clean = df.drop_duplicates()
```

<span style="font-size: 14px;">By default, it compares all columns and keeps the first occurrence. The same `keep` parameter controls which occurrence to retain:</span>

```python
df.drop_duplicates(keep='first')  # keep first (default)
df.drop_duplicates(keep='last')   # keep last
df.drop_duplicates(keep=False)    # remove ALL duplicates (keep none)
```

---

## <span style="font-size: 16px;">Subset-Based Deduplication</span>

<span style="font-size: 14px;">Often you want to deduplicate based on specific columns, not all columns:</span>

```python
# Keep one row per email address
df.drop_duplicates(subset=['email'])

# Keep one row per (name, date) combination
df.drop_duplicates(subset=['name', 'date'])
```

<span style="font-size: 14px;">The `subset` parameter accepts a column name or list of names. Rows are considered duplicates if they match on all subset columns, regardless of other column values.</span>

### <span style="font-size: 14px;">Choosing Which Row to Keep</span>

<span style="font-size: 14px;">When using subset-based deduplication, the choice of `keep` determines which row's non-subset values are retained. To keep the row with the most recent timestamp:</span>

```python
df_clean = (df
    .sort_values('timestamp', ascending=False)
    .drop_duplicates(subset=['user_id'], keep='first')
)
```

<span style="font-size: 14px;">Sort the DataFrame so the desired row appears first, then deduplicate with `keep='first'`.</span>

---

## <span style="font-size: 16px;">Index After Deduplication</span>

<span style="font-size: 14px;">After dropping duplicates, the index retains the original row numbers, creating gaps:</span>

```python
df = pd.DataFrame({'a': [1, 2, 2, 3]})
result = df.drop_duplicates()
print(result.index)  # [0, 1, 3] - index 2 is missing
```

<span style="font-size: 14px;">Use `.reset_index(drop=True)` to get a clean sequential index:</span>

```python
result = df.drop_duplicates().reset_index(drop=True)
```

---

## <span style="font-size: 16px;">Exact vs. Fuzzy Duplicates</span>

<span style="font-size: 14px;">`drop_duplicates()` finds exact matches only. For fuzzy duplicates (e.g., "John Smith" vs "john smith" vs "J. Smith"), you need preprocessing:</span>

```python
# Normalize before deduplicating
df['name_clean'] = df['name'].str.lower().str.strip()
df.drop_duplicates(subset=['name_clean'])
```

<span style="font-size: 14px;">For more complex fuzzy matching, consider libraries like `fuzzywuzzy` or `recordlinkage`.</span>

---

## <span style="font-size: 16px;">Deduplication in Practice</span>

### <span style="font-size: 14px;">Event Logs</span>

<span style="font-size: 14px;">Event pipelines often produce duplicate events due to at-least-once delivery guarantees:</span>

```python
events = events.drop_duplicates(subset=['event_id'])
```

### <span style="font-size: 14px;">Multi-Source Data</span>

<span style="font-size: 14px;">When combining data from overlapping sources:</span>

```python
combined = pd.concat([source_a, source_b])
combined = combined.drop_duplicates(subset=['record_id'], keep='last')
```

### <span style="font-size: 14px;">Time Windows</span>

<span style="font-size: 14px;">Keep the most recent record per entity within a time window:</span>

```python
df = df.sort_values('timestamp')
df = df.drop_duplicates(subset=['user_id'], keep='last')
```

---

## <span style="font-size: 16px;">Performance</span>

<span style="font-size: 14px;">`drop_duplicates()` uses hash-based comparison, giving it $O(n)$ average-case complexity. For DataFrames with many columns, specifying `subset` improves performance by reducing the number of columns that need to be hashed.</span>

<span style="font-size: 14px;">For very large DataFrames (millions of rows), consider whether the deduplication can be pushed upstream to the database (SQL `DISTINCT`) or the file reader (`pd.read_csv` with post-load dedup).</span>

---

## <span style="font-size: 16px;">Common Pitfalls</span>

* <span style="font-size: 14px;">**Floating-point duplicates**: Due to floating-point precision, values like 0.1 + 0.2 and 0.3 may not match. Round before deduplicating.</span>
* <span style="font-size: 14px;">**NaN equality**: Two NaN values are considered equal by `drop_duplicates()`. Rows [NaN, 1] and [NaN, 1] are duplicates. This differs from standard NaN comparison where NaN != NaN.</span>
* <span style="font-size: 14px;">**Forgetting to reset index**: After deduplication, the index has gaps. Operations that assume a contiguous integer index may break.</span>
* <span style="font-size: 14px;">**Losing important rows**: With `keep=False`, both the original and duplicate rows are removed. Use this only when all duplicated rows are suspect.</span>