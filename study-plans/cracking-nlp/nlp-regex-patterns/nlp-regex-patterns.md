# <span style="font-size: 20px;">Regex Patterns for NLP</span>

<span style="font-size: 14px;">Regular expressions (regex) are a foundational tool in Natural Language Processing for extracting structured information from unstructured text. Before machine learning models can process text, regex patterns often serve as the first line of defense for data cleaning, feature extraction, and text normalization.</span>

---

## <span style="font-size: 16px;">Why Regex Matters in NLP</span>

* <span style="font-size: 14px;">**Data preprocessing**: Extracting emails, URLs, and dates from raw text before feeding it to a model</span>
* <span style="font-size: 14px;">**Named Entity Recognition (NER) bootstrapping**: Regex can identify structured entities that ML models often miss, especially in domain-specific text</span>
* <span style="font-size: 14px;">**Text normalization**: Standardizing formats (dates, currency) before tokenization improves downstream model performance</span>
* <span style="font-size: 14px;">**Feature engineering**: Counting pattern occurrences (hashtags, mentions) creates useful features for classification</span>
* <span style="font-size: 14px;">**Evaluation and post-processing**: Validating model outputs against expected patterns</span>

---

## <span style="font-size: 16px;">The Five Pattern Types</span>

### <span style="font-size: 14px;">1. Emails</span>

<span style="font-size: 14px;">Pattern:</span> <span style="font-family:monospace; font-size:13px;">\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b</span>

* <span style="font-size: 14px;">Matches a local part (letters, digits, dots, underscores, percent, plus, hyphen) followed by @ and a domain with at least a 2-character TLD</span>
* <span style="font-size: 14px;">The</span> <span style="font-family:monospace; font-size:13px;">\b</span> <span style="font-size: 14px;">word boundary anchors ensure we do not match partial strings embedded in longer tokens</span>

### <span style="font-size: 14px;">2. URLs</span>

<span style="font-size: 14px;">Pattern:</span> <span style="font-family:monospace; font-size:13px;">https?://[^\s]+</span>

* <span style="font-size: 14px;">Matches</span> <span style="font-family:monospace; font-size:13px;">http://</span> <span style="font-size: 14px;">or</span> <span style="font-family:monospace; font-size:13px;">https://</span> <span style="font-size: 14px;">followed by any non-whitespace characters</span>
* <span style="font-size: 14px;">The</span> `?` <span style="font-size: 14px;">after</span> <span style="font-family:monospace; font-size:13px;">s</span> <span style="font-size: 14px;">makes the 's' optional, matching both HTTP and HTTPS</span>
* <span style="font-size: 14px;">A negative lookbehind can be added to strip trailing punctuation such as commas or periods</span>

### <span style="font-size: 14px;">3. Dates</span>

<span style="font-size: 14px;">Pattern:</span> <span style="font-family:monospace; font-size:13px;">\d{1,2}/\d{1,2}/\d{2,4}</span> <span style="font-size: 14px;">or</span> <span style="font-family:monospace; font-size:13px;">\d{4}-\d{2}-\d{2}</span>

* <span style="font-size: 14px;">The first alternative matches MM/DD/YYYY or M/D/YY formats with slash separators</span>
* <span style="font-size: 14px;">The second alternative matches ISO 8601 format (YYYY-MM-DD) with hyphen separators</span>
* <span style="font-size: 14px;">The alternation operator</span> `|` <span style="font-size: 14px;">combines both patterns into a single regex</span>

### <span style="font-size: 14px;">4. Money</span>

<span style="font-size: 14px;">Pattern:</span> <span style="font-family:monospace; font-size:13px;">\$\d+(?:\.\d{2})?</span>

* <span style="font-size: 14px;">Matches a dollar sign followed by one or more digits, optionally followed by a decimal point and exactly two digits</span>
* <span style="font-size: 14px;">The</span> <span style="font-family:monospace; font-size:13px;">(?:...)</span> <span style="font-size: 14px;">is a non-capturing group - it groups the decimal portion without creating a capture group</span>
* <span style="font-size: 14px;">Matches both</span> <span style="font-family:monospace; font-size:13px;">$100</span> <span style="font-size: 14px;">and</span> <span style="font-family:monospace; font-size:13px;">$99.99</span>

### <span style="font-size: 14px;">5. Hashtags</span>

<span style="font-size: 14px;">Pattern:</span> <span style="font-family:monospace; font-size:13px;">#\w+</span>

* <span style="font-size: 14px;">Matches a hash symbol followed by one or more word characters (letters, digits, underscores)</span>
* <span style="font-size: 14px;">The</span> <span style="font-family:monospace; font-size:13px;">\w</span> <span style="font-size: 14px;">shorthand is equivalent to</span> <span style="font-family:monospace; font-size:13px;">[A-Za-z0-9_]</span>

---

## <span style="font-size: 16px;">Practical Tips for Regex in NLP</span>

* <span style="font-size: 14px;">**Use raw strings**: In Python, prefix regex patterns with</span> `r` <span style="font-size: 14px;">(e.g.,</span> <span style="font-family:monospace; font-size:13px;">r"\d+"</span><span style="font-size: 14px;">) to avoid double-escaping backslashes</span>
* <span style="font-size: 14px;">**Use</span> `re.findall()` <span style="font-size: 14px;">for extraction**: It returns all non-overlapping matches as a list, which is ideal for pattern extraction tasks</span>
* <span style="font-size: 14px;">**Word boundaries matter**: Use</span> <span style="font-family:monospace; font-size:13px;">\b</span> <span style="font-size: 14px;">to prevent partial matches. Without boundaries, a digit pattern inside "abc123def" would match "123"</span>
* <span style="font-size: 14px;">**Non-capturing groups**: Use</span> <span style="font-family:monospace; font-size:13px;">(?:...)</span> <span style="font-size: 14px;">when you need grouping for quantifiers but do not want the group in your</span> `findall` <span style="font-size: 14px;">results</span>
* <span style="font-size: 14px;">**Compile for performance**: Use</span> `re.compile()` <span style="font-size: 14px;">when applying the same pattern to many strings - it avoids re-parsing the regex each time</span>

---

## <span style="font-size: 16px;">Common Interview Follow-ups</span>

* <span style="font-size: 14px;">**What is the difference between greedy and non-greedy matching?** Greedy quantifiers (</span><span style="font-family:monospace; font-size:13px;">*</span><span style="font-size: 14px;">,</span> <span style="font-family:monospace; font-size:13px;">+</span><span style="font-size: 14px;">) match as much text as possible, while non-greedy (</span><span style="font-family:monospace; font-size:13px;">*?</span><span style="font-size: 14px;">,</span> <span style="font-family:monospace; font-size:13px;">+?</span><span style="font-size: 14px;">) match as little as possible. For extracting content between delimiters, non-greedy is usually correct</span>
* <span style="font-size: 14px;">**Why use</span> $	exttt{re.findall()}$ <span style="font-size: 14px;">instead of</span> $	exttt{re.search()}$<span style="font-size: 14px;">?**</span> <span style="font-size: 14px;">findall returns all non-overlapping matches in the string, while search returns only the first match. For extraction tasks where multiple matches exist, findall is the right choice</span>
* <span style="font-size: 14px;">**When are regexes not appropriate for NLP?** Parsing nested structures (like HTML or programming languages) is beyond the capability of regular expressions. Context-free grammars or dedicated parsers should be used instead. Regexes also struggle with ambiguous natural language patterns</span>
* <span style="font-size: 14px;">**What is the time complexity of regex matching?** Standard NFA-based engines (Python's re module) can be exponential in pathological cases with nested quantifiers (catastrophic backtracking). DFA-based engines like RE2 guarantee linear time but lack some features like backreferences</span>

---