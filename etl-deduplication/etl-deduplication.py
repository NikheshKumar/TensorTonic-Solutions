def deduplicate(records, key_columns, strategy):
    """
    Deduplicate records by key columns using the given strategy.
    """
    # Write code here
    import numpy as np

    ans = []

    if strategy == "first":
      seen = set()
      for rec in records:
        key_val = tuple(rec.get(k) for k in key_columns)
        if key_val in seen:
          continue 
        else:
          ans.append(rec)
          seen.add(key_val)
        
      return ans
        
      
    if strategy == "last":
      seen = set()
      for rec in records[::-1]:     
        key_val = tuple(rec.get(k) for k in key_columns)
        if key_val in seen:
          continue 
        else:
          ans.append(rec)
          seen.add(key_val)
          
      return ans
    
        
    if strategy == "most_complete":
      sorted_rec = sorted(records, key=lambda x: sum(1 for v in x.values() if v is None or (isinstance(v, float) and np.isnan(v))) )
      seen = set()
      for rec in sorted_rec : 
        key_val = tuple(rec.get(k) for k in key_columns)
        if key_val in seen:
          continue 
        else:
          ans.append(rec)
          seen.add(key_val)
          
      return ans
        
      
      


    