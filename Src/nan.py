import pandas as pd

def any_nans(data):
    # .sum() on PyArrow columns is very fast
    null_counts = data.isnull().sum()
    null_counts = null_counts[null_counts > 0]
    
    if not null_counts.empty:
        print("NaNs with Columns Name and Count:")
        # Adding percentage helps put the 8.4M rows into perspective
        percent = (null_counts / len(data)) * 100
        out = pd.DataFrame({'Count': null_counts, 'Percentage': percent.round(4)})
        print(out)
    else:
        print("No NaNs found")