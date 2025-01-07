import numpy as np
import pandas as pd

#open feat_importance.csv
df = pd.read_csv('feat_importance.csv')
df.columns = ['Feature', 'Importance']

#now, go feature by feature and compute drop in importance. If the drop is greater than 0.3, print the index, feature name, and importance
for i in range(1, len(df.index)):
    drop = df['Importance'][i]/df['Importance'][i-1]
    #print(drop)
    if drop <= 0.70:
        print(i, df['Feature'][i], df['Importance'][i], drop)

        #bigger than 10
        if i >= 10:
           #break, then save a csv only with features aboce that value
            break

#save the csv
df = df.iloc[:i,:]
df.to_csv('rf_top_sign05.csv', index=False)



