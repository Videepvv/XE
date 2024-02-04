import re
def change_sybbols(dp_content):
    dp_content = dp_content.replace("equals"," = ").replace("does not equal"," != ").\
                        replace("is less than"," < ").replace("is more than"," > ").\
                        replace("plus"," + ").replace("ten"," 10 ").replace("twenty"," 20 ").replace("thirty"," 30 ").\
                        replace("forty"," 40 ").replace("fifty"," 50 ").replace("block","").replace("and", " , ") 
    
    dp_content = re.sub(r'\s+', ' ', dp_content)                   
    return dp_content.strip()

import pandas as pd

df = pd.read_csv('listOfPropositions.csv')
df['Propositions'] = df["Propositions"].apply(change_sybbols)
df.to_csv('correctedList.csv')