import pandas as pd
from generateProps import UtteranceEncoding
utt = UtteranceEncoding()
prosList = pd.DataFrame()
print(type(utt.props))
prosList["Propositions"] = utt.props
prosList.to_csv("listOfPropositions.csv")