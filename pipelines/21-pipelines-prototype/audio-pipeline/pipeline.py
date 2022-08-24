# # %%
# from transformers import pipeline
# sentiment  = pipeline("sentiment-analysis")
# sentiment("mamooli weather")

# %%
from transformers import pipeline
translator  = pipeline(task="translation_fr_to_en", model="Helsinki-NLP/opus-mt-fr-en")
print(translator("After decades as a martial arts practitioner and runner, Wes yoga in 2010."))


# %%
# from transformers import pipeline
# summerize  = pipeline("summarization")
# summerize("If hearings or trials cannot take place because there are no barristers present to represent defendants, there won't be any trials in which criminals are sent to prison and those who are innocent are acquitted. Victims, like defendants, will be left in limbo, unsure when they will see justice. Tentative plans to broker a deal behind the scenes by bringing forward payments have failed - partly because there is simply no trust between the profession and ministers who wont meet them.")

# %%



