import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity,euclidean_distances,pairwise_distances

df = pd.read_csv("Trans_Name.csv")

tfidfvect = TfidfVectorizer()
dtm = tfidfvect.fit_transform(df["Tran_Name"])

def cosine_ver(target):
    target_dtm = tfidfvect.transform([target])
    cosine_matrix = cosine_similarity(target_dtm, dtm)
    df_cosine = pd.DataFrame(cosine_matrix)
    print(df.loc[df_cosine.loc[0].argmax()])
    



def euclidean_ver(target):
    target_dtm = tfidfvect.transform([target])
    ec_matrix = 1 / euclidean_distances(target_dtm, dtm)
    df_ec = pd.DataFrame(ec_matrix)
    print(df.loc[df_ec.loc[0].argmax()])
    



def jaccard_ver(target):
    target_dtm = tfidfvect.transform([target])
    jac_matrix = 1 - pairwise_distances(dtm.toarray(), metric="jaccard")
    df_jc = pd.DataFrame(jac_matrix)
    print(df.loc[df_jc.loc[0].argmax()])
    


def pearson_ver(target):
    target_dtm = tfidfvect.transform([target])
    pearson_matrix = np.corrcoef(dtm.toarray())
    df_ps = pd.DataFrame(pearson_matrix)
    print(df.loc[df_ps.loc[0].argmax()])
    

cosine_ver("웃는")
euclidean_ver("웃는")
jaccard_ver('웃는')
pearson_ver('웃는')
