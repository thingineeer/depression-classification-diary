import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity,euclidean_distances,pairwise_distances


class Emoji_similarlity():
    def __init__(self):
        self.df = pd.read_csv("./Dataset/Trans_Name.csv")
        self.tfidfvect = TfidfVectorizer(analyzer="char_wb", ngram_range=(1,4))
        self.dtm = self.tfidfvect.fit_transform(self.df["Tran_Name"])


    def cosine_ver(self, target):
        target_dtm = self.tfidfvect.transform([target])
        df_cos = pd.DataFrame(cosine_similarity(self.dtm, target_dtm), columns=["cosine_ver"])

        return(pd.concat([self.df, df_cos], axis=1).nlargest(10, "cosine_ver"))


    def euclidean_ver(self, target):
        target_dtm = self.tfidfvect.transform([target])
        df_cos = pd.DataFrame(cosine_similarity(self.dtm, target_dtm), columns=["cosine_ver"])

        return(pd.concat([self.df, df_cos], axis=1).nlargest(10, "cosine_ver"))


    def jaccard_ver(self, target):
        target_dtm = self.tfidfvect.transform([target])
        df_cos = pd.DataFrame(1 - pairwise_distances(self.dtm.toarray(), metric="jaccard"))

        return(pd.concat([self.df, df_cos], axis=1).nlargest(10, "cosine_ver"))

    def pearson_ver(self, target):
        target_dtm = self.tfidfvect.transform([target])
        df_cos = pd.DataFrame(np.corrcoef(self.dtm.toarray()))

        return(pd.concat([self.df, df_cos], axis=1).nlargest(10, "cosine_ver"))