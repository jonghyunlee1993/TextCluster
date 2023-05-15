import os
import openai
import pandas as pd
from tqdm.auto import tqdm
from sklearn.cluster import KMeans

import torch 
from transformers import AutoTokenizer, AutoModelWithLMHead

from transformers import logging
logging.set_verbosity_warning()

import warnings
warnings.filterwarnings(action='ignore')

class TextAnalyzer:
    def __init__(self, project, column_of_interest, number_of_cluster):
        self.project = project
        self.column_of_interest = column_of_interest
        self.number_of_cluster = number_of_cluster
        
        with open("key/openai_key.txt", "r") as f:
            OPENAI_API_KEY = f.readlines()[0].rstrip()
            
        openai.api_key = OPENAI_API_KEY
        self.load_text_model()
        
        self.df = pd.read_excel(f"data/{self.project}.xlsx", engine="openpyxl")
    
    def load_text_model(self):
        # Base Model (108M)
        self.tokenizer = AutoTokenizer.from_pretrained("beomi/kcbert-base")
        self.model = AutoModelWithLMHead.from_pretrained("beomi/kcbert-base").eval()

    def run_analyze(self):
        print("Step 1/4: Encode text\n")
        features = self.get_feature()
        print("Step 2/4: Get cluster labels\n")
        self.get_cluster(features)
        print("Step 3/4: Get theme of each clusters\n")
        self.get_theme()
        print("Step 4/4: Write results excel file\n")
        self.write_results()
        
    def get_feature(self):
        feature = []

        for i, line in tqdm(self.df.iterrows(), total=len(self.df)):
            token = self.tokenizer(line[self.column_of_interest].replace("\n", " "), return_tensors="pt")
            out = self.model.bert(**token)
            cls = out['last_hidden_state'][:, 0, :].squeeze(0).detach().numpy().tolist()
            feature.append(cls)

        return feature

    def get_cluster(self, features):
        kmeans = KMeans(
            init="random",
            n_clusters=self.number_of_cluster,
            n_init=10,
            max_iter=100,
            random_state=42
        ).fit(features)

        self.df.loc[:, "cluster"] = kmeans.labels_

    def get_theme(self):
        self.theme_dict = {}

        for i in tqdm(range(self.df.cluster.max() + 1)):
            subset = self.df.loc[self.df.cluster == i, self.column_of_interest].values.tolist()
            if len(subset) > 50:
                import random 
                random.shuffle(subset)
                subset = subset[:50]
            
            answer = self.query_chatGPT(subset)
            self.theme_dict[i] = answer
        
        self.df.loc[:, "theme"] = self.df.cluster.map(lambda x: self.theme_dict[x])

    def query_chatGPT(self, query_text, LLM_model="gpt-3.5-turbo"):
        query = f"""
            다음 문장을 읽고 주제를 한 줄로 요약해라. 다음 템플릿을 사용하라.:

            TEXT: {str(query_text)}

            TEMPLATE:
            주제: {'write here the main theme of given texts'}
        """

        messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": query}
        ]

        response = openai.ChatCompletion.create(
            model=LLM_model,
            messages=messages
        )

        answer = response['choices'][0]['message']['content']

        return answer
        
    def write_results(self):
        df_stat = (self.df.loc[:, ["theme", self.column_of_interest]].groupby("theme").count() / len(self.df) * 100).round(2)
        
        os.makedirs('results', exist_ok=True)
        
        with pd.ExcelWriter(f"results/{self.project}_results.xlsx", engine='xlsxwriter') as writer:
            df_stat.to_excel(writer, sheet_name="Summary", index=True)
            workbook = writer.book
            worksheet = writer.sheets["Summary"]
            worksheet.set_column('A:A', 100)

            for i in range(self.df.cluster.max() + 1):
                df_subset = self.df.loc[self.df.cluster == i].iloc[:, :-2]
                df_subset.to_excel(writer, sheet_name=f"Cluster_{i}", index=False)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='필요한 변수를 입력하세요.')
    parser.add_argument('-project', help='파일 이름, 확장자 제외')
    parser.add_argument('-coi', help='컬럼 이름')
    parser.add_argument('-nc', type=int, help='클러스터의 개수')
    
    args = parser.parse_args()
    
    analyzer = TextAnalyzer(project=args.project, column_of_interest=args.coi, number_of_cluster=args.nc)
    analyzer.run_analyze()