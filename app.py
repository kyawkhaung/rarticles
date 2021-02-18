from flask import Flask, render_template, request
#import jsonify
import requests
import io
import numpy as np
import pandas as pd
from models.article import ResearchArticle
#from pyrebase import pyrebase
#import pyrebase


#import transformers
import torch
#from transformers import BertTokenizer, BertModel, BertConfig
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)


csv_url = 'https://firebasestorage.googleapis.com/v0/b/researcharticle-c8bb9.appspot.com/o/raw_train.csv?alt=media&token=2d22ff09-2da9-4e17-9db6-5bf2caf92301'
npy_url = 'https://firebasestorage.googleapis.com/v0/b/researcharticle-c8bb9.appspot.com/o/embeddings_title_search.npy?alt=media&token=aa00ac45-a7e1-4c6f-a812-c7e21d8488c0'

#df = response.content
df = pd.read_csv(csv_url)

response = requests.get(npy_url)
embeddings_title_search = np.load(io.BytesIO(response.content))
'''
commented out in order to use with firebase cloud storgae   
#with open('./models/embeddings_title_search.npy', 'rb') as fp:
 #   embeddings_title_search = np.load(fp)
    
#with open('./models/embeddings_recommendation.npy', 'rb') as fp:
 #   embeddings_recommendation = np.load(fp)
    
#df = pd.read_csv('./dataset/raw_train.csv')
'''
sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')

embeddings_recommendation = cosine_similarity(embeddings_title_search)

version=pd.__version__

def get_articles():
    list_projects = []
    
    for i in range(len(df[0:20])):
        
        researchArticle = ResearchArticle(df.iloc[i]['ID'], df.iloc[i]['TITLE'], df.iloc[i]['ABSTRACT'])
        list_projects.append(researchArticle)
        
    return list_projects
                        
@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html', version=version, articles = get_articles())

@app.route('/info',methods=['GET'])
def Info():
    return render_template('info.html', articles = get_articles())

def search_article_title(query_term, all_title_embeddings, topn=20):
   query_embedding = sbert_model.encode(query_term)
   cos_scores = util.pytorch_cos_sim(query_embedding, embeddings_title_search)[0]
   top_results = torch.topk(cos_scores, k= topn)


   
   list_articles = []
   for score, idx in zip(top_results[0], top_results[1]):
       researchArticle = ResearchArticle(idx.item(), df.iloc[idx.item()]['TITLE'], 
                                            df.iloc[idx.item()]['ABSTRACT'])
       list_articles.append(researchArticle)
     #print(df_raw.iloc[idx.item()]['TITLE'], ":"  , "(Score: {:.4f})".format(score.item()))
   return list_articles

def get_detail_article(articleid):
    if articleid:
        researchArticle = ResearchArticle(articleid, df.iloc[articleid]['TITLE'], 
                                            df.iloc[articleid]['ABSTRACT'])
    
    return researchArticle

def get_similar_articles(articleid):
    #embeddings_recommendation = []
    list_articles = []
    confidence_level = 0.75
    topn = 10
    #def most_similar(doc_id, similarity_matrix, matrix, topn, confidence_level=0.75):
    
    if articleid:
        #title_name = df.iloc[articleid]["TITLE"]
        similar_ix=np.argsort(embeddings_recommendation[articleid])[::-1]
      
        for idx in similar_ix[:topn + 1]:
            if idx == articleid:
                continue
            
            article_consine_score = embeddings_recommendation[articleid][idx]
            if article_consine_score > confidence_level:
                researchArticle = ResearchArticle(idx, 
                                                  df.iloc[idx]['TITLE'], 
                                            df.iloc[idx]['ABSTRACT'])
            list_articles.append(researchArticle)
    
    return list_articles
    
@app.route("/searchtitle", methods=['POST'])
def searchtitle():

    if request.method == 'POST':
        query = str(request.form['queryterm'])
        if query:
           articles_output = search_article_title(query, embeddings_title_search, 20)
        
        if len(articles_output) > 0:
            return render_template('index.html', query_text=query, articles = articles_output)
        
        else:
           return render_template('index.html', prediction_text="Failed!")
           # return render_template('index.html',prediction_text="Decision to get loan is {}".format(output))
    else:
        return render_template('index.html')
    


@app.route("/detail/<articleid>", methods=['GET'])
def detail(articleid):

    if request.method == 'GET':
        if articleid:
           article_detail = get_detail_article(int(articleid))
           articles_output = get_similar_articles(int(articleid))
        
        if len(articleid) > 0:
            return render_template('detail.html', article = article_detail, articles=articles_output)
        
        else:
           return render_template('detail.html', aid="No such article exists!")
           # return render_template('index.html',prediction_text="Decision to get loan is {}".format(output))
    else:
        return render_template('detail.html', aid="Himm, something went wrong. Try again!")


if __name__=="__main__":
    app.run(debug=True)

