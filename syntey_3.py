#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# plt.style.use("ggplot")
import docx2txt
import spacy
data = pd.read_csv("dataset.csv", encoding="latin1")
data = data.fillna(method="ffill")
# data.tail(10)
import json
import request
import flask
#from flask import json
import pickle
from flask import request
# from flask import request
# from flask import Flask
# from flask_cors import CORS, cross_origin

# # from flask_cors import CORS

# app = Flask(__name__)
# cors = CORS(app, resources={r"/api/*": {"origins": "*"}},allow_headers=["Access-Control-Allow-Origin","Content-Type", "Authorization", "Access-Control-Allow-Credentials"],
#       supports_credentials=True)
# app.config['CORS_HEADERS'] = 'Content-Type'
from flask import Flask
from flask_cors import CORS
#,cross_origin


app = Flask(__name__)
#CORS(app,support_credentials=True)

#app.config['SECRET_KEY'] = 'the quick brown fox jumps over the lazy   dog'
#app.config['CORS_HEADERS'] = 'Content-Type'

#cors = CORS(app, resources={r"/": {"origins": "http://138.197.1.51:5000/"}})

words = list(set(data["words"].values))
n_words = len(words)
padd = list(data['words'])


# In[2]:


tags = ['B-LOCATION', 'B-PERSON', 'I-UNI', 'I-PERSON', 'I-SKILLS', 'I-EDUCATION', 'I-CDATE', 'B-EDATE', 'COMMA', 'B-EDUCATION', 'I-EDATE', 'B-EMAIL', 'B-COMPANY', 'B-NUMBER', 'I-EXP', 'I-COMPANY', 'I-LOCATION', 'B-EXP', 'B-SKILLS', 'I-NUMBER', 'B-UNI', 'O', 'B-CDATE']
# tags.pop(0)
max_len = 10

n_tags = len(tags)
tag2idx = {t: i for i, t in enumerate(tags)}
# print(tag2idx)
idx2tag = {i: w for w, i in tag2idx.items()}
# print(idx2tag)

def to_matrix(padd, n):
    return [padd[i:i+n] for i in range(0, len(padd), n)]
padd_to_2d = list(to_matrix(padd,8))
new_X = []
for seq in padd_to_2d:
    new_seq = []
    for i in range(max_len):
        try:
            new_seq.append(seq[i])
        except:
            new_seq.append("__PAD__")
    new_X.append(new_seq)
padd_to_2d = new_X
padd_y = list(data['tag'])
def to_matrix_tag(padd, n):
    return [padd[i:i+n] for i in range(0, len(padd), n)]
padd_to_2d_tag = list(to_matrix_tag(padd_y,8))
y = [[tag2idx[w] for w in s] for s in padd_to_2d_tag]
max_len=10


# In[2]:


from keras.preprocessing.sequence import pad_sequences
y = pad_sequences(maxlen=max_len, sequences=y, padding="post", value=tag2idx["O"])

from sklearn.model_selection import train_test_split
X_tr, X_te, y_tr, y_te = train_test_split(padd_to_2d, y, test_size=0.1, random_state=2018)
batch_size = 32
import tensorflow as tf
import tensorflow_hub as hub
from keras import backend as K
sess = tf.Session()
K.set_session(sess)
elmo_model = hub.Module("elmo_hub", trainable=False)
# elmo_model = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
sess.run(tf.global_variables_initializer())
sess.run(tf.tables_initializer())


# In[4]:


def ElmoEmbedding(x):
    return elmo_model(inputs={
                            "tokens": tf.squeeze(tf.cast(x, tf.string)),
                            "sequence_len": tf.constant(batch_size*[max_len])
                      },
                      signature="tokens",
                      as_dict=True)["elmo"]
# print(padd_to_2d_tag)
from keras.models import Model, Input
from keras.layers.merge import add
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional, Lambda


# In[5]:


X_tr, X_val = X_tr[:121*batch_size], X_tr[-13*batch_size:]
y_tr, y_val = y_tr[:121*batch_size], y_tr[-13*batch_size:]
y_tr = y_tr.reshape(y_tr.shape[0], y_tr.shape[1], 1)
y_val = y_val.reshape(y_val.shape[0], y_val.shape[1], 1)
input_text = Input(shape=(max_len,), dtype="string")
embedding = Lambda(ElmoEmbedding, output_shape=(None, 1024))(input_text)
x = Bidirectional(LSTM(units=512, return_sequences=True,
                       recurrent_dropout=0.2, dropout=0.2))(embedding)
x_rnn = Bidirectional(LSTM(units=512, return_sequences=True,
                           recurrent_dropout=0.2, dropout=0.2))(x)
x = add([x, x_rnn])  # residual connection to the first biLSTM
out = TimeDistributed(Dense(n_tags, activation="softmax"))(x)
model = Model(input_text, out)
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')


# In[6]:


checkpoint_path = "trained_model/cp-15.ckpt"
# # In[35]:
model.load_weights(checkpoint_path)


# In[7]:


nlp = spacy.load('en_core_web_lg')


# In[8]:


# In[3]:


# my_text = docx2txt.process("resume_77.docx")

import re

def clean_text_round2(text):
    '''Get rid of some additional punctuation and non-sensical text that was missed the first time around.'''
    text = re.sub('‘’“”…]', ' ', text)
    text = re.sub('\n', ' ', text)
    text = re.sub('\t', ' ', text)
    text = re.sub(' +', ' ', text)
#     text = re.sub('[-]', ' ', text)
    text = re.sub("\s\s+", "",text)
    return text

def Punctuation(string): 
  
    # punctuation marks 
    punctuations = '''!();'''
  
    # traverse the given string and if any punctuation 
    # marks occur replace it with null 
    for x in string: 
        if x in punctuations: 
            string = string.replace(x, " ") 
  
    # Print string without punctuation 
    return string
  
# Driver program 
# string = "Welcome???@@##$ to#$% Geeks%$^for$%^&Geeks"
# round3 = str(Punctuation(round2))


# In[ ]:
# counts =0
# def find_full_data_of_all(college_date,experience,tag,word,counts,beginning_of_ent='B-EXP',end_of_ent='I-EXP'):
# #     global c_date
# #     global counts
# #     global tag
# #     global word
#     for idx, name_tag in enumerate(college_date):
#         for j in range(name_tag+1,name_tag+10):
#             if counts==0 and tag[j-1]==beginning_of_ent:
#                 experience+=" "+word[j-1]

# #             count+=1
#             elif tag[j-1]!=end_of_ent:
#                 counts+=1
#                 break
#             else:
#                 experience+=word[j-1]
# #             c.append(a)
# #             print('jpt')
#                 counts+=1
#         counts=0
#     return experience
def find_full_data_of_all(college_date,experience,tag,word,counts,beginning_of_ent,end_of_ent):
#     global c_date
#     global count
    for idx, name_tag in enumerate(college_date):
        if len(tag)<name_tag+10:
            for j in range(name_tag,len(word)):
                if counts==0 and tag[j]==beginning_of_ent and tag[j]==beginning_of_ent:
                    experience+=" "+word[j]
                    break
                elif counts==0 and tag[j]==beginning_of_ent:
                    experience+=" "+word[j]
                     
                

#             count+=1
                
                elif tag[j]!=end_of_ent:
                    counts+=1
                    break
                else:
                    experience+=word[j]
#             c.append(a)
#             print('jpt')
                    counts+=1
            counts=0
        else:
            for j in range(name_tag+1,name_tag+10):
                if counts==0 and tag[j-1]==beginning_of_ent and tag[j]==beginning_of_ent:
                    experience+=" "+word[j-1]
                    break
                elif counts==0 and tag[j-1]==beginning_of_ent:
                    experience+=" "+word[j-1]
                    

#                     count+=1
                
                elif tag[j-1]!=end_of_ent:
                    counts+=1
                    break
                else:
                    experience+=word[j-1]
#             c.append(a)
#             print('jpt')
                    counts+=1
            
            counts=0
            
        
    return experience

def display_deg_com_exp(pos_deg,pos_date,r,t):
#     global r
#     global t
#     q=""
    for k,v in pos_deg.items():
        if bool(pos_date)==False:
            r.append(v)
            # print("company is empty so data with designation is "+v)
        else:
#     print(v)
            for k2,v2 in pos_date.items():
                
                if k-7<k2<k+7:
                    q=v+"="+v2
                    r.append(q)
                
                
    return r

    


# In[43]:





# In[13]:

@app.after_request
def after_request(response):
  response.headers.add('Access-Control-Allow-Origin', 'https://www.cyntey.com')
  response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
  response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
  response.headers.add('Access-Control-Allow-Credentials', 'true')
  return response

def finaly(result_designation,pos_date):
    k=[]
    o=[]
    
    for dev,shar in result_designation.items():
        value=''
        count=0
        for tri,puja in pos_date.items():
            if dev-10<tri<dev+10:
                o.append(dev)
                k.append(puja)
            else:
                value = puja
                count+= 1
#             break
#         else:
        if(count == len(result_designation)):
            o.append(dev)
            k.append("null")
    dic =dict(zip(o,k))
#     print(dic)

    from collections import Counter

    res = Counter(dic.values())

#     print(res)

    w=[]
    x=[]
    w1=[]
    x1=[]
    for k,v in res.items():
        if v>1:
            w.append(k)
            x.append(v)
        else:
            w1.append(k)
            x1.append(v)
            
#     print(w,x)
#     print(w1,x1)

    dics = dic

    key=[]
    for i in w1:
        for k,v in dic.items():
            if i == v:
                key.append(k)
#     print(key)
                
    combine_key_val=dict(zip(key,w1))
#     print(combine_key_val)

    home=[]
    ho =[]
    for k3,v3 in dic.items():    # for name, age in dictionary.iteritems():  (for Python 2.x)
        for io in w:
            if io==v3:
                for k,v in pos_date.items():
                    if v == v3:
                        home.append(k)
            
    #                 else:
                        
                        
                        
                
#     print(ho)          
    #             print(k3,v3)
    #             home.append(k3)
    homes=sorted(set(home))

#     print(homes)
#     print(home)

    gurkha_nepal=[]

    for k3,v3 in dic.items():    # for name, age in dictionary.iteritems():  (for Python 2.x)
        for io in w:
            if io==v3:
                gurkha_nepal.append(k3)
            
                
#     print(gurkha_nepal)
                

    group_2=[]
    idx_1=0
    #Iterate over groups, and calculate indexes for slicing
    for y in x:
        group_2.append(gurkha_nepal[idx_1:idx_1+y])
        #Increment indexes accordingly
        idx_1+=y
    print(group_2)


    resultant=[abs(x - y) for x, y in zip(gurkha_nepal, home)]
#     print(resultant)

    group=[]
    idx=0
    #Iterate over groups, and calculate indexes for slicing
    for y in x:
        group.append(resultant[idx:idx+y])
        #Increment indexes accordingly
        idx+=y

#     print(group)

    baluwa=[]
    for i in group:
        baluwa.append(min(i))
#     print(baluwa)

    ma=[]
    for i,v in enumerate(baluwa):
        ma.append(group_2[i][group[i].index(v)])
    dic_2=[]
    for i in ma:
        dic_2.append(dic[i])
#     print(dic_2)

    dic2=dict(zip(ma,dic_2))
    merged_dic={**combine_key_val,**dic2}
#     print(merged_dic)

    sorted_key=[]
    sorted_value=[]
    for key in sorted(merged_dic.keys()):
        sorted_key.append(key)
        sorted_value.append(merged_dic[key])
            

    result_designation_1=dict(zip(sorted_key,sorted_value))
#     print(result_designation_1)
    sorted_keys_values=[]
    for i in sorted_key:
        sorted_keys_values.append(result_designation[i])
#     print(sorted_keys_values)
    final_dict=dict(zip(sorted_keys_values,sorted_value))
    return final_dict,result_designation_1
#     print(final_dict)


# In[19]:


graph = tf.get_default_graph()

cors = CORS(app, resources={r"/*": {"origins": "https://www.cyntey.com"}})
app.config['CORS_AUTOMATIC_OPTIONS'] = True
#app.config['CORS_SUPPORTS_CREDENTIALS'] = True
@app.route('/',methods=['POST','GET','OPTIONS'])
#@cross_origin()
#@cross_origin(allow_headers=['Content-Type','authorization','Access-Control-Allow-Origin'])
#@cross_origin(allow_headers=['authorization'])
#@cross_origin(origin="*",allow_headers=['Content-Type','authorization','Access-Control-Allow-Origin'])
#@cross_origin(supports_credentials=True)
def main_function():
    with graph.as_default():
#         file = request.files['docfile']
        file = request.files['docfile']
        # file.headers.add('Access-Control-Allow-Origin', '*')
        my_text = docx2txt.process(file)


        round2 =clean_text_round2(my_text)
        round3 = str(Punctuation(round2))
        token_1 = nlp(round3)
        token = re.sub("\s\s+", " ",round3)
        round4 = nlp(token)
            
        tokens = [token.text for token in round4]
        # print(tokens)
        token_2 = []
        for to in tokens:
            if to !='\xa0':
                if to!='_':
                    token_2.append(to)
        test_1 = token_2


        length = len(test_1)
        k=[]
        j=[]
        for i in range(length,5000):
            if i%8==0:
                j.append(i)
        b=[]
        #print(j)
        for a in j:
            a = int(a/8)
            if a%32==0:
                b.append(a)
                break
            
        # print(b)
        correct_length = sum(d * 10**i for i, d in enumerate(b[::-1])) 
        correct_length = correct_length*8
        # print(correct_length)
        for i in range(length+1,correct_length+1):
            test_1.append(',')
        # print(test_1)
        def to_matrix_senti(padd, n):
            return [padd[i:i+n] for i in range(0, len(padd), n)]
        padd_to_2d_senti = list(to_matrix_senti(test_1,8))
        # print(len(padd_to_2d_senti))
        new_matrix_senti = []
        for seq in padd_to_2d_senti:
            new_seq_senti = []
            for i in range(max_len):
                try:
                    new_seq_senti.append(seq[i])
                except:
                    new_seq_senti.append("__PAD__")
            new_matrix_senti.append(new_seq_senti)
        padd_to_2d_senti = new_matrix_senti
        padd_to_2d_senti = np.array(padd_to_2d_senti)

        # tag = ['B-LOCATION', 'B-PERSON', 'I-UNI', 'I-PERSON', 'I-SKILLS', 'I-EDUCATION', 'I-CDATE', 'B-EDATE', 'COMMA', 'B-EDUCATION', 'I-EDATE', 'B-EMAIL', 'B-COMPANY', 'B-NUMBER', 'I-EXP', 'I-COMPANY', 'I-LOCATION', 'B-EXP', 'B-SKILLS', 'I-NUMBER', 'B-UNI', 'O', 'B-CDATE']
        # tag2idxs = {t: i for i, t in enumerate(tag)}
        # idx2tags = {i: w for w, i in tag2idxs.items()}
        # print(idx2tag)
        padd_to_2d_senti = np.array(padd_to_2d_senti)

        # y = model.predict([padd_to_2d_senti])
        y = model.predict([padd_to_2d_senti])
        # p = np.argmax(y[:352], axis=-1)
        # # print(p)
        # flat_list_te = padd_to_2d_senti[:352]
        # flat_list_test = [item for sublist in flat_list_te for item in sublist]
        # # print(flat_list_test)
        # # flat_list = [item for sublist in y_te[i] for item in sublist]
        # flat_list_pred = [item for sublist in p for item in sublist]
        # flat_list_pred_value_in_words = []
        # for i in flat_list_pred:
        #     flat_list_pred_value_in_words.append(idx2tags[i])
        # # print(len(flat_list_pred_value_in_words))

        # # print(p)
        # print("{:15}||{:5}||{}".format("Word", "True", "Pred"))
        # print(30 * "=")
        # test_pred = dict(zip(flat_list_test,flat_list_pred_value_in_words))
        p = np.argmax(y[:416], axis=-1)
        # print(p)
        flat_list_te = padd_to_2d_senti[:416]
        flat_list_test = [item for sublist in flat_list_te for item in sublist]
        # print(flat_list_test)
        # flat_list = [item for sublist in y_te[i] for item in sublist]
        flat_list_pred = [item for sublist in p for item in sublist]
        skills_name =[]
        skills_tag=[]
        company_name =[]
        company_tag=[]
        person_name=[]
        person_tag=[]
        company_working_date_name=[]
        company_working_date_tag=[]
        education_name=[]
        education_tag=[]
        universities_name=[]
        universities_tag=[]
        college_date_name=[]
        college_date_tag=[]
        email_name=[]
        email_tag=[]
        location_name=[]
        location_tag=[]
        designation_name=[]
        designation_tag=[]
        phone_number=[]
        phone_number_tag=[]
        word =[]
        tag=[]
        for ws,preds in zip(flat_list_test,flat_list_pred):
            predicted_tag = idx2tag[preds]
#             print(ws,predicted_tag)
#             if ws!='__PAD__' and predicted_tag!='O':
#                 word.append(ws)
#                 tag.append(predicted_tag)
                
#         print(len(word),len(tag))
                
#                 print(ws)
        

            if predicted_tag !='O':
                if predicted_tag !='COMMA':
                    word.append(ws)
                    tag.append(predicted_tag)
                    # print(ws,predicted_tag)
        # print(word,tag)
        
        # print("COMPANY POSITION==============================")
        com_tag_pos = [i for i,x in enumerate(tag) if x == 'B-COMPANY']
       
        pos_of_date = [i for i,x in enumerate(tag) if x == 'B-EDATE']
        
        exp=''
        counts=0
        
        pos_of_designation = [i for i,x in enumerate(tag) if x == 'B-EXP']
        # print(pos_of_designation)
        pos_of_i_designation = [i for i,x in enumerate(tag) if x == 'I-EXP']
        pos_of_i_com= [i for i,x in enumerate(tag) if x == 'I-COMPANY']
        
        
        def split_list(n):
       
            return [(x+1) for x,y in zip(n, n[1:]) if y-x != 1]

        def get_sub_list(my_list):
        
            my_index = split_list(my_list)
            output = list()
            prev = 0
            for index in my_index:
                new_list = [ x for x in my_list[prev:] if x < index]
                output.append(new_list)
                prev += len(new_list)
            output.append([ x for x in my_list[prev:]])
            return output

# my_list = [1, 3, 4, 7, 8, 10, 11, 13, 14]
        pos_of_double_designation=get_sub_list(pos_of_i_designation)
        pos_of_double_company = get_sub_list(pos_of_i_com)
        
        pos_of_double_company_1 = []
        pos_of_double_designation_1=[]
        a=[]
        for i in pos_of_double_designation:
            if tag[i[0]-1]=='B-EXP':
                a.append(i)
        
        b=[]
        for i in pos_of_double_company:
            if tag[i[0]-1]=='B-COMPANY':
                b.append(i)
        # print("after removing beginining of company and designation")
        pos_of_double_designation = [item for sublist in a for item in sublist]
        # print(pos_of_double_designation)
        list1 = [ele for ele in pos_of_i_designation if ele not in pos_of_double_designation]
        # print(list1)
        pos_of_double_designation=get_sub_list(list1)
        # print('====================company')
        pos_of_double_company = [item for sublist in b for item in sublist]
        # print(pos_of_double_company)
        list2 = [ele for ele in pos_of_i_com if ele not in pos_of_double_company]
        # print(list2)
        pos_of_double_company=get_sub_list(list2)
        
        for z in pos_of_double_designation:
            pos_of_double_designation_1.append(z[0])
        for x in pos_of_double_company:
            pos_of_double_company_1.append(x[0])
        #FORMING THE STRING OF CONTINUATION FROM I-COMPANY
        string_com=""
        
        string =""
        count=0
        for s in pos_of_double_designation:
            for sub in s:
                if count==0 and tag[sub]=='I-EXP':
                    string+=word[sub]
                    count+=1
                count=0
            string+=" "
        
        for c in pos_of_double_company:
            for sub in c:
                if count==0 and tag[sub]=='I-COMPANY':
                    string_com+=word[sub]
                    count+=1
                count=0
            string_com+=" "
        
        experience_com=''
        experience_date=''
        exp_date=experience_date
#         exp_company=''
        exp_designation=''
    
#         # global counts
        # print("EXPERIENCE DATE =========================================")
        experience_date = find_full_data_of_all(pos_of_date,exp_date,tag,word,counts,beginning_of_ent='B-EDATE',end_of_ent='I-EDATE')
            
        experience_designation = find_full_data_of_all(pos_of_designation,exp_designation,tag,word,counts,beginning_of_ent='B-EXP',end_of_ent='I-EXP')
        
        experience_company = find_full_data_of_all(com_tag_pos,experience_com,tag,word,counts,beginning_of_ent='B-COMPANY',end_of_ent='I-COMPANY')
        
        deg = experience_designation.split(' ')
        deg.pop(0)
#         print(deg)
        deg_1 = string.split(' ')
        deg_1.pop(-1)
        # print(deg_1)
        company = experience_company.split(' ')
        # print(company)
        company.pop(0)
        # print("COMPANY NAME AFTER POP=========")
        # print(company)
        company_1 = string_com.split(' ')
        company_1.pop(-1)
        # print("COMPANY NAME 1 AND 2 ==================")
        # print(company,company_1)
        
#         print(deg)
# #         # print(deg)
        date = experience_date.split(' ')
        date.pop(0)
        # print(date)
#         com = experiene_company.split(' ')
#         com.pop(0)
#         print(com)
# #         print(com)
        pos_deg = dict(zip(pos_of_designation,deg))
        # print(pos_deg)
        # print('jpt')
        pos_deg_1 = dict(zip(pos_of_double_designation_1,deg_1))
        # print(pos_deg_1)
        pos_date=dict(zip(pos_of_date,date))
        # print(pos_date)
        pos_combine_deg={**pos_deg,**pos_deg_1}
        # print(pos_combine_deg)
        sorted_key=[]
        sorted_value=[]
        for key in sorted(pos_combine_deg.keys()):
            sorted_key.append(key)
            sorted_value.append(pos_combine_deg[key])
        

        result_designation=dict(zip(sorted_key,sorted_value))
        # print(result_designation)
        # print("THIS IS FOR COMPANY DICTIONARY MAKING BY COMBININI 2 DICT ===================")
        pos_com = dict(zip(com_tag_pos,company))
        
        pos_com_1=dict(zip(pos_of_double_company_1,company_1))
        
        pos_combine_com={**pos_com,**pos_com_1}
        # print(pos_combine_com)
        sorted_key_company=[]
        sorted_value_company=[]
        for key in sorted(pos_combine_com.keys()):
            sorted_key_company.append(key)
            sorted_value_company.append(pos_combine_com[key])
        

        result_company=dict(zip(sorted_key_company,sorted_value_company))
        
        r=[]
        t=[]
#         results = display_deg_com_exp(result_designation,pos_date,result_company,r,t)
        results = display_deg_com_exp(result_designation,pos_date,r,t)
        # print(results)
        final,final_1=finaly(result_designation,pos_date)
        designation=[]
        date=[]
        company_full_name=[]
        
#         final_2,jpt=finaly(final_1,result_company)
        kj=[]
        oj=[]
        
        for dev,shar in final_1.items():
            value = ""
            count = 0
            for tri,puja in result_company.items():
                if dev-10<tri<dev+10:
                    oj.append(dev)
                    kj.append(puja)
                else:
                    value = puja
                    count+= 1
#             break
#         else:
            if(count == len(result_company)):
                oj.append(dev)
                kj.append("null")
        dic =dict(zip(oj,kj))
        
        for i,j in final.items():
            designation.append(i)
            date.append(j)
        for k,l in dic.items():
            company_full_name.append(l)
        
        string_4=''
        for i in range(0,len(designation)):
            string_4+=designation[i]+"|"+date[i]+"|"+company_full_name[i]+"|"
    
            string_4+="||"
        
        pos_of_skills = [i for i,x in enumerate(tag) if x == 'B-SKILLS']
        pos_of_college_date = [i for i,x in enumerate(tag) if x == 'B-CDATE']
        pos_of_college_name = [i for i,x in enumerate(tag) if x == 'B-UNI']
        pos_of_college_degree = [i for i,x in enumerate(tag) if x == 'B-EDUCATION']
        experience_college_date=''
        experience_college_name=''
        experience_college_degree=''
       
        experience_skills=''
        counts=0
        career_skills = find_full_data_of_all(pos_of_skills,experience_skills,tag,word,counts,beginning_of_ent='B-SKILLS',end_of_ent='I-SKILLS')
        car_skills = career_skills.split(' ')
        car_skills.pop(0)
        college_date = find_full_data_of_all(pos_of_college_date,experience_college_date,tag,word,counts,beginning_of_ent='B-CDATE',end_of_ent='I-CDATE')
#         # print(college_date)
        college_name=find_full_data_of_all(pos_of_college_name,experience_college_name,tag,word,counts,beginning_of_ent='B-UNI',end_of_ent='I-UNI')
#         # print(experiene_company)
        college_degree = find_full_data_of_all(pos_of_college_degree,experience_college_degree,tag,word,counts,beginning_of_ent='B-EDUCATION',end_of_ent='I-EDUCATION')
        college_date_1 = college_date.split(' ')
        college_date_1.pop(0)
# #         # print(deg)
        college_name_1 = college_name.split(' ')
        college_name_1.pop(0)
# #         # print(date)
        college_degree_1 = college_degree.split(' ')
        college_degree_1.pop(0)
# #      
        skills_name =[]
        skills_tag=[]
        person_name=[]
        person_tag=[]
        email_name=[]
        email_tag=[]
        phone_number=[]
        phone_number_tag=[]
        for w,pred in zip(flat_list_test,flat_list_pred):
            predicted_tag = idx2tag[pred]
            
            if predicted_tag !='O':
                
                if predicted_tag=='B-PERSON' or predicted_tag=='I-PERSON':
                    person_name.append(w)
                    person_tag.append(predicted_tag)
                if predicted_tag=='B-NUMBER' or predicted_tag=='I-NUMBER':
                    phone_number.append(w)
                    phone_number_tag.append(predicted_tag)
                if predicted_tag=='B-EMAIL':
                    email_name.append(w)
                    email_tag.append(predicted_tag)
                
#         # test_pred = list(zip(a,b))
#         # skills = list(zip(skills_name,skills_tag))
        
        person = list(zip(person_name,person_tag))
        number = list(zip(phone_number,phone_number_tag))
        email = list(zip(email_name,email_tag))
        somedict = { 
                     "person_name": [ x[0] for x in person],
                     "person_tag" : [ x[1] for x in person],
                     "desig":result_designation,
                     "com":result_company,
                     "date":pos_date,
                     "numbers" : [ x[0] for x in number],
                     "number_tag" : [ x[1] for x in number],
                     "emails" : [ x[0] for x in email],
                     "emails_tag" : [ x[1] for x in email],
                     "designation_details":string_4,
                     "college_degree":college_degree_1,
                     "college_uni":college_name_1,
                     "college_date":college_date_1,
# #                      "education_details":results,
                     "skills":list(set(car_skills)),
        }
        test_predicted = json.dumps(somedict)
        return test_predicted


        
app.run(host="138.197.1.51",port=5000)

