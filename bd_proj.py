import streamlit as st
import pandas as pd
import os


import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

from nltk.chunk import conlltags2tree, tree2conlltags
from pprint import pprint

import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
nlp = en_core_web_sm.load()

import numpy as np

import matplotlib.pyplot as plt
#%matplotlib inline
import pandas as pd
import geopandas as gpd
from urllib import request
from geotext import GeoText

from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut

from shapely.geometry import Point, Polygon
import descartes

from PIL import Image
import urllib
import requests
from io import BytesIO

response = requests.get('https://gujarathighcourt.nic.in/hccms/sites/default/files/slides//01hcfront.jpg')
img = Image.open(BytesIO(response.content))

st.image(img, width=750)

st.title('Gujarat High Court Annotated Transcriptions')
st.subheader('')
files=['1st December, 2020','6th January, 2021','7th January, 2021','8th January, 2021','11th January, 2021','22nd December, 2020','23rd December, 2020','24th December, 2020','25th November, 2020','26th November, 2020','26th October, 2020','27th November, 2020','27th October, 2020','28th October, 2020']

option = st.sidebar.selectbox('Date of the proceeding',files)

f = open(f'txt_file/{option}.txt', "r")
x=f.readlines()

if st.sidebar.checkbox('Transcribed Text'):
    st.subheader('Transcribed Text')
    st.write(x[0])

links={"1st December, 2020":"https://www.youtube.com/watch?v=NGV1uwAySsQ","6th January, 2021":"https://www.youtube.com/watch?v=a0dbm5rmGeM",
"7th January, 2021":"https://www.youtube.com/watch?v=33-YqzFeifg","8th January, 2021":"https://www.youtube.com/watch?v=90ESUEYST2I",
"11th January, 2021":"https://www.youtube.com/watch?v=il5WDpy082M","22nd December, 2020":"https://www.youtube.com/watch?v=xvgnNDPEcV0",
"23rd December, 2020":"https://www.youtube.com/watch?v=q0Vo-ZJofg8","24th December, 2020":"https://www.youtube.com/watch?v=6iVZPuR9I2Q",
"25th November, 2020":"https://www.youtube.com/watch?v=whcn1UWUBmE","26th November, 2020":"https://www.youtube.com/watch?v=fveJomSLxuA",
"26th October, 2020":"https://www.youtube.com/watch?v=WpqQWBERB_Y","27th November, 2020":"https://www.youtube.com/watch?v=kXcTHbS4whE",
"27th October, 2020":"https://www.youtube.com/watch?v=RUIgWWYBZPQ","28th October, 2020":"https://www.youtube.com/watch?v=GOl6f8pJsKA"}
# In[1]:
import os
lenk = links[option]
os.system(f"youtube-dl {lenk} --get-description --skip-download --youtube-skip-dash-manifest > txt_file.txt")

f=open("txt_file.txt")
deta=f.readlines()
f.close()


buffDeta=[]
for i in deta:
    if "HONOURABLE" in i:
        buffDeta.append(i)
        
    
Judges=[]
for k in buffDeta:
    buff=k.split("HONO")
    for i in buff:
        if "URABLE" in i:
            buff1=i.split("URABLE")[-1]
            if '-' in i:
                buff1=i.split('-')[0]
                buff1=buff1.split('URABLE')[-1]
            Judges.append("HONOURABLE "+buff1.strip().replace("AND",""))
        


df=pd.DataFrame(Judges,columns=['Judge(s)'])
df.index = np.arange(1, len(df)+1)

if st.sidebar.checkbox('Name of the Judge(s)'):
    st.subheader('Name of the Judge(s)')
    st.write(df.replace({',': ''}, regex=True))

def preprocess(sent):
    sent = nltk.word_tokenize(sent)
    sent = nltk.pos_tag(sent)
    return sent

#sent = preprocess(x[0])
#st.write(pd.DataFrame(sent))

#pattern = 'NP: {<DT>?<JJ>*<NN>}'

#cp = nltk.RegexpParser(pattern)
#cs = cp.parse(sent)

#iob_tagged = tree2conlltags(cs)
#st.write(pd.DataFrame(iob_tagged))

doc = nlp(x[0])

#df = [(X.text, X.label_) for X in doc.ents]
#st.write(pd.DataFrame(df,columns=['Entity','Label']))

#df1 = [(X, X.ent_iob_, X.ent_type_) for X in doc]
#st.write(pd.DataFrame(df1,columns = ['Entity','Entity Tag','Entity Type']))

Names = []
for i in doc.ents:
    if i.label_ == 'PERSON':
        i = str(i)
        Names.append(i)

if st.sidebar.checkbox('List of Names'):
    st.subheader('List of Names')
    st.write(pd.DataFrame(Names,columns=['Names']))

#Person_list=[(X, X.ent_iob_) for X in doc if X.ent_type_=='PERSON']
#st.write(pd.DataFrame(Person_list,columns=['Name Entity','Entity Tag']))


Places = []
for i in doc.ents:
    if i.label_ == 'GPE':
        i = str(i)
        Places.append(i)

if st.sidebar.checkbox('List of Places'): 
    st.subheader('List of Places')
    st.write(pd.DataFrame(Places,columns=['Places']))

#df2 = [(X, X.ent_iob_) for X in doc if X.ent_type_=='GPE']
#st.write(pd.DataFrame(df2,columns=['Place Entity','Entity Tag']))

# Organization = []
# for i in doc.ents:
#     if i.label_ == 'ORG':
#         i = str(i)
#         Places.append(i)

# if st.sidebar.checkbox('List of Organizations'): 
#     st.subheader('List of Organizations')
#     st.write(pd.DataFrame(Places,columns=['Organization']))

cities = [i for i in doc.ents if i.label_ == 'GPE']  

geolocator = Nominatim(timeout=2,user_agent='BD-Project')
lat_lon = []
for city in cities: 
    try:
        location = geolocator.geocode(city)
        if location:
            print(location.latitude, location.longitude)
            lat_lon.append(location)
    except GeocoderTimedOut as e:
        print("Error: geocode failed on input %s with message %s"%
             (city, e))


#st.write(lat_lon)


df3 = pd.DataFrame(lat_lon, columns=['Place Name', 'Coordinates'])

# if st.sidebar.checkbox('List of Places with it\'s Co-ordinates'):
#     st.subheader('List of Places with it\'s Co-ordinates')
#     st.write(df3)

geometry = [(x[1], x[0]) for x in df3['Coordinates']]
df4 = pd.DataFrame(geometry,columns=['lon', 'lat'])

if st.sidebar.checkbox('Geographical Mapping of the all Places present in Transcription'):
    st.subheader('Geographical Mapping of the all Places present in Transcription')
    st.map(df4)


