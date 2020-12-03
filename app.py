#Necessary imports
import streamlit as st
import pandas as pd
import regex as re
from matplotlib import pyplot as plt
from textblob import TextBlob
from gensim.summarization.summarizer import summarize 
from gensim.summarization import keywords
import spacy
from spacy import displacy
from collections import Counter
import webbrowser
import en_core_web_sm

nlp = spacy.load('en_core_web_sm')


#Headings for Web Application
st.title("Natural Language Processing Web Application")
st.subheader("We have performed three major tasks in this Application :")
st.markdown("1. Sentiment Analysis : To identify whether the input sentence is positive or negative or neutral")
st.markdown("2. Entity Extraction : To breakdown your input into different parts")
st.markdown("3. Text Summarization : To summarize your large paragraph into its compact version\n")
st.subheader("Note: In order to use the third task we need to enter at least ten sentences(try copying a paragraph from wikipedia)")

st.sidebar.subheader('About the Creator:')
st.sidebar.markdown('Manav Nitin Kapadnis')
st.sidebar.markdown('Sophomore | IIT Kharagpur')

github_url = 'https://github.com/manavkapadnis'
if st.sidebar.button('Github'):
    webbrowser.open_new_tab(github_url)

linkedin_url = 'https://www.linkedin.com/in/manav-nitin-kapadnis-013b94192/'
if st.sidebar.button('LinkedIn'):
    webbrowser.open_new_tab(linkedin_url)

k_url = 'https://www.kaggle.com/darthmanav'
if st.sidebar.button('Kaggle'):
    webbrowser.open_new_tab(k_url)



#Picking what NLP task you want to do
option = st.selectbox('Which task would you like to see ?',('Sentiment Analysis', 'Entity Extraction', 'Text Summarization')) #option is stored in this variable

#Textbox for text user is entering
st.subheader("Enter the text you'd like to analyze:")
text = st.text_input('Please enter text which is more than 10 words') #text is stored in this variable
text_copy=text

#Display results of the NLP task
st.header("Results")

#Function to take in dictionary of entities, type of entity, and returns specific entities of specific type
def entRecognizer(entDict, typeEnt):
    entList = [ent for ent in entDict if entDict[ent] == typeEnt]
    return entList




# Function For Analysing Tokens and Lemma
@st.cache
def text_analyzer(my_text):
	nlp = spacy.load('en_core_web_sm')
	docx = nlp(my_text)
	# tokens = [ token.text for token in docx]
	allData = [('"Token":{},\n"Lemma":{}'.format(token.text,token.lemma_))for token in docx ]
	return allData





#Sentiment Analysis
if option == 'Sentiment Analysis':

    #Creating graph for sentiment across each sentence in the text inputted
    sents = text.split('.')
    entireText = TextBlob(text)
    sentScores = []
    for sent in sents:
        text = TextBlob(sent)
        score = text.sentiment[0]
        sentScores.append(score)

    if st.checkbox("Show Tokenization process"): 
    	st.subheader("Tokenize Your Text")
    	if st.button("Analyze"): 
    		nlp_result = text_analyzer(text_copy)
    		st.json(nlp_result)     
        #Plotting sentiment scores per sentencein linegraph 
    		#st.line_chart(sentScores)

    #Polarity and Subjectivity of the entire text inputted
    sentimentTotal = entireText.sentiment
    p=entireText.sentiment.polarity
    s=entireText.sentiment.subjectivity
    st.markdown("The sentiment of the overall text below.")
    st.write('Polarity means how positive or negative your sentence is, it varies between (-1,1)')
    st.markdown('Subjectivity means how subjective your sentence is, it varies between (0,1)')
    st.success(sentimentTotal)

    

#Named Entity Recognition
elif option == 'Entity Extraction':

    #Getting Entity and type of Entity
    entities = []
    entityLabels = []
    doc = nlp(text)
    for ent in doc.ents:
        entities.append(ent.text)
        entityLabels.append(ent.label_)
    entDict = dict(zip(entities, entityLabels)) #Creating dictionary with entity and entity types

    #Using function to create lists of entities of each type
    entOrg = entRecognizer(entDict, "ORG")
    entCardinal = entRecognizer(entDict, "CARDINAL")
    entPerson = entRecognizer(entDict, "PERSON")
    entDate = entRecognizer(entDict, "DATE")
    entGPE = entRecognizer(entDict, "GPE")

    #Displaying entities of each type
    st.write("Organization Entities: " + str(entOrg))
    st.write("Cardinal Entities: " + str(entCardinal))
    st.write("Personal Entities: " + str(entPerson))
    st.write("Date Entities: " + str(entDate))
    st.write("Geographical positional Entities: " + str(entGPE))

#Text Summarization
else:
    summWords = summarize(text)
    st.subheader("Summary")
    st.write(summWords)


