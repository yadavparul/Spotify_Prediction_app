import pandas as pd
import numpy as np
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import warnings
import streamlit as st
warnings.filterwarnings("ignore")

data_00 = pd.read_csv("dataset-of-00s.csv")
data_10 = pd.read_csv("dataset-of-10s.csv")
data_60 = pd.read_csv("dataset-of-60s.csv")
data_70 = pd.read_csv("dataset-of-70s.csv")
data_80 = pd.read_csv("dataset-of-80s.csv")
data_90 = pd.read_csv("dataset-of-90s.csv")

stacked = sorted(glob('dataset-of-*.csv'))
df = pd.concat((pd.read_csv(file).assign(filename = file)
          for file in stacked),ignore_index = True)

features_response = df.columns.tolist()
items_to_remove = ['track','uri','artist','filename','key','mode','time_sign','chorus_hit','sections']
features_response = [item for item in features_response if item not in items_to_remove]

data=df[features_response]
data=np.array(data)

X=data[1:, :-1]
y=data[1:,-1]

X.astype('int')
y.astype('int')

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=24)

rf = RandomForestClassifier(
    n_estimators=200, criterion='gini', max_depth=9,
    min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
    max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0,
    min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=None,
    random_state=4, verbose=0, warm_start=False, class_weight=None
)

rf.fit(X_train,y_train)

def predict_default(danceability,energy,loudness,speechiness,acousticness,instrumentalness,liveness,valence,tempo,duration_ms):
    input=np.array([[danceability,energy,loudness,speechiness,acousticness,instrumentalness,liveness,valence,tempo,duration_ms]]).astype(np.float64)
    prediction=rf.predict_proba(input)
    pred='{0:.{1}f}'.format(prediction[0][0],2)
    print(type(pred))
    return float(pred)

def main():
    html_temp="""
    <div style="background-color:#025246;padding:10px">
    <h2 style="color:white;text-align:center;">Spotify Hit Prediction</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    danceability=st.text_input("Danceability","")
    energy=st.text_input("Energy","")
    loudness=st.text_input("Loudness","")
    speechiness=st.text_input("Speechiness","")
    acousticness=st.text_input("Acousticness","")
    instrumentalness=st.text_input("Instrumentalness","")
    liveness=st.text_input("Liveness","")
    valence=st.text_input("Valence","")
    tempo=st.text_input("Tempo","")
    duration_ms=st.text_input("Duration_ms","")
    
    flop_html="""
        <div style="background-color:#F4D03F;padding:10px>
            <h2 style="color:white;text-align:center;">Song is less likely to go hit</h2>
            </div>
    """
    hit_html="""
        <div style="background-color:#F4D03F;padding:10px>
            <h2 style="color:white;text-align:center;">Song is more likely to get hit</h2>
            </div>
    """

    if st.button("Predict"):
        output=predict_default(danceability,energy,loudness,speechiness,acousticness,instrumentalness,liveness,valence,tempo,duration_ms)
        st.success('The hit probability is {}'.format(output))

        if output>0.5:
            st.markdown(hit_html,unsafe_allow_html=True)
        else:
            st.markdown(flop_html,unsafe_allow_html=True)

if __name__=='__main__':
    main()

