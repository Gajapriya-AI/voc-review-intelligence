"""Optional Deep Learning experiment. Install requirements-dl.txt before running.

This is deliberately separate from the lightweight default app so a small-storage
computer can run the baseline without installing TensorFlow unless desired.
"""
from pathlib import Path
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data'/'training_reviews.csv'
OUT=ROOT/'models'/'bilstm'


def main():
    try:
        import tensorflow as tf
        from tensorflow.keras.preprocessing.text import Tokenizer
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        from tensorflow.keras import Sequential
        from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
        from tensorflow.keras.callbacks import EarlyStopping
    except ImportError as exc:
        raise SystemExit('TensorFlow is optional. Install it with: pip install -r requirements-dl.txt') from exc
    df=pd.read_csv(DATA).dropna(subset=['review','sentiment'])
    df['review']=df['review'].astype(str).str.strip()
    df=df[(df.review!='') & df.sentiment.isin(['Positive','Negative','Neutral'])]
    X_train,X_test,y_train,y_test=train_test_split(df.review,df.sentiment,test_size=.2,random_state=42,stratify=df.sentiment)
    tokenizer=Tokenizer(num_words=20000,oov_token='<OOV>')
    tokenizer.fit_on_texts(X_train)
    xtr=pad_sequences(tokenizer.texts_to_sequences(X_train),maxlen=120,padding='post',truncating='post')
    xte=pad_sequences(tokenizer.texts_to_sequences(X_test),maxlen=120,padding='post',truncating='post')
    encoder=LabelEncoder(); ytr=encoder.fit_transform(y_train); yte=encoder.transform(y_test)
    model=Sequential([Embedding(20000,64, input_length=120),Bidirectional(LSTM(32)),Dropout(.3),Dense(32,activation='relu'),Dropout(.2),Dense(len(encoder.classes_),activation='softmax')])
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    model.fit(xtr,ytr,validation_split=.15,epochs=8,batch_size=16,callbacks=[EarlyStopping(monitor='val_loss',patience=2,restore_best_weights=True)],verbose=2)
    probs=model.predict(xte,verbose=0); pred=probs.argmax(axis=1)
    OUT.mkdir(parents=True,exist_ok=True)
    model.save(OUT/'bilstm_sentiment.keras')
    (OUT/'tokenizer.json').write_text(tokenizer.to_json(),encoding='utf-8')
    (OUT/'labels.json').write_text(json.dumps(encoder.classes_.tolist(),indent=2),encoding='utf-8')
    metrics={'test_rows':len(yte),'accuracy':float(accuracy_score(yte,pred)),'macro_f1':float(f1_score(yte,pred,average='macro')),'labels':encoder.classes_.tolist(),'classification_report':classification_report(yte,pred,target_names=encoder.classes_,output_dict=True,zero_division=0),'confusion_matrix':confusion_matrix(yte,pred).tolist(),'note':'Metrics are from the bundled synthetic starter dataset only; not evidence of real-world accuracy.'}
    (OUT/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in metrics.items() if k not in ('classification_report','confusion_matrix')},indent=2))

if __name__=='__main__': main()
