from joblib import load

# List of two verbal statements
input_text=["I was little nervous because I was never been on the stage before", "... and she approached me, and at that time that she had tripped over the rug and kind of bumped into me I was heading to the nursery, which is ammm ... I know you guys have all seen the layout, there's ahhh ... was a loveseat right there and a rocking chair right next to each other, and there's a little wall, and whenever she tripped over the rug she bumped into me and Grant approached her and grabbed her and kind of pulled her ... was pulling her back and just telling her to chill out, and she started fighting him, and whenever she did, that little wall right by the loveseat, she kicked it and whenever she did, they both went over the chair and landed on the floor, and I just went so ... I ran to the bedroom with Lily I didn't even stick around to see what happened, and when I got to the bedroom Little Grant was trying to come ... he was coming out of the door, and he asked me what that noise was and I told him that the chair fell and ... to come back into the bedroom, and ammm ... he asked where his daddy was and I told him he was picking up the chair."]

# load models which are TfidfVectorizer for ngram and SVM classifier for binary classification
tfidf_vect_ngram=load("tfidf_ngram.joblib")
classifier=load("svm_classifier.joblib")

def verbal_statement_classifier(tfidf_vect_ngram,classifier):
    # feature dimension of 6264 and use model for predictons
    feature_vector_valid = tfidf_vect_ngram.transform(input_text)
    predictions = classifier.predict(feature_vector_valid)
    return  predictions

# List of deception labels for each corresponding verbal statements
print(verbal_statement_classifier(tfidf_vect_ngram,classifier))
