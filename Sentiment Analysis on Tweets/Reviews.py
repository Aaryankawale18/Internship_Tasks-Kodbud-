import re
import string
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score)
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")

POSITIVE = [
    "Absolutely love this product! Works exactly as described and arrived quickly.",
    "Best purchase I've made this year. Quality is outstanding.",
    "Five stars! This exceeded all my expectations. Highly recommend.",
    "Amazing quality for the price. Very satisfied with this purchase.",
    "Great product, fast shipping, and excellent customer service.",
    "Wow, this is exactly what I needed. Works perfectly, no complaints!",
    "Incredible value. I bought two more for gifts. Everyone loves them.",
    "Super easy to set up and use. My kids absolutely love it.",
    "The quality is top notch. Feels premium and very durable.",
    "This product changed my daily routine. So much better than expected!",
    "Delivered on time and packaging was perfect. Product works great.",
    "Highly recommend to everyone! Life-changing little gadget.",
    "Beautiful design and works flawlessly. Couldn't be happier.",
    "Solid build quality, intuitive interface. Exactly what I was looking for.",
    "Outstanding performance. I've had it for 3 months and zero issues.",
    # Tweets
    "Just got my new headphones and omg the sound quality is INCREDIBLE!! 🎧❤️",
    "loving this new app update!! so much smoother and faster now 🙌",
    "This coffee maker is a game changer. Best mornings ever!! ☕😍",
    "Can't believe how good this product is for the price. Total win! 💯",
    "Finally found a skincare product that actually works!! My skin is glowing 🌟",
    "shoutout to @BrandX for the best customer service I've ever experienced!",
    "This book is absolutely amazing. I couldn't put it down all weekend! 📚",
    "New running shoes arrived and they feel like clouds ☁️ best purchase ever",
    "So impressed with the battery life on this laptop. Still going strong!",
    "Tried the new restaurant downtown — food was phenomenal! Will definitely return 🍽️",
    "This workout routine is genuinely life-changing. Feel amazing every day!",
    "The delivery was so fast and the packaging was beautiful. Very impressed!",
    "Update: 2 weeks later and this product still works perfectly. Love it!",
    "Honestly the best investment I made this year. Totally worth every penny.",
    "My productivity has doubled since using this app. Highly recommend! 🚀",
]

NEGATIVE = [
    "Terrible product. Broke after just two days of use. Complete waste of money.",
    "Very disappointed. Nothing like the photos. Cheap material and bad quality.",
    "Do not buy this. It stopped working after a week. Total junk.",
    "Worst purchase ever. Returned it immediately. Save your money.",
    "Arrived broken. Customer support was unhelpful and rude.",
    "Overpriced garbage. Fell apart in the first hour. Zero stars if I could.",
    "This product is a scam. Doesn't work as advertised at all.",
    "Huge disappointment. Waited 3 weeks and received a damaged item.",
    "Horrible quality. Smells terrible and looks nothing like the pictures.",
    "Would not recommend to anyone. Poor build quality and bad instructions.",
    "Returned immediately. It leaked everywhere and ruined my counter.",
    "False advertising. The product is tiny and doesn't do what it claims.",
    "Sent the wrong color AND wrong size. Won't be ordering again.",
    "Stopped working after 5 days. Absolute rubbish for this price.",
    "Customer service ignored my emails for 2 weeks. Still no refund.",
    # Tweets
    "This app is so buggy it crashed THREE times in one hour. Uninstalling now 😤",
    "just got my order and it's completely broken. this is unacceptable!! 😡",
    "Waited 45 min for customer support and they never helped. Never again!",
    "I can't believe how bad this product is. Total waste of money 💸",
    "Worst experience ever. The product stopped working on day one. AVOID",
    "Honestly so frustrated rn. This is the 3rd defective item I've received 🤬",
    "The new update completely broke my workflow. Thanks for nothing @AppX",
    "paid premium price for absolute garbage. so disappointed and angry rn",
    "This is ridiculous. The battery drains in 2 hours. Totally useless. 👎",
    "Returned it. Cheap plastic, bad smell, doesn't work. 0/10 would not recommend",
    "Been waiting 3 weeks for my order. No updates, no response. Awful service.",
    "This product made my problem WORSE. How does this have good reviews?!",
    "What a scam. The 'features' don't work and support is ghosting me.",
    "Never buying from this brand again. Absolutely terrible experience.",
    "The product looks nothing like advertised. Cheap knockoff quality. 😠",
]

data = pd.DataFrame({
    "review": POSITIVE + NEGATIVE,
    "label":  ["positive"] * len(POSITIVE) + ["negative"] * len(NEGATIVE)
})
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

STOPWORDS = {
    "i","me","my","myself","we","our","ours","ourselves","you","your","yours",
    "yourself","yourselves","he","him","his","himself","she","her","hers",
    "herself","it","its","itself","they","them","their","theirs","themselves",
    "what","which","who","whom","this","that","these","those","am","is","are",
    "was","were","be","been","being","have","has","had","having","do","does",
    "did","doing","a","an","the","and","but","if","or","because","as","until",
    "while","of","at","by","for","with","about","against","between","into",
    "through","during","before","after","above","below","to","from","up","down",
    "in","out","on","off","over","under","again","further","then","once","here",
    "there","when","where","why","how","all","both","each","few","more","most",
    "other","some","such","no","nor","not","only","own","same","so","than","too",
    "very","s","t","can","will","just","don","should","now","d","ll","m","o",
    "re","ve","y","ain","aren","couldn","didn","doesn","hadn","hasn","haven",
    "isn","ma","mightn","mustn","needn","shan","shouldn","wasn","weren","won","wouldn"
}

def simple_stem(word: str) -> str:
    suffixes = ["ingly","ation","ness","ment","ful","ing","ed","er","ly","es","s"]
    for suf in suffixes:
        if word.endswith(suf) and len(word) - len(suf) >= 3:
            return word[: -len(suf)]
    return word

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)          
    text = re.sub(r"@\w+", "", text)                      
    text = re.sub(r"#(\w+)", r"\1", text)                 
    text = re.sub(r"[^\w\s]", " ", text)                  
    text = re.sub(r"\d+", "", text)                      
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 2]
    tokens = [simple_stem(t) for t in tokens]             
    return " ".join(tokens)

data["cleaned"] = data["review"].apply(clean_text)
print("=" * 65)
print("      SENTIMENT ANALYSIS — Tweets & Product Reviews")
print("=" * 65)
print(f"\n📊 Dataset: {len(data)} samples  |  "
      f"Positive: {(data.label=='positive').sum()}  |  "
      f"Negative: {(data.label=='negative').sum()}")

print("\n🧹 TEXT CLEANING EXAMPLES:")
print("─" * 65)
samples = data.sample(3, random_state=7)
for _, row in samples.iterrows():
    print(f"  Original : {row['review'][:70]}...")
    print(f"  Cleaned  : {row['cleaned'][:70]}")
    print(f"  Label    : {row['label'].upper()}\n")

X_train, X_test, y_train, y_test = train_test_split(
    data["cleaned"], data["label"],
    test_size=0.25, random_state=42, stratify=data["label"]
)

tfidf_params = dict(
    ngram_range=(1, 2),
    max_features=8000,
    sublinear_tf=True,
    min_df=1
)

models = {
    "Logistic Regression": Pipeline([
        ("tfidf", TfidfVectorizer(**tfidf_params)),
        ("clf",   LogisticRegression(max_iter=1000, C=1.0, random_state=42))
    ]),
    "Naive Bayes": Pipeline([
        ("tfidf", TfidfVectorizer(**tfidf_params)),
        ("clf",   MultinomialNB(alpha=0.5))
    ]),
    "Linear SVM": Pipeline([
        ("tfidf", TfidfVectorizer(**tfidf_params)),
        ("clf",   LinearSVC(C=1.0, random_state=42, max_iter=2000))
    ]),
}

print("─" * 65)
print("📈 MODEL COMPARISON")
print("─" * 65)
print(f"  {'Model':<22} {'Accuracy':>9}  {'CV Mean':>9}  {'CV Std':>8}")
print(f"  {'─'*22:<22} {'─'*9:>9}  {'─'*9:>9}  {'─'*8:>8}")

best_model, best_acc = None, 0
results = {}

for name, pipe in models.items():
    pipe.fit(X_train, y_train)
    acc = accuracy_score(y_test, pipe.predict(X_test))
    cv  = cross_val_score(pipe, data["cleaned"], data["label"], cv=5, scoring="accuracy")
    results[name] = {"pipeline": pipe, "acc": acc, "cv": cv}
    star = " ⭐" if acc > best_acc else ""
    print(f"  {name:<22} {acc*100:>8.1f}%  {cv.mean()*100:>8.1f}%  ±{cv.std()*100:>5.1f}%{star}")
    if acc > best_acc:
        best_acc = acc
        best_model = name
best_pipe = results[best_model]["pipeline"]
y_pred    = best_pipe.predict(X_test)

print(f"\n🏆 Best Model: {best_model}  ({best_acc*100:.1f}% accuracy)")
print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred,
                             target_names=["negative", "positive"]))

cm = confusion_matrix(y_test, y_pred, labels=["negative", "positive"])
print("🧩 Confusion Matrix:")
print(f"                   Pred Negative  Pred Positive")
print(f"  Actual Negative      {cm[0][0]:<6}         {cm[0][1]}")
print(f"  Actual Positive      {cm[1][0]:<6}         {cm[1][1]}")

lr_pipe = results["Logistic Regression"]["pipeline"]
feat_names = lr_pipe.named_steps["tfidf"].get_feature_names_out()
coefs      = lr_pipe.named_steps["clf"].coef_[0]
top_n = 8

top_pos_idx = np.argsort(coefs)[-top_n:][::-1]
top_neg_idx = np.argsort(coefs)[:top_n]

print(f"\n🔑 TOP SENTIMENT KEYWORDS (Logistic Regression)")
print("─" * 65)
print(f"  {'✅ Positive words':<35} {'❌ Negative words'}")
print(f"  {'─'*33:<35} {'─'*28}")
for i in range(top_n):
    pw = feat_names[top_pos_idx[i]]
    nw = feat_names[top_neg_idx[i]]
    print(f"  {pw:<35} {nw}")

def predict_sentiment(text: str, pipe=best_pipe, model_name=best_model) -> dict:
    cleaned  = clean_text(text)
    pred     = pipe.predict([cleaned])[0]
    if hasattr(pipe.named_steps["clf"], "predict_proba"):
        proba = pipe.predict_proba([cleaned])[0]
        classes = pipe.classes_
        conf = dict(zip(classes, proba))
        confidence = f"{max(proba)*100:.1f}%"
    else:
        score = pipe.decision_function([cleaned])[0]
        prob_pos = 1 / (1 + np.exp(-score))
        confidence = f"{max(prob_pos, 1-prob_pos)*100:.1f}%"
        conf = {"positive": prob_pos, "negative": 1 - prob_pos}
    emoji = "😊 POSITIVE" if pred == "positive" else "😠 NEGATIVE"
    return {"text": text, "sentiment": emoji,
            "confidence": confidence, "cleaned": cleaned}

new_texts = [
    "This laptop is absolutely fantastic! Battery lasts all day and it's super fast.",
    "Terrible battery life and the screen flickered after one week. Returning it.",
    "Okay product I guess, nothing special but it works fine.",
    "OMG I love this mascara so much!! Makes my lashes look incredible 😍",
    "Ordered twice and both times the item arrived damaged. Never again!!!",
    "Pretty good phone for the price. Camera could be better but overall satisfied.",
    "just tried the new update and it completely broke my settings 😤 fix this NOW",
    "Incredible service!! They replaced my faulty unit within 24 hours. Impressed!",
]

print(f"\n🔍 PREDICTING NEW TEXT  (Model: {best_model})")
print("─" * 65)
for txt in new_texts:
    r = predict_sentiment(txt)
    print(f"\n  {r['sentiment']}  (confidence: {r['confidence']})")
    print(f"  → \"{r['text'][:72]}{'...' if len(r['text'])>72 else ''}\"")

print("\n" + "=" * 65)
print("  Pipeline: Clean → TF-IDF bigrams → Logistic Regression")
print("  NLP Steps: lowercase · URL/mention strip · punctuation")
print("             remove · stopword filter · suffix stemming")
print("=" * 65)
