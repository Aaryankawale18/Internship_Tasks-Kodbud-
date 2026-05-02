import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")

HAM_MESSAGES = [
    "Hey, are we still on for lunch tomorrow?",
    "Can you send me the project report by Friday?",
    "Happy birthday! Hope you have a wonderful day.",
    "The meeting has been rescheduled to 3 PM.",
    "Please find the attached invoice for your records.",
    "Thanks for your help with the presentation!",
    "I'll be home late tonight, don't wait up.",
    "Can we schedule a call to discuss the proposal?",
    "Your subscription has been renewed successfully.",
    "The package was delivered to your doorstep.",
    "Let me know if you need any changes to the document.",
    "Great job on finishing the task ahead of schedule!",
    "I've reviewed your code and left some comments.",
    "Dinner is at 7, see you there!",
    "Could you please review the attached contract?",
    "The conference starts at 9 AM on Monday.",
    "Your appointment is confirmed for next Tuesday.",
    "I enjoyed our conversation at the networking event.",
    "Please update the shared spreadsheet with your hours.",
    "Thanks for covering for me while I was out.",
    "The quarterly report is ready for your review.",
    "Just checking in — how are things going?",
    "Your feedback has been noted and we'll follow up.",
    "We're looking forward to meeting your team next week.",
    "The system maintenance window is tonight from 2-4 AM.",
    "Can you forward me the email chain from last week?",
    "Our weekly sync is moved to Wednesday.",
    "Please RSVP by end of day Thursday.",
    "Your order has been shipped and will arrive by Friday.",
    "Looking forward to your presentation at the summit.",
]

SPAM_MESSAGES = [
    "CONGRATULATIONS! You've won a $1,000 gift card! Click NOW!",
    "FREE iPhone! Limited time offer. Claim your prize today!",
    "You have been selected for an exclusive cash reward. Act fast!",
    "URGENT: Your bank account has been compromised. Verify immediately!",
    "Make $5000 a week working from home — no experience needed!",
    "Hot singles in your area are waiting to meet you!",
    "Buy cheap Viagra online — no prescription needed!",
    "Nigerian prince needs your help transferring $10 million!",
    "You are the lucky winner of our international lottery!",
    "CLICK HERE to claim your FREE vacation package!!!",
    "Lose 30 pounds in 30 days — doctors HATE this trick!",
    "Investment opportunity: 500% returns guaranteed!",
    "Your PayPal account has been suspended. Verify NOW.",
    "Exclusive deal: Designer bags 90% off. Limited stock!",
    "Earn money fast — join our network marketing team!",
    "You have a pending IRS refund. Claim it immediately!",
    "FREE credit score check — no strings attached!",
    "This is not a joke — you've won $50,000! Reply ASAP!",
    "Cheap meds online — save 80% on prescriptions!",
    "Make money fast with our proven crypto trading bot!",
    "Congratulations! You qualify for a pre-approved loan!",
    "LIMITED OFFER: Enlargement pills — buy 2 get 3 free!",
    "Your email was randomly selected. Claim your prize now!",
    "WARNING: Viruses detected on your computer. Call us NOW!",
    "Unlock the secret to passive income — join FREE today!",
    "Hot stock tip: Buy before the price explodes tomorrow!",
    "Your account will be deleted unless you verify immediately!",
    "Earn $500/day from home — no skills required!",
    "Free gift card waiting — click to redeem in 60 seconds!",
    "Act now — this once-in-a-lifetime offer expires tonight!",
]

data = pd.DataFrame({
    "message": HAM_MESSAGES + SPAM_MESSAGES,
    "label":   ["ham"] * len(HAM_MESSAGES) + ["spam"] * len(SPAM_MESSAGES)
})
data = data.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

print("=" * 60)
print("       SPAM EMAIL CLASSIFIER — scikit-learn")
print("=" * 60)
print(f"\n📊 Dataset: {len(data)} messages  |  "
      f"Ham: {(data.label=='ham').sum()}  |  "
      f"Spam: {(data.label=='spam').sum()}")

X_train, X_test, y_train, y_test = train_test_split(
    data["message"], data["label"],
    test_size=0.25, random_state=42, stratify=data["label"]
)
print(f"\n🔀 Train size: {len(X_train)}  |  Test size: {len(X_test)}")

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=5000,
        sublinear_tf=True   
    )),
    ("clf", MultinomialNB(alpha=0.1))   
])

pipeline.fit(X_train, y_train)
print("\n✅ Model trained successfully!")

y_pred = pipeline.predict(X_test)

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, pos_label="spam")
rec  = recall_score(y_test, y_pred, pos_label="spam")
f1   = f1_score(y_test, y_pred, pos_label="spam")

print("\n" + "─" * 60)
print("📈 EVALUATION METRICS")
print("─" * 60)
print(f"  Accuracy  : {acc*100:.1f}%")
print(f"  Precision : {prec*100:.1f}%")
print(f"  Recall    : {rec*100:.1f}%")
print(f"  F1 Score  : {f1*100:.1f}%")

print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred, target_names=["ham", "spam"]))

cm = confusion_matrix(y_test, y_pred, labels=["ham", "spam"])
print("🧩 Confusion Matrix:")
print(f"              Predicted Ham  Predicted Spam")
print(f"  Actual Ham       {cm[0][0]:<6}         {cm[0][1]}")
print(f"  Actual Spam      {cm[1][0]:<6}         {cm[1][1]}")

def classify(message: str) -> dict:
    pred   = pipeline.predict([message])[0]
    proba  = pipeline.predict_proba([message])[0]
    classes = pipeline.classes_
    prob_dict = dict(zip(classes, proba))
    return {
        "message":      message,
        "prediction":   pred.upper(),
        "ham_prob":     f"{prob_dict['ham']*100:.1f}%",
        "spam_prob":    f"{prob_dict['spam']*100:.1f}%",
        "verdict":      "🚨 SPAM" if pred == "spam" else "✅ HAM"
    }

new_messages = [
    "Congratulations! You've won a FREE iPhone. Click here to claim!",
    "Hi John, can we reschedule our 3 PM meeting to tomorrow?",
    "URGENT: Your account will be suspended unless you act NOW!!!",
    "Please review the attached quarterly report before Friday.",
    "Make $10,000 per week from home — guaranteed!!!",
    "Your order #ORD-2891 has been shipped and will arrive by Thursday.",
]

print("\n" + "─" * 60)
print("🔍 PREDICTIONS ON NEW MESSAGES")
print("─" * 60)
for msg in new_messages:
    result = classify(msg)
    print(f"\n  {result['verdict']}")
    print(f"  Message : \"{result['message'][:65]}{'...' if len(result['message'])>65 else ''}\"")
    print(f"  Ham: {result['ham_prob']}  |  Spam: {result['spam_prob']}")

print("\n" + "=" * 60)
print("  Pipeline: TF-IDF (bigrams, 5k features) → Naive Bayes")
print("=" * 60)
