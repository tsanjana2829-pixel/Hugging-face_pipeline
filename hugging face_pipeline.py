import pandas as pd
from transformers import pipeline


# ============================================================
# STEP 1 – Read Input Article
# ============================================================

df = pd.read_csv("dataset/input_text.csv")

text = df["text"].iloc[0]

print("Original Text:")
print(text)


# ============================================================
# STEP 2 – Sentiment Analysis
# ============================================================

sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    revision="714eb0f"
)

sentiment_result = sentiment_pipeline(
    text[:1000],
    truncation=True,
    max_length=512
)

print("\n========== SENTIMENT ANALYSIS ==========")
print(sentiment_result)


# Save Sentiment Result

sentiment_df = pd.DataFrame({
    "Text": [text],
    "Sentiment": [sentiment_result[0]["label"]],
    "Score": [sentiment_result[0]["score"]]
})

sentiment_df.to_csv(
    "dataset/sentiment_results.csv",
    index=False
)


# ============================================================
# STEP 3 – Text Summarization
# ============================================================

summarizer = pipeline(
    "summarization",
    model="sshleifer/distilbart-cnn-12-6"
)

summary = summarizer(
    text[:3000],
    max_length=100,
    min_length=30,
    do_sample=False
)

summary_text = summary[0]["summary_text"]

print("\n========== SUMMARY ==========")
print(summary_text)


# Save Summary

summary_df = pd.DataFrame({
    "Original_Text": [text],
    "Summary": [summary_text]
})

summary_df.to_csv(
    "dataset/summarized_text.csv",
    index=False
)


# ============================================================
# STEP 4 – English to French Translation
# ============================================================

translator_fr = pipeline(
    "translation",
    model="Helsinki-NLP/opus-mt-en-fr"
)

words = text.split()

chunks = []

for i in range(0, len(words), 200):
    chunks.append(" ".join(words[i:i+200]))

translated_fr_chunks = []

for chunk in chunks:
    result = translator_fr(
        chunk,
        max_length=512
    )

    translated_fr_chunks.append(
        result[0]["translation_text"]
    )

french_text = " ".join(translated_fr_chunks)

print("\n========== FRENCH TRANSLATION ==========")
print(french_text)


# ============================================================
# STEP 5 – English to Spanish Translation
# ============================================================

translator_es = pipeline(
    "translation",
    model="Helsinki-NLP/opus-mt-en-es"
)

translated_es_chunks = []

for chunk in chunks:
    result = translator_es(
        chunk,
        max_length=512
    )

    translated_es_chunks.append(
        result[0]["translation_text"]
    )

spanish = " ".join(translated_es_chunks)

print("\n========== SPANISH TRANSLATION ==========")
print(spanish)


# ============================================================
# STEP 6 – Save Translation Results
# ============================================================

translation_df = pd.DataFrame({
    "Original_Text": [text],
    "French_Translation": [french_text],
    "Spanish_Translation": [spanish]
})

translation_df.to_csv(
    "dataset/translated_text.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nTranslation file saved successfully!")


# ============================================================
# STEP 7 – Question Answering
# ============================================================

qa_pipeline = pipeline(
    "question-answering",
    model="distilbert/distilbert-base-cased-distilled-squad"
)

question = "What is one major advantage of AI in education?"

qa_result = qa_pipeline(
    question=question,
    context=text
)

print("\n========== QUESTION ANSWERING ==========")
print("Question:", question)
print("Answer:", qa_result["answer"])
print("Score:", qa_result["score"])


# Save QA Result

qa_df = pd.DataFrame({
    "Question": [question],
    "Answer": [qa_result["answer"]],
    "Score": [qa_result["score"]]
})

qa_df.to_csv(
    "dataset/qa_results.csv",
    index=False
)


# ============================================================
# STEP 8 – Final Comparison
# ============================================================

comparison_df = pd.DataFrame({
    "Original Text": [text],
    "Summary": [summary_text],
    "French Translation": [french_text],
    "Spanish Translation": [spanish],
    "Sentiment": [sentiment_result[0]["label"]],
    "Answer": [qa_result["answer"]]
})

print("\n========== FINAL COMPARISON ==========")
print(comparison_df)

comparison_df.to_csv(
    "dataset/final_comparison.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nFinal comparison saved successfully!")