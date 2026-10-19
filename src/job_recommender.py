
from pathlib import Path
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_DIR = Path(__file__).resolve().parent.parent
JOBS_FILE = PROJECT_DIR / "data" / "processed" / "cleaned_jobs.csv"

jobs = pd.read_csv(JOBS_FILE).fillna("")

# Remove records without useful text
jobs = jobs[
    (jobs["title"].astype(str).str.strip() != "") &
    (jobs["description"].astype(str).str.strip() != "")
].copy()

# Combine job title and description
jobs["job_text"] = (
    jobs["title"].astype(str) + " " +
    jobs["description"].astype(str)
)

# Sample resume for testing
resume_text = """
Python developer with experience in Python, SQL, Pandas,
machine learning, scikit-learn, and data analysis.
Built projects using Flask and data visualization tools.
"""

print("Preparing real job postings...")

# Convert job text to numerical features
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=20000,
    ngram_range=(1, 2)
)

job_vectors = vectorizer.fit_transform(jobs["job_text"])
resume_vector = vectorizer.transform([resume_text])

# Calculate similarity
scores = cosine_similarity(resume_vector, job_vectors).flatten()
jobs["match_score"] = scores * 100

# Display top 10 results
ranked_jobs = jobs.sort_values(
    "match_score", ascending=False
).head(10)

print("\n===== SMARTHIRE REAL JOB RECOMMENDATIONS =====")

for rank, (_, job) in enumerate(ranked_jobs.iterrows(), start=1):
    print(f"\nRank: {rank}")
    print("Job title:", job["title"])
    print("Company:", job["company_name"])
    print("Location:", job["location"])
    print(f"Similarity score: {job['match_score']:.2f}%")
    print("Description:", str(job["description"])[:250], "...")

print("\nTotal jobs searched:", len(jobs))
print("Recommendation engine completed!")
