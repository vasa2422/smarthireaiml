
from pathlib import Path
import pandas as pd
import re

PROJECT_DIR = Path(__file__).resolve().parent.parent
JOBS_FILE = PROJECT_DIR / "data" / "sample_jobs.csv"


def extract_skills(text, known_skills):
    """Find known skills mentioned in text."""
    text = str(text).lower()
    found = []

    for skill in known_skills:
        pattern = r"(?<!\w)" + re.escape(skill.lower()) + r"(?!\w)"
        if re.search(pattern, text):
            found.append(skill)

    return found


# Load sample jobs
jobs = pd.read_csv(JOBS_FILE).fillna("")

# Sample resume for testing
resume_text = """
Python developer with knowledge of Python, SQL,
and application development.
"""

# Select a target job
target_job_title = "Data Analyst"

matched_jobs = jobs[
    jobs["title"].str.lower() == target_job_title.lower()
]

if matched_jobs.empty:
    raise ValueError(f"Job not found: {target_job_title}")

job = matched_jobs.iloc[0]

# Extract required skills from the target job
required_skills = [
    skill.strip()
    for skill in str(job["skills"]).split(",")
    if skill.strip()
]

# Find skills in the resume
resume_skills = extract_skills(resume_text, required_skills)

# Identify missing skills
missing_skills = [
    skill for skill in required_skills
    if skill not in resume_skills
]

print("\n===== SMARTHIRE SKILL-GAP REPORT =====")
print("Target role:", job["title"])
print("Company:", job["company"])

print("\nRequired skills:")
print(", ".join(required_skills))

print("\nSkills found in resume:")
print(", ".join(resume_skills) if resume_skills else "None detected")

print("\nSkills to improve:")
print(", ".join(missing_skills) if missing_skills else "No gaps detected")

if required_skills:
    coverage = len(resume_skills) / len(required_skills) * 100
    print(f"\nDetected skill coverage: {coverage:.2f}%")
