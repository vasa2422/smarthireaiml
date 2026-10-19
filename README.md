# SmartHire — AI-Powered Resume Analysis & Career Guidance

An intelligent, Streamlit-based career guidance application that analyzes resumes, predicts suitable job categories, recommends relevant job listings, identifies skill gaps, and provides a personalized learning roadmap.

## Overview

SmartHire helps job seekers understand how their resumes align with job opportunities and identify skills they can develop for their target roles.

The application combines Natural Language Processing (NLP), machine learning, text similarity, and skill-keyword analysis to provide practical career guidance through an interactive web interface.

**Project type:** Machine Learning / NLP / Career Guidance  
**Application:** Streamlit Web App  
**Primary language:** Python

## Key Features

### 1. Resume Analysis
- Upload resumes in PDF, DOCX, or TXT format.
- Extract text from supported resume documents.
- Predict a resume's job category using a trained machine-learning classifier.
- Present analysis results through an interactive interface.

### 2. Matching Jobs
- Compare resume text against available job descriptions.
- Rank job listings using TF-IDF vectorization and cosine similarity or the application's configured text-similarity method.
- Display relevant information, such as job title, company, location, and similarity score.
- Allow users to explore job descriptions and listing details.

### 3. Skill Gap Analysis
- Select a target job role.
- Identify recognized skill keywords in the selected job description.
- Compare detected job requirements with keywords found in the resume.
- Display matched skills and skills to consider improving.
- Calculate skill coverage based on recognized required-skill keywords.

### 4. Personalized Learning Roadmap
- Use identified skill gaps to guide learning.
- Present a suggested four-week study routine.
- Organize the routine into fundamentals, guided practice, a mini-project, and review.
- Provide practical deliverables to help users apply their learning.

### 5. Interactive Dashboard
- Streamlit-based user interface.
- Resume upload and analysis workflow.
- Job-matching results and detailed job descriptions.
- Skill-gap metrics and learning recommendations.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Application development |
| Streamlit | Interactive web interface |
| Pandas | Dataset loading and manipulation |
| NumPy | Numerical operations where required |
| Scikit-learn | Machine learning and text processing |
| TF-IDF | Text feature extraction |
| Logistic Regression | Resume category classification |
| Cosine Similarity | Text similarity and job matching |
| Joblib / Pickle | Loading serialized machine-learning models |
| PyPDF | PDF text extraction |
| python-docx | DOCX text extraction |
| Matplotlib | Data visualization |
| Seaborn | Exploratory data analysis |
| Jupyter Notebook | Experimentation and model evaluation |

*The final dependency list should match the imports used by your actual application and training scripts.*

## Project Structure

```text
SMART_HIRE/
│
├── app/
│   ├── streamlit_app.py
│   └── streamlit_app_backup.py
│
├── data/
│   ├── raw/
│   │   ├── companies/
│   │   ├── jobs/
│   │   ├── mappings/
│   │   ├── UpdatedResumeDataSet.csv
│   │   └── sample_jobs.csv
│   │
│   ├── interim/
│   └── processed/
│       └── cleaned_jobs.csv
│
├── models/
│   ├── resume_classifier.pkl
│   └── resume_classifier_eda.pkl
│
├── notebooks/
│   └── 01_eda.ipynb
│
├── reports/
│
├── src/
│   ├── check_datasets.py
│   ├── clean_jobs.py
│   ├── inspect_data.py
│   ├── inspect_jobs.py
│   ├── job_recommender.py
│   ├── predict_resume.py
│   ├── preprocess_data.py
│   ├── skill_gap.py
│   └── train_classifier.py
│
├── requirements.txt
└── README.md
```

*This structure reflects the project files identified during development. Keep only files and paths that actually exist in your final repository.*

## Installation and Setup

### Prerequisites

Install the following before running SmartHire:

- Python 3.11 or another Python version compatible with your installed dependencies.
- Visual Studio Code or another Python editor.
- Git, if you plan to publish the project on GitHub.

### Step 1: Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd smart_hire
```

Replace the placeholder with your actual repository URL.

If you already have the project locally, open the `smart_hire` folder in VS Code instead.

### Step 2: Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can use Command Prompt:

```cmd
.venv\Scripts\activate.bat
```

### Step 3: Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Ensure that `requirements.txt` includes the packages required by the application and its supporting scripts.

### Step 4: Verify the required files

Before launching the application, verify that the model and processed job dataset exist at the paths configured in `app/streamlit_app.py`.

Expected paths:

```text
models/resume_classifier.pkl
data/processed/cleaned_jobs.csv
```

If either file is missing, run the appropriate training or data-preprocessing script after verifying its input paths.

### Step 5: Run the Streamlit application

From the project root directory, run:

```bash
python -m streamlit run app/streamlit_app.py
```

Streamlit will display a local URL in the terminal, usually:

```text
http://localhost:8501
```

Open the URL shown in your terminal to use SmartHire.

## How SmartHire Works

```text
Resume Upload
      |
      v
Text Extraction
      |
      v
Resume Category Prediction
      |
      v
Resume and Job Text Comparison
      |
      v
Ranked Job Recommendations
      |
      v
Skill Gap Analysis
      |
      v
Personalized Learning Roadmap
```

### Resume Classification

The resume classification workflow uses text preprocessing, TF-IDF feature extraction, and a trained Logistic Regression classifier.

TF-IDF transforms textual information into numerical features that a machine-learning model can process.

The classifier predicts a category based on patterns learned from its training data.

### Job Matching

The job-matching component compares resume content with job descriptions and ranks listings according to text similarity.

A higher similarity score indicates greater textual overlap under the configured method. It does not represent the probability of receiving an interview or job offer.

### Skill Gap Analysis

The application detects recognized skill keywords in the selected job description and compares them with those found in the uploaded resume.

Skill coverage is calculated as:

**Skill Coverage (%) = (Matched Required Skills / Detected Required Skills) × 100**

The report depends on the application's skill vocabulary and keyword aliases. It is not a complete assessment of technical proficiency or practical experience.

### Learning Roadmap

The roadmap translates identified skill gaps into a suggested learning routine:

- **Week 1 — Fundamentals:** Study concepts and prepare notes.
- **Week 2 — Guided Practice:** Complete exercises.
- **Week 3 — Mini-Project:** Apply skills to a practical task.
- **Week 4 — Review and Publish:** Test the work and document the project.

The plan is a starting point that users should adapt to their individual learning needs.

## Model Training and Evaluation

SmartHire includes a training workflow for its resume classification model.

To train the classifier, run the following command from the project root, provided the script's expected dataset paths are configured correctly:

```bash
python src/train_classifier.py
```

The training workflow should:

1. Load the resume dataset.
2. Prepare resume text and category labels.
3. Split the dataset into training and testing sets.
4. Extract TF-IDF text features.
5. Train the classification model.
6. Evaluate predictions using classification metrics.
7. Save the trained model to the path expected by the Streamlit application.

Evaluation can include accuracy, precision, recall, F1-score, and a confusion matrix.

**Evaluation note:** Model performance depends on the dataset, preprocessing, class distribution, and test split. Report the measured results from your final training run rather than assuming that previous results will reproduce exactly.

## Dataset Information

The project uses resume and job-listing data for classification, recommendation, and skill analysis.

The development workspace contains a resume dataset named `UpdatedResumeDataSet.csv` and job-related CSV files.

Before distributing or publishing datasets, verify their original sources, licensing terms, attribution requirements, and permission for redistribution. If redistribution is not permitted, provide instructions for obtaining the data instead of committing the files.

Do not include personal resumes, email addresses, phone numbers, credentials, or other sensitive information in a public repository.

## Testing Checklist

Before releasing SmartHire, test the following:

- [ ] A valid PDF resume uploads successfully.
- [ ] A valid DOCX resume uploads successfully.
- [ ] A valid TXT resume uploads successfully.
- [ ] Extracted text is displayed or processed correctly.
- [ ] Resume category prediction completes successfully.
- [ ] Relevant job listings are ranked and displayed.
- [ ] Job descriptions can be inspected.
- [ ] Skill-gap metrics and matched skills are displayed.
- [ ] Missing or unrecognized skill keywords are handled appropriately.
- [ ] The learning roadmap responds appropriately to the selected role.
- [ ] Invalid files, empty resumes, and missing datasets produce understandable errors.
- [ ] The application starts successfully from a clean environment.

## Limitations

- Resume categories are limited to the classes represented in the training data.
- Job similarity scores measure textual similarity, not hiring probability.
- Skill-gap results depend on recognized keywords and configured aliases.
- The quality of recommendations depends on the completeness and relevance of the job dataset.
- A detected keyword does not establish that a candidate has mastered the corresponding skill.
- The learning roadmap provides educational suggestions rather than guaranteed career outcomes.

SmartHire is an educational prototype. Its outputs should support, rather than replace, independent review and career decision-making.

## Future Enhancements

Potential improvements include:

- Expand the skill vocabulary and technical-skill aliases.
- Improve job relevance through semantic embeddings.
- Add filters for job location, experience level, and employment type.
- Add downloadable analysis reports.
- Introduce more detailed, role-specific learning roadmaps.
- Add automated tests and stronger error handling.
- Improve accessibility and responsive UI design.
- Deploy the application to a cloud hosting platform.
- Add privacy-conscious resume processing and data-retention controls.

## Contributing

Contributions and suggestions are welcome.

1. Fork the repository.
2. Create a feature branch.
3. Implement and test your changes.
4. Submit a pull request describing the improvement.

## Responsible Use and Privacy

SmartHire is intended for educational and career-guidance purposes. It should not be used as the sole basis for hiring, rejection, or other consequential employment decisions.

Process resumes responsibly, obtain appropriate permission before using personal information, and avoid committing sensitive files or secrets to version control.

## License

Choose a license before publishing the project publicly. Add the corresponding `LICENSE` file and update this section with the selected license. Do not claim an open-source license until you have selected and added one.

## Author

**SmartHire Project**

Developed as a machine-learning and career-guidance project using Python and Streamlit.

---

**SmartHire — Understand your skills. Discover opportunities. Plan your growth.**