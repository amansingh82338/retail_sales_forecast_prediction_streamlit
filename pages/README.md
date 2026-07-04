# A/B Testing Analysis & Business Recommendations 🧪📊

An end-to-end A/B testing pipeline demonstrating data cleaning, exploratory data analysis (EDA), statistical hypothesis testing, customer segmentation, and actionable business recommendations. 

This project goes beyond simple p-value calculations by integrating a full modular codebase, segmented behavioral analysis (to detect Simpson's Paradox), and an interactive Streamlit dashboard to communicate findings to stakeholders.

---

## 🎯 Business Objective
The core objective of this project is to evaluate the performance of a new landing page (Treatment) against the existing landing page (Control). The analysis determines if the new design drives a statistically significant increase in **Conversion Rate** and **Revenue**, without degrading the user experience for key demographic segments.

## 🛠️ Tech Stack & Skills Demonstrated
* **Language:** Python 3
* **Data Manipulation:** Pandas, NumPy
* **Statistical Analysis:** SciPy, Statsmodels (Z-Test, T-Test, Chi-Square, Confidence Intervals, Effect Size)
* **Data Visualization:** Matplotlib, Seaborn
* **Web Dashboard:** Streamlit
* **Concepts:** A/B Testing Methodology, Hypothesis Testing, Customer Segmentation, Data Storytelling, Modular Software Architecture

---

## 📂 Project Structure

```text
ab_testing_analysis/
│
├── data/
│   ├── raw/                        # Original, immutable dataset
│   └── processed/                  # Cleaned data and aggregated summaries
│
├── notebooks/                      # Jupyter notebooks for step-by-step analysis
│   ├── 01_data_cleaning.ipynb
│   ├── 02_exploratory_data_analysis.ipynb
│   ├── 03_statistical_testing.ipynb
│   ├── 04_customer_segmentation.ipynb
│   └── 05_business_recommendations.ipynb
│
├── src/                            # Modular Python scripts containing reusable logic
│   ├── data_cleaning.py
│   ├── eda.py
│   ├── statistics.py
│   ├── visualization.py
│   └── utils.py
│
├── dashboards/                     # Streamlit web application
│   ├── streamlit_app.py
│   ├── pages/
│   │   ├── 1_Overview.py
│   │   ├── 2_Experiment_Analysis.py
│   │   ├── 3_Segment_Insights.py
│   │   ├── 4_Statistical_Test.py
│   │   └── 5_Business_Recommendations.py
│   └── assets/                     # Custom CSS and images
│
├── reports/                        # Exported final reports and figures
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation
🚀 Setup & Installation
To run this project locally, follow these steps:

1. Clone the repository:

Bash
git clone [https://github.com/yourusername/ab_testing_analysis.git](https://github.com/yourusername/ab_testing_analysis.git)
cd ab_testing_analysis
2. Create a virtual environment (optional but recommended):

Bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
3. Install dependencies:

Bash
pip install -r requirements.txt
💻 How to Use
1. View the Analysis Notebooks
Navigate to the notebooks/ directory to view the step-by-step analytical process. The notebooks are designed to be read sequentially, starting from data cleaning and ending with executive recommendations.

2. Run the Streamlit Dashboard
To view the interactive dashboard, run the following command from the root directory:

Bash
streamlit run dashboards/streamlit_app.py
This will launch a local web server (usually at http://localhost:8501) where you can interact with the experiment KPIs, statistical results, and segment heatmaps.

📊 Key Findings (Executive Summary)
Overall Lift: The Treatment group demonstrated a statistically significant lift in conversion rate compared to the Control group.

Statistical Rigor: A Two-Proportion Z-Test confirmed the results with a p-value < 0.05, and Chi-Square tests confirmed that traffic was evenly and randomly distributed across devices.

Segment Insights: The new page performed exceptionally well among Desktop users and the 55+ age cohort, while slightly underperforming on Mobile devices.

Recommendation: Proceed with a phased rollout of the new landing page, while initiating a follow-up product investigation to optimize the mobile responsive design.

✉️ Author
Aman Singh Chauhan