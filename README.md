# Player Performance with Machine Learning

This repository contains the final report, presentation materials, and Python scripts for estimating player transfer value using machine learning models.

Demo video: https://youtu.be/ZUixDtCBw5w

Repository structure:
- src/ : Python scripts (Main7.py, Main8.py, Main9.py)
- data/ : dataset files used by the scripts (extended_player_data.csv)
- docs/ : report and presentation files (PDF/DOCX/PPTX)
- media/ : local media (ignored by git)

Setup:
1) Install Python 3
2) Install dependencies:
pip install -r requirements.txt

Run:
You can run scripts from the repository root:
python src/Main7.py
python src/Main8.py
python src/Main9.py

Or you can run them from inside the src folder:
cd src
python Main7.py
python Main8.py
python Main9.py

What each script does:
- Main7.py: Random Forest training and evaluation, correlation analysis, clustering, and correlation heatmap.
- Main8.py: Compares multiple models (Linear Regression, Random Forest, Gradient Boosting, SVR) using MAE and R², and visualizes feature importance.
- Main9.py: Random Forest feature importance and prediction performance (MAE) with an actual vs predicted plot.

Notes:
All scripts use the dataset at data/extended_player_data.csv. If the dataset is very small, metrics such as R² may be low/negative, which is expected and does not mean the code is broken.
