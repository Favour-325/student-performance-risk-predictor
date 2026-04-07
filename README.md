# Student Performance Risk Predictor 🎓🤖

An end-to-end, full-stack Machine Learning web application designed to identify students at academic risk. This project demonstrates a complete MLOps pipeline—from experimental data science in the "lab" to a production-grade cloud deployment using **FastAPI**, **AWS S3**, **Nginx**, and **Gunicorn**.

## 📁 Project Architecture & Structure

The repository is structured to showcase both the **Data Science** process and the **Software Engineering** implementation:

* **`app/`**: The production backend.
    * `ml_models/`: Contains the serialized models before cloud storage.
    * `static/`: The web interface. A clean, responsive HTML/CSS/JS frontend that allows users to input student metrics and receive real-time risk assessments.
    * `main.py`: The FastAPI core which handles prediction logic, manages **AWS S3** model fetching, and serves the static frontend.
    * `schema.py`: The Pydantic models used for validating the user inputs.
* **`artifacts/`**: Contains the graphs and other images illustrating various insights about the project.
* **`data/`**: Contains the dataset used for training the models.
* **`lab/`**: 🧪 **The Research Engine.** This folder contains the notebooks used for:
    * **Exploratory Data Analysis (EDA)**: Visualizing correlations in student data.
    * **Feature Engineering**: Building and testing custom preprocessors (encoders/scalers).
    * **Model Selection**: Implementing `GridSearchCV` to evaluate multiple algorithms (Logistic Regression, Random Forest, etc.).
    * **Validation**: Ensuring the model generalizes well before export.
    * `notebook.ipynb`: The clean final setup for the selected model.
    * `playground.ipynb`: Where all the dirty work was done 😂😋.
 * **`.gitignore`**: Specifies untracked files that Git should ignore.
 * **`requirements.txt`**: The project dependencies.


## 🏗️ Technical Workflow

1.  **Model Evolution**: In the `lab/` environment, I performed hyperparameter tuning via **GridSearchCV** to optimize for precision and recall, ensuring the model effectively identifies "High Risk" students.
2.  **Cloud-Native Storage**: The finalized model and preprocessors are decoupled from the code and stored in **AWS S3**.
3.  **Production Serving**: On server startup, the FastAPI backend dynamically pulls the latest model weights from S3.
4.  **Web Stack**: The application is served on an **AWS EC2** instance using **Gunicorn** with **Uvicorn** workers, sitting behind an **Nginx** reverse proxy for security and performance.

## 🛠️ Tech Stack

* **ML & Data Science**: Scikit-Learn, Pandas, NumPy, Matplotlib, Seaborn.
* **Backend**: FastAPI, Boto3 (AWS SDK), Python 3.10.
* **Frontend**: HTML5, CSS3, JavaScript (Fetch API).
* **DevOps**: AWS (EC2, S3), Nginx, Gunicorn, Systemd, Linux.

## 🚀 Installation & Setup

### Local Setup
1. **Clone the repo**:
   `git clone https://github.com/Favour-325/student-performance-risk-predictor.git`
   
   `cd student-performance-risk-predictor`

3. **Environment**:
   `python3 -m venv env
    source env/bin/activate
    pip install -r requirements.txt`

4. **Launch**:
  `uvicorn app.main:app --reload`

**Production Access**
The app is deployed on AWS EC2 (Public IP: [13.49.25.76](http://13.49.25.76/)) and managed via **Systemd** to ensure 99.9% uptime.

**API Docs**: `/docs` (Swagger UI)

**Web UI**: `/` (Root)

### 🤝 Key Insights
By integrating GridSearchCV during the development phase, I moved beyond basic model fitting to true performance optimization. This project highlights my ability to handle the entire ML lifecycle—from messy raw data in a notebook to a polished, cloud-hosted web application.

Developed by **Favour Eyong Tabi**
BTech CSC Cloud Computing | Jnr. Machine Learning Engineer | 
**LinkedIn:** [Favour Eyong Tabi](www.linkedin.com/in/favour-tabi)
