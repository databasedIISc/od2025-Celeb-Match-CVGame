# Which Bollywood Celebrity Do You Look Like?

Have you ever wondered which Bollywood celebrity you resemble the most? This project aims to answer that question by using machine learning techniques to analyze facial features and match them with a database of Bollywood celebrities. By comparing various aspects of your face, such as shape, position, and skin color, this tool provides an entertaining way to discover your Bollywood doppelgänger.


## Adapted from:
https://github.com/entbappy/Which-Bollywood-Celebrity-You-look-like

## Dataset Used

Download [this dataset](https://www.kaggle.com/sushilyadav1998/bollywood-celeb-localized-face-dataset) from Kaggle and place all 100 image folders into a single folder named `data`.

Repeat the process for [this dataset](https://www.kaggle.com/datasets/havingfun/100-bollywood-celebrity-faces) and place all 100 image folders into a single folder named `archive`.

## Steps to Run This Project

### Step 1: Clone the Repository

```bash
git clone https://github.com/databasedIISc/od2025-Celeb-Match-CVGame.git
```

### Step 2: Create an Environment

```bash
conda create -n celebrity python=3.7 -y
```

### Step 3: Install the Requirements

```bash
pip install -r requirements.txt
```

### Step 4: Initial Setup

Execute this command once if you are not changing the data:

```bash
python run.py
```

### Step 5: Run the Project

Use the following command to run the project:

```bash
python main.py
```



