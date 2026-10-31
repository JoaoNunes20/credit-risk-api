# OSS Message Classifier - NLP Project

This repository contains a Natural Language Processing (NLP) project developed for the course *Natural Language Processing* (MSc in Artificial Intelligence, 1st year, 2nd semester). The project focuses on classifying online messages from open-source software platforms (GitHub, Bugzilla, NNTP) into **request** and **non-request** categories using various machine learning models and word embeddings.

---

## 🚀 How to Run the Project

1. **Clone the repository:**

```bash
git clone https://github.com/your-username/oss-message-classifier.git
cd oss-message-classifier
```

2. **Install dependencies:**

Make sure you are using **Python 3.12**. Then install all required libraries using:

```bash
pip install -r requirements.txt
```

3. **Download FastText model (if using FastText embeddings):**

To use the FastText pre-trained embeddings, download the model from the following link:

📥 [Download FastText model](https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M-subword.zip)

After downloading, extract the `.zip` file and place the `.bin` model file (`crawl-300d-2M-subword.bin`) in the root directory of the project (or update the path in the notebook accordingly).

4. **Open and run the notebook:**

You can now open the `PLN.ipynb` notebook in Jupyter:

```bash
jupyter notebook PLN.ipynb
```

Run all cells in sequence to reproduce the results and visualizations.

---

## 📂 Dataset

The dataset used in this project can be downloaded from:

🔗 https://portulanclarin.net/repository/browse/oss-online-communication-messages/99f89cc203be11ebbc3f02420a0004e1c62281d86e7f4245adc76ab0fa808cfd/

---

Feel free to fork, adapt, or contribute!