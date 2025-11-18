# 🎓 Student Engagement Detection using ML & DL  
A complete pipeline to detect **student engagement** using **DAiSEE dataset**, **PyFeat facial behavior features**, and multiple **Machine Learning** and **Deep Learning** models.

---

## 🚀 Project Overview

This project focuses on analyzing student engagement using a **subset of 5,000 videos** from the **DAiSEE dataset**.  
Each video was processed using **PyFeat**, extracting:

- Facial Action Units (AUs)  
- 68 Facial Landmarks  
- Head Pose (Pitch, Roll, Yaw)  
- Emotion Probabilities  
- Additional face-related features  

These extracted features were used to develop and compare **Machine Learning** and **Deep Learning** models.

Machine Learning models (XGBoost, HGB, Random Forest, Logistic Regression) achieved higher accuracy compared to Deep Learning models (VGG19, ResNet, EfficientNet), showing that **engineered PyFeat features are highly effective** for engagement prediction.

---

## 📊 DAiSEE Dataset Summary

- 🎥 **9068 video snippets**, each **10 seconds long**  
- 🎓 **112 subjects (age 18–30)**  
- 🌍 Asian demographic  
- 👩‍🎓 32 female, 80 male  
- 🏠 Recorded in **6 real-world environments** (dorm, lab, library, etc.)  
- 💡 **3 lighting conditions**: light, dark, neutral  
- 📌 Labeled for **Boredom, Confusion, Engagement, Frustration**  
- ⭐ Each affect has **4 levels** → very low, low, high, very high  

This project uses the **Engagement** label and converts it to **Binary (Engaged vs Disengaged)**.

---

## 🧠 Features Extracted (PyFeat)

- 68 **Facial Landmarks**: x_0…x_67 and y_0…y_67  
- **Action Units (AUs)**: AU01, AU02, AU04, AU06, AU12, etc.  
- **Head Pose**: Pitch, Yaw, Roll  
- **Emotion Scores**: happiness, sadness, fear, anger, neutral...  
- **Face Metrics**: FaceRectX, FaceRectY, Width, Height, Score  
- **30 Frames extracted per video**, converted into **one final aggregated row** per clip  
- Final result exported into a **CSV**  

---

## 🧪 Machine Learning Models Used

### **1. Logistic Regression**
- Baseline classifier  
- Fast and interpretable  

### **2. Random Forest**
- Ensemble of decision trees  
- Captures non-linear patterns  
- Reduces overfitting  

### **3. Histogram Gradient Boosting (HGB)**
- Optimized for tabular datasets  
- Fast and accurate boosting model  

### **4. XGBoost**
- Regularized gradient boosting  
- Handles imbalance effectively  
- Best performing ML model in this project  

---

## 🤖 Deep Learning Models Used

### **ResNet**
- Skip connections  
- Handles deep architectures without vanishing gradients  

### **EfficientNet**
- Lightweight & scalable  
- Efficient depth/width resolution  

### **VGG19**
- Deep CNN with many layers  
- Good for fine-grained facial features  
- Computationally expensive  

---

## 📁 Project Structure

```bash
student-engagement/
├── data/
│ ├── daisee_videos/
│ └── features_extracted/
├── models/
│ ├── final_xgb_model.pkl
│ ├── random_forest.pkl
│ └── scaler.pkl
├── notebooks/
│ ├── feature_extraction.ipynb
│ ├── ml_training.ipynb
│ ├── dl_training.ipynb
├── src/
│ ├── preprocess.py
│ ├── train_ml.py
│ ├── train_dl.py
│ └── predict_video.py
└── README.md
```
---

## ⚙️ Installation

###  Clone the repository:

```bash
git clone https://github.com/yourname/student-engagement.git
cd student-engagement

Install dependencies:

pip install -r requirements.txt

```
---
# 📈 Results Summary

- Machine learning models significantly outperformed deep learning due to structured facial behavior features.

- ML Best Performance: XGBoost

- Accuracy: 93%

- F1 Score: 0.96

- Strong generalization

- Deep Learning Performance (VGG19/ResNet/EfficientNet)

- Lower accuracy due to limited dataset size per class

- Computationally heavy

- Overfitting observed

# 🧾 Conclusion

* PyFeat feature engineering + ML models provide highly reliable engagement prediction

* ML approaches outperform DL for small-to-medium sized datasets

* Real-time engagement recognition becomes possible due to   lightweight feature computation

# 🔮 Future Work

* Integrate boredom label in the model

* Build real-time engagement monitoring system using webcam

* Optimize MTCNN + PyFeat for multi-face detection

* Deploy model into a web dashboard

# 📝 License

* This project is for educational and research purposes.

# 🤝 Contribute

* Pull requests are welcome!

## ⭐ If you found this helpful, leave a star!
```
If you want, I can also:  
✅ Create a professional **Project Logo**  
✅ Create a **Project Poster**  
✅ Make a **GitHub project description**  
Just tell me!