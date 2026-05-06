
# CHAPTER 3: PROPOSED SYSTEM

## 3.1 Existing System

The existing approaches to depression detection can be broadly categorized into three types: clinical methods, unimodal automated systems, and limited multimodal prototypes. Each of these approaches has significant limitations that the MindScan system aims to address.

### 3.1.1 Clinical Methods

Traditional clinical depression detection relies on face-to-face consultations between patients and mental health professionals. Psychiatrists and psychologists use standardized diagnostic criteria from the DSM-5 (Diagnostic and Statistical Manual of Mental Disorders, 5th Edition) and validated screening tools such as the PHQ-9 (Patient Health Questionnaire-9), BDI (Beck Depression Inventory), and HAM-D (Hamilton Depression Rating Scale). While these methods are considered the gold standard for depression diagnosis, they suffer from several critical limitations:

- **Accessibility Barriers:** Mental health professionals are scarce, particularly in developing countries and rural areas. The WHO estimates that there are fewer than 2 psychiatrists per 100,000 people in most low- and middle-income countries.
- **Stigma and Reluctance:** Social stigma associated with mental health prevents many individuals from seeking professional help. Studies indicate that only 1 in 3 people with depression seeks treatment.
- **Subjectivity and Bias:** Clinical assessments depend heavily on the clinician's training, experience, and interpretation, introducing potential variability in diagnosis.
- **Temporal Limitations:** Clinical visits provide only a snapshot of the patient's mental state at the time of the visit, potentially missing fluctuating symptoms.
- **Cost Barriers:** Professional mental health consultations can be expensive and may not be covered by insurance in many regions.

### 3.1.2 Unimodal Automated Systems

Several automated systems have been developed that analyze a single data modality for depression detection:

- **Text-Only Systems:** Systems like those developed by De Choudhury et al. (2013) and Tadesse et al. (2019) analyze social media text using NLP techniques and machine learning classifiers. While effective, these systems miss non-verbal cues that are important indicators of depression, such as facial expressions, body language, and vocal patterns.
- **Image-Only Systems:** Systems based on facial emotion recognition, such as those using FACS analysis or deep learning-based emotion classifiers, can detect emotional states from facial images. However, they cannot capture the rich cognitive and emotional content expressed through language and may be affected by factors such as lighting conditions, pose variations, and image quality.
- **Audio-Only Systems:** Voice-based depression detection systems analyze prosodic features (pitch, energy, speaking rate) and spectral features. While promising, these systems require audio recordings, which may not be readily available in all contexts.

### 3.1.3 Limitations of Existing Systems

The key limitations of existing systems that motivate the development of MindScan include:

1. **Single-Modality Constraint:** Most systems analyze only one type of data, limiting their ability to capture the multifaceted nature of depression.
2. **Lack of User-Friendly Interfaces:** Many existing systems are research prototypes without intuitive interfaces, making them inaccessible to general users.
3. **No Real-Time Processing:** Most systems operate in batch mode and do not support interactive, real-time analysis.
4. **Limited Interpretability:** Many deep learning-based systems function as black boxes, providing limited insight into the factors driving their predictions.
5. **No Session Tracking:** Existing systems typically provide one-time analysis without tracking patterns over multiple sessions.

## 3.2 Proposed System

MindScan is proposed as an AI-powered multimodal depression detection system that addresses the limitations of existing systems by integrating text analysis and facial emotion recognition into a unified, web-based platform. The key features and advantages of the proposed system are:

### 3.2.1 Key Features

1. **Multimodal Analysis:** MindScan combines NLP-based text analysis with DeepFace-powered facial emotion recognition, providing a more comprehensive depression assessment than unimodal systems.

2. **Three Analysis Modes:**
   - **Text Analysis Mode:** Analyzes social media text (Reddit posts, tweets, etc.) for depressive indicators using TF-IDF vectorization and Logistic Regression classification.
   - **Face Analysis Mode:** Detects and classifies facial emotions from uploaded images or webcam captures, mapping emotions to depression risk weights.
   - **Combined Mode:** Fuses text and face analysis results using a weighted averaging approach (60% text, 40% face) for comprehensive assessment.

3. **Real-Time Web Interface:** A modern, responsive Flask-based web application enables real-time interactive analysis with instant results visualization.

4. **Comprehensive Results Dashboard:** The results dashboard displays:
   - Depression/Non-depression classification with confidence scores
   - Risk level assessment (Low, Low-Moderate, Moderate, High)
   - Detected emotional signals and sentiment analysis
   - Emotion distribution map for facial analysis
   - Signal breakdown visualization for combined analysis

5. **Session History and Trend Tracking:** The system maintains a session history of all analyses performed, with a confidence trend visualization that allows users to observe patterns over multiple analyses.

6. **Webcam Integration:** Users can capture facial images directly through the webcam for real-time face analysis, in addition to uploading pre-captured images.

7. **Interpretable Results:** Unlike black-box deep learning systems, MindScan provides interpretable results with clear confidence scores, emotion breakdowns, and risk level categorizations.

### 3.2.2 Advantages over Existing Systems

| Feature | Existing Systems | MindScan |
|---------|-----------------|----------|
| Modality | Single (text OR face OR audio) | Multimodal (text + face) |
| Interface | Command-line / Research prototype | Modern web UI |
| Real-time | Batch processing | Real-time interactive |
| Interpretability | Limited (black-box) | High (confidence, emotion map) |
| Session tracking | Not available | Full session history + trends |
| Webcam support | Rarely available | Built-in webcam integration |
| Deployment | Complex setup | Simple Flask deployment |
| Accessibility | Technical users only | General users |

## 3.3 System Architecture

### 3.3.1 High-Level Architecture

The MindScan system follows a three-tier architecture consisting of the Presentation Layer, Application Layer, and Data/Model Layer. The architecture is designed for modularity, scalability, and ease of deployment.

```
┌─────────────────────────────────────────────────────┐
│                 PRESENTATION LAYER                   │
│                                                     │
│   ┌──────────┐  ┌──────────┐  ┌──────────────┐    │
│   │ Text     │  │ Face     │  │ Combined     │    │
│   │ Input    │  │ Upload/  │  │ Input Panel  │    │
│   │ Panel    │  │ Webcam   │  │              │    │
│   └────┬─────┘  └────┬─────┘  └──────┬───────┘    │
│        │             │               │              │
│   ┌────┴─────────────┴───────────────┴───────┐     │
│   │          Results Dashboard                │     │
│   │  ┌──────┐ ┌──────────┐ ┌───────────┐    │     │
│   │  │Badge │ │Confidence│ │Emotion Map│    │     │
│   │  │Result│ │Bar       │ │           │    │     │
│   │  └──────┘ └──────────┘ └───────────┘    │     │
│   │  ┌──────────────┐ ┌──────────────────┐  │     │
│   │  │Session       │ │Confidence Trend  │  │     │
│   │  │History       │ │Visualization     │  │     │
│   │  └──────────────┘ └──────────────────┘  │     │
│   └──────────────────────────────────────────┘     │
└───────────────────┬─────────────────────────────────┘
                    │ HTTP/JSON (REST API)
┌───────────────────┴─────────────────────────────────┐
│                 APPLICATION LAYER                    │
│                  (Flask Server)                      │
│                                                     │
│   ┌─────────────────────────────────────────┐       │
│   │            Route Handlers               │       │
│   │  /predict/text  /predict/face           │       │
│   │  /predict/combined  /status             │       │
│   └────────┬────────────────┬───────────────┘       │
│            │                │                       │
│   ┌────────┴─────┐  ┌──────┴──────────┐           │
│   │ Text Analysis│  │ Face Analysis   │           │
│   │ Module       │  │ Module          │           │
│   │              │  │                 │           │
│   │ - clean_text │  │ - save_temp_img │           │
│   │ - vectorize  │  │ - analyze_face  │           │
│   │ - predict    │  │ - emotion_map   │           │
│   └────────┬─────┘  └──────┬──────────┘           │
│            │                │                       │
│   ┌────────┴────────────────┴───────────────┐       │
│   │         Fusion Module                   │       │
│   │   Weighted Average (60% text, 40% face) │       │
│   └─────────────────────────────────────────┘       │
└───────────────────┬─────────────────────────────────┘
                    │
┌───────────────────┴─────────────────────────────────┐
│               DATA / MODEL LAYER                     │
│                                                     │
│   ┌──────────────┐  ┌────────────────┐             │
│   │ model.pkl    │  │ vectorizer.pkl │             │
│   │ (Logistic    │  │ (TF-IDF       │             │
│   │  Regression) │  │  Vectorizer)  │             │
│   └──────────────┘  └────────────────┘             │
│   ┌──────────────────────────────────┐             │
│   │ DeepFace Models (TensorFlow)     │             │
│   │ - Emotion Recognition Model      │             │
│   └──────────────────────────────────┘             │
│   ┌──────────────────────────────────┐             │
│   │ Reddit Depression Dataset        │             │
│   │ (depression_dataset_reddit_      │             │
│   │  cleaned.csv)                    │             │
│   └──────────────────────────────────┘             │
└─────────────────────────────────────────────────────┘
```

### 3.3.2 Data Flow Diagram – Level 0 (Context Diagram)

The Context Diagram represents the highest level of abstraction, showing the MindScan system as a single process interacting with external entities.

```
                    ┌─────────┐
                    │  User   │
                    └────┬────┘
                         │
            Text / Image │ │ Depression Assessment
             Input       │ │ Results
                         ▼ │
                    ┌─────────────┐
                    │  MindScan   │
                    │  System     │
                    └──────┬──────┘
                           │
                    ┌──────┴──────┐
                    │ Pre-trained │
                    │ ML Models   │
                    └─────────────┘
```

### 3.3.3 Data Flow Diagram – Level 1

The Level 1 DFD decomposes the MindScan system into its major processes:

```
                         ┌─────────┐
                         │  User   │
                         └────┬────┘
                    Text      │      Image
              ┌───────────────┼──────────────┐
              ▼               │              ▼
    ┌──────────────┐          │    ┌──────────────┐
    │ P1: Text     │          │    │ P2: Face     │
    │ Preprocessing│          │    │ Analysis     │
    │ & Analysis   │          │    │              │
    └──────┬───────┘          │    └──────┬───────┘
           │                  │           │
           │ Text Result      │           │ Face Result
           │                  │           │
           ▼                  │           ▼
    ┌──────────────────────────────────────────┐
    │        P3: Fusion Module                  │
    │  (Weighted Average: 60% Text, 40% Face)  │
    └──────────────────┬───────────────────────┘
                       │
                       ▼ Combined Result
              ┌────────────────┐
              │ P4: Results    │
              │ Presentation   │
              └────────┬───────┘
                       │
                       ▼
                   ┌─────────┐
                   │  User   │
                   └─────────┘
```

### 3.3.4 Use Case Diagram

The Use Case Diagram identifies the primary actors and use cases of the MindScan system:

```
    Actor: User
    ┌──────────────────────────────────────────────┐
    │                MindScan System                │
    │                                              │
    │  ○ Enter/Paste Social Media Text             │
    │  ○ Upload Face Image                         │
    │  ○ Capture Webcam Photo                      │
    │  ○ Select Analysis Mode (Text/Face/Combined) │
    │  ○ Run Depression Analysis                   │
    │  ○ View Detection Results                    │
    │  ○ View Emotion Distribution Map             │
    │  ○ View Session History                      │
    │  ○ View Confidence Trend                     │
    │  ○ Try Example Text Inputs                   │
    │  ○ Clear Image                               │
    │  ○ Check System Status                       │
    └──────────────────────────────────────────────┘
```

### 3.3.5 Sequence Diagram for Text Analysis

```
    User          Frontend(JS)       Flask Server      ML Model
     │                │                   │                │
     │ Enter Text     │                   │                │
     ├───────────────>│                   │                │
     │                │ POST /predict/text│                │
     │                ├──────────────────>│                │
     │                │                   │ clean_text()   │
     │                │                   ├───────────────>│
     │                │                   │ vectorize()    │
     │                │                   ├───────────────>│
     │                │                   │ predict()      │
     │                │                   ├───────────────>│
     │                │                   │<───────────────┤
     │                │   JSON Response   │  prediction    │
     │                │<──────────────────┤                │
     │  Display Result│                   │                │
     │<───────────────┤                   │                │
     │                │                   │                │
```

## 3.4 Methodology

The development of MindScan follows a systematic methodology that encompasses data collection, preprocessing, feature extraction, model training, facial analysis integration, multimodal fusion, and web application deployment. Each phase is described in detail below.

### 3.4.1 Data Collection and Dataset Description

The primary dataset used for training the text classification model is the **Reddit Depression Dataset** (`depression_dataset_reddit_cleaned.csv`). This dataset consists of cleaned Reddit posts collected from depression-related subreddits (such as r/depression, r/SuicideWatch) and non-depression-related subreddits.

The dataset contains the following key attributes:
- **clean_text:** The preprocessed text content of the Reddit post
- **is_depression:** Binary label indicating whether the post is from a depression-related subreddit (1) or not (0)

The dataset exhibits class imbalance, with unequal numbers of depressive and non-depressive posts. This imbalance is addressed through upsampling of the minority class during the preprocessing phase.

### 3.4.2 Text Preprocessing Pipeline

The text preprocessing pipeline implemented in MindScan performs the following operations sequentially:

**Step 1: Lowercasing**
All text is converted to lowercase to ensure case-insensitive analysis:
```python
text = str(text).lower()
```

**Step 2: URL Removal**
URLs (http, https, www) are removed as they do not contribute to sentiment analysis:
```python
text = re.sub(r"http\S+|www\S+|https\S+", "", text)
```

**Step 3: Mention and Hashtag Removal**
Social media mentions (@username) and hashtags (#topic) are stripped:
```python
text = re.sub(r"@\w+|#\w+", "", text)
```

**Step 4: Special Character Removal**
All characters except lowercase alphabets and spaces are removed:
```python
text = re.sub(r"[^a-z\s]", "", text)
```

**Step 5: Stopword Removal**
Common English stopwords (e.g., "the," "is," "at," "on") are removed using NLTK's stopword corpus to reduce noise and dimensionality:
```python
words = [w for w in text.split() if w not in stop_words]
```

**Step 6: Lemmatization**
Words are reduced to their base/root form using NLTK's WordNetLemmatizer to normalize morphological variations:
```python
words = [lemmatizer.lemmatize(w) for w in words]
```

### 3.4.3 Feature Extraction – TF-IDF Vectorization

After preprocessing, the cleaned text is converted into numerical feature vectors using **Term Frequency–Inverse Document Frequency (TF-IDF)** vectorization. TF-IDF is a statistical measure that evaluates the importance of a word in a document relative to a corpus. It is computed as:

**TF(t, d) = (Number of times term t appears in document d) / (Total number of terms in document d)**

**IDF(t, D) = log(Total number of documents in corpus D / Number of documents containing term t)**

**TF-IDF(t, d, D) = TF(t, d) × IDF(t, D)**

The MindScan TF-IDF vectorizer is configured with the following parameters:
- **max_features = 10,000:** Limits the vocabulary to the top 10,000 most important terms
- **ngram_range = (1, 2):** Considers both unigrams (single words) and bigrams (pairs of consecutive words) to capture phrasal patterns
- **sublinear_tf = True:** Applies sublinear scaling (1 + log(tf)) to dampen the effect of high-frequency terms

```python
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), sublinear_tf=True)
X_train_v = vectorizer.fit_transform(X_train)
```

### 3.4.4 Class Balancing – Minority Upsampling

To address class imbalance in the dataset, the minority class (depressive posts) is upsampled using random oversampling with replacement to match the size of the majority class:

```python
minority_up = resample(minority, replace=True, n_samples=len(majority), random_state=42)
df_balanced = pd.concat([majority, minority_up])
```

This ensures that the classifier receives equal representation of both classes during training, preventing bias toward the majority class.

### 3.4.5 Model Training – Logistic Regression

The text classification model uses **Logistic Regression** with the following configuration:

```python
model = LogisticRegression(class_weight='balanced', max_iter=1000, C=1.0)
```

- **class_weight='balanced':** Automatically adjusts class weights inversely proportional to class frequencies, providing additional protection against class imbalance
- **max_iter=1000:** Sets the maximum number of iterations for the solver to converge
- **C=1.0:** Regularization parameter (inverse of regularization strength); C=1.0 provides moderate regularization

Logistic Regression was selected for the following reasons:
1. **Interpretability:** Logistic Regression provides probabilistic outputs that can be directly interpreted as confidence scores
2. **Efficiency:** It is computationally efficient and trains quickly, making it suitable for deployment
3. **Performance:** When combined with well-engineered TF-IDF features, Logistic Regression achieves competitive performance comparable to more complex models
4. **Probabilistic Output:** The `predict_proba()` method provides calibrated probability estimates, which are used as confidence scores in MindScan

### 3.4.6 Facial Emotion Recognition – DeepFace

The facial analysis module uses **DeepFace**, a lightweight facial analysis framework built on top of TensorFlow/Keras. DeepFace provides state-of-the-art facial emotion recognition capabilities using pre-trained deep learning models.

The facial analysis pipeline operates as follows:

1. **Image Input:** The user uploads a face image or captures one via webcam. The image is received as a base64-encoded string.
2. **Image Decoding and Saving:** The base64 string is decoded and saved as a temporary JPEG file:
```python
def save_temp_image(img_b64: str) -> str:
    img_bytes = base64.b64decode(img_b64)
    img = Image.open(BytesIO(img_bytes)).convert("RGB")
    img.save(tmp_path)
    return tmp_path
```
3. **Emotion Analysis:** DeepFace analyzes the image and returns emotion probability scores for seven emotions:
```python
result = DeepFace.analyze(
    img_path=tmp_path,
    actions=["emotion"],
    enforce_detection=False,
    silent=True
)
```
4. **Emotion Percentage Calculation:** Raw emotion scores are normalized to percentages:
```python
emo_pct = {k: round(float(v) / total * 100, 1) for k, v in emotions.items()}
```

### 3.4.7 Emotion-to-Depression Weight Mapping

Each detected emotion is mapped to a depression risk weight based on established research correlating specific emotions with depressive states:

**Table 3.1: Emotion-to-Depression Weight Mapping**

| Emotion | Depression Weight | Rationale |
|---------|------------------|-----------|
| Sad | 0.90 | Strongest indicator of depressive state |
| Fear | 0.70 | Strongly linked to anxiety and depression |
| Angry | 0.55 | Can co-occur with irritable depression |
| Disgust | 0.50 | May indicate negative rumination |
| Contempt | 0.45 | Social withdrawal indicator |
| Neutral | 0.25 | Flat affect — possible anhedonia |
| Surprise | 0.15 | Generally low-risk indicator |
| Happy | 0.05 | Positive emotional state, low risk |

The depression score from facial analysis is determined by the weight of the dominant (most intense) emotion:
```python
dep_score = EMOTION_DEPRESSION_WEIGHTS.get(dominant, 0.2)
is_dep = dep_score > 0.5
```

### 3.4.8 Multimodal Fusion Strategy

When both text and face analysis are performed (Combined Mode), MindScan uses a **weighted late fusion** strategy to combine the results:

```python
fused_conf = round(t["confidence"] * 0.6 + f["confidence"] * 0.4, 1)
fused_dep = t["is_depressive"] or f["is_depressive"]
```

**Fusion Rules:**
- **Confidence Fusion:** The fused confidence score is a weighted average of text confidence (60%) and face confidence (40%). The higher weight for text reflects the generally higher information density and reliability of textual analysis compared to single-frame facial analysis.
- **Depression Classification:** The fused depression classification uses an OR logic — if either modality indicates depression, the fused result is classified as depressive. This conservative approach ensures that potential depression indicators from either modality are not overlooked.

### 3.4.9 Web Application Architecture

The MindScan web application is built using the following architecture:

- **Backend:** Python Flask framework serving RESTful API endpoints
- **Frontend:** Single-page HTML5 application with embedded CSS3 and JavaScript
- **Communication:** JSON-based HTTP requests between frontend and backend
- **API Endpoints:**
  - `GET /` — Serves the main web interface
  - `GET /status` — Health check endpoint confirming DeepFace availability
  - `POST /predict/text` — Text-only depression analysis
  - `POST /predict/face` — Face-only depression analysis
  - `POST /predict/combined` — Combined multimodal analysis

## 3.5 Module Description

The MindScan system is organized into five primary modules, each responsible for a specific aspect of the system's functionality.

**Table 3.2: Module Description Summary**

### Module 1: Text Preprocessing Module

- **Purpose:** Cleans and normalizes raw social media text for analysis
- **Key Functions:** `clean_text(text)`
- **Operations:** Lowercasing, URL removal, mention/hashtag stripping, special character removal, stopword elimination, lemmatization
- **Input:** Raw social media text (string)
- **Output:** Cleaned, preprocessed text (string)
- **Libraries Used:** re, NLTK (stopwords, WordNetLemmatizer)

### Module 2: Text Classification Module

- **Purpose:** Classifies preprocessed text as depressive or non-depressive
- **Key Components:** TF-IDF Vectorizer (vectorizer.pkl), Logistic Regression Model (model.pkl)
- **Operations:** TF-IDF vectorization, model prediction, probability estimation
- **Input:** Cleaned text from Module 1
- **Output:** Binary classification (depressive/non-depressive), confidence score
- **Libraries Used:** scikit-learn (TfidfVectorizer, LogisticRegression)

### Module 3: Facial Emotion Recognition Module

- **Purpose:** Detects and classifies facial emotions from images
- **Key Functions:** `save_temp_image(img_b64)`, `analyze_face(tmp_path)`
- **Operations:** Base64 image decoding, image saving, DeepFace emotion analysis, emotion percentage calculation, depression weight mapping
- **Input:** Base64-encoded face image
- **Output:** Emotion distribution (percentages), dominant emotion, depression classification, confidence score
- **Libraries Used:** DeepFace, TensorFlow, PIL, base64

### Module 4: Multimodal Fusion Module

- **Purpose:** Combines text and face analysis results into a unified assessment
- **Key Logic:** Weighted average fusion (60% text, 40% face) with OR-based depression classification
- **Input:** Text analysis result, face analysis result
- **Output:** Fused depression classification, fused confidence score
- **Endpoint:** `/predict/combined`

### Module 5: Web Interface Module

- **Purpose:** Provides the user-facing web application for interaction
- **Key Components:** HTML5 structure, CSS3 styling (Outfit + Playfair Display fonts), JavaScript logic
- **Features:** Mode switching (Text/Face/Combined), text input with character count, image upload with drag-and-drop, webcam capture, results dashboard with confidence bars, emotion map, session history, confidence trend visualization
- **Communication:** Fetch API for asynchronous HTTP requests to Flask backend

---

\newpage
