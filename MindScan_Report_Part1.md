
# MINDSCAN: AI-POWERED MULTIMODAL DEPRESSION DETECTION SYSTEM

## A Project Report

**Submitted in partial fulfillment of the requirements for the award of the Degree of Bachelor of Technology in Computer Science and Engineering**

---

**Submitted by:**

[Student Name]
[Register Number]

**Under the guidance of:**

[Guide Name]
[Designation]

**Department of Computer Science and Engineering**
[College Name]
[University Name]
[Year: 2025–2026]

---

\newpage

## CERTIFICATE

This is to certify that the project report titled **"MindScan: AI-Powered Multimodal Depression Detection System"** is a bonafide record of the project work done by **[Student Name]** (Register No: [Register Number]) under my guidance and supervision, submitted in partial fulfillment of the requirements for the award of the degree of **Bachelor of Technology in Computer Science and Engineering** during the academic year **2025–2026**.

&nbsp;

**Internal Guide** &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; **Head of Department**

[Guide Name] &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; [HOD Name]

Date: &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; Date:

**External Examiner:**

---

\newpage

## DECLARATION

I hereby declare that the project report titled **"MindScan: AI-Powered Multimodal Depression Detection System"** submitted to [University Name] in partial fulfillment of the requirements for the award of the degree of **Bachelor of Technology in Computer Science and Engineering** is a record of original work done by me under the supervision and guidance of **[Guide Name]**, [Designation], Department of Computer Science and Engineering, [College Name].

I further declare that this project report has not been submitted to any other university or institution for the award of any degree or diploma.

&nbsp;

Place: [City]

Date: [Date]

&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; **[Student Name]**

---

\newpage

## ACKNOWLEDGEMENT

I would like to express my sincere gratitude to my project guide **[Guide Name]**, [Designation], Department of Computer Science and Engineering, for their valuable guidance, encouragement, and support throughout the development of this project.

I extend my heartfelt thanks to **[HOD Name]**, Head of the Department of Computer Science and Engineering, for providing the necessary infrastructure and facilities required for the successful completion of this project.

I am also grateful to the Principal, **[Principal Name]**, for providing an encouraging environment and for the opportunity to carry out this project work.

I would like to thank all the faculty members and staff of the Department of Computer Science and Engineering for their help and cooperation during the course of this project.

I also express my gratitude to my family and friends for their continuous support and encouragement throughout the project work.

Finally, I thank the Almighty for blessing me with health, strength, and perseverance to complete this project successfully.

&nbsp;

**[Student Name]**

---

\newpage

## ABSTRACT

Depression is one of the most prevalent mental health disorders worldwide, affecting an estimated 280 million people globally according to the World Health Organization (WHO). Early detection of depression is critical for timely intervention and treatment, yet traditional diagnostic methods rely heavily on clinical interviews and self-reporting, which are often subject to bias, stigma, and accessibility barriers. With the increasing use of social media platforms and digital communication, there is a growing opportunity to leverage artificial intelligence (AI) and machine learning (ML) techniques for automated depression detection.

**MindScan** is an AI-powered multimodal depression detection system that combines Natural Language Processing (NLP) for textual analysis and Deep Learning-based facial emotion recognition to identify potential indicators of depression. The system analyzes social media text (such as Reddit posts and tweets) using a Logistic Regression classifier trained on TF-IDF vectorized features extracted from a cleaned Reddit depression dataset. Simultaneously, the system employs DeepFace, a state-of-the-art facial analysis framework built on TensorFlow, to detect and classify facial emotions from uploaded images or webcam captures.

The text analysis module preprocesses input text through lowercasing, URL and mention removal, special character stripping, stopword elimination, and lemmatization using NLTK. The preprocessed text is then vectorized using TF-IDF with bigram support and classified using a balanced Logistic Regression model. The facial analysis module maps detected emotions to depression risk weights, where emotions like sadness (0.90) and fear (0.70) carry high depression correlation weights, while happiness (0.05) indicates low risk.

A novel fusion module combines both modalities using a weighted averaging approach (60% text, 40% face) to produce a comprehensive depression risk assessment. The system is deployed as a Flask web application with a modern, responsive user interface featuring real-time analysis, session history tracking, confidence trend visualization, and emotion distribution mapping.

Experimental results demonstrate that the text classification model achieves high accuracy on the balanced Reddit depression dataset, and the multimodal fusion approach provides more robust and reliable depression indicators compared to unimodal analysis alone. MindScan serves as a research and educational tool, emphasizing that it is not a substitute for professional clinical diagnosis.

**Keywords:** Depression Detection, Natural Language Processing, Facial Emotion Recognition, Multimodal Analysis, Machine Learning, TF-IDF, Logistic Regression, DeepFace, Flask, Sentiment Analysis

---

\newpage

## TABLE OF CONTENTS

| Chapter | Title | Page No. |
|---------|-------|----------|
| | Certificate | i |
| | Declaration | ii |
| | Acknowledgement | iii |
| | Abstract | iv |
| | Table of Contents | v |
| | List of Figures | vii |
| | List of Tables | viii |
| | List of Abbreviations | ix |
| **1** | **INTRODUCTION** | **1** |
| 1.1 | Background | 1 |
| 1.2 | Problem Statement | 3 |
| 1.3 | Objectives | 4 |
| **2** | **LITERATURE REVIEW** | **5** |
| 2.1 | Introduction to Literature Review | 5 |
| 2.2 | Review of Existing Systems and Papers | 5 |
| 2.3 | Comparative Analysis | 7 |
| **3** | **PROPOSED SYSTEM** | **9** |
| 3.1 | Existing System | 9 |
| 3.2 | Proposed System | 10 |
| 3.3 | System Architecture | 11 |
| 3.4 | Methodology | 15 |
| 3.5 | Module Description | 20 |
| **4** | **IMPLEMENTATION AND RESULTS** | **22** |
| 4.1 | Tools and Technologies Used | 22 |
| 4.2 | System Implementation | 22 |
| 4.3 | Working Model | 25 |
| 4.4 | Results and Outputs | 25 |
| 4.5 | Performance Analysis | 27 |
| **5** | **CONCLUSION** | **28** |
| 5.1 | Conclusion | 28 |
| 5.2 | Future Enhancements | 29 |
| | Appendices | 30 |
| | References | 35 |

---

\newpage

## LIST OF FIGURES

| Figure No. | Title | Page No. |
|------------|-------|----------|
| Figure 1.1 | Global Depression Statistics (WHO 2023) | 2 |
| Figure 1.2 | Traditional vs AI-based Depression Detection | 3 |
| Figure 3.1 | High-Level System Architecture of MindScan | 11 |
| Figure 3.2 | Data Flow Diagram – Level 0 (Context Diagram) | 12 |
| Figure 3.3 | Data Flow Diagram – Level 1 | 13 |
| Figure 3.4 | Use Case Diagram | 14 |
| Figure 3.5 | Sequence Diagram for Text Analysis | 14 |
| Figure 3.6 | Text Preprocessing Pipeline | 15 |
| Figure 3.7 | TF-IDF Vectorization Process | 16 |
| Figure 3.8 | Model Training Workflow | 17 |
| Figure 3.9 | Facial Emotion Recognition Pipeline | 18 |
| Figure 3.10 | Multimodal Fusion Architecture | 19 |
| Figure 3.11 | Emotion-to-Depression Weight Mapping | 19 |
| Figure 4.1 | MindScan Home Page – Text Analysis Mode | 25 |
| Figure 4.2 | MindScan – Face Image Analysis Mode | 25 |
| Figure 4.3 | MindScan – Combined Analysis Mode | 26 |
| Figure 4.4 | Depression Detection Result (Depressive) | 26 |
| Figure 4.5 | Depression Detection Result (Non-Depressive) | 26 |
| Figure 4.6 | Face Emotion Map Visualization | 27 |
| Figure 4.7 | Session History and Confidence Trend | 27 |
| Figure 4.8 | Confusion Matrix – Text Classification Model | 27 |

---

\newpage

## LIST OF TABLES

| Table No. | Title | Page No. |
|-----------|-------|----------|
| Table 2.1 | Comparative Analysis of Existing Systems | 7 |
| Table 3.1 | Emotion-to-Depression Weight Mapping | 19 |
| Table 3.2 | Module Description Summary | 20 |
| Table 4.1 | Tools and Technologies Used | 22 |
| Table 4.2 | Dataset Description | 23 |
| Table 4.3 | Model Hyperparameters | 24 |
| Table 4.4 | Classification Report – Text Model | 27 |
| Table 4.5 | Performance Comparison: Unimodal vs Multimodal | 28 |

---

\newpage

## LIST OF ABBREVIATIONS

| Abbreviation | Full Form |
|-------------|-----------|
| AI | Artificial Intelligence |
| ML | Machine Learning |
| NLP | Natural Language Processing |
| DL | Deep Learning |
| TF-IDF | Term Frequency – Inverse Document Frequency |
| CNN | Convolutional Neural Network |
| API | Application Programming Interface |
| WHO | World Health Organization |
| NLTK | Natural Language Toolkit |
| SVM | Support Vector Machine |
| GUI | Graphical User Interface |
| HTTP | Hypertext Transfer Protocol |
| JSON | JavaScript Object Notation |
| REST | Representational State Transfer |
| CSS | Cascading Style Sheets |
| HTML | HyperText Markup Language |
| URL | Uniform Resource Locator |
| FER | Facial Emotion Recognition |
| RGB | Red Green Blue |
| JPEG/JPG | Joint Photographic Experts Group |
| PNG | Portable Network Graphics |
| UI | User Interface |
| UX | User Experience |
| PHQ-9 | Patient Health Questionnaire-9 |
| DSM-5 | Diagnostic and Statistical Manual of Mental Disorders, 5th Edition |
| LSTM | Long Short-Term Memory |
| BERT | Bidirectional Encoder Representations from Transformers |
| ROC | Receiver Operating Characteristic |
| AUC | Area Under the Curve |

---

\newpage

# CHAPTER 1: INTRODUCTION

## 1.1 Background

Mental health has emerged as one of the most pressing global health concerns of the 21st century. Depression, clinically referred to as Major Depressive Disorder (MDD), is a debilitating mental health condition characterized by persistent feelings of sadness, hopelessness, loss of interest in activities, fatigue, difficulty concentrating, and in severe cases, suicidal ideation. According to the World Health Organization (WHO), depression affects approximately 280 million people worldwide, making it one of the leading causes of disability globally. The COVID-19 pandemic further exacerbated the global mental health crisis, with studies indicating a 25–27% increase in the prevalence of depression and anxiety disorders during 2020–2021.

The economic burden of depression is equally staggering. The World Economic Forum estimates that mental health conditions, including depression, will cost the global economy approximately $16 trillion by 2030 in lost productivity, healthcare expenditures, and reduced quality of life. In India alone, the National Mental Health Survey (NMHS) reported that nearly 1 in 20 individuals suffers from depression, yet the treatment gap remains as high as 85%, largely due to social stigma, lack of awareness, shortage of mental health professionals, and limited accessibility to diagnostic services.

Traditional methods of diagnosing depression rely primarily on clinical interviews conducted by trained psychiatrists and psychologists, standardized questionnaires such as the Patient Health Questionnaire-9 (PHQ-9), the Beck Depression Inventory (BDI), and the Hamilton Depression Rating Scale (HAM-D). While these instruments are well-validated and widely used, they suffer from several inherent limitations. First, they are heavily dependent on self-reporting, which can be influenced by social desirability bias, denial, or the patient's inability to accurately articulate their emotional state. Second, access to mental health professionals remains severely limited, particularly in rural and underserved areas. Third, the episodic nature of depression means that symptoms may not be apparent during a scheduled clinical visit, leading to underdiagnosis.

The rapid proliferation of social media platforms such as Reddit, Twitter (now X), Facebook, and Instagram has fundamentally transformed how individuals express their thoughts, emotions, and experiences. Research has consistently demonstrated that linguistic patterns in social media posts can serve as reliable indicators of mental health states. Individuals experiencing depression often exhibit distinct linguistic markers, including increased use of first-person singular pronouns (e.g., "I," "me," "my"), negative emotion words, absolutist language (e.g., "always," "never," "nothing"), references to sleep disturbances, social isolation, and reduced positive emotion vocabulary.

Simultaneously, advances in computer vision and deep learning have enabled the development of sophisticated facial emotion recognition (FER) systems capable of detecting and classifying human emotions from facial images and video streams. Research in affective computing has established strong correlations between facial expressions and emotional states, with specific facial Action Units (AUs) associated with depressive affect, including reduced facial expressiveness (flat affect), increased expression of sadness and contempt, and diminished expressions of happiness and surprise.

The convergence of these two domains — Natural Language Processing (NLP) for text analysis and Computer Vision for facial emotion recognition — presents a compelling opportunity to develop multimodal AI systems capable of providing automated, non-invasive, and scalable depression screening. Such systems can complement traditional clinical methods by offering continuous monitoring capabilities, reducing the burden on healthcare professionals, and reaching populations that may not otherwise have access to mental health services.

**MindScan** was conceived in this context as an AI-powered multimodal depression detection system that leverages both textual analysis and facial emotion recognition to identify potential indicators of depression. The system is designed to serve as a research and educational tool, providing preliminary depression risk assessments while emphasizing that it is not a substitute for professional clinical diagnosis and treatment.

The textual analysis component of MindScan processes social media text (such as Reddit posts, tweets, and other written content) using a machine learning pipeline that includes text preprocessing (lowercasing, URL removal, mention and hashtag stripping, special character removal, stopword elimination, and lemmatization), feature extraction using Term Frequency–Inverse Document Frequency (TF-IDF) vectorization with bigram support, and classification using a Logistic Regression model trained on a cleaned Reddit depression dataset. The model employs class balancing through upsampling to address the inherent class imbalance in mental health datasets.

The facial emotion recognition component utilizes DeepFace, a state-of-the-art facial analysis framework built on TensorFlow, to detect and classify seven primary emotions (sadness, fear, anger, disgust, contempt, surprise, neutral, and happiness) from facial images. Each detected emotion is then mapped to a depression risk weight, with emotions strongly associated with depression (such as sadness and fear) receiving higher weights and positive emotions (such as happiness) receiving lower weights.

A novel fusion module combines the outputs of both modalities using a weighted averaging approach, where text analysis contributes 60% and facial analysis contributes 40% of the final depression risk score. This weighting reflects the generally higher reliability and information density of textual content compared to single-frame facial analysis.

The system is deployed as a Flask-based web application with a modern, responsive user interface that supports three analysis modes: text-only analysis, face-only analysis, and combined multimodal analysis. The interface features real-time analysis capabilities, session history tracking, confidence trend visualization, emotion distribution mapping, and a comprehensive results dashboard.

## 1.2 Problem Statement

Despite the high global prevalence of depression and its significant impact on individuals, families, and societies, timely and accurate detection of depression remains a major challenge. The key problems that motivate the development of MindScan are outlined below:

1. **Accessibility Gap in Mental Health Services:** The global shortage of mental health professionals creates a significant accessibility gap, particularly in developing countries. The WHO reports that in low-income countries, there is less than 1 psychiatrist per 100,000 people. This shortage means that millions of individuals suffering from depression cannot access timely professional diagnosis and treatment.

2. **Limitations of Self-Reporting:** Traditional depression screening tools rely heavily on self-reported data, which is subject to recall bias, social desirability bias, and the patient's subjective interpretation of their symptoms. Many individuals either underreport their symptoms due to stigma or are unable to recognize the severity of their condition.

3. **Delayed Detection and Intervention:** Depression is often diagnosed only when symptoms become severe enough to prompt medical attention, by which time the condition may have significantly progressed. Early detection through automated screening could enable timely intervention and improve treatment outcomes.

4. **Unimodal Analysis Limitations:** Most existing automated depression detection systems rely on a single modality — either text analysis or facial recognition — which limits their accuracy and robustness. Text analysis alone may miss important non-verbal cues, while facial analysis alone cannot capture the richness of cognitive and emotional states expressed through language.

5. **Need for Non-invasive Screening Tools:** There is a growing need for non-invasive, scalable, and privacy-conscious depression screening tools that can complement traditional clinical methods. Such tools should be capable of analyzing readily available data sources (such as social media posts and facial images) without requiring specialized clinical equipment or trained personnel.

6. **Integration Challenge:** Existing research has largely focused on individual modalities in isolation, with limited work on developing practical, integrated multimodal systems that combine textual and facial analysis for depression detection. There is a need for systems that can effectively fuse information from multiple modalities to provide more comprehensive and reliable assessments.

## 1.3 Objectives

The primary objectives of the MindScan project are:

1. **To develop an NLP-based text analysis module** capable of preprocessing social media text and classifying it as depressive or non-depressive using machine learning techniques, specifically TF-IDF vectorization with Logistic Regression.

2. **To implement a facial emotion recognition module** using the DeepFace framework that can detect and classify facial emotions from uploaded images or webcam captures, and map these emotions to depression risk indicators.

3. **To design and implement a multimodal fusion algorithm** that combines the outputs of text analysis and facial emotion recognition using a weighted averaging approach (60% text, 40% face) to produce a comprehensive depression risk assessment.

4. **To build a user-friendly web-based interface** using Flask and modern web technologies (HTML5, CSS3, JavaScript) that enables users to perform text analysis, face analysis, or combined multimodal analysis through an intuitive and responsive dashboard.

5. **To evaluate the performance** of the text classification model using standard metrics (accuracy, precision, recall, F1-score) and assess the effectiveness of the multimodal fusion approach compared to unimodal analysis.

6. **To ensure ethical considerations** by clearly communicating that the system is a research and educational tool, not a clinical diagnostic instrument, and by emphasizing the importance of professional mental health support.

---

\newpage

# CHAPTER 2: LITERATURE REVIEW

## 2.1 Introduction to Literature Review

The field of automated depression detection has witnessed significant growth over the past decade, driven by advances in natural language processing (NLP), computer vision, deep learning, and the increasing availability of social media data. Researchers have explored various approaches to detect depression using textual, visual, acoustic, and multimodal data sources. This chapter provides a comprehensive review of existing literature, systems, and research papers relevant to the MindScan project, covering text-based depression detection, facial emotion recognition for mental health assessment, and multimodal fusion approaches.

The literature review is organized into three main sections: (1) a review of existing systems and research papers, categorized by modality and methodology, (2) a comparative analysis of the reviewed systems highlighting their strengths, limitations, and relevance to the MindScan project, and (3) identification of research gaps that the proposed system aims to address.

## 2.2 Review of Existing Systems and Papers

### 2.2.1 Text-Based Depression Detection

**Paper 1: De Choudhury et al. (2013) – "Predicting Depression via Social Media"**
De Choudhury and colleagues conducted pioneering research on using social media (Twitter) data to predict the onset of depression. They analyzed behavioral attributes including posting frequency, emotional content, linguistic style (increased use of first-person pronouns), social engagement, and mentions of medications. Using a Support Vector Machine (SVM) classifier, they achieved a classification accuracy of approximately 70%. This work established the foundational framework for using social media text as a proxy for mental health assessment and highlighted the potential of linguistic markers as depression indicators.

**Paper 2: Orabi et al. (2018) – "Deep Learning for Depression Detection of Twitter Users"**
Orabi et al. explored deep learning approaches for depression detection on Twitter, comparing Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (RNNs) with traditional machine learning methods. They demonstrated that deep learning models, particularly CNNs with word embeddings, outperformed traditional classifiers (SVM, Naive Bayes) in capturing complex linguistic patterns associated with depression. Their work highlighted the importance of contextual and semantic features beyond simple bag-of-words representations.

**Paper 3: Tadesse et al. (2019) – "Detection of Depression-Related Posts in Reddit Social Media Forum"**
Tadesse et al. focused specifically on the Reddit platform, analyzing posts from the r/depression subreddit and non-depression-related subreddits. They compared multiple machine learning classifiers including Logistic Regression, SVM, Random Forest, and ensemble methods, using TF-IDF features, LIWC (Linguistic Inquiry and Word Count) features, and a combination of both. Their results showed that the combination of TF-IDF and LIWC features with ensemble classifiers achieved the highest performance, with accuracy exceeding 91%. This work is particularly relevant to MindScan as it validates the effectiveness of TF-IDF-based features for Reddit depression detection.

**Paper 4: Trotzek et al. (2018) – "Utilizing Neural Networks and Linguistic Metadata for Early Detection of Depression Indications in Text Sequences"**
Trotzek et al. proposed a neural network architecture that combined linguistic metadata (such as post length, punctuation usage, and temporal patterns) with deep learning features for early depression detection. They demonstrated that incorporating metadata alongside textual features improved classification performance, suggesting that multimodal feature sets provide more robust depression indicators than text content alone.

**Paper 5: Yates et al. (2017) – "Depression and Self-Harm Risk Assessment in Online Forums"**
Yates et al. developed a CNN-based model for assessing depression and self-harm risk in online forum posts. They introduced the concept of "risk assessment" as a graded scale rather than binary classification, which aligns more closely with clinical approaches to depression severity assessment. Their work demonstrated the viability of deep learning for nuanced mental health assessment beyond simple depressive/non-depressive classification.

### 2.2.2 Facial Emotion Recognition for Depression Detection

**Paper 6: Girard et al. (2014) – "Nonverbal Social Withdrawal in Depression"**
Girard et al. investigated the relationship between nonverbal behavior (facial expressions, head movements, and gaze patterns) and depression severity. Using the FACS (Facial Action Coding System), they demonstrated that individuals with depression exhibited reduced facial expressiveness, decreased smiling intensity, increased expressions of contempt, and more frequent downward gaze. Their findings provide the theoretical foundation for using facial emotion recognition as a depression indicator.

**Paper 7: Jan et al. (2017) – "Artificial Intelligent System for Automatic Depression Level Analysis through Visual and Vocal Expressions"**
Jan et al. developed a multimodal system that combined facial expression analysis with voice analysis for depression level assessment. They used a deep CNN for facial feature extraction and achieved promising results in detecting depression severity levels. Their work demonstrated the value of combining visual and acoustic modalities for more accurate depression assessment.

**Paper 8: Yang et al. (2017) – "Hybrid Depression Classification and Estimation from Audio/Video Processing"**
Yang et al. proposed a hybrid approach that combined audio and video modalities for depression classification and severity estimation. They used facial Action Unit detection, head pose estimation, and gaze direction analysis from video, combined with prosodic and spectral features from audio. Their multi-modal fusion approach significantly outperformed unimodal baselines, validating the benefits of multimodal analysis for depression detection.

### 2.2.3 Multimodal Depression Detection Systems

**Paper 9: Williamson et al. (2016) – "Detecting Depression using Vocal, Facial and Semantic Features"**
Williamson et al. developed a comprehensive multimodal system that integrated vocal (prosodic and spectral), facial (AU intensities and dynamics), and semantic (word usage and sentiment) features for depression detection. They explored various fusion strategies including early fusion, late fusion, and decision-level fusion. Their results demonstrated that late fusion (combining individual modality predictions) generally outperformed early fusion (concatenating features before classification), achieving state-of-the-art performance on the DAIC-WOZ dataset.

**Paper 10: Gui et al. (2019) – "Cooperative Multimodal Approach to Depression Detection in Social Media"**
Gui et al. proposed a cooperative multimodal approach that combined textual and visual (profile images and posted images) information from social media for depression detection. They developed attention-based fusion mechanisms that learned to weight different modalities based on their informativeness, achieving superior performance compared to both unimodal baselines and simple concatenation-based fusion approaches.

## 2.3 Comparative Analysis

The following table presents a comparative analysis of the reviewed systems and their relevance to MindScan:

**Table 2.1: Comparative Analysis of Existing Systems**

| System/Paper | Year | Modality | Technique | Dataset | Accuracy | Limitations |
|-------------|------|----------|-----------|---------|----------|-------------|
| De Choudhury et al. | 2013 | Text | SVM | Twitter | ~70% | Low accuracy, limited features |
| Orabi et al. | 2018 | Text | CNN, RNN | Twitter | ~85% | High computational cost |
| Tadesse et al. | 2019 | Text | LR, SVM, RF | Reddit | ~91% | Text-only, no visual cues |
| Trotzek et al. | 2018 | Text + Metadata | Neural Networks | CLPsych | ~88% | No facial analysis |
| Yates et al. | 2017 | Text | CNN | Reddit | ~82% | Binary classification only |
| Girard et al. | 2014 | Face | FACS Analysis | Clinical | N/A | Requires clinical setting |
| Jan et al. | 2017 | Face + Voice | Deep CNN | AVEC | ~78% | No text analysis |
| Yang et al. | 2017 | Audio + Video | Hybrid DL | AVEC | ~80% | No social media text |
| Williamson et al. | 2016 | Voice + Face + Text | Multi-fusion | DAIC-WOZ | ~83% | Complex, requires interview |
| Gui et al. | 2019 | Text + Image | Attention-based | Twitter | ~86% | High complexity |
| **MindScan (Proposed)** | **2026** | **Text + Face** | **LR + DeepFace** | **Reddit** | **High** | **Research tool only** |

### Key Research Gaps Identified

Based on the comprehensive literature review, the following research gaps were identified that the MindScan project aims to address:

1. **Lack of Practical Multimodal Systems:** While academic research has demonstrated the benefits of multimodal depression detection, few practical, deployable systems exist that combine text and facial analysis in a user-friendly web application.

2. **Complexity Barrier:** Most existing multimodal systems require complex deep learning architectures, specialized hardware (GPUs), and extensive training data, making them difficult to deploy and use in real-world settings.

3. **Limited Accessibility:** Existing systems are typically designed as research prototypes without intuitive user interfaces, limiting their accessibility to non-technical users.

4. **Absence of Real-time Analysis:** Many existing systems operate in batch processing mode and do not support real-time, interactive analysis capabilities.

5. **Insufficient Fusion Strategies for Lightweight Systems:** While sophisticated fusion mechanisms (attention-based, cooperative learning) have been proposed, there is a need for simpler, interpretable fusion strategies suitable for lightweight deployment.

MindScan addresses these gaps by providing a practical, lightweight, and accessible multimodal depression detection system with a modern web interface, real-time analysis capabilities, and an interpretable weighted fusion strategy.

---

\newpage
