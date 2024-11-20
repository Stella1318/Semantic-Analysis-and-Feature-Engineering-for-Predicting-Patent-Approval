# Semantic Analysis and Feature Engineering for Predicting Patent Approval

## Abstract
The patent approval process is a cornerstone of intellectual property protection, incentivizing innovation and safeguarding inventors' rights. However, this process is often protracted and resource-intensive, presenting significant challenges for stakeholders such as inventors, investors, and policymakers. With an average approval wait time exceeding two years and substantial associated costs, there is a growing need for predictive tools to streamline patent evaluation. This paper investigates whether machine learning models can predict patent approval status using the textual content of patent abstracts and related application descriptions. By leveraging techniques such as TF-IDF vectorization and topic modeling through Latent Dirichlet Allocation (LDA), we extract relevant features for classification. Additionally, we engineer a similarity score to quantify overlaps with previously granted patents. Our results, obtained through a comparative analysis of multiple classification models, indicate that while text-based similarity offers predictive insights, the combination of abstract content and related text data yields the most accurate predictions. These findings highlight the potential for computational tools to enhance decision-making in the patent application 

## Introduction and Context

Innovation drives economic growth, particularly in technology-intensive industries where advancements rely heavily on research and development (R&D). Patents play a pivotal role in this ecosystem by granting inventors exclusive rights to their creations, thereby encouraging investment in novel ideas. However, the patenting process is fraught with challenges. Obtaining approval for a patent can take over two years, with significant costs incurred during the application process. In 2020 alone, the U.S. Patent and Trademark Office (USPTO) processed approximately 646,000 applications, granting only 388,900 patents—a 50% approval rate when factoring in wait times. These dynamics underscore the critical need for efficient evaluation mechanisms to support stakeholders in navigating the patenting landscape.

Predicting patent approval status can aid in strategic decision-making for resource allocation, investment planning, and intellectual property management. Given that patent offices assess applications based on criteria such as novelty, utility, and non-obviousness, we hypothesize that the content of patent abstracts is predictive of approval outcomes. Moreover, since the overlap with previously granted patents influences decisions, we explore whether the similarity between new and existing applications can enhance predictive accuracy.

Existing research in this domain has employed various methodologies, from statistical models to advanced machine learning techniques. While some studies have focused on patentability scores, others have integrated textual and non-textual features for predictive analysis. Our approach differs in two significant ways. First, we simplify the prediction task to a binary classification problem—Approved or Not Approved—enhancing interpretability. Second, we introduce a feature engineering technique to quantify textual similarity, reducing dimensionality while retaining meaningful information. This paper aims to contribute to the field by proposing a computational framework for patent approval prediction, leveraging machine learning to address this complex challenge.

## Features

- **Data Preprocessing**: Tokenization, stopword removal, stemming, and lemmatization.
- **Feature Engineering**: TF-IDF vectorization, LDA topic modeling, cosine similarity.
- **Model Training**: Logistic regression with GridSearchCV for hyperparameter tuning.
- **Evaluation**: Accuracy, classification report, and model performance visualization.

Here's the rewritten continuation:

---

**2. Data and Methodology**

**2.1 Data**

The dataset for this study was sourced from PatentsView, consisting of 5,000 observations and 42 variables, including textual features. Due to time and computational constraints, this sample size was deemed sufficient for our analysis. The primary variables of interest are `application_abstract`, which contains the patent abstract, and `rel_app_text`, which describes a related patent application. While the exact definition of `rel_app_text` is not explicitly stated, we assume it represents previously granted patents with significant overlap to the current application. This assumption aligns with the dataset's structure and relevance to our research objectives.

Given the theoretical focus on patent quality, we prioritize intrinsic characteristics of the patent rather than external factors. This approach ensures that the analysis remains rooted in the substantive content of the applications, which is directly linked to approval likelihood.

---

**2.2 Text Variable Vectorization**

Textual data must be transformed into numerical representations to serve as inputs for machine learning models. For this purpose, we applied Term Frequency-Inverse Document Frequency (TF-IDF) vectorization, a technique that balances word frequency and specificity. TF-IDF was chosen over alternatives like Bag-of-Words and word embeddings because it effectively captures the relevance of terms within the context of technical patent language.

Unlike general text data, patent abstracts often feature consistent terminology to describe identical or similar concepts. For instance, compounds with the same molecular structure may have different isomer names, as seen with Modafinil and Armodafinil. TF-IDF assigns higher weights to terms that are uniquely representative of a specific document while accounting for their prevalence across the corpus. This approach enhances the model's ability to discern nuanced differences in patent language, which are critical for approval decisions.

---

**2.3 Feature Engineering: Similarity Score**

To assess the impact of textual overlap between applications, we engineered a similarity score based on Latent Dirichlet Allocation (LDA) topic modeling. The process involved the following steps:

1. **Topic Extraction**: LDA was used to uncover thematic structures in the `application_abstract` and `rel_app_text` variables. Each document was represented as a distribution of topics, with the number of topics determined through coherence score optimization.
   
2. **Topic Distribution Representation**: For each document, a vector of topic proportions was generated, reflecting the relative prevalence of each theme.

3. **Cosine Similarity Calculation**: The cosine similarity between the topic distributions of `application_abstract` and `rel_app_text` was computed for each observation. Higher similarity values indicate greater thematic overlap, which may influence approval outcomes.

This process required extensive tuning of LDA hyperparameters to ensure consistent topic structures across both textual variables. Initially, discrepancies in topic counts necessitated exclusion restrictions, but subsequent resampling and tuning resolved these issues.

---

**2.4 Classification Models**

To predict patent approval status, we tested four models using three classifiers: Penalized Logistic Regression, K-Nearest Neighbors (KNN), and Random Forest. The models incorporated varying combinations of features, as summarized in Table 1:

| Model   | Feature 1            | Feature 2              | Feature 3          | Target               |
|---------|----------------------|------------------------|--------------------|----------------------|
| Model 1 | `application_abstract` | —                      | —                  | `approval_status`    |
| Model 2 | `rel_app_text`        | —                      | —                  | `approval_status`    |
| Model 3 | `application_abstract` | `similarity_score`     | —                  | `approval_status`    |
| Model 4 | `application_abstract` | `rel_app_text`         | `similarity_score` | `approval_status`    |

Class imbalance in the target variable (`approval_status`, with a 3300:1700 ratio) was addressed using undersampling during training and class weights in the classifiers. These methods mitigated overfitting and improved model performance on minority class predictions.

Grid search was employed to optimize hyperparameters for each classifier, ensuring robust model tuning. This systematic approach enhanced the reliability of our results.

---

**3. Results**

**3.1 Optimal Number of Topics Through LDA**

The optimal number of topics for both `application_abstract` and `rel_app_text` was determined to be 20, based on coherence score analysis. Figures 1.1 and 1.2 illustrate the coherence score trends, highlighting the stability of topics in abstracts compared to the variability in related application texts. This variability likely reflects the diverse nature of related patents and their terminology.

---

**3.2 Top Words for Each Topic**

Figures 3 and 5 display the top 10 words for each topic in `application_abstract` and `rel_app_text`, respectively. The consistency of terms in abstracts across training and test datasets underscores the robustness of LDA in capturing thematic structures. Conversely, the variability in `rel_app_text` suggests differences in the linguistic and conceptual content of related patents.

---

**3.3 Model Performance**

The accuracy and classification reports for the four models are summarized below:

- **Logistic Regression**: Model 4 achieved the highest accuracy (55.95%).
- **KNN Classifier**: Model 3 outperformed others with an accuracy of 58.1%.
- **Random Forest Classifier**: Model 4 yielded the best performance at 58.8%.

Despite these results, class imbalances persisted, as evidenced by disparities in precision and F1 scores across categories. The Random Forest Classifier's superior accuracy suggests its effectiveness in capturing complex patterns within the data.

---

**3.4 Discussion**

The inclusion of similarity scores marginally improved predictive performance, emphasizing the nuanced role of textual overlap in patent approval. Model 4 consistently outperformed others, highlighting the benefits of combining abstract content with related text data and similarity measures. However, the limitations of LDA in capturing technical distinctions, such as novel chemical compositions, may have constrained the model's predictive capabilities.

---

**4. Conclusion**

Our findings demonstrate the feasibility of predicting patent approval status using machine learning models, with a maximum accuracy of 58.8% achieved through a Random Forest Classifier. While text-based similarity provides valuable insights, its predictive power is limited by the granularity of topic modeling techniques like LDA. Future research could explore advanced algorithms, such as large language models, to capture deeper semantic and technical nuances in patent applications.

---

Let me know if you'd like additional refinements or edits to specific sections!
## License

MIT License

Copyright (c) 2024 [Stella Abelinde]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


## Contributor
- Harshikha Agarwal

## Acknowledgements

- The dataset used in this project is sourced from Patentview.
- Thanks to Prof.Marlene Koffi for guidance and support !
  

## Credits

This notebook is an original work. If you use this notebook or any part of this repository for your research or project, please credit this repository and the author. Proper citation includes linking back to this repository and mentioning the author's name. Thank you!
