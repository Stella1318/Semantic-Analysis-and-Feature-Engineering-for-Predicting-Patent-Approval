# Semantic-Analysis-and-Feature-Engineering-for-Predicting-Patent-Approval

This repository contains my final project for my undergraduate machine learning class, which was recognized as the top submission. In this project, I aimed to predict patent approval using the patent application abstract. I also developed a similarity score to capture the uniqueness of each patent application. For a detailed discussion of my model specifications, please continue reading below. It's important to note that this project is a work in progress. I plan to update it regularly as I gather new ideas to further enhance my model.

## Predicting Patent Approval

### `patent_prediction.ipynb`

#### Introduction and Context

Patent offices worldwide employ the concept of patentability, which encompasses six key criteria: Patentable Subject, Utility, Novelty, Quality, Non-Obviousness, and Enablement, guiding examiners in their patent grant decisions. It's intuitive to presume that inventors and patent lawyers endeavor to incorporate these criteria into patent applications and subsequent abstracts, suggesting the potential predictive power of abstracts in determining patent approval. This prompts our primary research question: **Can machine learning models effectively predict patent approval based on patent abstracts?**

Additionally, a significant determinant of patent approval is the overlap with previously granted patents. Thus, our secondary research question emerges: **Can patent approval be predicted based on an application's similarity to previously published patents?** 

A review of existing literature on patent approval is instrumental in situating this project within the broader academic and industry discourse. Prior studies have explored diverse methodologies, from traditional statistical models to advanced machine learning approaches. For instance, Carley et al. (2013) utilized statistical models to analyze patent approval rates, while Hido et al. (2012) employed textual mining techniques and logistic regression to predict patent approval. In a similar vein, Lin et al. (2018) evaluated patent quality using deep learning models.

Our project diverges from previous approaches in two key aspects. Firstly, we simplify the problem from regression to classification, employing a binary target—Approved or Not Approved—enhancing interpretability and eliminating the need for threshold scores. Secondly, we adopt a more straightforward yet sophisticated approach to determine patent overlap and similarity, aiming to reduce computational complexity.

#### Data and Methodology

##### Data

We sourced our dataset from Patentview, selecting a sample size of 5000 observations and 42 columns, including text variables, due to time and computational constraints. Upon further analysis, we identified the intrinsic quality of the product/process as the sole relevant predictor.

Our primary variables of interest are `application_abstract` and `rel_app_text`. The former denotes the abstract of the patent application, while the latter refers to a related patent application description. Although not explicitly stated, we inferred that `rel_app_text` pertains to previously published patents with significant overlap with the corresponding patent application.

##### Text Vectorization

Our methodology commences with the construction of a vector representation for our text variables (`application_abstract` and `rel_app_text`) post pre-processing. This transformation is imperative for utilizing text variables as predictors in machine learning models, which typically demand numerical input. We opted for TF-IDF vectorization due to its suitability in capturing semantic information relevant to our problem domain.

##### Feature Engineering: Similarity Score

To address our secondary research question regarding patent approval based on overlap and similarity, we engineered a similarity score variable. Initially, we utilized Latent Dirichlet Allocation (LDA) to generate topic distributions for both `application_abstract` and `rel_app_text`, revealing underlying themes in our document corpus. Subsequently, we calculated the cosine similarity between the topic distributions of each pair of documents, signifying their thematic resemblance. Notably, ensuring uniformity in LDA model hyperparameters between `application_abstract` and `rel_app_text` was paramount, requiring meticulous tuning to attain congruent topic distributions. This step, albeit time-consuming, proved essential in accurately assessing document similarity.

#### Model Selection and Training

- **Introduction to Various Models**: Implementation of various machine learning algorithms suitable for the classification task, such as Logistic Regression, Decision Trees, Random Forest, Gradient Boosting, and Support Vector Machines (SVM).
- **Training and Hyperparameter Tuning**: Training each model on the prepared dataset and tuning hyperparameters to optimize performance. Use of cross-validation to evaluate the robustness of the models and avoid overfitting.

#### Model Evaluation

- **Evaluation Metrics**: Evaluation of the trained models using metrics such as accuracy, precision, recall, F1-score, and ROC-AUC. 
- **Model Comparison**: Comparison of model performance on the training and testing datasets to identify the best-performing model.
- **Visualization**: Visualization of the ROC curves, precision-recall curves, and confusion matrices to assess the models' performance in detail.

#### Advanced Techniques

- **Ensemble Methods**: Application of ensemble methods, such as Bagging and Boosting, to improve model accuracy and robustness.
- **Model Interpretability**: Discussion on model interpretability, including feature importance and SHAP (SHapley Additive exPlanations) values.
- **Computational Efficiency**: Examination of the trade-offs between model complexity and performance.

#### Results and Insights

- **Summary of Findings**: Summary of the findings from the patent approval prediction analysis.
- **Insights**: Insights derived from the models, including key patterns and relationships identified in the data.
- **Discussion**: Discussion on the limitations of the analysis and potential areas for future work.

#### Conclusion

- **Recap of Objectives**: Recap of the objectives and main findings from the notebook.
- **Reflection**: Reflection on the importance of predictive modeling in the context of patent approval and its applications in various domains.
- **Future Work**: Plans for future updates and enhancements to the model.

