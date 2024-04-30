# Legal Text Classification for Document Categorization in Law

## 1. Introduction
The Legal Text Classification System is an advanced natural language processing (NLP) solution designed to automate the categorization of legal documents. Leveraging state-of-the-art deep learning techniques, the system streamlines document management and analysis processes in legal domains, offering improved efficiency and accuracy. This project is a collaborative effort by Sowjanya Linga and Vamshi Thatikonda, pursuing their Master's degrees in Data Science at the University of New Haven.

## 2. Project Overview
Our research introduces a Legal Text Classification System aimed at automating document categorization in legal management. By leveraging advanced NLP techniques, the system efficiently categorizes legal documents into predefined classes, streamlining retrieval processes. We review complexities in legal document processing and advancements in text classification, employing various algorithms like LSTM, SVM, and Random Forest. Utilizing tools such as Python, scikit-learn, and TensorFlow, along with data preprocessing and deep learning architectures, our project delivers a robust NLP model and intuitive interface. Our evaluation methodology includes model performance metrics, comparative analysis, and user feedback integration, contributing to NLP advancements and offering a valuable tool for legal professionals.

## 3. Key Features
- **Advanced Text Classification**: Utilizes bidirectional LSTM (Long Short-Term Memory) networks, Support Vector Machines (SVM), and Random Forest algorithms for accurate document classification.
- **Flexible Architecture**: Allows for customization of model architecture and hyperparameters to adapt to diverse legal document datasets.
- **Comprehensive Evaluation**: Incorporates various performance metrics such as accuracy, precision, recall, and F1-score for thorough model assessment.
- **Interactive Interface**: Provides an intuitive interface for legal professionals to input text and receive predictions using the trained models.
- **Ease of Deployment**: Models are saved for future use and can be easily deployed in practical legal workflows.

## 4. Getting Started
To get started with the Legal Text Classification System, follow these steps:
1. Clone the repository from [GitHub](https://github.com/SowjanyaLinga/NLP_Project).
2. Install the required dependencies using `pip install .
3. Prepare your legal text dataset in CSV format, ensuring it includes both text content and corresponding labels.
4. Train and evaluate the models using the provided code scripts, adjusting hyperparameters as needed for optimal performance.
5. Save the trained models' state dictionaries for future use and deployment.

## 5. Usage Instructions
### LSTM Model
- Initialize a `TextClassifier` object with specified input, embedding, hidden, and output dimensions.
- Train the model using the provided training loop, optimizing with the Adam optimizer and cross-entropy loss function.
- Evaluate the trained model on a held-out test set using various evaluation metrics.
- Save the trained model's state dictionary for future use and deployment.

### SVM Model
- Utilize the `LinearSVC` classifier from scikit-learn for training and classification.
- Preprocess the text data, vectorize using TF-IDF, and train the SVM classifier.
- Evaluate the model's performance using accuracy, precision, recall, and F1-score.

### Random Forest Model
- Train a `RandomForestClassifier` using TF-IDF vectorized features.
- Predict outcomes for sample legal text using the trained model.
- Evaluate the model's performance using accuracy, precision, recall, and F1-score.

## 6. Conclusion
The Legal Text Classification System offers a powerful solution for automating document categorization in the legal domain. With its advanced NLP capabilities and flexible model architecture, the system provides legal professionals with an efficient and reliable tool for managing and analyzing large volumes of legal texts.

## 7. Future Work
In the future, we aim to explore more sophisticated NLP methods such as BERT-based models for contextual understanding and transformer architectures for improved document representation. Fine-tuning these models on legal text corpora and incorporating domain-specific embeddings can enhance the system's ability to capture nuanced semantic relationships and improve classification accuracy.

## 8. Video Explanation
watch the accompanying video tutorial. 
The video provides insights into the key concepts, implementation strategies, and performance evaluation.
[Video Explanation](https://youtu.be/qOqGKSXeNCE)

## 9. References
- [IEEE Xplore - Legal Text Classification for Document Categorization in Law](https://ieeexplore.ieee.org/document/9207211)
- [ResearchGate - Preparing Legal Documents for NLP Analysis: Improving the Classification of Text Elements by Using Page Features](https://www.researchgate.net/publication/358028171_Preparing_Legal_Documents_for_NLP_Analysis_Improving_the_Classification_of_Text_Elements_by_Using_Page_Features)
- [ResearchGate - Evaluating Text Classification in the Legal Domain Using BERT Embeddings](https://www.researchgate.net/publication/375654387_Evaluating_Text_Classification_in_the_Legal_Domain_Using_BERT_Embeddings)
- [Springer - Legal Text Classification Using Machine Learning Techniques](https://link.springer.com/chapter/10.1007/978-981-99-8181-6_9)
- [ScienceDirect - Text Classification in Legal Domain](https://www.sciencedirect.com/science/article/abs/pii/S0306457321002764)


