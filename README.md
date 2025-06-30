# Task 5: Decision Trees and Random Forests

🎯 Objective
Learn and implement tree-based machine learning models for classification and evaluate them using various performance metrics.

 🧰 Tools & Libraries
  Python 3.11
  Scikit-learn
  Pandas
  Matplotlib

📊 Dataset
We use the *built-in Breast Cancer dataset* provided by Scikit-learn, which contains 30 numeric features describing characteristics of cell nuclei in breast cancer biopsies.

📝 Steps Performed

1. Load Dataset
  Loaded Breast Cancer dataset from sklearn.datasets.load_breast_cancer.
  Converted it into a Pandas DataFrame for easy handling.
  
2. Preprocessing
 No missing values or categorical features.
 Data split into training and test sets (70:30).

3. Train Decision Tree Classifier
  A Decision Tree model was trained using max_depth=4 to prevent overfitting.
  Accuracy and classification report generated.
  
4. Visualize Decision Tree
  Used plot_tree() to visualize the decision-making flow of the model.

5. Train Random Forest Classifier
  Trained an ensemble Random Forest model with 100 trees.
  Compared its accuracy and performance to the Decision Tree.

6. Feature Importance
  Plotted feature importance based on the trained Random Forest model.

7. Cross-Validation
  Evaluated the Random Forest using 5-fold cross-validation.

 📈 Results
  Decision Tree Accuracy:* ~93%
  Random Forest Accuracy:* ~96%
  Random Forest performed better in generalization and consistency.
  Feature importance highlights the most influential input features.

