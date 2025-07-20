In this project, I developed a gender classification model using interaction data from an e-commerce platform. The dataset included user actions over time, such as product views with attributes like category, brand, and product gender.

I conducted extensive feature engineering, aggregating categorical counts (e.g., most-viewed brands, product gender distributions, category frequency), and binary flags (e.g., whether a user viewed women’s products). The goal was to transform raw event logs into meaningful user-level features.

To handle class imbalance, I used a Random Forest model with class_weight="balanced", ensuring fair learning across both gender classes. I applied a stratified train-test split to preserve the class distribution and evaluated performance using ROC AUC and Balanced Error Rate (BER). These were averaged for a custom hybrid score.
