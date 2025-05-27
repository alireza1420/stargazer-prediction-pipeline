# Evaluating the accuracy of the prediction of stargazers in open source projects
- Task: To study which prediction model has the highest accuracy in predicting the number of stars for a GitHub repository

## Main Architecture and Final Deliverable Code of Our Project
- The main architecture and final deliverable code of our project is available in this folder !['0_ci_cd'](https://github.com/alireza1420/stargazer-prediction-pipeline/tree/main/0_ci_cd). We have provided instruction that how our system works.


## Initial Experimental Results (Update from 9th May, 2025)
- A total of 24 experiments have been applied to the prepared dataset using basic machine learning models. Here is the comparison among them.
- ![Model Comparison](https://github.com/alireza1420/stargazer-prediction-pipeline/blob/main/2_prediction_techniques/model_comparison.png)
- Experimental analysis: The results appear overfitted for most models due to data leakage. Further tuning is necessary.
- Visualize the features to identify similarities.

## Second Stage Experimental Results (Update from 15th May, 2025)
- The fields 'watchers' and 'star' show similarity. Leakage was found.
- Updated results are here.
- ![Model Comparison](https://github.com/alireza1420/stargazer-prediction-pipeline/blob/main/model_training/model_comparison.png)
- Do experiments on DNN.

## Third Stage Experimental Results (Update from 22nd May, 2025)
- The dataset has been applied with a total of 144 variations on 20 machine learning models.
- Updated results are here.
- ![Model Comparison](https://github.com/alireza1420/stargazer-prediction-pipeline/blob/main/3_more_operations_on_new_dataset_features/top_20_model_comparison.png)
- Got a stable result.
