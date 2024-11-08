library(tidyverse)
library(tidymodels)
library(vroom)
library(ggplot2)
library(embed)
ghouls_train <- vroom("C:/Users/Josh/Documents/stat348/ghouls-goblins-and-ghosts-boo/train.csv")
ghouls_train_missing <- vroom("C:/Users/Josh/Documents/stat348/ghouls-goblins-and-ghosts-boo/trainWithMissingValues.csv")
ghouls_test <- vroom("C:/Users/Josh/Documents/stat348/ghouls-goblins-and-ghosts-boo/test.csv")

ghouls_train_missing <- ghouls_train_missing %>%
  mutate(color = factor(color)) %>%
  mutate(type = factor(type))


missing_recipe <- recipe(type~.,data=ghouls_train_missing) %>%
  step_impute_bag(rotting_flesh, impute_with = imp_vars('hair_length','bone_length','has_soul','color'), trees = 100) %>%
  step_impute_bag(hair_length, impute_with = imp_vars('rotting_flesh','bone_length','has_soul','color'), trees = 100) %>%
  step_impute_bag(bone_length, impute_with = imp_vars('hair_length','rotting_flesh','has_soul','color'), trees = 100) %>%
  step_impute_bag(color, impute_with = imp_vars('hair_length','bone_length','has_soul','rotting_flesh'), trees = 100) %>%
  step_impute_bag(has_soul, impute_with = imp_vars('hair_length','bone_length','rotting_flesh','color'), trees = 100)
  
ghouls_prepped_missing_recipe <- prep(missing_recipe)
baked_ghouls_missing <- bake(ghouls_prepped_missing_recipe, new_data=ghouls_train_missing)


rmse_vec(ghouls_train[is.na(ghouls_train_missing)],
         baked_ghouls_missing[is.na(ghouls_train_missing)])


## fit a model
## HW 24 multinomial KNN model
library(kknn)

# create the recipe, making sure to normalize for when we calculate distances
knn_recipe_ggg <- recipe(type~.,data= ghouls_train) %>%
  step_mutate_at(all_nominal_predictors(), fn = factor) %>%
  step_lencode_glm(all_nominal_predictors(), outcome = vars(type)) %>%
  step_normalize(all_numeric_predictors())

# fit the model
knn_model_ggg <- nearest_neighbor(neighbors=tune()) %>%
  set_mode("classification") %>%
  set_engine("kknn")

# workflow
knn_wf_ggg <- workflow() %>%
  add_recipe(knn_recipe_ggg) %>%
  add_model(knn_model_ggg)

# tuning grid
knn_tuning_grid_ggg <- grid_regular(neighbors(),
                                levels = 5)
#folds
knn_folds_ggg <- vfold_cv(ghouls_train, v = 10, repeats= 1)

# cross-validation
knn_CV_results_ggg <- knn_wf_ggg %>%
  tune_grid(resamples=knn_folds_ggg,
            grid=knn_tuning_grid_ggg,
            metrics=metric_set(accuracy))

# pick the best tuning parameter
best_knn_tune_ggg <- knn_CV_results_ggg %>%
  select_best(metric = "accuracy")

# finalize the workflow
final_knn_wf_ggg <-
  knn_wf_ggg %>%
  finalize_workflow(best_knn_tune_ggg) %>%
  fit(data=ghouls_train)

# make the predictions
knn_predictions_ggg <- predict(final_knn_wf_ggg,
                           new_data=ghouls_test,
                           type="class")

# create the file to submit to kaggle
knn_submission_ggg <- knn_predictions_ggg %>%
  bind_cols(.,ghouls_test) %>%
  select(id, .pred_class) %>%
  rename(type=.pred_class)

vroom_write(x=knn_submission_ggg, file ="./KNN_Preds_ggg.csv", delim=",")





## Naive bayes Classifier
install.packages("discrim")
install.packages("naivebayes")

# create the recipe
nb_recipe_ggg <- recipe(type~.,data= ghouls_train) %>%
  step_mutate_at(all_nominal_predictors(), fn = factor) %>%
  step_lencode_glm(all_nominal_predictors(), outcome = vars(type))

# nb model 
nb_model_ggg <- naive_Bayes(Laplace=tune(), smoothness=tune()) %>%
  set_mode("classification") %>%
  set_engine("naivebayes")

nb_wf_ggg <- workflow() %>%
  add_recipe(nb_recipe_ggg) %>%
  add_model(nb_model_ggg)

# tune smoothness and laplace
# tuning grid
nb_tuning_grid_ggg <- grid_regular(Laplace(),
                               smoothness(),
                               levels = 5)
#folds
nb_folds_ggg <- vfold_cv(ghouls_train, v = 10, repeats= 1)

# cross-validation
nb_CV_results_ggg <- nb_wf_ggg %>%
  tune_grid(resamples=nb_folds_ggg,
            grid=nb_tuning_grid_ggg,
            metrics=metric_set(roc_auc, accuracy))

# pick the best tuning parameter
best_nb_tune_ggg <- nb_CV_results_ggg %>%
  select_best(metric = "roc_auc")

# # Finalize the workflow and fit it
final_nb_wf_ggg <- nb_wf_ggg %>%
  finalize_workflow(best_nb_tune_ggg) %>%
  fit(ghouls_train)

# predictions
nb_predictions_ggg <- predict(final_nb_wf_ggg,
                               new_data=ghouls_test,
                               type="class")

# make the file to submit
nb_submission_ggg <- nb_predictions_ggg %>%
  bind_cols(.,ghouls_test) %>%
  select(id, .pred_class) %>%
  rename(type=.pred_class)

vroom_write(x=nb_submission_ggg, file ="./nb_Preds_ggg.csv", delim=",")
