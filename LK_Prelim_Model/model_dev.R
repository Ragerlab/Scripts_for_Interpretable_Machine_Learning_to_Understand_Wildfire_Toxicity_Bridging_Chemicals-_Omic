rm(list=ls())

library(janitor)
library(reshape2)
library(tidyverse)
library(factoextra)
library(scales)
library(gridExtra)
library(randomForest)
library(caret)
library(parsnip)
library(gt)
library(mltools)
library(naniar)
library(viridis)
library(readxl)
library(Metrics)

#Set working directory
setwd("LK_Lab_Notebook/Projects/13_Cloud")

#Read in  and format mouse tox data
tox <- read_xlsx("input/ChemistrywTox_MouseMap_042821.xlsx", sheet=3)
neutro <- tox %>%
  rename("Exposure"="Exposure...1") %>%
  filter(Exposure!="LPS" & Exposure!="Saline") %>%
  mutate(Link=paste0(Exposure,"_",MouseID)) %>%
  select(Exposure, Link,  Neutrophil)

#Isolate injury protein marker (outcome var) from tox dataset 
injury <- tox %>%
  rename("Exposure"="Exposure...1") %>%
  filter(Exposure!="LPS" & Exposure!="Saline") %>%
  mutate(Link=paste0(Exposure,"_",MouseID)) %>%
  select(Exposure, Link, Injury_Protein) 

#Read in  and format burn chemistry data (predictor vars)
chem <- read_xlsx("input/ChemistrywTox_MouseMap_042821.xlsx", sheet=2)
exps <- colnames(select(chem, contains("Flaming") | contains("Smoldering")))
chem <- chem %>%
  select(Chemical, all_of(exps)) %>%
  column_to_rownames("Chemical") %>%
  t() %>%
  as.data.frame() %>%
  rownames_to_column("Exposure")




############################################################################################
### RF to predict injury protein response
############################################################################################

# Very simple train controls
ctrl <- trainControl(method='LOOCV',
                     search='grid')



# Merge injury protein markers with chemistry data so that each out outcome injury marker protein is associated
# with the chemistry data for the corresponding burn. 
injury_df <- merge(injury, chem, by="Exposure", all.x=TRUE)
injury_df <- injury_df %>% column_to_rownames("Link") %>% select(!Exposure) %>% mutate(across(everything(), as.numeric))


# Set seed and establish train and test sets to use throughout
set.seed(9)
tt_indices <- createDataPartition(y=injury_df$Injury_Protein, p=0.6, list=FALSE)

train_x <- injury_df[tt_indices,]
train_y <- train_x["Injury_Protein"]
train_x <- train_x %>% select(!Injury_Protein)
# write.csv(train_set,"output/training_data.csv")


test_x <- injury_df[-tt_indices,]
test_y <- test_x["Injury_Protein"]
test_x <- test_x %>% select(!Injury_Protein)
# write.csv(test_set, "output/testing_data.csv")

# Grid of different number of predictors to try in trees
p_injury <- ncol(train_x)
tunegrid_injury <- expand.grid(mtry = c(floor(sqrt(p_injury)), p_injury/2, p_injury))

# Train model
set.seed(17)
Sys.time()
rf_gridsearch_injury <- train(x=train_x,
                              y = train_y$Injury_Protein,
                              trControl=ctrl,
                              method = 'rf',
                              importance=TRUE,
                              ntree=500,
                              tuneGrid = tunegrid_injury)
Sys.time()

#Extract the best performing model from grid search
rf_final_injury <- rf_gridsearch_injury$finalModel


#Look at variable importance plot
varImpPlot(rf_final_injury)
var_imp_rf_injury <- as.data.frame(importance(rf_final_injury))
# write.csv(var_imp_rf_injury, "output/injury_protein_var_imp.csv")

#Apply model to test set
set.seed(17)
rf_res_injury <- predict(rf_final_injury, test_x)
rmse(test_y$Injury_Protein, rf_res_injury)



