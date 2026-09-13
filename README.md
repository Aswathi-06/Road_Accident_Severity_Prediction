#Road Accident Severity Prediction\
\
Road Accident Severity Prediction uses Machine Learning to classify road traffic accidents into three severity levels: Slight Injury, Serious Injury, and Fatal Injury, based on driver, vehicle, road, environment, and casualty details recorded at the time of the accident.\
\
The dataset used here is the "Road Traffic Accidents" dataset from Kaggle (saurabhshahane/road-traffic-accidents), downloaded using the kagglehub library.\
It contains 12,316 accident records with 32 input features covering driver information (age, sex, driving experience), vehicle information (type, ownership, defects), road and environment conditions (junction type, road surface, lighting, weather), and casualty details.\
\
Three classification models — Random Forest, Logistic Regression, and Decision Tree — are trained and compared to predict Accident_severity.\
The dataset is divided into training and testing sets using an 80-20 split, with stratification on the target column to preserve class proportions across all three severity levels.\
The performance of each model is evaluated using Accuracy Score, Classification Report, and a Confusion Matrix. \
\
Install Required Libraries

The project uses numpy, pandas, seaborn, matplotlib, scikit-learn and kagglehub.\
The Kaggle API token was entered securely using getpass() and stored in the KAGGLE_API_TOKEN environment variable.\
\
Load Dataset

The dataset is downloaded from Kaggle using kagglehub.dataset_download("saurabhshahane/road-traffic-accidents"), and the file RTA Dataset.csv is loaded into a pandas DataFrame using pd.read_csv(). \
The dataset has 12,316 rows and 32 columns, viewed initially with df.head() and df.shape.\
\
EDA (Exploratory Data Analysis)

Missing values are checked using df.isnull().sum(). Columns with heavy missing data (Defect_of_vehicle, Service_year_of_vehicle, Work_of_casuality, Fitness_of_casuality) are filled with 'Unknown', while columns with fewer missing values (such as Educational_level, Driving_experience, Type_of_vehicle, Road_surface_type, Type_of_collision, etc.) are filled with their mode (most frequent value).\
Any remaining inconsistent null-like entries ('na', 'NA', 'unknown') are standardized and filled as 'Unknown'. \
The distribution of Accident_severity is visualized using a Seaborn count plot to check class balance. \
\
The target column is label-encoded and all remaining categorical features are One-Hot Encoded, expanding the dataset to 1,253 numerical features. \
A correlation heatmap is plotted to identify the top features most correlated with accident severity.\
\
Split Dataset

The dataset is split into training and testing sets with an 80-20 ratio, random_state=42 for reproducibility, and stratify=y to maintain the same proportion of Slight, Serious, and Fatal cases in both sets.\
\
Train

Three classification models Random Forest Classifier, Logistic Regression and Decision Tree Classifier are trained on the training set\
 \
class_weight='balanced' is used across all models because the severity classes are highly imbalanced (most accidents are "Slight Injury"), and this setting helps the models pay more attention to the minority classes (Serious and Fatal).\
\
Evaluate the Model

Each model is evaluated on the test set using Accuracy Score, Classification Report and Confusion Matrix\
\
A Decision Tree plot (top 2 levels) is also generated using plot_tree() to visually interpret the rules the model uses to classify accident severity.\
\
Finally, the trained model is used to predict the severity of a new accident by providing its driver, vehicle, road, and casualty details as input.
