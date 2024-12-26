# Car Prediction Model

This project builds a machine learning model to predict car prices based on various features such as road condition, years of usage, mileage, horsepower, and more. The model is developed using TensorFlow and Keras.

## Features

- **Exploratory Data Analysis (EDA)**: Visualizes data relationships using Seaborn.
- **Data Preprocessing**: Shuffles and normalizes the dataset.
- **Deep Learning Model**: Utilizes a Sequential neural network with three dense layers to predict car prices.
- **Evaluation**: Assesses model performance using metrics like Root Mean Squared Error (RMSE) and Mean Squared Error (MSE).
- **Visualization**: Plots training and validation losses and compares actual vs predicted car prices.

## Dataset

The dataset (`train.csv`) is expected to have the following features:

- `v.id`: Unique vehicle identifier
- `on road old`: Previous road condition
- `on road now`: Current road condition
- `years`: Age of the car in years
- `km`: Mileage of the car
- `rating`: Rating based on user feedback
- `condition`: Current condition of the car
- `economy`: Fuel economy
- `top speed`: Maximum speed
- `hp`: Horsepower
- `torque`: Engine torque
- `current price`: Actual current price of the car (target variable)

## Dependencies

The following Python libraries are required:

- `tensorflow`
- `pandas`
- `numpy`
- `seaborn`
- `matplotlib`

Install them using:


pip install tensorflow pandas numpy seaborn matplotlib


## Usage

Clone the repository:

git clone https://github.com/yourusername/car-prediction-model.git
cd car-prediction-model
Place the dataset (train.csv) in the project directory.

Run the script:
python car_prediction_model.py

## The script performs the following:

Loads and preprocesses the data.
Trains a deep learning model.
Evaluates the model on the test set.
Displays training history and predicted vs actual car prices.

## Results

The model's performance is evaluated using RMSE and MSE. Visualizations are provided to analyze:
Loss vs Epochs
RMSE vs Epochs
Actual vs Predicted car prices

## Future Improvements

Experiment with different model architectures.
Add hyperparameter tuning.
Incorporate feature engineering to improve model accuracy.

## Acknowledgments

TensorFlow and Keras for providing the deep learning framework.
The dataset used for this project.
Open-source tools and resources for visualization and data analysis.

