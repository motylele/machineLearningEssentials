import pandas as pd
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt


def read_data(file_path):
    """Read the data from a CSV file and return a DataFrame"""
    df = pd.read_csv(file_path)
    return df


def forecast_temperature(df, target_year):
    """Perform linear regression to forecast the temperature for the target year"""
    linear_regression = stats.linregress(x=df['Year'], y=df['Temperature'])
    predicted_temp = linear_regression.slope * target_year + linear_regression.intercept
    return predicted_temp


def linear_regression_visualization(df):
    """Visualize the time series linear regression as a plot"""
    sns.set(style="darkgrid")
    plt.figure(figsize=(10, 6))
    sns.regplot(x='Year', y='Temperature', data=df, scatter_kws={'s': 80}, line_kws={'color': 'red'})
    plt.xlabel('Year')
    plt.ylabel('Temperature (degrees Celsius)')
    plt.title('Linear Regression: Year vs. Temperature')
    plt.show()


def main():
    # Read data from CSV file and rename columns
    # Data are January temperatures from Washington, D. C. from 1953-2018
    data_file = 'data.csv'
    df = read_data(data_file)
    df.columns = ['Year', 'Temperature']

    # Print first five records from the data frame
    print("First five records:")
    print(df.head())

    # Print descriptive statistics for temperature
    print("\nTemperature descriptive statistics:")
    print(df['Temperature'].describe())

    # Forecast temperature for the year 2022
    target_year = 2022
    predicted_temp = forecast_temperature(df, target_year)
    print("\nForecast average maximum temperature in January {} is {:.2f} degrees Celsius".format(target_year, predicted_temp))

    # Provide the actual temperature for January 2022
    actual_temp = -0.27
    print("The actual temperature in January {} is {} degrees Celsius".format(target_year, actual_temp))

    # Calculate the forecasting error
    forecasting_error = abs(predicted_temp - actual_temp)
    print("Forecasting error is {:.2f} degrees Celsius".format(forecasting_error))

    # Visualizing the forecast
    linear_regression_visualization(df)


if __name__ == "__main__":
    main()
