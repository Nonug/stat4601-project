# stat4601-project
Time Series Analysis on the Average Temperature of Stanley

## Background
Given the importance of drawing accurate temperature forecasts, our group is interested in finding a good fit and suitable time series model for studying and making forecasts for the weather temperature of Hong Kong
## Dataset
 extracted the daily temperature dataset of Hong Kong (Stanley) from data.gov.hk that comprises the mean daily temperature of Stanley, Hong Kong

## Directory
Documents of interest are: 
```
├───data                          # source data and the processed data
├───plots                         # images of our report/presentation
│   ├───eda       
│   ├───models                    # Outputs of model summaries
│   ├───temperature
├───scripts                       # utility scripts
│   ├───fetch.py                  # for obtaining data from data.gov.hk
│   ├───plot.py                   # for plotting data
│   └───stats.py                  # for computing statistics
├───eda.ipynb                     # Exploratory data analysis
├───model_fitting.ipynb           # Model training, diagnostics and forecasting
```