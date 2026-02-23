# Ex.No:04   FIT ARMA MODEL FOR TIME SERIES
# Date: 



### AIM:
To implement ARMA model in python.
### ALGORITHM:
1. Import necessary libraries.
2. Set up matplotlib settings for figure size.
3. Define an ARMA(1,1) process with coefficients ar1 and ma1, and generate a sample of 1000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

4. Display the autocorrelation and partial autocorrelation plots for the ARMA(1,1) process using
plot_acf and plot_pacf.
5. Define an ARMA(2,2) process with coefficients ar2 and ma2, and generate a sample of 10000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

6. Display the autocorrelation and partial autocorrelation plots for the ARMA(2,2) process using
plot_acf and plot_pacf.
### PROGRAM:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


data = pd.read_csv("/kaggle/input/time-series/daily-minimum-temperatures-in-me.csv", on_bad_lines='skip')
data.columns = ["Date", "Temp"]
data["Date"] = pd.to_datetime(data["Date"])
data["Temp"] = pd.to_numeric(data["Temp"], errors="coerce")
data.dropna(inplace=True)

# Extract time series
X = data["Temp"].values
N = len(X)

plt.rcParams['figure.figsize'] = [12, 6]
plt.plot(X)
plt.title("Original Melbourne Temperature Data")
plt.xlabel("Time Index")
plt.ylabel("Temperature")
plt.grid(True)
plt.show()


plt.subplot(2, 1, 1)
plot_acf(X, lags=50, ax=plt.gca())
plt.title("Original Data ACF")

plt.subplot(2, 1, 2)
plot_pacf(X, lags=50, ax=plt.gca())
plt.title("Original Data PACF")
plt.tight_layout()
plt.show()

arma11_model = ARIMA(X, order=(1, 0, 1)).fit()

# Show parameter names & values
print("Parameter names (ARMA 1,1):", arma11_model.param_names)
print("Parameter values:", arma11_model.params)

# Extract coefficients using integer index
phi11_arma11 = arma11_model.params[1]    # AR(1)
theta11_arma11 = arma11_model.params[2]  # MA(1)


ar1 = np.array([1, -phi11_arma11])
ma1 = np.array([1, theta11_arma11])

ARMA_1 = ArmaProcess(ar1, ma1).generate_sample(nsample=N)

plt.plot(ARMA_1)
plt.title("Simulated ARMA(1,1) Process")
plt.xlim([0, 500])
plt.grid(True)
plt.show()

plot_acf(ARMA_1, lags=50)
plt.title("ACF of ARMA(1,1)")
plt.show()

plot_pacf(ARMA_1, lags=50)
plt.title("PACF of ARMA(1,1)")
plt.show()


arma22_model = ARIMA(X, order=(2, 0, 2)).fit()

# Show parameters
print("Parameter names (ARMA 2,2):", arma22_model.param_names)
print("Parameter values:", arma22_model.params)

# Extract AR & MA coefficients
phi1_arma22 = arma22_model.params[1]   # AR(1)
phi2_arma22 = arma22_model.params[2]   # AR(2)

theta1_arma22 = arma22_model.params[3] # MA(1)
theta2_arma22 = arma22_model.params[4] # MA(2)


ar2 = np.array([1, -phi1_arma22, -phi2_arma22])
ma2 = np.array([1, theta1_arma22, theta2_arma22])

ARMA_2 = ArmaProcess(ar2, ma2).generate_sample(nsample=N*10)

plt.plot(ARMA_2)
plt.title("Simulated ARMA(2,2) Process")
plt.xlim([0, 500])
plt.grid(True)
plt.show()


plot_acf(ARMA_2, lags=50)
plt.title("ACF of ARMA(2,2)")
plt.show()

plot_pacf(ARMA_2, lags=50)
plt.title("PACF of ARMA(2,2)")
plt.show()
```
OUTPUT:
<img width="1282" height="561" alt="image" src="https://github.com/user-attachments/assets/525027ac-9daa-4e1c-b3a6-57288511b68c" />
<img width="980" height="501" alt="image" src="https://github.com/user-attachments/assets/9410d152-5b05-4203-81dc-a9d363fab503" />
<img width="1019" height="513" alt="image" src="https://github.com/user-attachments/assets/49801e0c-fae9-48e0-9068-8b199cbf4517" />
<img width="1083" height="505" alt="image" src="https://github.com/user-attachments/assets/d100a14e-0135-4010-bd04-53e3b3c74539" />
<img width="1040" height="506" alt="image" src="https://github.com/user-attachments/assets/95baa27a-5d66-4751-aff1-5ef44ba7617e" />
<img width="1026" height="508" alt="image" src="https://github.com/user-attachments/assets/5a8765fe-2483-49d7-9296-5f0c90212429" />
<img width="1099" height="502" alt="image" src="https://github.com/user-attachments/assets/72694124-4f50-4f31-88dd-ffeb3a19d5e7" />

RESULT:
Thus, a python program is created to fir ARMA Model successfully.
