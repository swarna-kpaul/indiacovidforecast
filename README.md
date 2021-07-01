# indiacovidforecast
Reference paper - https://www.medrxiv.org/content/10.1101/2020.10.19.20215665v1

All intermediate and output files will be stored in workingdir

```
import indiacovidforecast as CI
workingdir = "/content/drive/My Drive/covid19Indiatest/"
```

## Refresh data
###### no GPU needed
```
CI.create_test_model(workingdir=workingdir,refresh_data = 'Y',preparegrid='N',prepapreframesdata='N',createmodel='N',testmodel='N',testspan=1)
```

## Create grid data
###### no GPU needed
```
CI.create_test_model(workingdir=workingdir,refresh_data = 'N',preparegrid='Y',prepapreframesdata='N',createmodel='N',testmodel='N',testspan=1)
```
## Create frames data for a testperiod of most recent 100 days
###### no GPU needed
```
CI.create_test_model(workingdir=workingdir,refresh_data = 'N',preparegrid='N',prepapreframesdata='Y',createmodel='N',testmodel='N',testspan=100)
```
## Train a model for a testperiod of most recent 100 days
###### GPU needed
```
CI.create_test_model(workingdir=workingdir,refresh_data = 'N',preparegrid='N',prepapreframesdata='N',createmodel='Y',testmodel='N',testspan=100)
```
## Test the model created for a testperiod of most recent 100 days
###### GPU needed
```
CI.create_test_model(workingdir=workingdir,refresh_data = 'N',preparegrid='N',prepapreframesdata='N',createmodel='N',testmodel='Y',testspan=100)
```

## Create frames data for forecasting
###### no GPU needed
```
CI.create_test_model(workingdir=workingdir,refresh_data = 'N',preparegrid='N',prepapreframesdata='Y',createmodel='N',testmodel='N',testspan=5)
```
## Train a model for a forecasting
###### GPU needed
```
CI.create_test_model(workingdir=workingdir,refresh_data = 'N',preparegrid='N',prepapreframesdata='N',createmodel='Y',testmodel='N',testspan=5)
```
## Forecast for next 100 days
###### GPU needed
```
forecast_frame=CI.forecast_cases(workingdir=workingdir,refresh_data = 'N',preparedata='N',createmodel='N',forecast='Y',forecastspan=105)
```
