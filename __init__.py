
from getdata import fetch_india_patientdata
from dataprep import *
from model import *
###########get data
def create_test_model(workingdir,refresh_data = 'Y',preparedata='Y',createmodel='Y',testmodel='Y',testspan=60):
	if refresh_data == 'Y':
		fetch_india_patientdata(workingdir)
	if preparedata == 'Y':
		country_dataprep(workingdir,workingdir,country='India',testspan=testspan)
	if createmodel == 'Y':
		train_country_model(workingdir,workingdir,country='India')
	if testmodel == 'Y':
		KL_div,MAPE_grid,MAPE_countrytotal,averageerror = test_country_model(workingdir,workingdir,country='India',testspan=testspan)
		return (KL_div,MAPE_grid,MAPE_countrytotal,averageerror)
		
	
	
def forecast_cases(workingdir,refresh_data = 'Y',preparedata='Y',createmodel='Y',forecast='Y'):
	testspan = 1
	if refresh_data == 'Y':
		fetch_india_patientdata(workingdir)
	if preparedata == 'Y':
		country_dataprep(workingdir,workingdir,country='India',testspan=testspan)
	if createmodel == 'Y':
		train_country_model(workingdir,workingdir,country='India')
	if forecast == 'Y':
		forecast_frame = forecast_country_cases(workingdir,country='India')
		return forecast_frame