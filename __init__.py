
from covid19India.getdata import fetch_india_patientdata
from covid19India.dataprep import *
from covid19India.model import *
###########get data
def create_test_model(workingdir,refresh_data = 'Y',preparegrid='Y',prepapreframesdata='Y', createmodel='Y',testmodel='Y',testspan=60,qt=-1):
	country = "India"
	if refresh_data == 'Y':
		fetch_india_patientdata(workingdir)
	if preparegrid == 'Y':
		qt = country_dataprep(workingdir,workingdir,country='India',testspan=testspan)
		print("qt=",qt)
	if prepapreframesdata == 'Y':
		frames_grid = pd.read_csv(workingdir+country+"framesgrid.csv")
		frames_grid = pickle.loads(pickle.dumps(reduce_mem_usage(frames_grid),-1))
		#df_pixel_county= pd.read_csv(workingdir+country+"pixel_counties.csv")
		prep_image(frames_grid,minframe=10,testspan=testspan,src_dir=workingdir,extframes = ['norm_pop'])
	if createmodel == 'Y':
		train_country_model(workingdir,workingdir,country='India')
	if testmodel == 'Y':
		KL_div,MAPE_grid,MAPE_countrytotal,averageerror = test_country_model(workingdir,workingdir,country='India',span=testspan)
		return (KL_div,MAPE_grid,MAPE_countrytotal,averageerror)
		
	
	
def forecast_cases(workingdir,refresh_data = 'Y',preparedata='Y',createmodel='Y',forecast='Y',forecastspan=100):
	testspan = 1
	if refresh_data == 'Y':
		fetch_india_patientdata(workingdir)
	if preparedata == 'Y':
		country_dataprep(workingdir,workingdir,country='India',testspan=testspan)
	if createmodel == 'Y':
		train_country_model(workingdir,workingdir,country='India')
	if forecast == 'Y':
		forecast_frame = forecast_country_cases(workingdir,country='India', span=forecastspan)
		return forecast_frame
		
