import numpy as np
import math
import pandas as pd
import pandasql as ps
import matplotlib.pyplot as plt
import pickle
import multiprocessing
## Function to divide the GRID Area into Pixels
## Parameter Needed - 1. pixlatmax - float - Maximum Value of Lattitude( GRID Boundary) 2. pixlatmin - float - Minimum value of the lattitudes( GRID Boundary)
##						3. pixlonmax - float - Maximum value of Longitude( GRID Boundary) 4. pixlonmin - float - Minimum value of longitude( GRID Boundary)
##						5. pixelsize - Number - Size of Earch Pixel in GRID(Number of Pixel in Grid)	6. Grid No - Number - The Id of Grid

def GetPixelDF(pixlatmin,pixlatmax,pixlonmin,pixlonmax,pixelsize,grid_no):
	fact=100000000
	latmin = np.int(pixlatmin*fact)
	latmax = np.int(pixlatmax*fact)
	longmin = np.int(pixlonmin*fact)
	longmax = np.int(pixlonmax*fact)
	pixelLatRangeStep = np.int((latmax-latmin)/(pixelsize))
	pixelLonRangeStep = np.int((longmax-longmin)/(pixelsize))
	pixlatvals = list(np.round(np.arange(latmin,latmax,pixelLatRangeStep)/fact,5))
	if len(pixlatvals) == pixelsize:
		pixlatvals.append(pixlatmax)
	pixlonvals = list(np.round(np.arange(longmin,longmax,pixelLonRangeStep)/fact,5))
	if len(pixlonvals) == pixelsize:
		pixlonvals.append(pixlonmax)
	ret_df = []
	pixno = 1
	for i in range(len(pixlatvals)-1):
		minlat = pixlatvals[i]
		maxlat = pixlatvals[i+1]
		for j in range(len(pixlonvals)-1):
			minlong = pixlonvals[j]
			maxlong = pixlonvals[j+1]
			ret_df.append([grid_no,pixno,minlat,maxlat,minlong,maxlong])
			pixno +=1 
	ret_df = pd.DataFrame(ret_df,columns =['grid','pixno','minlat','maxlat','minlong','maxlong'])
	return ret_df

## Function to divide the whole country into GRIDS and Pixels
## Parameter Needed - 1. latlongrange - Tuple -	Coordinate boundary of the country(south, north, west, east) 2. latstep - number -Number of division under lattitude range
##						3. longstep - Number - Number of division under longitude range	4. margin - Number - Overlapping adjustment for pixel boundaries
##						5. pixelsize - Number - Pixelsize of each subpixel	6. counties - Dataframe - The county Dataframe containing the lattitude longitude and population data

def get_The_Area_Grid(latlongrange,latstep,longstep,margin,pixelsize, counties):
	fact=100000000
	(min_lat,max_lat,min_long,max_long) = latlongrange#(23, 49,-124.5, -66.31)
	min_lat = np.int(min_lat*fact)
	max_lat = np.int(max_lat*fact)
	min_long = np.int(min_long*fact)
	max_long = np.int(max_long*fact)
	range_of_longitude = max_long - min_long
	range_of_latitude = max_lat - min_lat
	block_longitude = np.int(range_of_longitude/(longstep))
	block_latitude = np.int(range_of_latitude/(latstep))
	lattitudes = list(np.round(np.arange(min_lat,max_lat,block_latitude)/fact,5))
	if len(lattitudes) == latstep:
		lattitudes.append(max_lat/fact)
	longitudes = list(np.round(np.arange(min_long,max_long,block_longitude)/fact,5))
	if len(longitudes) == longstep:
		longitudes.append(max_long/fact)
	print(len(lattitudes),len(longitudes))
	#print(longitudes)
	Area_Grid =	{}
	Area_pixel_Grid = pd.DataFrame()
	
	Area_Grid['lattitudes'] = lattitudes
	Area_Grid['longitudes'] = longitudes
	grid_no = 1
	for a in range(len(Area_Grid['lattitudes'])-1):
		pixlatmin = Area_Grid['lattitudes'][a] -(block_latitude*margin)/(fact*pixelsize)
		pixlatmax = Area_Grid['lattitudes'][a+1] + (block_latitude*margin)/(fact*pixelsize)
		for b in range(len(Area_Grid['longitudes'])-1):
			pixlonmin = Area_Grid['longitudes'][b] - (block_longitude*margin)/(fact*pixelsize)
			pixlonmax = Area_Grid['longitudes'][b+1] + (block_longitude*margin)/(fact*pixelsize)
			Area_pixel_Grid = Area_pixel_Grid.append(GetPixelDF(pixlatmin,pixlatmax,pixlonmin,pixlonmax,pixelsize+2*margin,grid_no))
			grid_no +=1
	Area_pixel_Grid = ps.sqldf("""select a.*, sum(ifnull(b.pop,0)) pop from Area_pixel_Grid a left outer join counties b
		on b.lat between a.minlat and a.maxlat and b.long between a.minlong and a.maxlong 
		group by a.grid,a.pixno,a.minlong,a.maxlong,a.minlat,a.maxlat""", locals())
	return Area_pixel_Grid

## Function to validate the frames based on the time series patient data
## Parameter Needed - 1. frames_grid - Dataframe - Will contain the dataframe population data for each and every frame 2. df_pop_pat - Dataframe - Population and patient data of country
## 					 3. margin - number - Overlapping adjustment for pixels
def validate_frames(frames_grid,df_pop_pat,margin):
	days = np.max(frames_grid['day'])+1
	pixno = np.int(np.max(frames_grid['pixno']))
	pix = np.int(math.sqrt(pixno))
	print(pix)
	print(np.sum(frames_grid['new_pat']),np.sum(df_pop_pat['new_pat']),len(frames_grid),
		 len(set(frames_grid['grid'])),len(frames_grid)/(len(set(frames_grid['grid']))*days) )
	a = np.reshape(range(1,pixno+1),(pix,pix))
	a=np.flip(a,0)
	start = np.int(margin)
	end = np.int(pix-margin)
	a=a[start:end,start:end].flatten()
	print(np.sum(frames_grid[frames_grid['pixno'].isin(a)]['new_pat']))

## Creates the Frame DF from Area DF and Patient Data
## Parameter Needed - 1. df_pop_pat - Dataframe - country specific pixel level patient and population data for country 2. Area_df - DataFrame-	Pixel level coridnate data with population for each pixel
## Creates the Frame DF from Area DF and Patient Data
from numpy import percentile
## Parameter Needed - 1. df_pop_pat - Dataframe - country specific pixel level patient and population data for country 2. Area_df - DataFrame-	Pixel level coridnate data with population for each pixel
def frames_df(df_pop_pat,Area_df):
	days = ps.sqldf("select distinct Date from df_pop_pat order by 1",locals())
	days['day'] = np.arange(len(days))
	Area_df['key'] = 1
	days['key'] = 1
	#Area_day_df =Area_df.merge(days, on='key')
	frames_grid = pd.DataFrame()
	for grid in set(Area_df['grid']):
		Area_df_grid = Area_df[Area_df['grid']==grid]
		Area_day_df_grid = pickle.loads(pickle.dumps(Area_df_grid.merge(days, on='key'),-1))
		df_pop_pat_grid = pickle.loads(pickle.dumps(df_pop_pat[(df_pop_pat['lat'] >= np.min(Area_day_df_grid['minlat'])) & (df_pop_pat['lat'] < np.max(Area_day_df_grid['maxlat']))
									 & (df_pop_pat['long'] >= np.min(Area_day_df_grid['minlong'])) & (df_pop_pat['long'] < np.max(Area_day_df_grid['maxlong']))],-1))
		if len(df_pop_pat_grid) == 0:
			continue
		manager = multiprocessing.Manager()
		return_dict = manager.dict()
		p=multiprocessing.Process(target =get_grid_frames, args =(grid,Area_day_df_grid,df_pop_pat_grid,return_dict,))
		p.start()
		p.join()
		frames = return_dict[1]    
		#maxpop = max(frames['pop'])	
		#frames['pixel'] = np.array((np.log(frames[['new_pat']].values.astype(float)+1)/np.log(frames[['sus_pop']].values.astype(float)+2)))
		frames_grid = frames_grid.append(frames,ignore_index=True)
	qt = percentile(frames_grid[frames_grid['no_pat']>0]['beta'],95)
	frames_grid.loc[frames_grid['beta'] > qt, ['beta']] = qt
	frames_grid['pixel'] = frames_grid['beta']/qt
	frames_grid['day'] = frames_grid['day'] -min(frames_grid['day'])
	frames_grid = frames_grid.sort_values(['grid','pixno','day'])
	frames_grid['norm_pop'] = np.log(frames_grid['pop']+1)/np.log(max(frames_grid['pop']))
	del Area_df_grid
	del Area_day_df_grid
	del df_pop_pat_grid
	del Area_df
	return (frames_grid,qt)


def get_grid_frames(grid,Area_day_df_grid,df_pop_pat_grid,return_dict):
	if len(df_pop_pat_grid) == 0:
		return_dict[1] = pd.DataFrame()
		return  
	frames = ps.sqldf("""select a.grid,a.day,a.pixno,a.Date,
					sum(ifnull(b.Tot_Infected,0)) no_pat,
					sum(ifnull(a.pop,0))*0.7 pop,
					sum(ifnull(b.delta_I,0)) new_pat, 
					sum(ifnull(b.delta_R,0)) + sum(ifnull(b.delta_D,0)) new_removed_pat,
					sum(ifnull(a.pop,0))*0.7 - sum(ifnull(b.Tot_Infected,0)) S, 
					sum(ifnull(b.Tot_Infected,0)) - sum(ifnull(b.Tot_Recovered,0)) - sum(ifnull(b.Tot_Deceased,0)) I,
					max(ifnull(b.lat,0)) lat, min(ifnull(b.long,0)) long         
					from Area_day_df_grid a left outer join df_pop_pat_grid b on a.Date = b.Date and
					b.lat between a.minlat and a.maxlat and b.long between a.minlong and a.maxlong
					group by a.grid,a.day,a.pixno""",locals())
	frames['pop'] = frames.groupby(['grid','pixno'])['pop'].transform('max')
	frames = frames.sort_values(['grid','pixno','day'])  
	frames.loc[frames['I']<0, ['I']]= 0 
	frames['Iperc'] = frames['I']/(frames['pop']+1)
	frames['Sperc'] = frames['S']/(frames['pop']+1)			
	frames['invI'] = 1/(frames['I']+1) 
	frames['SI'] = frames['S']*frames['I']
	frames['beta'] = frames['new_pat'].div(frames.groupby(['grid','pixno'])['SI'].shift(1)+1)*(frames['I']+1)
	frames['gamma'] = frames['new_removed_pat'].div(frames.groupby(['grid','pixno'])['I'].shift(1)+1)
	frames = frames.dropna()   
	return_dict[1] = frames

## Prepares the Training Images for the Neural network injestion after Test Train Validation if the results meets proper Threshold
## Parameter Needed - 1. frames_grid - Output of frames_df function 2. minframe - Minimum no of frames required 3. channel - no of feature variable (population,patients) 4. extframes - Array External parameters like (no of testing,VMT if needed, defult - None)
def prep_image(frames_grid,minframe,testspan=5,src_dir="./", extframes = []):
	country = "India"
	days = np.max(frames_grid['day'])
	pixno = np.int(np.max(frames_grid['pixno']))
	pix = np.int(math.sqrt(pixno))
	train = []
	output = []
	test =[]
	testoutput = []
	test_gridday = dict()
	train_gridday = dict()
	testseq = 0
	trainseq = 0
	for grid in sorted(set(frames_grid['grid'])):
		train_samp = []
		output_samp = []
		grid_frames_grid = pickle.loads(pickle.dumps(frames_grid[(frames_grid['grid']==grid)][extframes+['pixel','pixno','day']],-1))
		for day in range(days,0,-1):
			trainframes = pickle.loads(pickle.dumps(grid_frames_grid[(grid_frames_grid['day']==day-1)].sort_values(['pixno']),-1))
			outputframes = pickle.loads(pickle.dumps(grid_frames_grid[(grid_frames_grid['day']==day)].sort_values(['pixno']),-1))
			trainframe,testframe = prepare_training_samp(trainframes,outputframes,pix,extframes)
			train_samp.append(trainframe)
			output_samp.append(testframe)
		train_samp = np.array(train_samp)
		output_samp = np.array(output_samp)
		if train_samp.shape[0]< minframe:
			continue	
		test.append(np.flip(train_samp[testspan:minframe+testspan,::,::,::],0))
		testoutput.append(np.flip(output_samp[:testspan,::,::,::],0))
		test_gridday[testseq] = (grid,testspan)
		testseq += 1
		######################create training records
		for i in range(testspan,train_samp.shape[0]-minframe,int(minframe/5)):
			if np.sum(train_samp[i:i+minframe,::,::,0]) == 0 and np.sum(output_samp[i:i+minframe,::,::,0]) == 0:
				continue
			train.append(np.flip(train_samp[i:i+minframe,::,::,::],0))
			output.append(np.flip(output_samp[i:i+minframe,::,::,::],0))
			train_gridday[trainseq] = (grid,i)
			trainseq += 1
	test = np.array(test)
	testoutput = np.array(testoutput) 

	train = np.array(train)
	output = np.array(output) 
	with open(src_dir+country+"prepdata.pkl", 'wb') as filehandler:
		pickle.dump((train,output,test,testoutput,test_gridday,train_gridday),filehandler)
		
		
def prepare_training_samp(trainframes,testframes,pix,extframes):
	trainframe = np.array(trainframes['pixel']).reshape(pix,pix)
	trainframe = np.flip(trainframe,0)
	trainframe = trainframe[::,::,np.newaxis]
			################# add any external frames
	for newcol in extframes:
		newframe = np.array(trainframes[newcol]).reshape(pix,pix)
		newframe = np.flip(newframe,0)
		trainframe = np.concatenate((trainframe,newframe[::,::,np.newaxis]),axis = 2)
	#return_dict['trainsamp'] = frame
	
	testframe = np.array(testframes['pixel']).reshape(pix,pix)
	testframe = np.flip(testframe,0)
	testframe = testframe[::,::,np.newaxis]
	for newcol in extframes:
		newframe = np.array(testframes[newcol]).reshape(pix,pix)
		newframe = np.flip(newframe,0)
		testframe = np.concatenate((testframe,newframe[::,::,np.newaxis]),axis = 2)
	#return_dict['outputsamp'] = frame
	del newframe
	return (trainframe,testframe)

# """Compute softmax values for each sets of scores in x."""
def softmax(x):
	if np.max(x) > 1:
		e_x = np.exp(x/np.max(x))
	else:
		e_x = np.exp(x - np.max(x))
	return e_x / e_x.sum()



def country_dataprep(src_dir,tgt_dir,country='India',testspan = 100,channel = 2,minframe=10,margin=4,pixelsize=8):
	pix = pixelsize+2*margin
	gridpix = np.flip(np.array(range(1,pix**2+1)).reshape(pix,pix),0)
	gridpix = gridpix[margin:pix-margin,margin:pix-margin].flatten()

	if country == 'USA':
		df_pop_pat = pd.read_csv(src_dir+"/USA_covid_data_final.csv")
		counties = pd.read_csv(src_dir+"/USA_counties.csv")
		df_pop_pat = df_pop_pat[df_pop_pat['data_date']>20200307]
		_df_pop_pat = df_pop_pat.groupby(['cfips'])['new_pat'].sum().reset_index()
		area=(23, 49,-124.5, -66.31)
		M=18
		N=30
		Area_df = get_The_Area_Grid(area,M,N,margin=margin,pixelsize=pixelsize,counties=counties)
		_df_area_county = ps.sqldf("""select b.cfips, b.county,b.lat,b.long,b.pop,ifnull(c.new_pat,0) no_pat from counties b 
								left outer join _df_pop_pat c on b.cfips = c.cfips""",locals())
		df_pixel_county = ps.sqldf("""select a.grid,a.pixno,d.cfips cfips,d.county county,d.no_pat, d.pop from Area_df a 
									join _df_area_county d
									on d.lat between a.minlat and a.maxlat and d.long between a.minlong and a.maxlong""",locals())
		df_pixel_county = df_pixel_county[df_pixel_county['pixno'].isin(gridpix)]
		df_pixel_county['ratio']=df_pixel_county.groupby(['grid','pixno'])['no_pat'].apply(lambda x: softmax(x))
	elif country == 'India':
		df_pop_pat = pd.read_csv(src_dir+"/India_Covid_Patient.csv")
		counties = pd.read_csv(src_dir+"/India_counties.csv")
		_df_pop_pat = df_pop_pat.groupby(['District','State'])['delta_I'].sum().reset_index()
		area=(6.665, 36.91,68, 97.77)
		M=21
		N=18
		Area_df = get_The_Area_Grid(area,M,N,margin=margin,pixelsize=pixelsize,counties=counties)
		_df_area_county = ps.sqldf("""select b.District, b.State,b.lat,b.long,b.pop,sum(ifnull(c.delta_I,0)) Tot_I from counties b 
								left outer join df_pop_pat c on b.District = c.District and b.State = c.State
								group by b.District, b.State,b.lat,b.long,b.pop""",locals())
		df_pixel_county = ps.sqldf("""select a.grid,a.pixno,d.District District,d.State State,d.Tot_I, d.pop from Area_df a 
									join _df_area_county d
									on d.lat between a.minlat and a.maxlat and d.long between a.minlong and a.maxlong""",locals())
		df_pixel_county = df_pixel_county[df_pixel_county['pixno'].isin(gridpix)]
		df_pixel_county['ratio']=df_pixel_county.groupby(['grid','pixno'])['Tot_I'].apply(lambda x: softmax(x))
	
	frames_grid,qt = frames_df(df_pop_pat,Area_df)
	frames_grid.to_csv(tgt_dir+country+"framesgrid.csv",index=False)
	df_pixel_county.to_csv(tgt_dir+country+"pixel_counties.csv",index=False)
	return qt
	#with open(tgt_dir+country+"framesdata.pkl", 'wb') as filehandler:
	#	pickle.dump((frames_grid,df_pixel_county,qt),filehandler)