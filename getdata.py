import pandas as pd
import urllib.request
import numpy as np
import shapefile
from datetime import datetime
from zipfile import ZipFile
import pandasql as ps
import requests
import json
import pkg_resources
import io

def softmax(x):
	if np.max(x) > 1:
		e_x = np.exp(x/np.max(x))
	else:
		e_x = np.exp(x - np.max(x))
	return e_x / e_x.sum()

def norm(x):
	s = np.sum(x)
	if s == 0:
		s =1 
	x = x/s
	return x

def add_Recovery_Data_Usa(tgtdir,covid_Country_Patient_Data_USA):
	Usa_County_Wise_Covid_Data = covid_Country_Patient_Data_USA
	url='https://covidtracking.com/api/v1/states/daily.csv'
	urllib.request.urlretrieve(url,tgtdir+'USA_State_Wise_Daily_Tests.csv')
	latest_test_data = pd.read_csv(tgtdir+'USA_State_Wise_Daily_Tests.csv')
	latest_test_data = latest_test_data[['date','state','recovered']]
	latest_test_data = latest_test_data.fillna(0)
	usa_county_state = pd.read_excel('/content/drive/My Drive/covid19/data/'+'usa_county_state_fips.xlsx')
	USA_County_State_Recov = ps.sqldf(''' select a.*,b.cfips,b.county from latest_test_data a left join usa_county_state b 
	on a.state = b.state''',locals())
	Usa_County_Wise_Covid_Data = ps.sqldf(''' select a.*, ifnull(b.state,"Data Miss") as state, ifnull(b.recovered,0) recovered
	 from Usa_County_Wise_Covid_Data a left join USA_County_State_Recov b on a.cfips = b.cfips and a.data_date = b.date''',locals())
	Usa_County_Wise_Covid_Data['patratio'] = Usa_County_Wise_Covid_Data.groupby(['state','data_date'])['no_pat'].apply(lambda x: norm(x))
	Usa_County_Wise_Covid_Data['r_pat'] = Usa_County_Wise_Covid_Data['patratio']*Usa_County_Wise_Covid_Data['recovered']  
	Usa_County_Wise_Covid_Data['r_pat'] = Usa_County_Wise_Covid_Data['r_pat'].apply(lambda x : round(x))
	Usa_County_Wise_Covid_Data = Usa_County_Wise_Covid_Data.sort_values(['cfips','data_date']) 
	Usa_County_Wise_Covid_Data['r_pat'] = Usa_County_Wise_Covid_Data.groupby(['cfips'])['r_pat'].apply(lambda x: x.cummax())
	#Usa_County_Wise_Covid_Data['r_pat'] = Usa_County_Wise_Covid_Data.groupby(['cfips'])['r_pat'].apply(lambda x: x.cumsum())
	return Usa_County_Wise_Covid_Data
# The below function used to get the USA Patient Data Automatically from HARVARD DATABASE COVID Patient Database and will create a timeseries patient file along with population of the Area at county along with a USA County file
## Parameter Needed - Target Directory to save the File
def transform_us_rawdata(latest_data,value_name):
	allcols = list(latest_data.columns)
	datecols = allcols[allcols.index('HHD10')+1:]
	latest_data = latest_data[['COUNTY', 'NAME']+datecols]
	datecolsmod=[datetime.strptime(i,'%m/%d/%Y').strftime('%Y%m%d') for i in datecols]
	latest_data.columns = ['cfips', 'county']+datecolsmod
	latest_data = latest_data.melt(id_vars=['cfips', 'county'], var_name='data_date', value_name=value_name)
	latest_data['county']=latest_data['county'].apply(lambda x : x.split(' County')[0])	 
	latest_data = latest_data.sort_values(['cfips','data_date'])
	latest_data[value_name] = latest_data.groupby(['cfips'])[value_name].apply(lambda x: x.cummax())
	return latest_data

def fetch_us_patientdata(tgtdir):
	url='https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/HIDLTK/WPDN1Q'
	urllib.request.urlretrieve(url,tgtdir+'/us_county_confirmed_cases.tab')
	confirmedcases_data = pd.read_csv(tgtdir+'/us_county_confirmed_cases.tab',sep='\t')
	confirmedcases_data = transform_us_rawdata(confirmedcases_data,'no_pat')
	url='https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/HIDLTK/KG4VAV'
	urllib.request.urlretrieve(url,tgtdir+'/us_county_death_cases.tab')
	deceased_data = pd.read_csv(tgtdir+'/us_county_death_cases.tab',sep='\t')
	deceased_data =  transform_us_rawdata(deceased_data,'d_pat')
	url='https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/HIDLTK/USGG3W'
	urllib.request.urlretrieve(url,tgtdir+'/us_county_recovered_cases.tab')
	recovered_data = pd.read_csv(tgtdir+'/us_county_recovered_cases.tab',sep='\t')
	recovered_data =  transform_us_rawdata(recovered_data,'r_pat')
	latest_data = pd.merge(confirmedcases_data,deceased_data, on=['cfips','county','data_date'])
	latest_data = ps.sqldf("""select a.*,ifnull(b.r_pat,0) r_pat from latest_data a left outer join 
  recovered_data b on a.cfips = b.cfips and a.data_date = b.data_date""",locals()) 	
	#latest_data = pd.merge(latest_data,recovered_data, on=['cfips','county','data_date']) 

	url='https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/HIDLTK/OFVFPY'
	urllib.request.urlretrieve(url,tgtdir+'/COUNTY_MAP.zip')
	zip = ZipFile(tgtdir+'/COUNTY_MAP.zip')
	zip.extractall(tgtdir)
	sf = shapefile.Reader(tgtdir+"/CO_CARTO")
	shape_df = pd.DataFrame()
	shapes = sf.shapes()
	records = sf.records()
	for eachrec in range(len(records)):
		eachRec = {}
		shapebbbox = shapes[eachrec].bbox
		shapelat = (shapebbbox[1] + shapebbbox[3]) / 2
		shapelong = (shapebbbox[0] + shapebbbox[2]) / 2
		eachRec['lat'] = [shapelat]
		eachRec['long'] = [shapelong]
		eachRec['county_fips'] = [records[eachrec][0]]
		eachRec['county_name'] = [records[eachrec][1]]
		eachRec['POP'] = [records[eachrec][10]]
		eachRec['HHD'] = [records[eachrec][11]]
		shape_df = shape_df.append(pd.DataFrame.from_dict(eachRec))

	us_counties = shape_df
	us_counties['county_name'] = us_counties['county_name'].apply(lambda x: x.split(' County')[0])
	us_counties['county_fips'] = us_counties['county_fips'].apply(lambda x: int(x))
	us_counties.columns = ['lat','long', 'cfips', 'county', 'pop', 'HHD']
	full_data = pd.merge(latest_data, us_counties, on=['cfips', 'county'])
	if sum(full_data['no_pat']) != sum(latest_data['no_pat']):
		print("fetch failed")
		raise
	#full_data['no_pat'] = full_data.groupby(['cfips'])['no_pat'].apply(lambda x: x.cummax())
	full_data = add_Recovery_Data_Usa(tgtdir,full_data)
	full_data['new_pat'] = full_data.groupby(['lat','long'])['no_pat'].diff()
	full_data['new_r_pat'] = full_data.groupby(['lat','long'])['r_pat'].diff()
	full_data['new_d_pat'] = full_data.groupby(['lat','long'])['d_pat'].diff()
	full_data = full_data.dropna()
	us_counties.to_csv(tgtdir+'USA_counties.csv',index=False)
	full_data.to_csv(tgtdir+'USA_covid_data_final.csv',index=False)
	print(' USA Patient Data Created under Directory :'+tgtdir)
	#return full_data 

## Below function will create the China COVID19 time series Patient file by abosrving data from Harvard Database and it will create County file along with Population Data by county/province
## Parameter Needed - Target Directory to save the File

def load_support_data(filename,type = 'xl'):
	# This is a stream-like object. If you want the actual info, call
	# stream.read()
	stream = pkg_resources.resource_stream(__name__, 'data/'+filename)
	if type == 'xl':
		return pd.read_excel(stream)
	elif type == 'csv':
		return pd.read_csv(stream)


## The below function will get the Indian COVID-19 time series patient Data and District level details over India, Along with the population file
## Parameter Needed - Target Directory to save the File
def fetch_india_patientdata(tgtdir):
	India_Raw_Data = requests.get('https://api.covid19india.org/csv/latest/districts.csv')
	India_full_Data = pd.read_csv(io.StringIO(India_Raw_Data.content.decode('utf-8')),delimiter=",")
	India_full_Data = India_full_Data[India_full_Data['State'] != '']
	India_full_Data['Date'] = pd.to_datetime(India_full_Data['Date'],format = '%Y/%m/%d')
	India_full_Data['Date'] = India_full_Data['Date'].dt.strftime("%Y%m%d").astype(int)
	lat_long_df = requests.get('https://raw.githubusercontent.com/swarna-kpaul/indiacovidforecast/main/data/India_District_Wise_Population_Data_with_Lat_Long.csv')
	lat_long_df = pd.read_csv(io.StringIO(lat_long_df.content.decode('utf-8')),delimiter=",")
	#lat_long_df = load_support_data('India_District_Wise_Population_Data_with_Lat_Long.csv','csv')
	lat_long_df['detecteddistrict'] = lat_long_df['detecteddistrict'].apply(lambda x: x.strip())
	lat_long_df['detectedstate'] = lat_long_df['detectedstate'].apply(lambda x: x.strip())
	
	India_Full_Merge_Data = ps.sqldf(
			''' select a.Date,a.State,a.District,a.Confirmed,a.Recovered,a.Deceased,ifnull(b.population,0) as pop,ifnull(b.long,0) as long,ifnull(b.lat,0) as lat from India_full_Data a left join lat_long_df b on lower(a.district) = lower(b.detecteddistrict) and lower(a.state) = lower(b.detectedstate)
			order by a.State,a.District,a.Date ''',
			locals())
	for i in range(len(India_Full_Merge_Data)-1):
		if India_Full_Merge_Data.iloc[i]['Confirmed']<India_Full_Merge_Data.iloc[i]['Recovered']+India_Full_Merge_Data.iloc[i]['Deceased']:
			India_Full_Merge_Data.loc[i,'Confirmed']  = India_Full_Merge_Data.iloc[i]['Recovered']+India_Full_Merge_Data.iloc[i]['Deceased']   
		if India_Full_Merge_Data.iloc[i+1]['State'] == India_Full_Merge_Data.iloc[i]['State'] and India_Full_Merge_Data.iloc[i+1]['District'] == India_Full_Merge_Data.iloc[i]['District']:
			if India_Full_Merge_Data.iloc[i+1]['Confirmed'] < India_Full_Merge_Data.iloc[i]['Confirmed']:
				India_Full_Merge_Data.loc[i+1,'Confirmed'] = India_Full_Merge_Data.iloc[i]['Confirmed']
			if India_Full_Merge_Data.iloc[i+1]['Recovered'] < India_Full_Merge_Data.iloc[i]['Recovered']:
				India_Full_Merge_Data.loc[i+1,'Recovered'] = India_Full_Merge_Data.iloc[i]['Recovered']      
			if India_Full_Merge_Data.iloc[i+1]['Deceased'] < India_Full_Merge_Data.iloc[i]['Deceased']:
				India_Full_Merge_Data.loc[i+1,'Deceased'] = India_Full_Merge_Data.iloc[i]['Deceased']  

	India_Full_Merge_Data['delta_I'] = India_Full_Merge_Data.groupby(['State','District'])['Confirmed'].diff()
	India_Full_Merge_Data['delta_R'] = India_Full_Merge_Data.groupby(['State','District'])['Recovered'].diff()	
	India_Full_Merge_Data['delta_D'] = India_Full_Merge_Data.groupby(['State','District'])['Deceased'].diff()	
	India_Final_Merge_Data = India_Full_Merge_Data[India_Full_Merge_Data['pop']>0]
	India_Final_Merge_Data =India_Final_Merge_Data.dropna()	
	India_Final_Merge_Data.columns = ['Date', 'State', 'District', 'Tot_Infected', 'Tot_Recovered','Tot_Deceased','pop','long','lat','delta_I','delta_R','delta_D']
	India_district = India_Final_Merge_Data[['District', 'State', 'lat', 'long', 'pop']].drop_duplicates()
	India_Final_Merge_Data.to_csv(tgtdir + '/India_Covid_Patient.csv', index=False)
	India_district.to_csv(tgtdir + '/India_counties.csv', index=False)
	print(' India Patient Data Created under Directory :' + tgtdir)
	return  India_Final_Merge_Data