from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Conv3D, Conv2D
from tensorflow.keras.layers import ConvLSTM2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import losses
import numpy as np
import pandas as pd
import random
import pandasql as ps
import pickle
from scipy.stats import entropy


########## Create ConvLSTM network ##############
 
from tensorflow.keras.layers import LayerNormalization
def create_model(pixel,filters,channel,hiddenlayers = 4):
	seq = Sequential()
	#seq.add(BatchNormalization(trainable=False))
	seq.add(ConvLSTM2D(filters=filters, kernel_size=(3, 3),
				   input_shape=(None, pixel, pixel, channel),
				   padding='same', return_sequences=True))#activation = 'tanh', recurrent_activation = 'tanh')),activation = 'elu'
	#seq.add(BatchNormalization(trainable=False))
	for layer in range(hiddenlayers-1):
		seq.add(ConvLSTM2D(filters=filters, kernel_size=(3, 3),
				   padding='same', return_sequences=True))# activation = 'tanh', recurrent_activation = 'tanh'))
	seq.add(ConvLSTM2D(filters=filters, kernel_size=(3, 3),
				   padding='same', return_sequences=False)) #activation = 'tanh', recurrent_activation = 'tanh'))

	seq.add(Conv2D(filters=1, kernel_size=(3, 3),
			   activation='elu',
			   padding='same', data_format='channels_last'))
	#seq.add(BatchNormalization(trainable=False))
	seq.compile(loss='mean_squared_error', optimizer='adam',metrics=['mae'])
	return seq

import pandas as pd
import statsmodels.formula.api as sm

def get_localdist(trainX,spatialboundary,ST,boundmargin,span,channel):
	trainx_dist = []
	for day in range(span):
		if day <= boundmargin:
			_trainx_dist = trainX[0:ST,spatialboundary[0]:spatialboundary[1],spatialboundary[2]:spatialboundary[3],::]
		elif day >= span - boundmargin-1:
			_trainx_dist = trainX[span-ST:span,spatialboundary[0]:spatialboundary[1],spatialboundary[2]:spatialboundary[3],::]
		else:
			_trainx_dist = trainX[day-boundmargin:day+boundmargin+1,spatialboundary[0]:spatialboundary[1],spatialboundary[2]:spatialboundary[3],::]
		_trainx_dist = _trainx_dist.reshape(ST**3,channel)
		_trainx_dist = np.std(_trainx_dist, axis = 0)
		trainx_dist.append(_trainx_dist)
	trainx_dist = np.array(trainx_dist)
	return (trainx_dist)

def get_localranddist(trainx_dist,span,channel,spatial):
	randomlist = np.array(random.sample(range(-5, 5), span))[::,np.newaxis]
	for j in range(1,channel):
		if j in spatial:
			a = random.randint(-5,5) 
			_randomlist = np.array([a for i in range(10)])[::,np.newaxis]
		else:
			_randomlist = np.array(random.sample(range(-5, 5), span))[::,np.newaxis]
		randomlist = np.concatenate((randomlist,_randomlist),axis = 1)
	randomlist[randomlist == 0 ] =1
	return (trainx_dist/randomlist)

import statsmodels.api as sm

def run_ST_lime_pixel(model,trainX,trainx_dist,samp,span,channel,spatial,ST,r,c,channellist,incubation):
	trainx = []
	trainy = []
	#print(r,c)
	incubation_span = span - incubation
	for i in range(samp):
		rand_trainx_dist = get_localranddist(trainx_dist,span,channel,spatial)
		_trainx = pickle.loads(pickle.dumps(trainX , -1))
		#if (r,c) == (5,6):
		#	print(_trainx[::,r,c,4])
		temp = _trainx[::,r,c,::]+rand_trainx_dist
		rand_trainx_dist[np.where((temp <0) | (temp >1) )] = rand_trainx_dist[np.where((temp <0) | (temp >1) )] * -1
		_trainx[(incubation_span - ST):incubation_span,r,c,channellist] = _trainx[(incubation_span - ST):incubation_span,r,c,channellist]+rand_trainx_dist[(incubation_span - ST):incubation_span,channellist]	
		#print(_trainx[::,r,c,4])    
		for C in spatial:
			_trainx[::,::,::,C] = _trainx[incubation_span-1,::,::,C]
		_trainy = model.predict(_trainx[np.newaxis,::,::,::,::])
		_trainy = _trainy[0,::,::,0]
		trainx.append(_trainx)
		trainy.append(_trainy)
	trainx = np.array(trainx)[::,::,r,c,::]
	#print(trainx[::,::,4].shape)  
	trainy = np.array(trainy)[::,r,c]
	traindata = pd.DataFrame()
	for C in channellist:
		if C in spatial:
			traindata['C'+str(C)] = trainx[::,span-1,C].flatten()
		else:
			for T in range(incubation+1,incubation+ST+1):
				traindata['C'+str(C)+'_T'+str(T)] = trainx[::,span-T,C].flatten()
	traindata['Y'] = trainy.flatten()
	traindata = traindata[traindata.sum(axis=1)>0]
	X=list(traindata.columns)
	X.remove('Y')
	#X.remove('index')
	_traindata = pickle.loads(pickle.dumps(traindata,-1))
	for x in X:
		_traindata[x] = (_traindata[x] - _traindata[x].mean())/_traindata[x].std()
	_traindata['Y'] = (_traindata['Y'] - _traindata['Y'].mean())/_traindata['Y'].std()
	
	try:
		res = sm.OLS(_traindata['Y'],_traindata[X]).fit()
	except:
		print(channellist)
		print(traindata.iloc[0]) #trainx[::,span-4,4].flatten()) #trainx[::,span-1,2].flatten())
		raise
	return(res,traindata)
	
import itertools
def run_regression(model,grid,train,train_gridday,frames_grid,exclude_channel = [0],spatial = [1],start=0,ST=3,margin = 4,samp= 500, incubation = 3,offset=10):
	trainsamp = []
	maxday = max(frames_grid['day'])
	span = train.shape[1]
	channel = train.shape[-1]
	channellist = list(set(range(channel)) - set(exclude_channel))
	pix = np.int(np.sqrt(max(frames_grid['pixno'])))
	_gridpix = np.flip(np.array(range(1,max(frames_grid['pixno'])+1)).reshape(pix,pix),0)
	gridpix = _gridpix[margin:pix-margin,margin:pix-margin].flatten()
	allowedgridpix = frames_grid[(frames_grid['no_pat']>10) & (frames_grid['grid'] == grid)].groupby(['grid','pixno'])['day'].count().reset_index()
	allowedgridpix = allowedgridpix[allowedgridpix.day > 30 ][['grid','pixno']]
	gridpix = np.intersect1d(gridpix,np.array(allowedgridpix['pixno']))
	train_xplain = pd.DataFrame()
	gridtraindata_xplain= pd.DataFrame()
	for k,(_grid,T) in train_gridday.items():
		if _grid == grid:
			trainsamp.append(k)
			
	for T_span in itertools.islice(trainsamp[0:span], None, None, ST):# trainsamp[start:start+ST]:
		trainX = train[T_span,::,::,::,::]
		g,day = train_gridday[T_span]
	
		for pixno in gridpix:
			(r,c) = np.array((np.where(_gridpix==pixno))).reshape(2)
			_boundmargin = np.int((ST-1)/2)
			spatialboundary = (r-_boundmargin,r+_boundmargin+1,c - _boundmargin, c+_boundmargin+1)
			trainx_dist = get_localdist(trainX,spatialboundary,ST,_boundmargin,span,channel)
			print("pixno",pixno,"Tspan",T_span)   
			res,traindata_explain = run_ST_lime_pixel(model,trainX,trainx_dist,samp,span,channel,spatial,ST,r,c, channellist,incubation)
			traindata_explain['grid'] = grid; traindata_explain['pixno'] = pixno; traindata_explain['day'] = maxday - day;
			gridtraindata_xplain = gridtraindata_xplain.append(traindata_explain, ignore_index = True)
			#print(res.summary())   
			fnames = list(res.params.index.values); coef = list(res.params); pvalue = list(res.pvalues)
			fnames.append('beta');coef.append(np.mean(trainX[span-ST:,r,c,0])); pvalue.append(0)
			for C in channellist:
				fnames.append('act_C_'+str(C))
				coef.append(np.mean(trainX[span-ST:,r,c,C]))
				pvalue.append(0)
			temp_df = pd.DataFrame({'fnames':fnames,'coef':coef,'pvalue':pvalue})
			temp_df['grid'] = grid; temp_df['pixno'] = pixno; temp_df['day'] = maxday - day; 
			train_xplain = train_xplain.append(temp_df, ignore_index = True)
	return(train_xplain,gridtraindata_xplain)

# """Compute softmax values for each sets of scores in x."""
def softmax(x):
	if np.max(x) > 1:
		e_x = np.exp(x/np.max(x))
	else:
		e_x = np.exp(x - np.max(x))
	return e_x / e_x.sum()


	
########## Convert image pixel values to number of infection cases ########
def convert_image_to_data(image,margin,sus_pop):  
	frame = image
	frame[frame<0.001] = 0
	pix = frame.shape[0]
	frame = frame[margin:pix-margin,margin:pix-margin]
	_sus_pop = np.log(sus_pop +2)
	frame = np.multiply(frame,_sus_pop)
	popexists_size = len(sus_pop[sus_pop>0])
	frame = np.exp(frame) -1
	frame = np.round(frame,0)
	return (frame,popexists_size)
	
def forecast(model,input_sequence,frames_grid,test_gridday,span,qt,in_grid=-1,epsilon_T = -1,margin=4,spatial_channel=[],calculate_channel={},pixno=-1):	
	pix = np.int(np.sqrt(max(frames_grid['pixno'])))
	_gridpix = np.flip(np.array(range(1,max(frames_grid['pixno'])+1)).reshape(pix,pix),0)
	gridpix = _gridpix[margin:pix-margin,margin:pix-margin]
	forecastframe = pd.DataFrame()
	channels = input_sequence.shape[-1]
	_span = 10
	forecast_frames_grid = frames_grid[frames_grid['day'] <= max(frames_grid['day'])-_span]
	print(max(frames_grid['day'])-_span)
	for k,(grid,_filler) in test_gridday.items():
		if in_grid >-1 and in_grid != grid:
			continue
		track = input_sequence[k]
		totpop = track[0,::,::,1] 
		pix = totpop.shape[0]
		print(grid)    
		I_0 = np.log(np.array(forecast_frames_grid[(forecast_frames_grid.grid == grid) & (forecast_frames_grid.day == max(forecast_frames_grid.day))].sort_values(['pixno'])['I'])+1)
		I_0 = np.flip(I_0.reshape(pix,pix),0)

		popexists = pickle.loads(pickle.dumps(totpop[::,::],-1))
		popexists[popexists>0] = 1
		######## for each prediction day
		for i in range(span):
			new_pos = model.predict(track[np.newaxis, ::, ::, ::, ::])
			new = new_pos[::, ::, ::, ::]
			new = np.multiply(new[0,::,::,0],popexists)[np.newaxis,::,::,np.newaxis]
			I_0 = np.multiply(I_0,popexists)
			new[new<0] = 0
			new[new>1] = 1
			if i > 0:
				#print(max( forecast_frames_grid[(forecast_frames_grid.grid == grid)]['day']))
				sum_beta_gamma = forecast_frames_grid[(forecast_frames_grid.grid == grid) & (forecast_frames_grid.day >41 )][['pixno','beta','gamma']].groupby(['pixno']).sum()
				sum_beta = np.flip(np.array(sum_beta_gamma.beta).reshape(pix,pix),0)
				sum_gamma =np.flip(np.array(sum_beta_gamma.gamma).reshape(pix,pix),0);         
			if epsilon_T > 1 and i > 0:
				#pass				
				Iperc = pickle.loads(pickle.dumps(track[-1,::,::,4],-1)); Iperc[Iperc==0]=1        
				gamma1 = I_0*(i+1)/epsilon_T+  new[0,::,::,0]*qt/Iperc + sum_beta -sum_gamma; gamma1[gamma1>0.2] = 0.2
			else:
				gamma = forecast_gamma(forecast_frames_grid,grid,5)
			if pixno > -1 and i > 0:
				gamma = forecast_gamma(forecast_frames_grid,grid,5)
				pixcor = np.where(_gridpix == pixno)		
				gamma[pixcor] = gamma[pixcor]	
			elif i > 0 and epsilon_T>1:
				gamma = gamma															
			_forecast_frames_grid = calculate_future_SIR(forecast_frames_grid,grid,forecastbeta = new[0,::,::,0],forecastgamma = gamma,qt = qt)
			forecast_frames_grid = forecast_frames_grid.append(_forecast_frames_grid, ignore_index=True)
			#print(span, max( forecast_frames_grid[(forecast_frames_grid.grid == grid)]['day']))
			########### append channels
			newtrack = new
			for channel in range(1,channels):
				if channel in spatial_channel:
					channel_data = track[0,::,::,channel]
					newtrack = np.concatenate((newtrack,channel_data[np.newaxis,::,::,np.newaxis]),axis = 3)
				elif channel in calculate_channel:
					channel2 = np.flip(np.array(_forecast_frames_grid[calculate_channel[channel]]).reshape(pix, pix), 0)
					newtrack = np.concatenate((newtrack,channel2[np.newaxis,::,::,np.newaxis]),axis = 3)		
			
			track = np.concatenate((track, newtrack), axis=0)
			predictframe = np.squeeze(new,0)[::,::,0][margin:pix-margin,margin:pix-margin]
			_forecastframe = pd.DataFrame({'pixno':gridpix[totpop[margin:pix-margin,margin:pix-margin]>0].flatten(), 
			'predict':predictframe[totpop[margin:pix-margin,margin:pix-margin]>0].flatten()}) 
			_forecastframe['day'] = i
			_forecastframe['grid'] = grid   
			forecastframe = forecastframe.append(_forecastframe)   		
	return (forecastframe,forecast_frames_grid)
	

def calculate_future_SIR(forecast_frames_grid,grid,forecastbeta,forecastgamma,qt):
	_forecast_frames_grid = forecast_frames_grid[forecast_frames_grid['grid'] == grid]
	_forecast_frames_grid = _forecast_frames_grid[_forecast_frames_grid['day'] == max(_forecast_frames_grid['day'])].sort_values(['pixno'])
	beta = np.flip(forecastbeta,0).flatten()
	gamma = np.flip(forecastgamma,0).flatten()
	_forecast_frames_grid.loc[:,'pixel'] = beta
	_forecast_frames_grid.loc[:,'beta'] = beta*qt/_forecast_frames_grid.Iperc
	_forecast_frames_grid.loc[:,'gamma'] = gamma
	_forecast_frames_grid.loc[:,'new_pat'] = np.round(qt*beta *_forecast_frames_grid['SI']/(_forecast_frames_grid['I']+1)) #(_forecast_frames_grid['I']+1))
	_forecast_frames_grid.loc[:,'no_pat'] = _forecast_frames_grid['new_pat'] + _forecast_frames_grid['no_pat']
	_forecast_frames_grid.loc[:,'S'] = _forecast_frames_grid['S'] - _forecast_frames_grid['new_pat']
	_forecast_frames_grid.loc[:,'new_removed_pat'] = np.round(gamma * (_forecast_frames_grid['I']+1))
	_forecast_frames_grid.loc[_forecast_frames_grid['new_removed_pat']>_forecast_frames_grid['I'],['new_removed_pat']] = 0
	_forecast_frames_grid.loc[:,'I'] = _forecast_frames_grid['I'] + _forecast_frames_grid['new_pat'] - _forecast_frames_grid['new_removed_pat']
	_forecast_frames_grid.loc[:,'SI'] = _forecast_frames_grid['I'] * _forecast_frames_grid['S']
	temp_I = np.array(_forecast_frames_grid['I'])
	temp_I[temp_I<1] = 1
	_forecast_frames_grid.loc[:,'Iperc'] = _forecast_frames_grid['I']/(_forecast_frames_grid['pop']+1) #1/temp_I
	_forecast_frames_grid.loc[:,'Sperc'] = _forecast_frames_grid['S']/(_forecast_frames_grid['pop']+1)
	_forecast_frames_grid.loc[:,'day'] = _forecast_frames_grid['day'] +1

	return _forecast_frames_grid

from statsmodels.tsa.arima_model import ARIMA	
def forecast_gamma_model(frames_grid,span):
	gamma_model = {}
	T = max(frames_grid['day']) - span
	pix = max(frames_grid['pixno'])
	for grid in frames_grid['grid'].unique():
		for pixno in range(1,pix+1):
			t_series = np.array(frames_grid[(frames_grid['grid'] == grid) & (frames_grid['pixno'] == pixno) & (frames_grid['no_pat'] >0)]['gamma'])
			if len(t_series) > 10 :   
				gamma_model[(grid,pixno)] = ARIMA(t_series, (2,1,2))
				gamma_model[(grid,pixno)].fit()
	return gamma_model


def forecast_gamma(forecast_frames_grid,grid,span):
	_forecast_frames_grid = forecast_frames_grid[forecast_frames_grid['grid'] == grid]
	_forecast_frames_grid = _forecast_frames_grid[_forecast_frames_grid['day'] >= max(_forecast_frames_grid['day']) - span]
	gamma = np.array(_forecast_frames_grid.groupby(['pixno'])['gamma'].mean())
	pix = np.int(np.sqrt(max(_forecast_frames_grid['pixno'])))
	gamma = np.flip(gamma.reshape(pix,pix),0)
	return gamma

def validate(ensemble,test,testout,test_gridday,frames_grid,margin, qt, spatial_channel = [], forecast_channel=[], calculate_channel = {}):
	errorsum = 0
	averagetotalerror = 0
	cnt = 1
	channels = test.shape[-1]
	predicttotal = pd.DataFrame()
	pix = np.int(np.sqrt(max(frames_grid['pixno'])))
	gridpix = np.flip(np.array(range(1,max(frames_grid['pixno'])+1)).reshape(pix,pix),0)
	gridpix = gridpix[margin:pix-margin,margin:pix-margin]
	errorframe = pd.DataFrame()  
	minpop = min(frames_grid['norm_pop']) 
	span = 10#test_gridday[0][1]
	forecast_frames_grid = frames_grid[frames_grid['day'] <= max(frames_grid['day'])-span]
	for k,(grid,span) in test_gridday.items():
		######## for each test grid
		track = test[k]
		totpop = track[0,::,::,1] 
		pix = totpop.shape[0]
		print(grid)		
		popexists = pickle.loads(pickle.dumps(totpop[::,::],-1))
		popexists[popexists>0] = 1
		popexists_size = len(popexists[popexists>0].flatten())
		out = testout[k]
		######## for each prediction day
		for i in range(span):
			new_pos = ensemble.predict(track[np.newaxis, ::, ::, ::, ::])
			#new_pos = ensemble_predict(ensemble,track[np.newaxis, ::, ::, ::, ::])
			new = new_pos[::, ::, ::, ::]
			new = np.multiply(new[0,::,::,0],popexists)[np.newaxis,::,::,np.newaxis]
			new[new<0] = 0
			new[new>1] = 1
			gamma = forecast_gamma(forecast_frames_grid,grid,10)
			_forecast_frames_grid = calculate_future_SIR(forecast_frames_grid,grid,forecastbeta = new[0,::,::,0],forecastgamma = gamma,qt = qt)
			forecast_frames_grid = forecast_frames_grid.append(_forecast_frames_grid, ignore_index=True)
			########### append channels
			newtrack = new
			for channel in range(1,channels):
				if channel in spatial_channel:
					channel_data = track[i,::,::,channel]
					newtrack = np.concatenate((newtrack,channel_data[np.newaxis,::,::,np.newaxis]),axis = 3)
				elif channel in forecast_channel:
					channel_data = out[i,::,::,channel]
					newtrack = np.concatenate((newtrack,channel_data[np.newaxis,::,::,np.newaxis]),axis = 3)
				elif channel in calculate_channel:
					channel2 = np.flip(np.array(_forecast_frames_grid[calculate_channel[channel]]).reshape(pix, pix), 0)
					newtrack = np.concatenate((newtrack,channel2[np.newaxis,::,::,np.newaxis]),axis = 3)
			#print(channels,spatialorforecast_channel,newtrack.shape,track.shape)		
			track = np.concatenate((track, newtrack), axis=0)
			predictframe = np.squeeze(new,0)[::,::,0][margin:pix-margin,margin:pix-margin]
			actualframe = out[i,::,::,0][margin:pix-margin,margin:pix-margin]
			notzeroframe = pickle.loads(pickle.dumps(actualframe, -1))
			notzeroframe[notzeroframe == 0] =1	 
			_errorframe = pd.DataFrame({'pixno':gridpix[totpop[margin:pix-margin,margin:pix-margin]>0].flatten(), 
                               'predict':predictframe[totpop[margin:pix-margin,margin:pix-margin]>0].flatten(), 
                               'actual':actualframe[totpop[margin:pix-margin,margin:pix-margin]>0].flatten()}) 
			_errorframe['day'] = i
			_errorframe['grid'] = grid   
			errorframe = errorframe.append(_errorframe)   			
			error = np.sum(np.absolute((predictframe - actualframe)/notzeroframe))/(popexists_size+1)
			averagetotalerror += np.sum(np.absolute((predictframe - actualframe)))/(popexists_size+1)
			errorsum +=error
			cnt +=1
	averageerror = errorsum/cnt
	averagetotalerror /=	cnt
	return (averageerror,averagetotalerror,forecast_frames_grid)
	
	############ Test ensemble model foor Italy ####################
############ Test ensemble model foor Italy ####################
############ Test ensemble model foor Italy ####################
############ Test ensemble model foor Italy ####################
def test_model(model,test,testoutput,test_gridday,frames_grid,qt,spatial_channel,forecast_channel,calculate_channel = {},span=5,margin=4):
	test_gridday_span = {}
	pix = np.int(np.sqrt(max(frames_grid['pixno'])))
	gridpix = np.flip(np.array(range(1,max(frames_grid['pixno'])+1)).reshape(pix,pix),0)
	gridpix = gridpix[margin:pix-margin,margin:pix-margin].flatten()
	for i,v in test_gridday.items():
		if True: #v[0] == grid:		
			test_gridday_span[i] = (v[0],span)
	#print(test_gridday_span)	 
	(averageerror,averagetotalerror,forecast_frames_grid) = validate(model,test,testoutput,test_gridday_span,frames_grid,margin=4,qt=qt, spatial_channel = spatial_channel, forecast_channel = forecast_channel, calculate_channel = calculate_channel)
	predict = forecast_frames_grid[(forecast_frames_grid['day']>max(forecast_frames_grid['day'])-span) ][['grid','day','pixno','new_pat','no_pat','S','I','beta','new_removed_pat']]
	predict = predict[predict.pixno.isin(gridpix)]
	actual = frames_grid[(frames_grid['day']>max(frames_grid['day'])-span)][['grid','day','pixno','new_pat','no_pat','S','I','beta','new_removed_pat','pop']]
	actual = actual[actual.pixno.isin(gridpix)]
	errorframe = pd.merge(predict,actual,on=['grid','pixno','day'])
	errorframe = errorframe[errorframe['pop'] > 0 ]
	#errorframe['beta_xx'] = errorframe['beta_x']*errorframe['pop_x']/errorframe['I_x'];errorframe['beta_yy'] = errorframe['beta_y']*errorframe['pop_y']/errorframe['I_y'];	
	KL_div = entropy( softmax(errorframe['beta_x']), softmax(errorframe['beta_y']) )
	total_errorframe = errorframe.groupby(['day']).sum().reset_index()
	grid_total_errorframe = errorframe.groupby(['grid','day']).sum().reset_index()
	MAPE_countrytotal = np.mean(np.absolute((total_errorframe['no_pat_x'] - total_errorframe['no_pat_y'])/(total_errorframe['no_pat_y'])))
	MAPE_grid = np.mean(np.absolute((grid_total_errorframe['no_pat_x'] - grid_total_errorframe['no_pat_y'])/(grid_total_errorframe['no_pat_y'])))
	
	return(KL_div,MAPE_grid,MAPE_countrytotal,averageerror,errorframe)


def train_country_model(src_dir,model_dir,country,epochs = 20,hiddenlayers=4,batch_size = 50, channel = 2 , pixel = 16, filters = 32):
	with open(src_dir+country+'prepdata.pkl', 'rb') as filehandler:
		indata,df_pixel_county = pickle.load(filehandler)
	(train,output,test,testoutput,test_gridday,frames_grid,qt) = indata
	with open(src_dir+country+'testprepdata.pkl', 'wb') as filehandler:
		pickle.dump((test,testoutput,test_gridday,frames_grid,qt), filehandler)
	print(country+" test data have been saved in "+src_dir+country+'testprepdata.pkl')
	if country == 'USA':
		pass
	else:
		out = output[::,-1,::,::,0]
		out = out[::,::,::,np.newaxis]
		model = create_model(pixel=pixel,filters=filters,channel=channel,hiddenlayers=hiddenlayers) 
		hist = model.fit(train, out, batch_size=batch_size,epochs=epochs, validation_split=0.05)
		model.save(model_dir+country+"model.h5py")
		print(country+" model have been generated and saved")
		return
		
		
def test_country_model(src_dir,model_dir,country,span,margin=4):
	with open(src_dir+country+'testprepdata.pkl', 'rb') as filehandler:
		(test,testoutput,test_gridday,frames_grid,qt) = pickle.load(filehandler)
	if country == 'USA':
		pass
	else:
		model = keras.models.load_model(model_dir+country+"model.h5py")
		if span > test_gridday[0][1]:
			print("span should be less than ",test_gridday[0][1]+1)
			raise
		KL_div,MAPE_grid,MAPE_countrytotal,averageerror,errorframe = test_model(model,test,testoutput,test_gridday,frames_grid,qt,spatial_channel = [1],forecast_channel = [], calculate_channel = {},span=span)
		errorframe.to_csv(src_dir+country+"errorframe.csv")
	return (KL_div,MAPE_grid,MAPE_countrytotal,averageerror)

def forecast_country_cases(src_dir,country,span=100,margin=4):
	with open(src_dir+country+'testprepdata.pkl', 'rb') as filehandler:
		(test,testoutput,test_gridday,frames_grid,qt) = pickle.load(filehandler)
	counties = pd.read_csv(src_dir+country+"_counties.csv")
	forecast_frame = pd.DataFrame()
	if country == 'USA':

		for group, (train,output,test,testoutput,test_gridday,frames_grid) in enumerate(indata):
			ensemble = load_ensemble('USA_group_'+str(group),src_dir)
			forecast_frame = forecast_frame.append(forecast(ensemble,test,frames_grid,test_gridday,span,margin))
	else:
		model = keras.models.load_model(model_dir+country+"model.h5py")
		forecastframe,forecast_frames_grid=forecast(model,test,frames_grid,test_gridday,span=span,qt=qt,spatial_channel=[1])
	forecast_frame = ps.sqldf("""select a.*, b.day,b.Date, a.ratio*b.no_pat Total_case,a.ratio*b.pop population,a.ratio*b.new_pat New_case, 
					a.ratio*b.new_removed_pat New_removed from df_pixel_county a join forecast_frames_grid b on a.grid = b.grid and a.pixno = b.pixno """,locals())
	if country == 'USA':
		forecast_frame['total_pat'] = forecast_frame.groupby(['cfips'])['predicted'].apply(lambda x: x.cumsum())
	elif country == 'Italy':
		forecast_frame['total_pat'] = forecast_frame.groupby(['ProvinceName','RegionName'])['predicted'].apply(lambda x: x.cumsum())
		
	#forecast_frame.loc[:,['total_pat']] = forecast_frame['total_pat'] +forecast_frame['no_pat']
	forecast_frame.to_csv(src_dir+country+"forecastcases.csv")
	return forecast_frame
