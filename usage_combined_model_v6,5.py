import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import sklearn
from sklearn.metrics import mean_squared_error
from math import sqrt
from datetime import timedelta
from dateutil.parser import parse
from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

import warnings
warnings.filterwarnings("ignore")

''' Database querying '''

def getDatabaseConnection():
    print('Reading database connection...')

    import os
    import yaml
    from sqlalchemy import create_engine

    db_conn = os.path.join(os.getcwd(), "db_connections/db_connections.yml")

    with open(db_conn, 'r') as ymlfile:
        db_conns = yaml.load(ymlfile)

    redshift = db_conns['Redshift']
    # megavolt = db_conns['Megavolt']

    engine = 'postgresql://{username}:{password}@{host}:{port}/{database}'.format(username=redshift['username'],
                                                                                  password=redshift['password'],
                                                                                  host=redshift['host'],
                                                                                  port=redshift['port'],
                                                                                  database=redshift['database'])
    conn = create_engine(engine, connect_args={'sslmode': 'require'})
    return conn


def queryDatabase():

    def queryInvoiceData(conn):
        p1 = "with cte as (select a.iethicalid, a.cesuniqueid, a.szip, it.* from ee.accounts a join ee.v_invoicestemp it on it.uniqueaccountid = a.cesuniqueid)"
        p2 = "select distinct iethicalid, szip, invoicefromdt, invoicetodt, kwh from cte order by iethicalid"
        q = p1 + p2
        df = pd.read_sql_query(q, con=conn)
        return df

    conn = getDatabaseConnection()
    print('Querying invoice data...')
    invoiceDataframe = queryInvoiceData(conn)
    print('Querying closest station data...')
    closestStationsDataframe = pd.read_sql_query("select * from public.weather_stations where ord=1;", con=conn)
    print('Querying weather data...')
    weather_data = pd.read_sql_query("select wban, yearmonthday, tavg from consumption.noaa_data", con=conn)

    merged_invoice_df = pd.merge(invoiceDataframe, closestStationsDataframe, left_on='szip', right_on='zip')
    merged_invoice_df['invoicefromdt'] = pd.to_datetime(merged_invoice_df['invoicefromdt'])
    merged_invoice_df['invoicetodt'] = pd.to_datetime(merged_invoice_df['invoicetodt'])

    cust_database = CustomerDatabase(merged_invoice_df, weather_data)
    return cust_database


''' Utility functions '''

def shaveData(dataframe, col, finalTime):
    # Shave off time intervals after specified final time
    found = False
    finalTimeIdx = len(dataframe.index)

    for idx in dataframe.index:
        if (dataframe[col][idx] > finalTime) and (found == False):
            finalTimeIdx = idx
            found = True

    return dataframe[0:finalTimeIdx]


def addTemporalValues(dataframe):
    # split timestamp into year, month, and day values, for regression
    dataframe['year'] = ""
    dataframe['month'] = ""
    dataframe['day'] = ""
    for date_idx in dataframe.index:
        dataframe.loc[[date_idx], 'year'] = dataframe['invoicetodt'][date_idx].year
        dataframe.loc[[date_idx], 'month'] = str(dataframe['invoicetodt'][date_idx].month)
        dataframe.loc[[date_idx], 'day'] = str(dataframe['invoicetodt'][date_idx].day)

        # Add a "days passed" column and iterate over dataframe to complete it
        dataframe['days_passed'] = ""
    for date_idx in dataframe.index:
        dataframe.loc[[date_idx], 'days_passed'] = \
            (dataframe['invoicetodt'][date_idx] - dataframe['invoicetodt'][0]).days

    return dataframe



def massage(data):
    # massage the data to make pandas happy
    data = np.asmatrix(data)
    data = data.reshape(data.size, 1)
    return data


def reset_x(data):
    x1 = data['tavg_intervalSum']
    x2 = data['days_passed']
    x3 = pd.Categorical(data['month'])
    x4 = data['prev_pd_kwh']
    x5 = data['prev_prev_pd_kwh']

    x1 = massage(x1)
    x2 = massage(x2)
    x4 = massage(x4)
    x5 = massage(x5)

    return x1, x2, x3, x4, x5


def reset_y(data):
    y = data['kwh']
    y = massage(y)
    return y


''' Classes '''

class Customer:
    def __init__(self, id, data):
        self.id = id
        self.data = data

    def formatData(self):
        # sort by invoicetodt to make sure there is proper temporal ordering
        self.data = self.data.sort_values(by='invoicetodt')

        # add avg_kwh column
        self.data['avg_kwh'] = self.data['kwh'].sum() / len(self.data)

        # Shave off time intervals after February 25th, 2017
        # We do not currently have temperature data after this point
        col = 'invoicetodt'
        self.data = shaveData(self.data, col, pd.Timestamp(datetime(2017, 2, 25)))
        self.data = self.data.reset_index(drop=True)

        try:
            # add temporal information for models to use
            self.data = addTemporalValues(self.data)
        except KeyError:
            print("in formatting data, could not add temporal information to customer ID: " + str(self.id))
            raise ValueError("Temporal information formatting exception")

        try:
            # add one-period time lag
            self.data['prev_pd_kwh'] = ""
            for date_idx in self.data.index:
                self.data.loc[[date_idx], 'prev_pd_kwh'] = self.data['kwh'][
                    max(date_idx - 1, 0)]

            # add two-period time lag
            self.data['prev_prev_pd_kwh'] = ""
            for date_idx in self.data.index:
                self.data.loc[[date_idx], 'prev_prev_pd_kwh'] = self.data['kwh'][max(date_idx - 2, 0)]

            # Shave off first two values, because of the time lag
            # TODO: Ask Mark -- is this necessary?
            self.data = self.data[2:]
        except KeyError:
            print("in formatting data, could not add time lags to customer ID: " + str(self.id))
            raise  ValueError("Time lag formatting exception")

        # add to data which contains customer ID for each instance
        self.data['iethicalid'] = self.id

    def displayTempVsUsagePlots(self):
        plt.figure(figsize=(10, 4))
        plt.plot(self.data['invoicetodt'], self.data['kwh'], 'ro')
        plt.plot(self.data['invoicetodt'], self.data['tavg_intervalSum'], 'ro', color='green', )
        plt.title('Temperature and kWH values over time')
        plt.show()

        plt.figure(figsize=(10, 4))
        plt.plot(self.data['invoicetodt'], self.data['kwh'])
        plt.plot(self.data['invoicetodt'], self.data['tavg_intervalSum'], color='green', )
        plt.title('Temperature and kWH values over time')
        plt.show()

    def getTrainingSet(self, init_trainsize, date_idx, includeTemp):
        x1, x2, x3, x4, x5 = reset_x(self.data)
        y = reset_y(self.data)
        x1 = x1[0: init_trainsize + date_idx]
        x2 = x2[0: init_trainsize + date_idx]
        x3 = x3[0: init_trainsize + date_idx]
        x4 = x4[0: init_trainsize + date_idx]
        x5 = x5[0: init_trainsize + date_idx]
        if (includeTemp):
            train_X = np.column_stack((x1, x2, x3, x4, x5))
        else:
            train_X = np.column_stack((x2, x3, x4, x5))
        train_y = y[0: init_trainsize + date_idx]
        return train_X, train_y

    def getTestSet(self, init_trainsize, date_idx, includeTemp):
        x1, x2, x3, x4, x5 = reset_x(self.data)
        oos_x1 = x1[init_trainsize + date_idx]
        oos_x2 = x2[init_trainsize + date_idx]
        oos_x3 = x3[init_trainsize + date_idx]
        oos_x4 = x4[init_trainsize + date_idx]
        oos_x5 = x5[init_trainsize + date_idx]
        if (includeTemp):
            oos_X = np.column_stack((oos_x1, oos_x2, oos_x3, oos_x4, oos_x5))
        else:
            oos_X = np.column_stack((oos_x2, oos_x3, oos_x4, oos_x5))
        return oos_X

    def setInitialVals(self):
        self.data = self.data.reset_index()
        self.init_trainsize = int(len(self.data) * .8)
        all_dates = self.data['invoicetodt']
        self.init_oos_dates = all_dates[self.init_trainsize:, ]
        self.init_oos_dates = self.init_oos_dates.reset_index()

        self.data['predicted_kwh_ols'] = None
        self.data['predicted_kwh_tree'] = None
        self.data['predicted_kwh_rf'] = None
        self.data['predicted_kwh_gbr'] = None

        columns = ['regression approach', 'RMSQE', 'AE', 'MAE']
        self.errorFrame = pd.DataFrame(columns=columns)
        self.errorFrame.loc[0] = ['predicted_kwh_ols', 0, 0, 0]
        self.errorFrame.loc[1] = ['predicted_kwh_tree', 0, 0, 0]
        self.errorFrame.loc[2] = ['predicted_kwh_rf', 0, 0, 0]
        self.errorFrame.loc[3] = ['predicted_kwh_gbr', 0, 0, 0]

    def forecastOn(self, date_idx, includeTemp):

        train_X, train_y = self.getTrainingSet(self.init_trainsize, date_idx, includeTemp)
        target_X = self.getTestSet(self.init_trainsize, date_idx, includeTemp)
        target_y = self.data['kwh'][self.init_trainsize + date_idx]

        def fitModel(train_X, train_y, model):
            model.fit(train_X, train_y)
            return model

        models = [linear_model.LinearRegression(fit_intercept=True), DecisionTreeRegressor(),
                  RandomForestRegressor(n_estimators=150, min_samples_split=2),
                  GradientBoostingRegressor(n_estimators=100, max_depth=3, loss='ls')]

        # TODO: Abstract this!

        # OLS
        model_ols = fitModel(train_X, train_y, models[0])
        predicted_y = int(model_ols.predict(target_X))
        self.data['predicted_kwh_ols'][self.init_trainsize + date_idx] = predicted_y
        error = int(abs(target_y - predicted_y))
        self.errorFrame.loc[[0], 'AE'] += error
        self.model_ols = model_ols

        # Decision Tree
        model_tree = fitModel(train_X, train_y, models[1])
        predicted_y = int(model_tree.predict(target_X))
        self.data['predicted_kwh_tree'][self.init_trainsize + date_idx] = predicted_y
        error = int(abs(target_y - predicted_y))
        self.errorFrame.loc[[1], 'AE'] += error
        self.model_tree = model_tree

        # Random Forest
        model_rf = fitModel(train_X, train_y, models[2])
        predicted_y = int(model_rf.predict(target_X))
        self.data['predicted_kwh_rf'][self.init_trainsize + date_idx] = predicted_y
        error = int(abs(target_y - predicted_y))
        self.errorFrame.loc[[2], 'AE'] += error
        self.model_rf = model_rf

        # Stochastic Gradient Boosting
        model_gbr = fitModel(train_X, train_y, models[3])
        predicted_y = int(model_gbr.predict(target_X))
        self.data['predicted_kwh_gbr'][self.init_trainsize + date_idx] = predicted_y
        error = int(abs(target_y - predicted_y))
        self.errorFrame.loc[[3], 'AE'] += error
        self.model_gbr = model_gbr

    def runModels(self, includeTemp):
        self.setInitialVals()
        for date_idx in self.init_oos_dates.index:
            self.forecastOn(date_idx, includeTemp)
        self.errorFrame.loc[[0], 'MAE'] = self.errorFrame['AE'][0]/len(self.init_oos_dates)
        self.errorFrame.loc[[1], 'MAE'] = self.errorFrame['AE'][1] / len(self.init_oos_dates)
        self.errorFrame.loc[[2], 'MAE'] = self.errorFrame['AE'][2] / len(self.init_oos_dates)
        self.errorFrame.loc[[3], 'MAE'] = self.errorFrame['AE'][3] / len(self.init_oos_dates)
        self.errorFrame = self.errorFrame.sort_values(by='AE')
        self.errorFrame = self.errorFrame.reset_index(drop=True)

    def displayConclusions(self):
        # Plot the algorithm results
        plt.figure(figsize=(15, 4))
        plt.plot(self.data['invoicetodt'], self.data['kwh'], label='actual usage')
        plt.plot(self.data['invoicetodt'], self.data['tavg_intervalSum'], color='green',
                 label='temperature')
        plt.plot(self.data['invoicetodt'], self.data['predicted_kwh_ols'], color='red',
                 label='predicted usage OLS')
        plt.plot(self.data['invoicetodt'], self.data['predicted_kwh_tree'], color='brown',
                 alpha=0.5, label='predicted usage tree')
        plt.plot(self.data['invoicetodt'], self.data['predicted_kwh_rf'], color='orange',
                 alpha=0.9, label='predicted usage rf')
        plt.plot(self.data['invoicetodt'], self.data['predicted_kwh_gbr'], color='brown',
                 alpha=0.9, label='predicted usage gbr')
        plt.legend(loc='best')
        plt.title('Customer ' + str(self.id) + ': Temperature, kWH, and estimated values over entire time period')
        plt.show()

        plt.figure(figsize=(15, 4))
        plt.plot(self.data['invoicetodt'], self.data['kwh'], label='actual usage')
        plt.plot(self.data['invoicetodt'],
                 self.data[(self.errorFrame['regression approach'][0])],
                 color='green',
                 label=(self.errorFrame['regression approach'][0]))
        plt.legend(loc='best')
        plt.title('Customer ' + str(self.id) + ': Best fit')
        plt.show()

        plt.figure(figsize=(15, 4))
        plt.plot(self.data['invoicetodt'], self.data['kwh'], label='actual usage')
        plt.plot(self.data['invoicetodt'],
        self.data[(self.errorFrame['regression approach'][len(self.errorFrame) - 1])],
        color='red', label=(self.errorFrame['regression approach'][len(self.errorFrame) - 1]))
        plt.legend(loc='best')
        plt.title('Customer ' + str(self.id) + ': Worst fit')
        plt.show()

    def displayRF(self):
        # Plot the algorithm results
        plt.figure(figsize=(15, 4))
        plt.plot(self.data['invoicetodt'], self.data['kwh'], label='actual usage')
        plt.plot(self.data['invoicetodt'], self.data['tavg_intervalSum'], color='green',
                 label='temperature')
        plt.plot(self.data['invoicetodt'], self.data['predicted_kwh_rf'], color='black',
                 alpha=0.9, label='predicted usage rf')
        plt.legend(loc='best')
        plt.title('Customer ' + str(self.id) + ': Temperature, kWH, and estimated values over entire time period')
        plt.show()

    def getMaximum(self, column):
        return self.data[column].max()

    def getMinimum(self, column):
        return self.data[column].min()

    def getValueAtMax(self, column_max, column_target):
        d = self.data.sort_values(by=column_max, ascending=False)
        return d.reset_index()[column_target][0]

    def getValueAtMin(self, column_min, column_target):
        d = self.data.sort_values(by=column_min, ascending=True)
        return d.reset_index()[column_target][0]


class CustomerDatabase:
    def __init__(self, invoice_df, weather_df):
        self.invoice_df = invoice_df
        self.weather_df = weather_df

    def merge(self, cust_slice, other_slice):
        cust_slice.listSufficient.extend(other_slice.listSufficient)
        cust_slice.listInsufficient.extend(other_slice.listInsufficient)
        merged_slice = CustomerSlice(cust_slice.listSufficient, cust_slice.listInsufficient)
        return merged_slice

    def getSpecificCustomerDataframe(self, selected_id):
        # Internal methods
        def calculatePeriod(fromdt, todt):
            return pd.period_range(fromdt, todt, freq='D');

        def cleanVals(x):
            if x is 'M':
                x = None
            else:
                x = pd.to_numeric(x, errors='coerce')
            return x

        customer_usage_df = ((self.invoice_df.loc[self.invoice_df['iethicalid'] == selected_id]).sort_values(by='invoicefromdt')).reset_index()
        # Shave off unnecessary values (unnecessary, for now!)
        customer_usage_df_final = customer_usage_df[['invoicetodt', 'kwh']].copy()
        # check to make sure weather station does not change
        if (customer_usage_df['wban'].nunique()) == 1:
            weather_station = int(customer_usage_df['wban'][0])
        else:
            raise ValueError('Customer changes weather stations')

        # Convert all values from strings to numerics, and change M's to NaNs
        weather_df_slice = self.weather_df[self.weather_df['wban'] == weather_station]
        weather_df_slice['tavg'] = list(map(cleanVals, weather_df_slice['tavg']))

        for interval_idx in customer_usage_df_final.index:
            days = calculatePeriod(customer_usage_df['invoicefromdt'][interval_idx],
                                   customer_usage_df['invoicetodt'][interval_idx])
            plotframe = pd.DataFrame(days, columns=['day'])

            def to_dateQuery(date):
                tempString = str(date.year)
                if date.month < 10:
                    tempString += "0"
                tempString += str(date.month)

                if date.day < 10:
                    tempString += "0"
                tempString += str(date.day)
                return int(tempString)

            plotframe['dateQuery'] = list(map(to_dateQuery, plotframe['day']))

            combined_usage_data = pd.merge(plotframe, weather_df_slice, left_on='dateQuery', right_on='yearmonthday')
            interval_temp_sum = np.sum(combined_usage_data['tavg'])

            # Last robustness check - otherwise set the tavg interval sum
            if (interval_temp_sum == 0):
                customer_usage_df_final.loc[[interval_idx], 'tavg_intervalSum'] = None
            else:
                customer_usage_df_final.loc[[interval_idx], 'tavg_intervalSum'] = interval_temp_sum

        return customer_usage_df_final

    def getSpecificCustomer(self, selected_id):
        customer_usage_df = self.getSpecificCustomerDataframe(selected_id)

        # drop NA's
        if (customer_usage_df.isnull().values.any()):
            customer_usage_df = customer_usage_df.dropna()

        length = len(customer_usage_df)
        if (length < 9):
            raise ValueError('insufficient data on customer: ID' + str(selected_id))

        customer = Customer(selected_id, customer_usage_df)
        return customer

    def selectSliceByIDs(self, ID_list):
        print('Selecting slice by IDs...')
        numProcessedCustomers = 0
        numDiscardedCustomers = 0
        numCustomers = len(ID_list)

        listSufficient = []
        listInsufficient = []

        for uniqueID in ID_list:
            percent = ((numProcessedCustomers + numDiscardedCustomers) / numCustomers) * 100
            print('Percent completed:  ' + str(percent) + '%')
            try:
                customer = self.getSpecificCustomer(uniqueID)
                customer.formatData()
                print('Customer selected: ' + str(uniqueID))
                listSufficient.append(customer)
                numProcessedCustomers += 1
            except ValueError:
                print("Customer discarded, insufficient data: " + str(uniqueID))
                listInsufficient.append(uniqueID)
                numDiscardedCustomers += 1

        print('selectSliceByIDs terminated successfully')
        print('Processed customers: ' + str(numProcessedCustomers))
        print('Discarded customers: ' + str(numDiscardedCustomers))

        cust_slice = CustomerSlice(listSufficient, listInsufficient)
        return cust_slice

    def selectAllCustomers(self):
        ID_list = self.invoice_df['iethicalid'].unique()
        cust_slice = self.selectSliceByIDs(ID_list)

        return cust_slice

    def selectRandomSlice(self, custCount):
        from random import shuffle
        ID_list = self.invoice_df['iethicalid'].unique().copy()

        if (custCount > len(ID_list)):
            raise ValueError('Cannot request more customers than exists in database')

        shuffle(ID_list)
        ID_list = ID_list[0:custCount]
        cust_slice = self.selectSliceByIDs(ID_list)
        return cust_slice

    ## TODO: Problem, this leaves the possibility of non-unique customers being merged
    def selectRandomHardSlice(self, custCount):
        cust_slice = self.selectRandomSlice(custCount)
        numProcessed = len(cust_slice.listSufficient)
        while (numProcessed < custCount):
            cust_slice = self.merge(cust_slice, self.selectRandomSlice(custCount-numProcessed))
            numProcessed = len(cust_slice.listSufficient)
        return cust_slice


class CustomerSlice:
    def __init__(self, listSufficient, listInsufficient):
        self.listSufficient = listSufficient
        self.listInsufficient = listInsufficient
        self.iter = 0
        columns = ['regression approach', 'RMSQE', 'AE', 'MAE']
        self.grand_errorFrame = pd.DataFrame(columns=columns)
        self.grand_errorFrame.loc[0] = ['predicted_kwh_ols', 0, 0, 0]
        self.grand_errorFrame.loc[1] = ['predicted_kwh_tree', 0, 0, 0]
        self.grand_errorFrame.loc[2] = ['predicted_kwh_rf', 0, 0, 0]
        self.grand_errorFrame.loc[3] = ['predicted_kwh_gbr', 0, 0, 0]

    def get_ID_list(self):
        ID_list = []
        for customer in self.listSufficient:
           ID_list.append(customer.id)
        return ID_list

    def merge(self, other_slice):
        self.listSufficient.extend(other_slice.listInsufficient)
        self.listInsufficient.extend(other_slice.listInsufficient)

    def reset_grand_errorFrame(self):
        for idx in self.grand_errorFrame.index:
            self.grand_errorFrame.loc[[idx], 'AE'] = 0
            self.grand_errorFrame.loc[[idx], 'MAE'] = 0

    def runModels(self, includeTemp):
        numCustomers = len(self.listSufficient)
        count = 0
        self.reset_grand_errorFrame()
        for customer in self.listSufficient:
            print('Running model on customer ID: ' + str(customer.id))
            customer.runModels(includeTemp)

            self.grand_errorFrame.loc[[0], 'AE'] += int(customer.errorFrame[customer.errorFrame['regression approach'] == 'predicted_kwh_ols']['AE'])
            self.grand_errorFrame.loc[[1], 'AE'] += int(customer.errorFrame[customer.errorFrame['regression approach'] == 'predicted_kwh_tree']['AE'])
            self.grand_errorFrame.loc[[2], 'AE'] += int(customer.errorFrame[customer.errorFrame['regression approach'] == 'predicted_kwh_rf']['AE'])
            self.grand_errorFrame.loc[[3], 'AE'] += int(customer.errorFrame[customer.errorFrame['regression approach'] == 'predicted_kwh_gbr']['AE'])

            self.grand_errorFrame.loc[[0], 'MAE'] += int(
                customer.errorFrame[customer.errorFrame['regression approach'] == 'predicted_kwh_ols']['MAE'])
            self.grand_errorFrame.loc[[1], 'MAE'] += int(
                customer.errorFrame[customer.errorFrame['regression approach'] == 'predicted_kwh_tree']['MAE'])
            self.grand_errorFrame.loc[[2], 'MAE'] += int(
                customer.errorFrame[customer.errorFrame['regression approach'] == 'predicted_kwh_rf']['MAE'])
            self.grand_errorFrame.loc[[3], 'MAE'] += int(
                customer.errorFrame[customer.errorFrame['regression approach'] == 'predicted_kwh_gbr']['MAE'])

            count += 1
            percent = (count / numCustomers) * 100
            print('Percent completed:  ' + str(percent) + '%')

        self.grand_errorFrame.loc[[0], 'MAE'] = self.grand_errorFrame['MAE'][0] / len(self.listSufficient)
        self.grand_errorFrame.loc[[1], 'MAE'] = self.grand_errorFrame['MAE'][1] / len(self.listSufficient)
        self.grand_errorFrame.loc[[2], 'MAE'] = self.grand_errorFrame['MAE'][2] / len(self.listSufficient)
        self.grand_errorFrame.loc[[3], 'MAE'] = self.grand_errorFrame['MAE'][3] / len(self.listSufficient)


    # TODO: How best to do this?
    def getSummaryStats(self, column):
        numCustomers = len(self.listSufficient)
        count = 0

        # average
        sum = 0
        for customer in self.listSufficient:
            sum += customer.data[column].mean()
            count += 1
            percent = (count / numCustomers) * 100
            print('Percent completed:  ' + str(percent) + '%')
        avg = sum/count
        return avg

    def displayErrors(self):
        plt.figure(figsize=(8, 4))
        y_pos = np.arange(len(self.grand_errorFrame['regression approach']))
        plt.bar(y_pos, self.grand_errorFrame['AE'])
        plt.xticks(y_pos, self.grand_errorFrame['regression approach'])
        plt.legend(loc='best')
        plt.title('Aggregate errors in slice')
        plt.show()

    def resetIter(self):
        self.iter = 0

    def iterNext(self):
        if (self.iter+1 < len(self.listSufficient)):
            self.iter += 1
        else:
            print('Resetting iter, reached end of list')
            self.resetIter()

    def displayAndNext(self):
        self.listSufficient[self.iter].displayConclusions()
        self.iterNext()

    def classificationTest(self):
        df = pd.DataFrame(columns=['customer', 'id', 'class'])
        df['customer'] = self.listSufficient
        # TODO: Still need to figure out how to do these vectorized -- ask Mark?
        for idx in df.index:
            df['id'][idx] = df['customer'][idx].id
            t_at_high = df['customer'][idx].getValueAtMax('kwh', 'tavg_intervalSum')
            t_at_low = df['customer'][idx].getValueAtMin('kwh', 'tavg_intervalSum')
            if (t_at_high > t_at_low):
                df['class'][idx] = "summer peaker"
            elif (t_at_high < t_at_low):
                df['class'][idx] = "winter peaker"
            else:
                df['class'][idx] = "pattern unclear"

        return df

class TempComparisonTest:

    def __init__(self, custCount):
        self.cust_database = queryDatabase()
        self.cust_slice = self.cust_database.selectRandomHardSlice(custCount)
        self.cust_slice_copy = self.cust_database.selectSliceByIDs(self.cust_slice.get_ID_list())
        self.cust_slice.runModels(True)
        self.cust_slice_copy.runModels(False)

    def error_frames(self):
        print('RF MAE with Temp: ' + str(self.cust_slice.grand_errorFrame['MAE'][2]))
        print('RF MAE without temp: ' + str(self.cust_slice_copy.grand_errorFrame['MAE'][2]))

    def avg_kwh(self):
        print('Average kwh used: ' + str(self.cust_slice.getSummaryStats('kwh')))

    def resetIter(self):
        self.iter = 0

    def iterNext(self):
        if (self.iter+1 < len(self.cust_slice.listSufficient)):
            self.iter += 1
        else:
            print('Resetting iter, reached end of list')
            self.resetIter()

    def compareAndNext(self):
        self.cust_slice.listSufficient[self.iter].displayRF()
        self.cust_slice_copy.listSufficient[self.iter].displayRF()
        self.iterNext()