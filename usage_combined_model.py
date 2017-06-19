import warnings
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore")

''' Database querying '''


def getDatabaseConnection():
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
    invoice_dataframe = queryInvoiceData(conn)
    closest_stations_dataframe = pd.read_sql_query("select * from public.weather_stations where ord=1;", con=conn)
    weather_data = pd.read_sql_query("select wban, yearmonthday, tavg from consumption.noaa_data", con=conn)
    merged_invoice_df = pd.merge(invoice_dataframe, closest_stations_dataframe, left_on='szip', right_on='zip')
    merged_invoice_df['invoicefromdt'] = pd.to_datetime(merged_invoice_df['invoicefromdt'])
    merged_invoice_df['invoicetodt'] = pd.to_datetime(merged_invoice_df['invoicetodt'])
    cust_database = CustomerDatabase(merged_invoice_df, weather_data)
    return cust_database


''' Utility functions '''


def shaveData(dataframe, col, final_time):
    # Shave off time intervals after specified final time
    found = False
    final_time_idx = len(dataframe.index)
    for idx in dataframe.index:
        if (dataframe[col][idx] > final_time) and not found:
            final_time_idx = idx
            found = True
    return dataframe[0:final_time_idx]


def addTemporalValues(dataframe):
    # split timestamp into year, month, and day values, for regression
    dataframe['year'] = list(map(lambda x: x.year, dataframe['invoicetodt']))
    dataframe['month'] = list(map(lambda x: x.month, dataframe['invoicetodt']))
    dataframe['day'] = list(map(lambda x: x.day, dataframe['invoicetodt']))

    # Add a "days passed" column and iterate over dataframe to complete it'
    dataframe['days_passed'] = list(map(lambda x: (x - dataframe['invoicetodt'][0]).days, dataframe['invoicetodt']))

    return dataframe


def massage(data):
    # massage the data to make pandas happy
    length = len(data)
    data = np.asmatrix(data)
    data = data.reshape(length, 1)
    return data


def copyCustomer(customer):
    c = Customer(customer.id, customer.data.copy())
    c.createEval()
    return c


''' Classes '''


def merge(slice_1, slice_2):
    slice_1.listSufficient.extend(slice_2.listSufficient)
    slice_1.listInsufficient.extend(slice_2.listInsufficient)
    merged_slice = CustomerSlice(slice_1.listSufficient, slice_1.listInsufficient)
    return merged_slice


class CustomerDatabase:
    def __init__(self, invoice_df, weather_df):
        self.invoice_df = invoice_df
        self.weather_df = weather_df

    def getSpecificCustomerDataframe(self, selected_id: str) -> pd.DataFrame:
        # Internal methods
        def calculatePeriod(fromdt, todt):
            return pd.period_range(fromdt, todt, freq='D')

        def cleanVals(x):
            if x is 'M':
                x = None
            else:
                x = pd.to_numeric(x, errors='coerce')
            return x

        customer_usage_df = ((self.invoice_df.loc[self.invoice_df['iethicalid'] == selected_id]).sort_values(
            by='invoicefromdt')).reset_index()
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
                temp_string = str(date.year)
                if date.month < 10:
                    temp_string += "0"
                temp_string += str(date.month)

                if date.day < 10:
                    temp_string += "0"
                temp_string += str(date.day)
                return int(temp_string)

            plotframe['dateQuery'] = list(map(to_dateQuery, plotframe['day']))

            combined_usage_data = pd.merge(plotframe, weather_df_slice, left_on='dateQuery', right_on='yearmonthday')
            interval_temp_sum = np.sum(combined_usage_data['tavg'])

            # Last robustness check - otherwise set the tavg interval sum
            if interval_temp_sum == 0:
                customer_usage_df_final.loc[[interval_idx], 'tavg_intervalSum'] = None
            else:
                customer_usage_df_final.loc[[interval_idx], 'tavg_intervalSum'] = interval_temp_sum

        return customer_usage_df_final

    def getSpecificCustomer(self, selected_id):
        customer_usage_df = self.getSpecificCustomerDataframe(selected_id)

        # drop NA's
        if customer_usage_df.isnull().values.any():
            customer_usage_df = customer_usage_df.dropna()

        length = len(customer_usage_df)
        if length < 5:
            raise ValueError('insufficient data on customer: ID' + str(selected_id))

        customer = Customer(selected_id, customer_usage_df)
        return customer

    def selectSliceByIDs(self, id_list):
        print('Selecting slice by IDs...')
        num_processed_customers = 0
        num_discarded_customers = 0
        num_customers = len(id_list)

        list_sufficient = []
        list_insufficient = []

        for uniqueID in id_list:
            percent = ((num_processed_customers + num_discarded_customers) / num_customers) * 100
            print('Percent completed:  ' + str(percent) + '%')
            try:
                customer = self.getSpecificCustomer(uniqueID)
                customer.formatData()
                print('Customer selected: ' + str(uniqueID))
                list_sufficient.append(customer)
                num_processed_customers += 1
            except ValueError:
                print("Customer discarded, insufficient data: " + str(uniqueID))
                list_insufficient.append(uniqueID)
                num_discarded_customers += 1

        print('selectSliceByIDs terminated successfully')
        print('Processed customers: ' + str(num_processed_customers))
        print('Discarded customers: ' + str(num_discarded_customers))

        cust_slice = CustomerSlice(list_sufficient, list_insufficient)
        return cust_slice

    def selectAllCustomers(self):
        id_list = self.invoice_df['iethicalid'].unique()
        cust_slice = self.selectSliceByIDs(id_list)
        return cust_slice

    def selectRandomSlice(self, cust_count):
        from random import shuffle
        id_list = self.invoice_df['iethicalid'].unique().copy()

        if cust_count > len(id_list):
            raise ValueError('Cannot request more customers than exists in database')

        shuffle(id_list)
        id_list = id_list[0:cust_count]
        cust_slice = self.selectSliceByIDs(id_list)
        return cust_slice

    # TODO: Problem, this leaves the possibility of non-unique customers being merged
    def selectRandomHardSlice(self, cust_count):
        cust_slice = self.selectRandomSlice(cust_count)
        num_processed = len(cust_slice.listSufficient)
        while num_processed < cust_count:
            cust_slice = merge(cust_slice, self.selectRandomSlice(cust_count - num_processed))
            num_processed = len(cust_slice.listSufficient)
        return cust_slice


class CustomerSlice:
    def __init__(self, list_sufficient, list_insufficient):
        self.listSufficient = list_sufficient
        self.listInsufficient = list_insufficient
        self.iter = 0
        columns = ['regression approach', 'RMSQE', 'AE', 'MAE']
        self.grand_errorFrame = pd.DataFrame(columns=columns)

    def get_ID_list(self):
        id_list = []
        for customer in self.listSufficient:
            id_list.append(customer.id)
        return id_list

    def copy(self):
        new_slice = CustomerSlice(list(map(copyCustomer, self.listSufficient)), self.listInsufficient)
        return new_slice

    def reset_grand_errorFrame(self, algorithms):
        for idx in algorithms.index:
            self.grand_errorFrame.loc[idx] = ['predicted_kwh_' + str(algorithms['name'][idx]), 0, 0, 0]
        for idx in self.grand_errorFrame.index:
            self.grand_errorFrame.loc[[idx], 'AE'] = 0
            self.grand_errorFrame.loc[[idx], 'MAE'] = 0

    def runModels(self, train_columns, target_column, algorithms):
        num_customers = len(self.listSufficient)
        count = 0
        self.reset_grand_errorFrame(algorithms)
        # TODO: VECTORIZE THE FOR LOOPS ?
        for customer in self.listSufficient:
            customer.runModels(train_columns, target_column, algorithms)
            for idx in algorithms.index:
                self.grand_errorFrame.loc[[idx], 'AE'] += \
                    int(customer.errorFrame[customer.errorFrame['regression approach']
                                            == 'predicted_kwh_' + str(algorithms['name'][idx])]['AE'])
                self.grand_errorFrame.loc[[idx], 'MAE'] += \
                    int(customer.errorFrame[customer.errorFrame['regression approach']
                                            == 'predicted_kwh_' + str(algorithms['name'][idx])]['MAE'])
            count += 1
            percent = (count / num_customers) * 100
            print('Percent completed:  ' + str(percent) + '%')

        for idx in algorithms.index:
            self.grand_errorFrame.loc[[idx], 'MAE'] = self.grand_errorFrame['MAE'][idx] / len(self.listSufficient)

    def getSummaryStats(self, column):
        num_customers = len(self.listSufficient)
        count = 0

        # average
        sum = 0
        for customer in self.listSufficient:
            sum += customer.data[column].mean()
            count += 1
        avg = sum / count
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
        if self.iter + 1 < len(self.listSufficient):
            self.iter += 1
        else:
            print('Resetting iter, reached end of list')
            self.resetIter()

    def displayAndNext(self, algorithms):
        self.listSufficient[self.iter].displayConclusions(algorithms)
        self.iterNext()

    def classificationTest(self):
        df = pd.DataFrame(columns=['customer', 'id', 'class'])
        df['customer'] = self.listSufficient
        for idx in df.index:
            df['id'][idx] = df['customer'][idx].id
            t_at_high = df['customer'][idx].getValueAtMax('kwh', 'tavg_intervalSum')
            t_at_low = df['customer'][idx].getValueAtMin('kwh', 'tavg_intervalSum')
            if t_at_high > t_at_low:
                df['class'][idx] = "summer peaker"
            elif t_at_high < t_at_low:
                df['class'][idx] = "winter peaker"
            else:
                df['class'][idx] = "pattern unclear"
        return df

    def createAggregateFrame(self):
        count = 1
        aggFrame = self.listSufficient[0].non_eval
        while (count < len(self.listSufficient)):
            aggFrame = aggFrame.append(self.listSufficient[count].non_eval)
            count = count + 1
        self.aggFrame = aggFrame.sort_values(by='invoicetodt').reset_index().drop('index', 1)

    def createTestFrame(self, algorithms):
        count = 1
        testFrame = self.listSufficient[0].eval
        while (count < len(self.listSufficient)):
            testFrame = testFrame.append(self.listSufficient[count].eval)
            count = count + 1
        self.testFrame = testFrame.reset_index()
        for idx in algorithms.index:
            self.testFrame['predicted_kwh_' + str(algorithms['name'][idx])] = None

    def create_aggregate_ErrorFrame(self, algorithms):
        columns = ['regression approach', 'RMSQE', 'AE', 'MAE']
        self.agg_errorFrame = pd.DataFrame(columns=columns)
        for idx in algorithms.index:
            self.agg_errorFrame.loc[idx] = ['predicted_kwh_' + str(algorithms['name'][idx]), 0, 0, 0]

    def createTrainSets(self, train_columns, target_column):
        train_X = self.aggFrame.as_matrix(columns=train_columns)
        train_y = self.aggFrame.as_matrix(columns=[target_column])
        return train_X, train_y

    def createTestSets(self, train_columns, target_column):
        test_X = self.testFrame.as_matrix(columns=train_columns)
        test_y = self.testFrame.as_matrix(columns=[target_column])
        return test_X, test_y

    def run_aggregate_models(self, train_columns, target_column, algorithms):
        self.createAggregateFrame()
        self.createTestFrame(algorithms)
        self.create_aggregate_ErrorFrame(algorithms)

        train_X, train_y = self.createTrainSets(train_columns, target_column)
        target_X, target_y = self.createTestSets(train_columns, target_column)

        algorithms['fitted_model'] = algorithms['algorithm']
        # TODO: Vectorize
        for idx in algorithms.index:
            current_algo =  algorithms['algorithm'][idx]
            algorithms['fitted_model'][idx] = current_algo.fit(train_X, train_y)
            for cust_idx in self.testFrame.index:
                self.testFrame['predicted_kwh_' + str(algorithms['name'][idx])][cust_idx] = int(current_algo.predict(target_X[cust_idx]))
                self.agg_errorFrame.loc[[idx], 'AE'] += int(abs(target_y[cust_idx] - int(current_algo.predict(target_X[cust_idx]))))
            self.agg_errorFrame.loc[[idx], 'MAE'] = self.agg_errorFrame['AE'][idx] / len(self.testFrame)


class Customer:
    def __init__(self, id, data):
        self.id = id
        self.data = data
        columns = ['regression approach', 'RMSQE', 'AE', 'MAE']
        self.errorFrame = pd.DataFrame(columns=columns)
        self.init_trainsize = int(len(self.data) * .8)

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
            self.data['prev_pd_kwh'] = list(map(lambda idx: self.data['kwh'][max(idx - 1, 0)], self.data.index))
            # add two-period time lag
            self.data['prev_prev_pd_kwh'] = list(map(lambda idx: self.data['kwh'][max(idx - 2, 0)], self.data.index))
            # Shave off first two values, because of the time lag
            self.data = self.data[2:]
        except KeyError:
            print("in formatting data, could not add time lags to customer ID: " + str(self.id))
            raise ValueError("Time lag formatting exception")

        # add to data which contains customer ID for each instance
        self.data['iethicalid'] = self.id
        self.createEval()

    def createEval(self):
        # Separate last row from rest of data - this will be used for prediction evaluation
        self.eval = self.data[len(self.data) - 1:]
        self.non_eval = self.data[0: len(self.data) - 1]



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

    def getTrainingSet(self, init_trainsize, date_idx, train_columns, target_column):
        X = self.data.as_matrix(columns=train_columns)
        y = self.data.as_matrix(columns=[target_column])
        train_X = X[: init_trainsize + date_idx]
        train_y = y[0: init_trainsize + date_idx]
        return train_X, train_y

    def getTestSet(self, init_trainsize, date_idx, train_columns):
        X = self.data.as_matrix(columns=train_columns)
        oos_X = X[init_trainsize+date_idx]
        return oos_X

    def setInitialVals(self, algorithms):
        self.data = self.data.reset_index()
        all_dates = self.data['invoicetodt']
        # Manually resetting the init_trainsize because it messes up for an unknown reason...
        self.init_trainsize = int(len(self.data) * .8)
        self.init_oos_dates = all_dates[self.init_trainsize:, ]
        self.init_oos_dates = self.init_oos_dates.reset_index()

        # TODO: Vectorize!??
        for idx in algorithms.index:
            self.data['predicted_kwh_' + str(algorithms['name'][idx])] = None
            self.errorFrame.loc[idx] = ['predicted_kwh_' + str(algorithms['name'][idx]), 0, 0, 0]

    def forecastOn(self, date_idx, train_columns, target_column, algorithms):
        train_X, train_y = self.getTrainingSet(self.init_trainsize, date_idx, train_columns, target_column)
        target_X = self.getTestSet(self.init_trainsize, date_idx, train_columns)
        target_y = self.data['kwh'][self.init_trainsize + date_idx]


        algorithms['fitted_model'] = algorithms['algorithm']
        # TODO: Vectorize
        for idx in algorithms.index:
            algorithms['fitted_model'][idx] = algorithms['algorithm'][idx].fit(train_X, train_y)
            predicted_y = int(algorithms['fitted_model'][idx].predict(target_X))
            self.data['predicted_kwh_' + str(algorithms['name'][idx])][self.init_trainsize + date_idx] = predicted_y
            error = int(abs(target_y - predicted_y))
            self.errorFrame.loc[[idx], 'AE'] += error

    def runModels(self, train_columns, target_column, algorithms):
        self.setInitialVals(algorithms)
        # TODO: Vectorize!
        for date_idx in self.init_oos_dates.index:
            self.forecastOn(date_idx, train_columns, target_column, algorithms)
        for idx in algorithms.index:
            self.errorFrame.loc[[idx], 'MAE'] = self.errorFrame['AE'][idx] / len(self.init_oos_dates)
        self.errorFrame = self.errorFrame.sort_values(by='AE')
        self.errorFrame = self.errorFrame.reset_index(drop=True)

    def displayBestFit(self):
        plt.figure(figsize=(15, 4))
        plt.plot(self.data['invoicetodt'], self.data['kwh'], label='actual usage')
        plt.plot(self.data['invoicetodt'],
                 self.data[(self.errorFrame['regression approach'][0])],
                 color='green',
                 label=(self.errorFrame['regression approach'][0]))
        plt.legend(loc='best')
        plt.title('Customer ' + str(self.id) + ': Best fit')
        plt.show()

    def displayWorstFit(self):
        plt.figure(figsize=(15, 4))
        plt.plot(self.data['invoicetodt'], self.data['kwh'], label='actual usage')
        plt.plot(self.data['invoicetodt'],
                 self.data[(self.errorFrame['regression approach'][len(self.errorFrame) - 1])],
                 color='red', label=(self.errorFrame['regression approach'][len(self.errorFrame) - 1]))
        plt.legend(loc='best')
        plt.title('Customer ' + str(self.id) + ': Worst fit')
        plt.show()

    def displayConclusions(self, algorithms):
        # Plot the algorithm results
        plt.figure(figsize=(15, 4))
        plt.plot(self.data['invoicetodt'], self.data['kwh'], label='actual usage')
        plt.plot(self.data['invoicetodt'], self.data['tavg_intervalSum'], color='green',
                 label='temperature')
        # TODO: Assign a color for each algorithm!!!! AND VECTORIZE!
        for idx in algorithms.index:
            plt.plot(self.data['invoicetodt'],
                     self.data['predicted_kwh_' + str(algorithms['name'][idx])],
                     color='red',
                     label='predicted usage ' + str(algorithms['name'][idx]))
        plt.legend(loc='best')
        plt.title('Customer ' + str(self.id) + ': Temperature, kWH, and fitted values over entire time period')
        plt.show()

        self.displayBestFit()
        self.displayWorstFit()

    def displaySpecificAlgorithm(self, specified_algorithm_string):
        # Plot the algorithm results
        plt.figure(figsize=(15, 4))
        plt.plot(self.data['invoicetodt'], self.data['kwh'], label='actual usage')
        plt.plot(self.data['invoicetodt'], self.data['tavg_intervalSum'], color='green',
                 label='temperature')
        plt.plot(self.data['invoicetodt'], self.data['predicted_kwh_' + specified_algorithm_string], color='black',
                 alpha=0.9, label='predicted usage rf')
        plt.legend(loc='best')
        plt.title('Customer ' + str(self.id) + ': Temperature, kWH, and fitted values over entire time period')
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


class TempComparisonTest:
    def __init__(self, cust_count):
        self.iter = 0
        print('Querying database...')
        self.cust_database = queryDatabase()
        self.cust_slice = self.cust_database.selectRandomHardSlice(cust_count)
        self.cust_slice_minus_temp = self.cust_slice.copy()
        my_algorithms_table = pd.DataFrame(columns=['name', 'algorithm'])
        my_algorithms_table['name'] = ['rf']
        my_algorithms_table['algorithm'] = [RandomForestRegressor(n_estimators=150, min_samples_split=2)]

        my_train_columns = ['tavg_intervalSum', 'prev_pd_kwh', 'prev_prev_pd_kwh', 'month', 'days_passed']
        self.cust_slice.runModels(my_train_columns, 'kwh', my_algorithms_table)

        my_train_columns_minus_temp = ['prev_pd_kwh', 'prev_prev_pd_kwh', 'month', 'days_passed']
        self.cust_slice_minus_temp.runModels(my_train_columns_minus_temp, 'kwh', my_algorithms_table)

    def error_frames(self):
        print('RF MAE with Temp: ' + str(self.cust_slice.grand_errorFrame['MAE'][0]))
        print('RF MAE without temp: ' + str(self.cust_slice_minus_temp.grand_errorFrame['MAE'][0]))

    def avg_kwh(self):
        print('Average kwh used: ' + str(self.cust_slice.getSummaryStats('kwh')))

    def resetIter(self):
        self.iter = 0

    def iterNext(self):
        if self.iter + 1 < len(self.cust_slice.listSufficient):
            self.iter += 1
        else:
            print('Resetting iter, reached end of list')
            self.resetIter()

    def compareAndNext(self):
        self.cust_slice.listSufficient[self.iter].displaySpecificAlgorithm('rf')
        self.cust_slice_minus_temp.listSufficient[self.iter].displaySpecificAlgorithm('rf')
        self.iterNext()
