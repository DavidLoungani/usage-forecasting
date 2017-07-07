/* Initial temp tables, to be combined into invoice data */
CREATE TEMP TABLE temp1 AS (
with cte as (select a.iethicalid, a.cesuniqueid, a.szip, it.* from ee.accounts a join ee.v_invoicestemp it on it.uniqueaccountid = a.cesuniqueid)
select distinct iethicalid, szip, invoicefromdt, invoicetodt, kwh from cte order by iethicalid
);

CREATE TEMP TABLE temp2 AS (
  SELECT *
  FROM public.weather_stations WHERE ord=1
);

/* Weather data */
CREATE TEMP TABLE w AS (
  SELECT wban, yearmonthday, tavg
  FROM consumption.noaa_data
);

/* Invoice data */
CREATE TEMP TABLE m AS (
  SELECT *
  FROM temp1
    INNER JOIN temp2 ON (temp1.szip = temp2.zip)
);

/* Create timestamps for weather data */
ALTER TABLE w
ADD yearmonthday_fix text;
UPDATE w
SET yearmonthday_fix = left(yearmonthday, 4) || '-' || left(right(yearmonthday, 4), 2) || '-' || right(yearmonthday, 2);

ALTER TABLE w
ADD yearmonthday_timestamp timestamp;
UPDATE w
SET yearmonthday_timestamp = yearmonthday_fix::timestamp;

/* Create timestamps and duration for invoice data */
ALTER TABLE m ADD COLUMN duration int;
UPDATE m
SET duration = invoicetodt - invoicefromdt;

ALTER TABLE m ADD COLUMN invoicefromdt_timestamp timestamp;
ALTER TABLE m ADD COLUMN invoicetodt_timestamp timestamp;

UPDATE m
SET invoicefromdt_timestamp = invoicefromdt::timestamp;
UPDATE m
SET invoicetodt_timestamp = invoicetodt::timestamp;

/* Merge the invoice and weather data, call this big_frame */
CREATE TEMP TABLE big_frame AS (
 SELECT m.iethicalid, m.zip, m.invoicefromdt_timestamp, m.invoicetodt_timestamp, m.duration, m.kwh, m.wban, m.dist, w.yearmonthday_timestamp, w.tavg
 FROM m JOIN w on (w.wban = m.wban AND w.yearmonthday_timestamp >= m.invoicefromdt_timestamp AND w.yearmonthday_timestamp <= m.invoicetodt_timestamp)
 );
 
/* Reduce the big_frame so that temperature columns are aggregated */
CREATE TEMP TABLE stats_frame_2 AS (
        SELECT 
                DISTINCT iethicalid, zip, invoicefromdt_timestamp, invoicetodt_timestamp, duration, kwh, wban, dist,
                SUM(CASE tavg WHEN 'M' then '0' else tavg end) AS tavg_interval_sum,
                AVG(CASE tavg WHEN 'M' then '0' else tavg end) AS tavg_interval_avg,
                SUM(CASE tavg WHEN 'M' then 1 else 0 end) AS num_missing
        FROM (select * from big_frame) GROUP BY iethicalid, zip, invoicefromdt_timestamp, invoicetodt_timestamp, duration, kwh, wban, dist
);

/* Add some markers for missing temperature data */
ALTER TABLE stats_frame_2 ADD COLUMN tavg_interval_sum_corrected int;
UPDATE stats_frame_2
SET tavg_interval_sum_corrected = tavg_interval_sum + (tavg_interval_avg * num_missing);

UPDATE stats_frame_2
SET duration = duration + 1;
ALTER TABLE stats_frame_2 ADD COLUMN perc_missing decimal(5, 4);
UPDATE stats_frame_2
SET perc_missing = ((num_missing::decimal)/(duration::decimal));

/* Add some columns with relevant information for analysis */
ALTER TABLE stats_frame_2 ADD COLUMN avg_kwh_day_over_invoice decimal(6, 2);
UPDATE stats_frame_2
SET avg_kwh_day_over_invoice = (kwh::decimal) / (duration::decimal);

ALTER TABLE stats_frame_2 ADD COLUMN todt_month int;
UPDATE stats_frame_2
SET todt_month = EXTRACT(MONTH from invoicetodt_timestamp);

ALTER TABLE stats_frame_2 ADD COLUMN todt_day int;
UPDATE stats_frame_2
SET todt_day = EXTRACT(DAY from invoicetodt_timestamp);

CREATE TEMP TABLE stats_frame_3 AS (
        SELECT *,
                ROW_NUMBER() OVER (PARTITION BY iethicalid ORDER BY invoicetodt_timestamp) AS invoice_num, 
                lag(kwh, 1) over (partition by iethicalid order by invoicetodt_timestamp)  AS  kwh_lag,
                lag(kwh, 2) over (partition by iethicalid order by invoicetodt_timestamp)  AS  kwh_lag_2,
                lag(avg_kwh_day_over_invoice, 1) over (partition by iethicalid order by invoicetodt_timestamp)  AS  daily_kwh_lag,
                lag(avg_kwh_day_over_invoice, 2) over (partition by iethicalid order by invoicetodt_timestamp)  AS  daily_kwh_lag_2
                FROM stats_frame_2
);

CREATE TEMP TABLE stats_frame_4 AS (
        SELECT *,
                avg(kwh_lag) over (partition by iethicalid order by invoicetodt_timestamp, iethicalid rows unbounded preceding) AS kwh_updating_avg,
                avg(daily_kwh_lag) over (partition by iethicalid order by invoicetodt_timestamp, iethicalid rows unbounded preceding) AS daily_kwh_updating_avg,
                first_value(invoicefromdt_timestamp) over (partition by iethicalid order by invoice_num rows between unbounded preceding and unbounded following) AS first_date
                from stats_frame_3
);

ALTER TABLE stats_frame_4 ADD COLUMN days_passed int;
UPDATE stats_frame_4
SET days_passed = EXTRACT (DAY from (invoicetodt_timestamp - first_date)); 


CREATE TEMP TABLE stats_frame_5 AS (
        SELECT *,
        avg(avg_kwh_day_over_invoice) OVER (partition by iethicalid, todt_month order by invoice_num, iethicalid rows between unbounded preceding and 1 preceding) AS month_avg_preceding
        FROM stats_frame_4
);


ALTER TABLE stats_frame_5 ADD COLUMN month_avg_preceding_flag int;
UPDATE stats_frame_5
set month_avg_preceding_flag = CASE WHEN month_avg_preceding is null then 0 else 1 end;

SELECT * FROM stats_frame_5;

/* Create the table */
CREATE TABLE ee.david_invoices_weather_matched_4 AS (SELECT * from stats_frame_5 ORDER BY iethicalid, invoicetodt_timestamp);