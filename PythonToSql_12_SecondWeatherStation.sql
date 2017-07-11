/* Initial temp tables, to be combined into invoice data */
CREATE TEMP TABLE temp1 AS (
with cte as (select a.iethicalid, a.cesuniqueid, a.szip, it.* from ee.accounts a join ee.v_invoicestemp it on it.uniqueaccountid = a.cesuniqueid)
select distinct iethicalid, szip, invoicefromdt, invoicetodt, kwh from cte order by iethicalid
);

CREATE TEMP TABLE temp2 AS (
  SELECT zip, wban as wban_1, dist as dist_1
  FROM public.weather_stations WHERE ord=1
);

CREATE TEMP TABLE temp3 AS (
   SELECT zip, wban as wban_2, dist as dist_2
   FROM public.weather_stations WHERE ord = 2
);

CREATE TEMP TABLE temp4 AS (
   SELECT temp2.zip, temp2.wban_1, temp2.dist_1, temp3.wban_2, temp3.dist_2
   FROM temp2
     INNER JOIN temp3 ON (temp2.zip = temp3.zip)
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
    INNER JOIN temp4 ON (temp1.szip = temp4.zip)
);

/* Customer personal data */
CREATE TEMP TABLE p AS (
        SELECT iethicalid, upper(scity), sstate, szip, sgender, sbillingaddress2, iutilityid, stype, countyfips, srateclass
        FROM ee.accounts ORDER BY iethicalid
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
 SELECT m.iethicalid, m.zip, m.invoicefromdt_timestamp, m.invoicetodt_timestamp, m.duration, m.kwh, m.wban_1, m.dist_1, 
        w.yearmonthday_timestamp as yearmonthday_timestamp_1, w.tavg as tavg_1, m.wban_2, m.dist_2
 FROM m JOIN w on (w.wban = m.wban_1 AND w.yearmonthday_timestamp >= m.invoicefromdt_timestamp AND w.yearmonthday_timestamp <= m.invoicetodt_timestamp)
 );
 
CREATE TEMP TABLE big_frame_2 AS (
 SELECT m.iethicalid, m.zip, m.invoicefromdt_timestamp, m.invoicetodt_timestamp, m.duration, m.kwh, m.wban_2, m.dist_2, 
        w.yearmonthday_timestamp as yearmonthday_timestamp_2, w.tavg as tavg_2
 FROM m JOIN w on (w.wban = m.wban_2 AND w.yearmonthday_timestamp >= m.invoicefromdt_timestamp AND w.yearmonthday_timestamp <= m.invoicetodt_timestamp)
 );

CREATE TEMP TABLE big_frame_3 AS (
 SELECT big_frame.iethicalid, big_frame.zip, big_frame.invoicefromdt_timestamp, big_frame.invoicetodt_timestamp, big_frame.duration, big_frame.kwh,
        big_frame.wban_1, big_frame.dist_1, big_frame.yearmonthday_timestamp_1, big_frame.tavg_1,
        big_frame_2.wban_2, big_frame_2.dist_2, big_frame_2.yearmonthday_timestamp_2, big_frame_2.tavg_2
 FROM big_frame LEFT JOIN big_frame_2 on (big_frame.iethicalid = big_frame_2.iethicalid AND 
                                     big_frame.wban_2 = big_frame_2.wban_2 AND 
                                     big_frame.yearmonthday_timestamp_1 = big_frame_2.yearmonthday_timestamp_2)
);
 
CREATE TEMP TABLE big_frame_4 AS (
        SELECT big_frame_3.iethicalid, big_frame_3.zip, big_frame_3.invoicefromdt_timestamp, 
               big_frame_3.invoicetodt_timestamp, big_frame_3.duration, big_frame_3.kwh, 
               big_frame_3.wban_1, big_frame_3.dist_1, big_frame_3.yearmonthday_timestamp_1 as yearmonthday_timestamp, big_frame_3.tavg_1,
               big_frame_3.wban_2, big_frame_3.dist_2, big_frame_3.tavg_2,
               p.upper, p.sstate, p.szip, p.sgender, p.sbillingaddress2, p.iutilityid, p.stype,
               p.countyfips, p.srateclass
        FROM big_frame_3 JOIN p on (big_frame_3.iethicalid = p.iethicalid)
);

/* Reduce the big_frame so that temperature columns are aggregated */
CREATE TEMP TABLE stats_frame_2 AS (
        SELECT 
                DISTINCT iethicalid, zip, invoicefromdt_timestamp, invoicetodt_timestamp, duration, kwh, wban_1, dist_1, wban_2, dist_2, upper, sstate, szip, sgender,
                         sbillingaddress2, iutilityid, stype, countyfips, srateclass,
                SUM(CASE tavg_1 WHEN 'M' then '0' else tavg_1 end) AS tavg_interval_sum_1,
                AVG(CASE tavg_1 WHEN 'M' then '0' else tavg_1 end) AS tavg_interval_avg_1,
                SUM(CASE tavg_1 WHEN 'M' then 1 else 0 end) AS num_missing_1,
                SUM(CASE tavg_2 WHEN NULL THEN '0' WHEN 'M' THEN '0' else tavg_2 end) AS tavg_interval_sum_2,
                AVG(CASE tavg_2 WHEN NULL THEN '0' WHEN 'M' THEN '0' else tavg_2 end) AS tavg_interval_avg_2,
                SUM(CASE tavg_2 WHEN NULL THEN 1 WHEN 'M' THEN  1 else 0 end) AS num_missing_2
        FROM (select * from big_frame_4) GROUP BY iethicalid, zip, invoicefromdt_timestamp, invoicetodt_timestamp, duration, kwh, 
                                                  wban_1, dist_1, wban_2, dist_2,
                                                  upper, sstate, szip, sgender, sbillingaddress2, iutilityid, stype, countyfips, srateclass
);

/* Add some markers for missing temperature data */
ALTER TABLE stats_frame_2 ADD COLUMN tavg_interval_sum_corrected_1 int;
ALTER TABLE stats_frame_2 ADD COLUMN tavg_interval_sum_corrected_2 int;
UPDATE stats_frame_2
SET tavg_interval_sum_corrected_1 = tavg_interval_sum_1 + (tavg_interval_avg_1 * num_missing_1);
UPDATE stats_frame_2
SET tavg_interval_sum_corrected_2 = tavg_interval_sum_2 + (tavg_interval_avg_2 * num_missing_2);

UPDATE stats_frame_2
SET duration = duration + 1;
ALTER TABLE stats_frame_2 ADD COLUMN perc_missing_1 decimal(5, 4);
ALTER TABLE stats_frame_2 ADD COLUMN perc_missing_2 decimal(5, 4);
UPDATE stats_frame_2
SET perc_missing_1 = ((num_missing_1::decimal)/(duration::decimal));
UPDATE stats_frame_2
SET perc_missing_2 = ((num_missing_2::decimal)/(duration::decimal));

/* Choosing weather station 2 if the first has more than 50% missing data! */
ALTER TABLE stats_frame_2 DROP COLUMN dist_preferred;

ALTER TABLE stats_frame_2 ADD COLUMN wban_preferred varchar;
ALTER TABLE stats_frame_2 ADD COLUMN dist_preferred decimal(4, 2);
ALTER TABLE stats_frame_2 ADD COLUMN tavg_interval_avg_preferred int;
ALTER TABLE stats_frame_2 ADD COLUMN perc_missing_preferred decimal(5, 4);

UPDATE stats_frame_2
SET wban_preferred = wban_1 WHERE perc_missing_1 <= 0.5;
UPDATE stats_frame_2
SET wban_preferred = wban_2 WHERE perc_missing_1 > 0.5;

UPDATE stats_frame_2
SET dist_preferred = dist_1 WHERE perc_missing_1 <= 0.5;
UPDATE stats_frame_2
SET dist_preferred = dist_2 WHERE perc_missing_1 > 0.5;

UPDATE stats_frame_2
SET tavg_interval_avg_preferred = tavg_interval_avg_1 WHERE perc_missing_1 <= 0.5;
UPDATE stats_frame_2
SET tavg_interval_avg_preferred = tavg_interval_avg_2 WHERE perc_missing_1 > 0.5;

UPDATE stats_frame_2
SET perc_missing_preferred = perc_missing_1 WHERE perc_missing_1 <= 0.5;
UPDATE stats_frame_2
SET perc_missing_preferred = perc_missing_2 WHERE perc_missing_1 > 0.5;


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

/* Add flags for missing data */
ALTER TABLE stats_frame_5 ADD COLUMN month_avg_preceding_flag int;
UPDATE stats_frame_5
set month_avg_preceding_flag = CASE WHEN month_avg_preceding is null then 0 else 1 end;

ALTER TABLE stats_frame_5 ADD COLUMN sgender_flag int;
UPDATE stats_frame_5
set sgender_flag = CASE WHEN sgender is null then 0 else 1 end;

ALTER TABLE stats_frame_5 ADD COLUMN sbillingaddress2_flag int;
UPDATE stats_frame_5
set sbillingaddress2_flag = CASE WHEN sbillingaddress2 is null then 0 else 1 end;

ALTER TABLE stats_frame_5 ADD COLUMN countyfips_flag int;
UPDATE stats_frame_5
set countyfips_flag = CASE WHEN countyfips is null then 0 else 1 end;

ALTER TABLE stats_frame_5 ADD COLUMN srateclass_flag int;
UPDATE stats_frame_5
set srateclass_flag = CASE WHEN srateclass is null then 0 else 1 end;

SELECT * FROM stats_frame_5;
SELECT COUNT(*) from (SELECT DISTINCT iethicalid from stats_frame_5);

/* Create the table */
CREATE TABLE ee.david_invoices_weather_matched_6 AS (SELECT * from stats_frame_5 ORDER BY iethicalid, invoicetodt_timestamp);