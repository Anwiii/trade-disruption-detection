-- Run in the BigQuery sandbox. Save the result as gdelt_monthly_events.csv
SELECT
  DIV(SQLDATE, 100) AS period,
  country,
  EventRootCode,
  COUNT(*) AS n_events,
  SUM(NumMentions) AS n_mentions,
  AVG(GoldsteinScale) AS avg_goldstein,
  AVG(AvgTone) AS avg_tone
FROM `gdelt-bq.gdeltv2.events_partitioned`,
  UNNEST(ARRAY(
    SELECT DISTINCT c
    FROM UNNEST([Actor1CountryCode, Actor2CountryCode]) AS c
    WHERE c IS NOT NULL
  )) AS country
WHERE _PARTITIONTIME >= TIMESTAMP('2017-01-01')
  AND _PARTITIONTIME <  TIMESTAMP('2026-01-01')
  AND country IN ('RUS','UKR','BLR','USA','BRA','CAN','AUS','IND','CHN','EGY',
                  'TUR','DEU','FRA','IDN','ARG','KAZ','NGA','BGD','PAK','SAU',
                  'ZAF','VNM','MEX','JPN','KOR')
GROUP BY period, country, EventRootCode
