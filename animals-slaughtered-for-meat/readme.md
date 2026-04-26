# Yearly number of animals slaughtered for meat - Data package

This data package contains the data that powers the chart ["Yearly number of animals slaughtered for meat"](https://ourworldindata.org/grapher/animals-slaughtered-for-meat?v=1&csvType=full&useColumnShortNames=false) on the Our World in Data website.

## CSV Structure

The high level structure of the CSV file is that each row is an observation for an entity (usually a country or region) and a timepoint (usually a year).

The first two columns in the CSV file are "Entity" and "Code". "Entity" is the name of the entity (e.g. "United States"). "Code" is the OWID internal entity code that we use if the entity is a country or region. For most countries, this is the same as the [iso alpha-3](https://en.wikipedia.org/wiki/ISO_3166-1_alpha-3) code of the entity (e.g. "USA") - for non-standard countries like historical countries these are custom codes.

The third column is either "Year" or "Day". If the data is annual, this is "Year" and contains only the year as an integer. If the column is "Day", the column contains a date string in the form "YYYY-MM-DD".

The remaining columns are the data columns, each of which is a time series. If the CSV data is downloaded using the "full data" option, then each column corresponds to one time series below. If the CSV data is downloaded using the "only selected data visible in the chart" option then the data columns are transformed depending on the chart type and thus the association with the time series might not be as straightforward.


## Metadata.json structure

The .metadata.json file contains metadata about the data package. The "charts" key contains information to recreate the chart, like the title, subtitle etc.. The "columns" key contains information about each of the columns in the csv, like the unit, timespan covered, citation for the data etc..

## About the data

Our World in Data is almost never the original producer of the data - almost all of the data we use has been compiled by others. If you want to re-use data, it is your responsibility to ensure that you adhere to the sources' license and to credit them correctly. Please note that a single time series may have more than one source - e.g. when we stich together data from different time periods by different producers or when we calculate per capita metrics using population data from a second source.

### How we process data at Our World In Data
All data and visualizations on Our World in Data rely on data sourced from one or several original data providers. Preparing this original data involves several processing steps. Depending on the data, this can include standardizing country names and world region definitions, converting units, calculating derived indicators such as per capita measures, as well as adding or adapting metadata such as the name or the description given to an indicator.
[Read about our data pipeline](https://docs.owid.io/projects/etl/)

## Detailed information about each time series


## Number of cattle slaughtered for meat – UN FAO
Last updated: February 25, 2026  
Next update: February 2027  
Date range: 1961–2024  
Unit: animals  


### How to cite this data

#### In-line citation
If you have limited space (e.g. in data visualizations), you can use this abbreviated in-line citation:  
Food and Agriculture Organization of the United Nations (2025) – with major processing by Our World in Data

#### Full citation
Food and Agriculture Organization of the United Nations (2025) – with major processing by Our World in Data. “Number of cattle slaughtered for meat – UN FAO” [dataset]. Food and Agriculture Organization of the United Nations, “Production: Crops and livestock products” [original data].
Source: Food and Agriculture Organization of the United Nations (2025) – with major processing by Our World In Data

### How is this data described by its producer - Food and Agriculture Organization of the United Nations (2025)?
Item: Meat of cattle with the bone, fresh or chilled

Description: Meat of bovine animals, fresh or chilled, with bone in. Common trade names are beef and veal. (Unofficial definition)

Metric: Producing Animals/Slaughtered

### Source

#### Food and Agriculture Organization of the United Nations – Production: Crops and livestock products
Retrieved on: 2026-02-25  
Retrieved from: http://www.fao.org/faostat/en/#data/QCL  


## Number of goats slaughtered for meat – UN FAO
Last updated: February 25, 2026  
Next update: February 2027  
Date range: 1961–2024  
Unit: animals  


### How to cite this data

#### In-line citation
If you have limited space (e.g. in data visualizations), you can use this abbreviated in-line citation:  
Food and Agriculture Organization of the United Nations (2025) – with major processing by Our World in Data

#### Full citation
Food and Agriculture Organization of the United Nations (2025) – with major processing by Our World in Data. “Number of goats slaughtered for meat – UN FAO” [dataset]. Food and Agriculture Organization of the United Nations, “Production: Crops and livestock products” [original data].
Source: Food and Agriculture Organization of the United Nations (2025) – with major processing by Our World In Data

### How is this data described by its producer - Food and Agriculture Organization of the United Nations (2025)?
Item: Meat of goat, fresh or chilled

Description: Meat of goat, fresh or chilled This subclass includes: -  meat of goat, animals of subclass 02123, fresh or chilled This subclass does not include: -  meat of goat, frozen, cf. 21136 - edible offal of goat, cf. 21156

Metric: Producing Animals/Slaughtered

### Source

#### Food and Agriculture Organization of the United Nations – Production: Crops and livestock products
Retrieved on: 2026-02-25  
Retrieved from: http://www.fao.org/faostat/en/#data/QCL  


## Number of chickens slaughtered for meat – UN FAO
Last updated: February 25, 2026  
Next update: February 2027  
Date range: 1961–2024  
Unit: animals  


### How to cite this data

#### In-line citation
If you have limited space (e.g. in data visualizations), you can use this abbreviated in-line citation:  
Food and Agriculture Organization of the United Nations (2025) – with major processing by Our World in Data

#### Full citation
Food and Agriculture Organization of the United Nations (2025) – with major processing by Our World in Data. “Number of chickens slaughtered for meat – UN FAO” [dataset]. Food and Agriculture Organization of the United Nations, “Production: Crops and livestock products” [original data].
Source: Food and Agriculture Organization of the United Nations (2025) – with major processing by Our World In Data

### How is this data described by its producer - Food and Agriculture Organization of the United Nations (2025)?
Item: Meat of chickens, fresh or chilled

Description: Meat of chickens, fresh or chilled This subclass includes: -  meat of chickens, Gallus domesticus, birds of subclass 02151, fresh or chilled This subclass does not include: -  meat of chickens, frozen, cf. 21141 - edible offal of chicken, cf. 21160

Metric: Producing Animals/Slaughtered

### Source

#### Food and Agriculture Organization of the United Nations – Production: Crops and livestock products
Retrieved on: 2026-02-25  
Retrieved from: http://www.fao.org/faostat/en/#data/QCL  


## Number of turkeys slaughtered for meat – UN FAO
Last updated: February 25, 2026  
Next update: February 2027  
Date range: 1961–2024  
Unit: animals  


### How to cite this data

#### In-line citation
If you have limited space (e.g. in data visualizations), you can use this abbreviated in-line citation:  
Food and Agriculture Organization of the United Nations (2025) – with major processing by Our World in Data

#### Full citation
Food and Agriculture Organization of the United Nations (2025) – with major processing by Our World in Data. “Number of turkeys slaughtered for meat – UN FAO” [dataset]. Food and Agriculture Organization of the United Nations, “Production: Crops and livestock products” [original data].
Source: Food and Agriculture Organization of the United Nations (2025) – with major processing by Our World In Data

### How is this data described by its producer - Food and Agriculture Organization of the United Nations (2025)?
Item: Meat of turkeys, fresh or chilled

Description: Meat of turkeys, fresh or chilled This subclass includes: -  meat of turkeys, birds of subclass 02152, fresh or chilled This subclass does not include: -  meat of turkeys, frozen, cf. 21144 - edible offal of turkey, cf. 21160

Metric: Producing Animals/Slaughtered

### Source

#### Food and Agriculture Organization of the United Nations – Production: Crops and livestock products
Retrieved on: 2026-02-25  
Retrieved from: http://www.fao.org/faostat/en/#data/QCL  


## Number of pigs slaughtered for meat – UN FAO
Last updated: February 25, 2026  
Next update: February 2027  
Date range: 1961–2024  
Unit: animals  


### How to cite this data

#### In-line citation
If you have limited space (e.g. in data visualizations), you can use this abbreviated in-line citation:  
Food and Agriculture Organization of the United Nations (2025) – with major processing by Our World in Data

#### Full citation
Food and Agriculture Organization of the United Nations (2025) – with major processing by Our World in Data. “Number of pigs slaughtered for meat – UN FAO” [dataset]. Food and Agriculture Organization of the United Nations, “Production: Crops and livestock products” [original data].
Source: Food and Agriculture Organization of the United Nations (2025) – with major processing by Our World In Data

### How is this data described by its producer - Food and Agriculture Organization of the United Nations (2025)?
Item: Meat of pig with the bone, fresh or chilled

Description: Meat, with the bone in, of domestic or wild pigs (e.g. wild boars), whether fresh or chilled. (Unofficial definition)

Metric: Producing Animals/Slaughtered

### Source

#### Food and Agriculture Organization of the United Nations – Production: Crops and livestock products
Retrieved on: 2026-02-25  
Retrieved from: http://www.fao.org/faostat/en/#data/QCL  


## Number of sheep slaughtered for meat – UN FAO
Last updated: February 25, 2026  
Next update: February 2027  
Date range: 1961–2024  
Unit: animals  


### How to cite this data

#### In-line citation
If you have limited space (e.g. in data visualizations), you can use this abbreviated in-line citation:  
Food and Agriculture Organization of the United Nations (2025) – with major processing by Our World in Data

#### Full citation
Food and Agriculture Organization of the United Nations (2025) – with major processing by Our World in Data. “Number of sheep slaughtered for meat – UN FAO” [dataset]. Food and Agriculture Organization of the United Nations, “Production: Crops and livestock products” [original data].
Source: Food and Agriculture Organization of the United Nations (2025) – with major processing by Our World In Data

### How is this data described by its producer - Food and Agriculture Organization of the United Nations (2025)?
Item: Meat of sheep, fresh or chilled

Description: Meat of sheep, fresh or chilled This subclass is defined through the following headings/subheadings of the HS 2007: 0204.10 - .23.

Metric: Producing Animals/Slaughtered

### Source

#### Food and Agriculture Organization of the United Nations – Production: Crops and livestock products
Retrieved on: 2026-02-25  
Retrieved from: http://www.fao.org/faostat/en/#data/QCL  


## Number of ducks slaughtered for meat – UN FAO
Last updated: February 25, 2026  
Next update: February 2027  
Date range: 1961–2024  
Unit: animals  


### How to cite this data

#### In-line citation
If you have limited space (e.g. in data visualizations), you can use this abbreviated in-line citation:  
Food and Agriculture Organization of the United Nations (2025) – with major processing by Our World in Data

#### Full citation
Food and Agriculture Organization of the United Nations (2025) – with major processing by Our World in Data. “Number of ducks slaughtered for meat – UN FAO” [dataset]. Food and Agriculture Organization of the United Nations, “Production: Crops and livestock products” [original data].
Source: Food and Agriculture Organization of the United Nations (2025) – with major processing by Our World In Data

### How is this data described by its producer - Food and Agriculture Organization of the United Nations (2025)?
Item: Meat of ducks, fresh or chilled

Description: Meat of ducks, fresh or chilled This subclass includes: -  meat of ducks, birds of subclass 02154, fresh or chilled This subclass does not include: -  meat of ducks, frozen, cf. 21142 - fatty liver of duck, cf. 21160 - edible offal of duck, cf. 21160

Metric: Producing Animals/Slaughtered

### Source

#### Food and Agriculture Organization of the United Nations – Production: Crops and livestock products
Retrieved on: 2026-02-25  
Retrieved from: http://www.fao.org/faostat/en/#data/QCL  


    