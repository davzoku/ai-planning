Workflow 
1. CSV file upload 
2. select/ input sku_id
3. read files, get comp data based on sku_id, calculate price discount, sales lag, sum columns, store data
4. Filter dataset inputs - year from, year to and year test
5. 




Price discount calc
- calculate lower bound which is 95% of the max price 
- calculate median for each group of time_year
- relative discount = median price / price = x
- calculate z_scores = x - mean of x / std of x
- get outlier indices 
- update price col

TO DO: 

Drop down for categories
drop down for store ID
Filter SKUs based on categories and have a multi select SKUs


Input tab and analysis tab 
input tab for historical data 
- append new data differently from histrical database 
	- input data is the demand data and time.csv 
	- show time.csv file as calendar data table 

Workflow: 
Page 1 - upload data based on suggested format
page 2 - generate demand 
- select data range 
- select store (input dd)
- select category (input dd)
- select SKU (input dd)
- generate demand (button)

Page 3 - Run promotional optimzation 
- select data range 
- run GA 

Page 4 - Results 
- visualizations on revenue increase - post analysis 



