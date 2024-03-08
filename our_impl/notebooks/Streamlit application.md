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


