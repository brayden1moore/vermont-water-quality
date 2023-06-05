# Vermont-Water-Quality
Model to estimate nitrogen and phosphorus for a given Vermont coordinate, address, or body of water.

TO USE:

* Download git repository.
* Create a .py or .ipynb the folder, and import vermont-water-quality as vwq.
* Call the <code>estimate()<code> method with a coordinate tuple, a body of water name string, or an address string as the argument.
* Add '''n_estimators''' or '''max_depth''' to your arguments to alter the GradientBoostingRegressor's hyperparameters.

Parcel data sourced from https://geodata.vermont.gov/datasets/09cf47e1cf82465e99164762a04f3ce6_0/explore?location=43.864753%2C-72.459770%2C8.93
<br>

Water sample data sourced from https://www.waterqualitydata.us//#countrycode=US&statecode=US%3A50&mimeType=csv&sorted=no&providers=NWIS&providers=STEWARDS&providers=STORET
<br>

Geospatial data sourced from
https://www.sciencebase.gov/catalog/item/4f70a58ce4b058caae3f8ddb
<br><br>

**Objective** <br>
Create a model that finds upstream polluters (farms, quarries, factories, mills, etc.) of bodies of water, and uses size, elevation, and proximity data to estimate the nitrogen and phosphorus concentrations downstream of them. <br><br>

**Method** <br>
In the training data, for every water sample, measure the distance between the sample and nearby potential polluters. Then ensure that these potential polluters are upstream of the sample by examining their position relative to the sample in a topographical map. If they have a higher elevation than the sample AND if the vector of elevations between them and the sample never exceeds the polluter's elevation, then consider them upstream of the sample. Then using scikit-learn's GradientBoostingRegressor with total nearby upstream farm area (scaled for distance) and count, total upstream industry area and count, distance and elevation difference from the nearest polluters, and season as features, predict nitrogen and phosphorus concentration. <br>
The "estimate" function can take a partial or full name of a body of water (i.e. "battenkill"), address (i.e. "87 River Rd, Rutland, VT"), or coordinate pair (i.e. (44.5,-73.0)) and returns an estimation of N and P concentrations as well as visualization of the prediction intervals, nearby polluters, and most recent nearby water sample. <br><br>

**Findings**<br>
I'm not finished yet, so I'll update later. Interesting thing I've found is that there are a number of samples in the Northeast Kingdom with N and P concentrations that cannot be explained by the model, mostly the tributaries of Lake Memphremagog.
