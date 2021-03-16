# Netflix Programs

I use netflix a lot for my entertainment and binge watching - Usually wasting too much time by clicking suggested tv shows and movies😕.

Then i thought why not analyze netflix content dataset to see, if i can find any intersting stuff.

Downloaded dataset(https://www.kaggle.com/shivamb/netflix-shows) as of 2019 from kaggle and setup an conda environment with numpy, pandas and matplotlib for analysis and visualisation.

### Netflix Show/Movie `release_year` based analysis

On Analysing the data, number of shows/movies has taken a extremley huge spike from the halfway mark in dataset. From 20 in 1980 to more than 1750 in 2020.
This increase might be owing to the addition few reasons
  * Content from different countries after the beginning of 2000's - We can confirm this by analyzing `country` further
  * Content not saved due to less technological advancements during the time period

*Plot's visualzied based on the release year seperated at midpoint*
<img src="/images/ry-1940-80.png">
<img src="/images/ry-1981-2020.png">
