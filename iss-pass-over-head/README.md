# ISS Pass Overhead

The [International Space Station](https://en.wikipedia.org/wiki/International_Space_Station) is a space station in low Earth orbit. The ISS programme is a joint project between five participating space agencies: NASA, Roscosmos, JAXA, ESA, and CSA. The ownership and use of the space station is established by intergovernmental treaties and agreements.

## Inspiration
I have this habit of exploring the night sky with the help of mobile apps like
[Star Walk](https://apps.apple.com/us/app/star-walk-2-night-sky-map/id892279069)
. Space has always been a subject of fascination for me.

The ISS circles the Earth in roughly 92 minutes and completes 15.5 orbits per
day. I collected this data with the help of [IFTTT mobile app](https://apps.apple.com/us/app/ifttt/id660944635).

![IFTTT ISS Setup](https://i.imgur.com/JpC3NOy.jpg)


For this project I used the following location of Chandigarh - 


![ISS Pass Overhead Location](https://i.imgur.com/MXik6Gf.jpg)

## Preparing the execution environment.

### Install dependencies.
It is advisable to create a separate [Python 3 virtual environment](https://help.dreamhost.com/hc/en-us/articles/115000695551-Installing-and-using-virtualenv-with-Python-3) for this project. Execute the following commands for installing virtualenv-

```
python3 -m pip install --upgrade pip
pip3 install virtualenv
```

### Create Virtual Environment

```
virtualenv venv
```

### Activate Virtual Environment

```
source venv/bin/activate
```

### Install dependencies using [requirements.txt](https://github.com/abhaystoic/machine-learning/blob/master/iss-pass-over-head/requirements.txt).

```
pip install -r requirements.txt
```

## Execute driver code.

```
python driver.py
```

## Output:

```
Data size:  Rows = 5469, Coumns = 2
********************
                               End Time  Duration
Start Time                                       
2017-09-21 23:39:00 2017-09-21 23:41:00       130
2017-09-22 02:54:00 2017-09-22 03:01:00       412
2017-09-22 04:29:00 2017-09-22 04:40:00       632
2017-09-22 06:06:00 2017-09-22 06:15:00       522
2017-09-22 19:29:00 2017-09-22 19:39:00       588

Intercept = [311.37406621]
Slope = [[1.18167149e-16]]
      Actual   Predicted
0         76  311.374066
1        446  311.374066
2        644  311.374066
3        502  311.374066
4        612  311.374066
...      ...         ...
1089     441  311.374066
1090     131  311.374066
1091     552  311.374066
1092     647  311.374066
1093     232  311.374066

[1094 rows x 2 columns]
Mean Absolute Error: 214.42090062326577
Mean Squared Error: 57020.775847633246
Root Mean Squared Error: 238.79023398714037
```

## Visual Outputs:


### Year-wise and Month-wise Box Plot


![Yearly and Monthly Distribution](https://i.imgur.com/SGOULjj.png)


### Start Time vs Duration


![Start Time vs Duration](https://i.imgur.com/qP8eMnb.png)


### Start Time vs Predicted Duration


![Start Time vs Predicted Duration](https://i.imgur.com/drybnEO.png)
