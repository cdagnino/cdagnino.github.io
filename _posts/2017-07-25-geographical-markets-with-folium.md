---
layout: post
title: Geographical Markets with folium
date:   2017-07-25 17:35:10
categories: visualization, geodata
image: /assets/article_images/telepizza_map.png
---

Imagine you're analyzing the market for pizzas in your city. You wonder if Chain A is somehow different. Maybe they tend to locate more on the richer parts of the city. Maybe they compete a lot with Chain B (similar locations), but not so much with Chain C.

Analyzing the behavior of a chain requires you to think about the geographical market of each pizza place: that is, the area in the city that they influence. How do we define these markets? For a pizza place, a sorrounding area of 1km seems too little. 10km is probably too much, but anything in between might be reasonable. On the other hand, for a big hospital 10 or even 20km might be reasonable.

In this post, I want to show how you go from having addresses for a few chains to producing an interactive map that shows the geographical markets that belong to each store.


I'll be using Folium, which is a nice wrapper for Leaflet, a javascript library for making interactive maps.


In this example, I'll consider two competing chains of delivery pizza in Santiago: Telepizza and Pizza Hut.

I got the addresses from a quick check on [800.cl](http://www.800.cl/) and then copy/pasted on a text file (get them [here](/assets/article_data/dires_telepizza.txt) and [here](/assets/article_data/dires_pizza_hut.txt)). I doubt this list is very thorough, but as this is just an example it won't matter. If we were seriously analyzing the pizza delivery market in Santiago, I would have to resort to scraping all the competitors.

So our starting point is two text files with the addresses for both of this chains. We'll need three steps

1. Load and clean the data
2. Geocode it using the google api (we need those Latitudes and Longitudes to map!)
3. Map with Folium

## 1. Load and clean the data

```py
import pandas as pd
import requests
def load_and_clean(file_n):
    df = pd.read_csv("data/{}.txt".format(file_n), header=None)
    df.columns = ['address']
    df['address'].str.replace('.', ',') + ', Santiago, Chile'
    return df
pizza_hut, telepizza = map(load_and_clean, ['dires_pizza_hut', 'dires_telepizza'])
```

## 2. Geocode

The `get_result` function below is a quick and dirty way of getting a few addresses. If you're geocoding many addresses, I'd suggest you write a script that has better exception handling and saves intermediate results. [Check this gist for ideas](https://gist.github.com/shanealynn/033c8a3cacdba8ce03cbe116225ced31)

I don't think you need a API key to geocode this, but, in any case, you can easily get one if you have a gmail acount (see [here](https://console.developers.google.com/apis/))

```py
api_key = 'your API key here'
base = "https://maps.googleapis.com/maps/api/geocode/json?address="

def get_result(addr, base_add=base):
    geo_url = str(base_add) + str(addr)
    if api_key is not None:
        geo_url = geo_url + '&key={0}'.format(api_key)
    result = requests.get(geo_url)
    result = result.json()
    
    # Handle the case with zero results
    if len(result['results']) == 0:
        output = {
            "formatted_address": None,
            "latitude": None,
            "longitude": None,
            "accuracy": None,
            "google_place_id": None,
        }
    else:
        answer = result['results'][0]
        output = {
            "formatted_address": answer.get('formatted_address'),
            "latitude": answer.get('geometry').get('location').get('lat'),
            "longitude": answer.get('geometry').get('location').get('lng'),
            "accuracy": answer.get('geometry').get('location_type'),
            "google_place_id": answer.get("place_id"),
        }
    # Append some other stuff
    output['input_string'] = addr
    output['number_of_results'] = len(result['results'])
    output['status'] = result.get('status')

    return output


results_telepizza = telepizza['address'].map(get_result)
df_telepizza = pd.DataFrame.from_records(results_telepizza)

results_pizzahut = pizza_hut['address'].map(get_result)
df_pizzahut = pd.DataFrame.from_records(results_pizzahut)
```

## 3. Make maps!



First, we need some extra imports:

```py
import geopandas as gpd
from shapely.geometry import Point
from collections import OrderedDict
import folium
from functools import partial
import os
EPSG_POST = 'epsg:32719'
# http://spatialreference.org/ref/epsg/32719/
```

This function will turn the `pd.DataFrame` of the previous step into a geopandas `DataFrame` with the appropriate projeciton

```py
def clean_df(df, epsg_post=EPSG_POST):
    """
    Takes a pandas dataframe with latitud and longitude
    and returns a cleaned geopandas dataframe with EPSG_POST projection

    This EPSG_POST is chosen so that the units are meters
    """
    df = df[df.status != 'ZERO_RESULTS']
    #Check longitude and latitude make sense
    df = df[(df.longitude < -65) & (df.latitude < -10)]
    pts = gpd.GeoSeries([Point(x, y) for x, y in zip(df.longitude, df.latitude)])
    df = gpd.GeoDataFrame(df, geometry=pts)
    df = df.dropna(subset=['geometry'])
    df = df[df.formatted_address.str.split(',').map(lambda x: x[-1].strip()) == 'Chile']
    #  4326 is "default" for naive latitude, longitude
    df.crs = {'init': 'epsg:4326'}
    df = df.to_crs({'init': epsg_post})
    return df
```

Now we write a function that produces the markets. In order to plot them, we need to produce polygons.

There are at least two ways we can go about this. The simple way is to just create a "buffer" ([see shapely docs on buffer](http://toblerity.org/shapely/shapely.geometry.html)), which creates a circle with a certain radius around a store. However, it might be that within that radius you find multiple stores from the same chain. In this case, it makes more sense to take a market as the intersection of all the buffers. The `get_polys` functions allows you to define both type of markets.

Also, we should bear in mind that a radius might be misleading, because what is truly relevant is the time distance. Since all these stores are within the same city (and not mixing urban and rural locations), the distance in km is probably quite correlated with the time distance. However, if you want to be careful, the best way is to generate isochrones around each store. These are all the areas that are reachable within a certain amount of time. Obtaining this polygons is a bit more involved, but you can check an example [here](http://drewfustin.com/isochrones/)




```py
def get_polys(chain, radius, market_type):
    """
    Produces polygons of a certain radius around stores of chain

    chain: pd.DataFrame of the chain
    radius: radius of influence zone of a market
    market_type : simple buffer or union of buffers
                    unary_union merges the buffers that intersect
    """
    if market_type == "buffer":
        return chain.buffer(radius)
    elif market_type == "union":
        unions = gpd.GeoSeries(list(chain.buffer(radius).unary_union))
        unions.crs = {'init': EPSG_POST}
        return unions
```

Once you have these polygons, you can calculate some descriptive statistics for them. For instance, we can ask how many stores of Telepizza are within the markets defined by the stores of PizzaHut. `count_stores_per_polyg` is an example of how to achieve this:


```py
def count_stores_per_polyg(polyg, chain_dict):
    """
    For each polygon, it finds the number of stores
    from each chain that lie within that polygon
    """
    d = OrderedDict()
    for name, chain in chain_dict.items():
        # Count the number of intersections
        d[name] = chain.intersects(polyg).sum()
    return pd.Series(d)

geo_pizzahut = clean_df(df_pizzahut)
geo_telepizza = clean_df(df_telepizza)

polyg_pizzahut = get_polys(geo_pizzahut, 1.5, "union")
chain_d = {'Telepizza': geo_telepizza, 'Pizzahut': geo_pizzahut}

counts_per_market = polyg_pizzahut.apply(count_stores_per_polyg, chain_dict=chain_d)
counts_per_market.head(3)

#Gives:
    Pizzahut	Telepizza
0	1	          0
1	1	          1
2	1	          1
```

The way to read this dataframe is that, in market 0 (or polygon 0), there is one Pizzahut store and 0 Telepizza store. In market 1, theres is one Pizzahut store and 1 Telepizza store.

Not the most exciting of results. If we had data for multiple chains (and not just two) we could see whether Pizzahut tends to have a similar distribution to, say ChainB over ChainC.

However you decide to produce the market polygons for each store, we can plug those into a folium map. The `make_interactive_map` produces polygons with a certain radius distance and plots them as a `Leaflet` map. It will be useful to put a marker for each store so you can check, say, the address. It's also convenient to also put a marker that shows some info of each market. In the function below, I'm only adding an the `id`, but it's easy to add other any other information.


```py
def make_interactive_map(dist_in_km, file_name, market_type, chain):
    """
    Saves an interactive map in .html format

    :param dist_in_km: size of radius in km
    :param file_name: file of the name to be saved
    :param market_type: "union" or "buffer"
    :param chain: pd.DataFrame of the chain
    :return: None
    """
    # Santiago Latitude, Longitude
    m = folium.Map([-33.4691, -70.642], zoom_start=11)
    rad = dist_in_km * 1000
    polys = get_polys(chain, rad, market_type)

    # Adds a Popup for each store 
    for index, row in chain.iterrows():
        lat, long = row['latitude'], row['longitude']
        text = row['formatted_address']
        folium.Marker(location=[lat, long], popup=text).add_to(m)
    folium.GeoJson(polys).add_to(m)

    # Add popup of id for each polygon
    polys = polys.to_crs({'init': 'epsg:4326'})
    for i in range(len(polys)):
        center = polys.geometry.centroid[i]
        lat, long = center.y, center.x
        folium.RegularPolygonMarker(
            location=[lat, long], popup="Index " + str(polys.index[i])).add_to(m)
    full_filename = file_name + '_' + market_type + ".html"
    if os.path.exists(full_filename):
        os.remove(full_filename)
    m.save(full_filename)
```

Now, generating the maps is as easy as:

```py
geo_telepizza = clean_df(df_telepizza)    
make_interactive_map(1.5, "telepizza_map", "union", geo_telepizza)

geo_pizzahut= clean_df(df_pizzahut)    
make_interactive_map(1.5, "pizzahut_map", "union", geo_pizzahut)

```

Note I've used `1.5` km as the radius. I experimented with a few lengths until I found something that was reasonable giving my knowledge of the city. A smaller radius won't deliver any intersecting markets, whereas a bigger one might make the whole city a single market.

![Static Version of the map](/assets/article_images/telepizza_map.png "Static Version of the map")

[Click here for the interactive map](/assets/article_images/telepizza_map_union.html). You can zoom in and out. Be sure to click on the markers too.