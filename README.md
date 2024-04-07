# OathRadar

## What is PathRadar?

PathRadar is an algorithm that classifies ASes into core and shell categories and uses the progressive learning process (PLP) for AS path inference, which enables the inference of AS-level paths, prefix-level paths, as well as non-valley-free paths. You can learn more about PathRadar in IMC 2024.

## Quickstart

To get started using PathRadar, clone or download this [GitHub repo](https://github.com/Zitong-Jin/PathRadar).

__Install Python dependencies__

```sh
$ pip install --user -r requirements.txt
```

__Download AS to Organization Mapping Dataset from CAIDA__

https://www.caida.org/data/as-organizations/

__Download PeeringDB Dataset from CAIDA__

Before March 2016: http://data.caida.org/datasets/peeringdb-v1/

After March 2016: http://data.caida.org/datasets/peeringdb-v2/

__Prepare BGP paths from Route Views and RIS__

You can prepare BGP paths from [BGPStream](https://bgpstream.caida.org/) or download rib file from [Route Views](http://archive.routeviews.org/) and [RIS](http://data.ris.ripe.net/). 

Noting that PathRadar only use IPv4 AS paths. Here is an example to extract AS paths from rib file:

```sh
prefix = re.search(r'PREFIX: ([^\n]*)\n', block).group(1).strip()
if prefix:
    aspath = re.search(r'ASPATH: ([^\n]*)\n', block).group(1).strip()
    if '{' in aspath or '(' in aspath:
        continue
    if ":" not in temp_prefix:
        output.append(aspath.replace(' ', '|'))
```

### Basic inference

The ASes on each BGP path should be delimited by '|' on each line, for example, AS1|AS2|AS3.

__Parse downloaded BGP paths__

```sh
$ python uniquePath.py -i=<aspaths file> -p=<peeringdb file>
# e.g. python uniquePath.py -i=aspaths_2019.txt -p=peeringdb_2019.json
# Output is written to 'aspaths.txt'.
```

__Run AS-Rank algorithm to bootstrap PathRadar__

```sh
$ perl asrank.pl aspaths.txt > asrel.txt
```

__Choose valleyfree paths__

```sh
$ python3 clean_alldata.py
# Output is written to 'valleyfree.txt'.
```

__Divide train dataset, validation dataset and test dataset__

```sh
$ python3 split_train_test.py
# Output is written to 'rib_data.txt', 'rib_train.txt', 'rib_test.txt', 'rib_validation.txt' and 'rib_origintest.txt'.
```

__Organize datasets according to various destination ASes__

```sh
$ python3 combine_path.py
# Output is written to 'rib/', 'data/', 'train/', 'test/', 'validation/' and 'origintest/' folders.
```

__Generate false samples with the K-GR model__

```sh
$ python3 gen_false_sample.py -m=<max path number>
# e.g. python3 gen_false_sample.py -m=10
# Output is written to the 'shortest' folder.
```

__Train the PathRadar classifier (round 0)__

```sh
$ python3 pathradar_train.py -r=<rounds> -q=<quick mode>
# e.g. python3 pathradar_train.py -r=0 -q=1
# Output is written to the 'models' folder.
# Please note that using the quick_mode options will greatly speed up our training process, but will slightly affect the accuracy of PathRadar
```

__Predict paths with PathRadar (round 0)__

```sh
$ python3 pathradar_prediction.py -r=<rounds> -t=<threshold> -m=<train or test> -q=<quick mode>
# e.g. python3 pathradar_prediction.py -r=0 -t=0.5 -m=train -q=1
# Output is written to the 'result_round0' folder.
# Please note that using the quick_mode options will only generate paths towards 1,000 destination ASes. It will greatly speed up our prediction process, but will slightly affect the final results of PathRadar
```

You can repeat the above process to improve the accuracy of PathRadar (by updating the rounds parameter), and an accurate result can be got after you finish the round 2.

Finally, you can use the following code to get the final result of PathRadar.

__Predict paths with PathRadar (round N)__

```sh
$ python3 pathradar_prediction.py -r=<round N> -t=<threshold> -m=<train or test> -q=<quick mode>
# e.g. python3 pathradar_prediction.py -r=2 -t=0.7 -m=test -q=0
# Use train_or_test=test to test PathRadar and use quick_mode=0 to generate paths towards all destination ASes in the test dataset
```

__Output data format__

\<source AS\>|\<AS1\>|\<AS2\>|...|\<destination AS\> \<path score\>

## Contact 

You can contact us at <jinzt19@mails.tsinghua.edu.cn>.
