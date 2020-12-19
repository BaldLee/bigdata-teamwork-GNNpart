# bigdata-teamwork-GNNpart
## Depedency
### Option 1
- python 3.8.5
- dgl-cuda10.2
- pytorch >= 1.3
- pandas
- requests
- install above packages with pip or conda
### Option 2
``` shell
conda create --name <env> --file requirements.txt
```
- It will help you to create a environment to run GCNtrain.py.
- PS: There are some useless packages in "requirements.txt" (such as Jupyter, matplotlib, etc.)
## Data
- To run GCNtrain.py, you need to put "rank_id.csv" and "knowledge_aquisition_reference.csv" into "/data" dir.