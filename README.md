# ModelChimp: PySpark Example

This repository contains example code of using ModelChimp with PySpark


## ModelChimp Installation


```shell
pip3 install modelchimp
```

Its assumed that PySpark is already installed


## Define the project

Create the project in modelchimp and copy the project key to the code in main.py

![alt text](https://docs.modelchimp.com/doc_project_definition.png )
![alt text](https://docs.modelchimp.com/doc_project_key.png )

```python
tracker = Tracker('<PROJECT KEY>', host='localhost:8000', experiment_name='MNIST Classification') #MODELCHIMP
```

## Execute the code
Execute the code with the following command

```bash
python3 main.py
```
