To run this you must have ElasticSearch 6.X installed and available on the path. All the python packages required are specified as a conda environment in `environment.yaml`

We have scripts that start and configure or stop ES that can be run with:

```bash
python qanta/ir.py start
python qanta/ir.py stop
```

The IR experiments in our paper can be reproduced by running `asr_experiments.py` and copying the output. In the file we enumerate all the experiments. If you don't want to run everything then commenting out the appropriate lines in `train_test_files` may help.