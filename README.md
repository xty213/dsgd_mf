## 10605 Homework 7
### Distributed SGD for Matrix Factorization on Spark

## AWS Setup
On master:

`./start-master.sh`

On slaves:

`./start-slave.sh org.apache.spark.deploy.worker.Worker spark://ip-172-31-44-204.ec2.internal:7077`

## Usage
### Experiment 1:
`./spark-submit --master spark://ip-172-31-44-204.ec2.internal:7077 /home/hadoop/spark/bin/dsgd_mf.py 20 10 100 0.6 0.1 hdfs://ip-172-31-44-204.ec2.internal:9000/nf_subsample.csv w.csv h.csv 2 > spark_dsgd.log 1 > eval_acc.log`

### Experiment 2:
`./spark-submit --master spark://ip-172-31-44-204.ec2.internal:7077 /home/hadoop/spark/bin/dsgd_mf.py 20 5 30 0.6 0.1 hdfs://ip-172-31-44-204.ec2.internal:9000/nf_subsample.csv w.csv h.csv 2 > spark_dsgd.log 1 > eval_acc.log`

### Experiment 3:
`./spark-submit --master spark://ip-172-31-44-204.ec2.internal:7077 /home/hadoop/spark/bin/dsgd_mf.py 30 10 30 0.6 0.1 hdfs://ip-172-31-44-204.ec2.internal:9000/nf_subsample.csv w.csv h.csv 2 > spark_dsgd.log 1 > eval_acc.log`

### Experiment 4:
`./spark-submit --master spark://ip-172-31-44-204.ec2.internal:7077 /home/hadoop/spark/bin/dsgd_mf.py 20 10 30 0.8 0.1 hdfs://ip-172-31-44-204.ec2.internal:9000/nf_subsample.csv w.csv h.csv 2 > spark_dsgd.log 1 > eval_acc.log`

### Evaluation:
`python eval2.pyc eval_acc.log ~/Downloads/spark-1.2.1-bin-hadoop2.3/bin/spark-submit ~/Documents/605/hw/7/dsgd_mf.py 50 2 50 0.6 1.0 ~/Documents/605/hw/7/autolab_train.csv w.csv h.csv > spark_dsgd.log`
