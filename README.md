Introduction：
Federated learning is a distributed machine learning technique that allows various data sources to work together to train models while keeping their raw data private. However, federated learning faces many challenges when dealing with non-independent and identically distributed (Non-IID) data, especially the problem of data heterogeneity, which can significantly degrade model performance. To address this challenge, we propose a new algorithm for personalized federated learning, known as pfedCluster. The core of the pfedCluster algorithm is to dynamically cluster clients using hierarchical tree clustering, which ensures minimal intra-cluster distance and maximal inter-cluster distance, thus optimizing the clustering effect. Additionally, the algorithm facilitates knowledge transfer between clusters through knowledge distillation, further enhancing model performance. This method improves model personalization by dynamically adjusting the clustering structure to suit varying data distributions. Experimental results show that pfedCluster effectively improves model performance on MNIST and CIFAR-10 datasets, demonstrating significant advantages in dealing with data heterogeneity compared to traditional federated learning algorithms.


**Note:** This repo is still in progress.

## Dependencies
* PyTorch = 1.12.0

## Quick Start

- Run FedAvg algorithm on CIFAR-10, skew partition, 10 clients, local iteration number is 200:

```console
python fedavg.py --gpu "7" --dataset 'cifar10' --partition 'noniid-skew' --n_parties 10 --num_local_iterations 200
```

- For fair comparisons on CIFAR-10, you can run the bash file. 

```console
sh run_cifar10.sh
```
- pfedCluster_cosine.py runs the experiment that we regularize the local model with the normalized aggregated model.



```
