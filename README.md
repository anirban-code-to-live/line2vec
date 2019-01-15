# pen-line2vec
Information network edge representation learning using edge-to-vertex dual graphs (a.k.a line graph). In addition to that, an optimisation problem is solved efficiently to generate the edge embeddings.

python main.py --input ../data/karate/karate.edgelist --dataset karate --output ../embed/karate/karate_10.embed --dimensions=4 --line-graph ../data/karate/karate_line.edgelist --l2v-iter=20 --iter=2
