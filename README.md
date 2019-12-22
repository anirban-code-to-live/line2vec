## line2vec
The *line2vec* algorithm learns continuous representations for edges in any (un)directed, (un)weighted graph. 
It is an information network edge representation learning using edge-to-vertex dual graphs (a.k.a line graph). In addition to that, an optimisation problem is solved efficiently to generate the edge embeddings.

### Environment Set-up

**Use python version 2.7**

- Clone the repository.
- Navigate to the base directory of line2vec (the download location)
- Create a virtual environment using the following command:<br/>
``virtualenv venv``<br/>
(If **virtualenv** package is not installed, please install using pip)
- Activate the environment using:<br/>
``source venv/bin/activate``
- Install required python modules to run the code.<br/>
``pip install -r requirements.txt``

Congratulations!! You are now setup to run the code.
  

### Basic Usage

#### Input

- Look at the sample dataset karate inside *data* directory. If you want to experiment on different datasets, create a folder with name of your dataset.
- The **edgelist** formatted file corresponding to the link structure of the information network is required to run and generate the embedding.
- Naming convention for the edgelist file: *<dataset_name>.edgelist*.
- Note : If you are running the code for the *first* time, include the **scratch** flag in the run command.


#### Example
To run *line2vec* on Zachary's karate club network, execute the following command, navigate to the **src** directory inside the project's home path and run the following command:<br/><br/>
	``python main.py --input ../data/karate/karate.edgelist --dataset karate --output ../embed/karate/karate.embed --dimensions=8 --line-graph ../data/karate/karate_line.edgelist --l2v-iter=20 --iter=2``

#### Options
You can check out the other options available to use with *line2vec* using:<br/>
	``python src/main.py --help``

#### Input
The supported input format is an edgelist:

	node1_id_int node2_id_int <weight_float, optional>
		
The graph is assumed to be undirected and unweighted by default. These options can be changed by setting the appropriate flags.

#### Output
The output file has *n+1* lines for a graph with *n* edges. 
The first line has the following format:

	num_of_edges dim_of_representation

The next *n* lines are as follows:
	
	edge_id dim1 dim2 ... dimd

where dim1, ... , dimd is the *d*-dimensional representation learned by *line2vec*.