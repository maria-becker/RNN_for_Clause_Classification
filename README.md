This is a Classifier for situation entity types as described in Becker et al., 2017. These clause types depend on a combination of syntactic-semantic and contextual features. 
We explore this task in a deeplearning framework, where tuned word representations capture lexical, syntactic and semantic features. 
We introduce an attention mechanism that pinpoints relevant context not only for the current instance, but also for the larger context.
The advantage of our neural model is that it avoids the need to reproduce linguistic features for other languages and is thus more easily transferable. 

We provide code for the basic local model (GRU), the local model with attention (GRU+attention), and our best performing context model which uses labels of previous clauses and genre information (GRU+attention+label+genre).

The data we used for our experiments can be found here, and we used the same train-dev-test split: https://github.com/annefried/sitent/tree/master/annotated_corpus

References:
Classifying Semantic Clause Types: Modeling Context and Genre Characteristics with Recurrent Neural Networks and Attention.
Maria Becker, Michael Staniek, Vivi Nastase, Alexis Palmer, and Anette Frank.
In: Proceedings of the 6th Joint Conference on Lexical and Computational Semantics (*SEM 2017), pages 230–240.
https://www.aclweb.org/anthology/S17-1027