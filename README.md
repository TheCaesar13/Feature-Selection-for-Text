# Feature-Selection-for-Text
An implementation of the new Slime Mold Algorithm, a heuristic optimization technique, in combination with filter/statistical fitness functions for selecting the best subset of features from text datasets.

The purpose of this paper is to develop a method relying on a population-
based search strategy (Slime Mould Algorithm), which has proven its ability to find optimal solutions
in large search spaces, and a statistical evaluation metric (Chi-squared) which is also widely used in
the field. The ambitions of the project were to find a sweet spot between very effective and very
efficient feature selection algorithms, however the experiments on two benchmark datasets and four
popular machine learning algorithms led to the conclusion that combining this specific heuristic
search algorithm and this particular statistical measure only obtain average result from both
perspectives.

4.1 Data preprocessing and transformation
Text data is often found under a semi-structured or unstructured form, which makes it
difficult, perhaps even impossible for machine learning algorithms to search and extract
patterns. At base machine learning algorithms have mathematical principles, equations
which take into consideration distances or probabilities, therefor they are appropriate for
numerical data. In order to harvest their analytical power the text data should be
transformed. There are various ways of representing text data; this study will consider a
BagofWords (BoW) representation due to its simple implementation and flexible
functionality. The datasets chosen for implementation and testing are benchmark corpuses
which have been used extensively in the literature and a certain amount of structure has
been added to them. Thus the starting point for us is a directory with as many subdirectories
as categories, each subdirectory holding all the files/documents in that class.
Firstly the corpus files are loaded into the memory of the Integrated Development
Environment (IDE) by calling a predefined function from the sklearn.datasets module. The
operation returns a list of strings values where every string represents the entire text found
in a document, it also returns a list with the names of the categories fetched from the titles
of the subdirectories mentioned earlier. Next the documents stored as string values have to
be processed in order to extract the words which will later form the features of the dataset.
This process is named tokenization, and it counts the occurrences of every term in each
document. The result is a matrix of size D x F, where D is the total number of documents and
F is the number of unique terms in the corpus. For a lower memory consumption when
storing the dataset, considering the large dimension that our corpuses have, sparse matrices
were utilized. For a rudimentary elimination of the irrelevant term, the special characters
and numbers were filtered out of the corpuses.
4.2 Slime Mould Algorithm – Search Strategy
Slime mould algorithm is a novel metaheuristic optimization method proposed by Li et. al.
(2020). It is based on the behaviour of Physarum polycephalum, which is a eukaryote
bacteria that has a dynamically adjustable food search strategy. The slime mould naturally
implements the concepts of exploration and exploitation found in population-based
optimization algorithms. When the bacteria finds a high quality food source it startsexploiting the nearby areas, on the contrary if the food source quality is low, it starts
exploring other paths.
The mathematical model of slime mould mainly consists of two stages, the food approach
and the food wrapping. The first stage is modelled by the following equations:
where →
(
→
( )
and →
)
and →
( )
( )
represent the mould population at iteration t and t+1 respectively,
are randomly selected individuals, →
( )
is the individual with the best fitness
from the current iteration, → is decreasing linearly from one to zero, → is a value between [-
a,a], r is a random value between [0,1] and → is the weight of the slime mold.
a = arctanh (− (
) + 1)
p = tanh|(i) − DF|
where DF is the best fitness obtained in all iterations so far and S(i) represents the fitness of the
population X(t).
where condition means that (i) is greater than half of the population,bF is the best fitness
obtained in this iteration, wF denotes the worst fitness resulted during this iteration and
SmellIndex is a sorted array of all the fitness values.
The food wrapping stage incorporates one more equation which limits the search range by
setting a lower boundary (LB) and an upper boundary (UB) and updating the position of the
slime mould based on the random value between [0,1] (rand) and an empirically determined
parameter z. Considering that the search agents are represented by fixed dimension vectors,
this third equation is not actually required for the binary version of the algorithm.Originally the algorithm was design for continuous functions which means it cannot be
applied to text feature selection due to the discrete nature of the latter. In order to utilize
the principles of the slime mould search strategy, the population encoding requires a
modification to allow only 0 (absent) and 1 (present) values for the array that stores the
features subsets. Thus a sigmoid activation function will be applied:
where x is the element in the
U(0, 1).
position of the i
individual from the population and σ ∼
The computational complexity of the slime mould algorithm is (N ∗ (1 + T ∗ N ∗ (1 + log N +
2 ∗ D))), where N is the number of individuals considered, T is the number of iterations and
D is the dimension of the search space.
4.3 Chi-Squared (χ2) – Fitness function
The Chi-Squared measure has its origins in statistical analysis and it was used to measure the
difference between observations of occurring events and the expectations of occurrence
according to a hypothesis. With regards to text categorization, the metric provides insights
on the independence between a word occurring in a document and a class that could be
assigned to the document. A Chi-Squared low value indicates a high independence, for
example if the word appears in all the classes then it cannot be used to differentiate them.
On the contrary, a Chi-Squared high value indicates strong dependence, if a word appears in
only one class then it can be used as separation criteria.
In the field of feature selection, Chi-Squared has proved to be a good metric and became
one of the popular choices for practitioners. One of its highlights is the ability to showcase
both association of a word with a category and exclusion of a category based on the
dependence/independence level.
The mathematical formula is presented in the following equation:
χ
(c
N ∗ (c ∗ c̅ ̅ c ̅ ∗ c̅ )
c̅ ) ∗ (c
c ̅ ) ∗ (c
c̅ ̅ ) ∗ (c̅ ̅
c ̅ )
where N is the total number of documents in the dataset, c is the number of documents
from class c that contain the word w, c̅ ̅ represents the number of documents that are not
from class c and do not contain the word w, c ̅ is the number of documents which are part
of class c but the word w is not present and c̅ is the number of documents that have the
word w but are not part of class c.The fitness function was built around the chi-square value of each pair attribute-class. Thus
every search agent would have a number of chi-square values, for every selected feature,
equal to the number of classes. The maximum value of every feature is chosen, this means
every selected attribute in the search agent will have associate a score corresponding to the
class it can best differentiate. Finally all the scores are summed up to form the fitness of a
subset.
4.4 Slime Mould Pseudo-code
Initialize the parameters pop_size, max_iteraition;
Initialize the positions of search agents Xi(i
1,2, ... , n);
While (t ≤ max_iteraition)
calculate the fitness of all search agents;
update bestFitness, bestPosition;
calculate the Weight by Eq. (2.5);
For each search agent
update p, vb, vc;
update positions by Eq. (2.7);
End For
t = t + 1;
End While
Return bestPosition;
