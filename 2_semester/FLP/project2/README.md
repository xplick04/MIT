### Maxim Plička (xplick04)
### Academic Year 2023/24
### Minimum Spanning Tree

# Method Description
The project deals with finding the minimum spanning tree of a graph.

After input is read, edges are stored in a database using a dynamic predicate representing an edge. These edges are stored in both directions since the input is an undirected graph.

The main traversal and finding of all paths involve selecting the first node from the database (if the graph is connected, it doesn't matter which node to start from). From this node, the traversal progresses by recursively diving into previously defined edges. During traversal, explored nodes are gradually added to the `Visited` list, preventing cycling. Traversal not only makes depth-first steps but also breadth-first, ensuring that the output contains all possible spanning trees. After exploring each neighbor, the traversal starts to backtrack, and edges are gradually added to the `Solution` list in alphabetical order. I chose this method of backtracking to avoid distinguishing between connected and disconnected graphs. The entire traversal is initiated using `setof`, obtaining all possible non-duplicate paths.

Paths are then filtered (keeping only paths containing all graph nodes) and printed in the desired format.

Regarding computation time, the algorithm can compute smaller tasks (provided test files) in a few milliseconds.

# Usage Guide
The project is compiled using the `make` command.

After compilation, the project can be run using `./flp23-log < TESTFILE`, where `TESTFILE` is a path to file containing input to the program.

Provided tests are located in the tests directory.

# Limitations
No input validation is performed, although empty lines are ignored.