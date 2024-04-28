/** 
* Project: 2. projekt FLP
* Author: Maxim Plička (xplick04)
* Date: 2024-04-25
*/

% ------------------------------------------------------------------------------
% Reading the input

read_line(L,C) :-
	get_char(C),
	(
		isEOFEOL(C), L = [], !;
		C == ' ', read_line(L,_); % ignore spaces
		read_line(LL,_),
		[C|LL] = L
	).


isEOFEOL(C) :-
	C == end_of_file;
	(char_code(C,Code), Code==10).


read_lines(Ls) :-
	read_line(L,C),
	( C == end_of_file, Ls = [] ;
	  read_lines(LLs), Ls = [L|LLs]
	).

% ------------------------------------------------------------------------------
% Database and helper functions

:- dynamic(edge/2). % Dynamic predicate for storing the edges


filter_empty([],[]).
filter_empty([[]|T],R) :- filter_empty(T,R).
filter_empty([H|T],[H|R]) :- filter_empty(T,R).


write_edges_to_db([]).
write_edges_to_db([[V1,V2]|Edges]) :-
    assert(edge(V1, V2)), % Assert the edge to the database dynamically
	assert(edge(V2, V1)), % Assert the edge reversed (nonoriented graph)
    write_edges_to_db(Edges).


get_first_vertex(FirstVertex) :-
    edge(FirstVertex, _),!. % Get the first vertex in database


get_unique_vertices(UniqueVertices) :- % Get all unique vertices from the database
	findall(V, edge(V, _), Vertices),
	findall(V2, edge(_, V2), Vertices2),
	append(Vertices, Vertices2, VerticesAll),
	list_to_set(VerticesAll, UniqueVertices).


list_to_set([], []).
list_to_set([H|T], S) :- list_to_set(T, S), member(H, S), !.
list_to_set([H|T], [H|S]) :- list_to_set(T, S).


convert_to_list([], []).
convert_to_list([(V1, V2)], [V1, V2]).
convert_to_list([(V1, V2)|T], [V1, V2|T2]) :- convert_to_list(T, T2).

% ------------------------------------------------------------------------------
% Solving the problem

solve(AllSolutions) :-
	get_first_vertex(FirstVertex), % Get the first vertex
	setof(Solution, search(FirstVertex, [FirstVertex], Solution), AllSolutions). % Find all paths


search(_, _, []).
search(Current, Visited, SolutionOut) :- % Maintain the order of the edges
    edge(Current, Next),
	not(member(Next, Visited)),
	(
		search(Current, [Next | Visited], Solution); % Recursively call for the next vertex (BFS)
		search(Next, [Next | Visited], Solution)	% Recursively call for the next vertex (DFS)
	),
	(
		Current @< Next -> % Store in alhabetic order
		sort([(Current, Next) | Solution], SolutionOut) % Append in the correct order (A,B) < (A,C)
		;
		sort([(Next, Current) | Solution], SolutionOut)
	).


% ------------------------------------------------------------------------------
% Printing the solution

print_spanning_tree([]).
print_spanning_tree([(V1, V2)]):- format('~w-~w', [V1, V2]). % Print last edge of solution
print_spanning_tree([(V1, V2)|T]) :-
	format('~w-~w ', [V1, V2]),
	print_spanning_tree(T).


print_solution([]).
print_solution([Tree|Trees]) :-
    print_spanning_tree(Tree),
	nl,
    print_solution(Trees). % Recursively call for the next tree

filter_solution([], []).
filter_solution([Tree|Trees], [Tree|FilteredTrees]) :-
	get_unique_vertices(VerticesGraph), % Get all vertices from input graph
	length(VerticesGraph, LengthGraph),	% Get the number of vertices
	convert_to_list(Tree, VerticesSol), % Converts edge tuples to one list
	sort(VerticesSol, VerticesSolution), % Remove duplicates
	length(VerticesSolution, LengthSolution),
	LengthSolution is LengthGraph,	% Check if the solution contains all vertices
	filter_solution(Trees, FilteredTrees).
filter_solution([_|Trees], FilteredTrees) :- % Skip tree that do not contain all vertices
	filter_solution(Trees, FilteredTrees).


% ------------------------------------------------------------------------------
% Main function

start :-
    prompt(_, ''),
    read_lines(LL),
	filter_empty(LL, Edges), % filter out empty rows
    (   
		Edges = [] ->  % If the input is empty, halt
        halt
    	;   
        write_edges_to_db(Edges), % Write the edges to the database
        solve(Solution), % Solve the problem
		filter_solution(Solution, FilteredSolution), % Filter out the paths not containing all vertices
        print_solution(FilteredSolution) % Filter out the paths not containing all vertices
    ).