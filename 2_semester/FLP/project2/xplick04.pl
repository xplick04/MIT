/** 
* Project: 2. projekt FLP
* Author: Maxim Plička (xplick04)
* Date: 2024-04-25
*/

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


get_unique_vertices(UniqueVertices) :-
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


solve(AllPaths) :-
	get_first_vertex(FirstVertex), % Get the first vertex
	setof(Path, search(FirstVertex, [FirstVertex], Path), AllPaths).


search(_, _, []).
search(Current, Visited, Solution2) :- % Maintain the order of the edges
    edge(Current, Next),
	Current @< Next, % Alphabetical order
    not(member(Next, Visited)),
	(
		search(Current, [Next | Visited], Solution); % Recursively call for the next vertex
		search(Next, [Next | Visited], Solution)	% Recursively call for the next vertex
	),
	sort([(Current, Next) | Solution], Solution2).

search(Current, Visited, Solution2) :-	% Maintain the order of the edges
    edge(Next, Current),
	Current @> Next, % Alphabetical order
    not(member(Next, Visited)),
	(
		search(Current, [Next | Visited],Solution);
		search(Next, [Next | Visited], Solution)
	),
	sort([(Next, Current) | Solution], Solution2).

% ------------------------------------------------------------------------------


print_solution([]).
print_solution([(V1, V2)]):- format('~w-~w\n', [V1, V2]). % Print last edge of solution
print_solution([(V1, V2)|T]) :-
	format('~w-~w ', [V1, V2]),
	print_solution(T).


print_spanning_trees([]).
print_spanning_trees([Tree|Trees]) :-
	get_unique_vertices(VerticesGraph), % Get all vertices from input graph
	length(VerticesGraph, LengthGraph),	% Get the number of vertices
	convert_to_list(Tree, VerticesSol), % Converts edge tuples to one list
	sort(VerticesSol, VerticesSolution), % Remove duplicates
    length(VerticesSolution, LengthSolution),
    LengthSolution is LengthGraph,
    print_solution(Tree),
    print_spanning_trees(Trees). % Recursively call for the next tree
print_spanning_trees([_|Trees]) :- % Skip tree that do not contain all vertices
    print_spanning_trees(Trees).

% ------------------------------------------------------------------------------


start :-
    prompt(_, ''),
    read_lines(LL),
    (   
		LL = [] ->  % If the input is empty, halt
        halt
    	;   
		filter_empty(LL, Edges), % filter out empty rows
        write_edges_to_db(Edges), % Write the edges to the database
        solve(P), % Solve the problem
        print_spanning_trees(P) % Filter out the paths not containing all vertices
    ).