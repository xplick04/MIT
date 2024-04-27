/** cte radky ze standardniho vstupu, konci na LF nebo EOF */
read_line(L,C) :-
	get_char(C),
	(
		isEOFEOL(C), L = [], !;
		C == ' ', read_line(L,_); % ignore spaces
		read_line(LL,_),
		[C|LL] = L
	).


/** testuje znak na EOF nebo LF */
isEOFEOL(C) :-
	C == end_of_file;
	(char_code(C,Code), Code==10).


read_lines(Ls) :-
	read_line(L,C),
	( C == end_of_file, Ls = [] ;
	  read_lines(LLs), Ls = [L|LLs]
	).

filter_empty([],[]).
filter_empty([[]|T],R) :- filter_empty(T,R).
filter_empty([H|T],[H|R]) :- filter_empty(T,R).


check_input([]).
check_input([H|T]) :- length(H, 2), check_input(T).
check_input([H|_]) :- length(H, _), write('Error: invalid input'), nl, halt.


:- dynamic(edge/2). % Declare edge/2 predicate as dynamic

write_edges_to_db([]).
write_edges_to_db([[V1,V2]|Edges]) :-
    assert(edge(V1, V2)), % Assert the edge to the database dynamically
	assert(edge(V2, V1)),
    write_edges_to_db(Edges).


get_first_vertex(FirstVertex) :-
    edge(FirstVertex, _),!. 


get_unique_vertices(UniqueVertices) :-
	findall(V, edge(V, _), Vertices),
	list_to_set(Vertices, UniqueVertices).

list_to_set([], []).
list_to_set([H|T], S) :- list_to_set(T, S), member(H, S), !.
list_to_set([H|T], [H|S]) :- list_to_set(T, S).


solve(P) :-
	get_first_vertex(FirstVertex), % Get the first vertex
	get_unique_vertices(UniqueVertices), % Get the unique vertices
	length(UniqueVertices, Num), % Number of vertices
	dfs(FirstVertex, [FirstVertex], P1, Num), % 2nd is visited
	reverse(P1, P). 



dfs(Current, Visited, Solution, UniqueVertices) :-
	length(Visited, L), L is UniqueVertices. % goal


dfs(Current, Visited, Solution, UniqueVertices) :-
	edge(Current, Next),
	\+ member(Next, Visited),
	append(Visited, [Next], NewVisited),
	dfs(Next, NewVisited, Solution, UniqueVertices).





start :-
    prompt(_, ''),

    read_lines(LL),
    filter_empty(LL, Edges), % filter out empty rows
    write_edges_to_db(Edges), % Write the edges to the database

	solve(P), % Solve the problem
	write(P), nl, % Write the solution

    halt.