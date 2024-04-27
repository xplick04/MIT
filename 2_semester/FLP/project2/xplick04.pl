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
	assert(edge(V2, V1)), % Assert the edge to the database dynamically
    write_edges_to_db(Edges).


get_first_vertex(FirstVertex) :-
    edge(FirstVertex, _),!. 


get_unique_vertices(UniqueVertices) :-
	findall(V, edge(V, _), Vertices),
	findall(V2, edge(_, V2), Vertices2),
	append(Vertices, Vertices2, Vertices3),
	list_to_set(Vertices3, UniqueVertices).

list_to_set([], []).
list_to_set([H|T], S) :- list_to_set(T, S), member(H, S), !.
list_to_set([H|T], [H|S]) :- list_to_set(T, S).


solve(AllPaths) :-
	get_first_vertex(FirstVertex), % Get the first vertex
	setof(Path, search(FirstVertex, [FirstVertex], Path), AllPaths).


search(_, Visited,[]) :-
	get_unique_vertices(UniqueVertices), % Get the unique vertices
    length(UniqueVertices, U),
    length(Visited, L),
    L is U.

search(Current, Visited, Solution2) :- % Maintain the order of the edges
    edge(Current, Next),
	Current @< Next,
    not(member(Next, Visited)),
	(
		dfs(Current, [Next | Visited], Solution); % Recursively call for the next vertex
		dfs(Next, [Next | Visited], Solution)	% Recursively call for the next vertex
	),
	sort([(Current, Next) | Solution], Solution2).

search(Current, Visited, Solution2) :-	% Maintain the order of the edges
    edge(Next, Current),
	Current @> Next,
    not(member(Next, Visited)),
	(
		search(Current, [Next | Visited],Solution);
		search(Next, [Next | Visited], Solution)
	),
	sort([(Next, Current) | Solution], Solution2).


print_solution([(V1, V2)]):- format('~w-~w\n', [V1, V2]). % Print last edge of solution
print_solution([(V1, V2)|T]) :-
	format('~w-~w ', [V1, V2]),
	print_solution(T).

start :-
    prompt(_, ''),

    read_lines(LL),
    filter_empty(LL, Edges), % filter out empty rows
    write_edges_to_db(Edges), % Write the edges to the database

	solve(P), % Solve the problem
	forall(member(X, P), print_solution(X)), % Print the solution

    halt.