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


isElement(X, [X|_]).
isElement(X, [_|T]) :- isElement(X, T).	


convert_to_list([], []).
convert_to_list([H|T], L) :- convert_to_list(T, L1), append(H, L1, L).


get_unique([], []).
get_unique([H|T], L) :- isElement(H, T), get_unique(T, L).
get_unique([H|T], [H|L]) :- get_unique(T, L).


start :-
		prompt(_, ''),
		read_lines(LL),
		filter_empty(LL,Edges), % filter out empty rows	
		check_input(Edges),	% check that there are two elements in each row
		write(Edges),
		nl,
		convert_to_list(Edges, List), % convert to one list
		get_unique(List, Vertices), % get unique vertices
		
		nl,
		halt.