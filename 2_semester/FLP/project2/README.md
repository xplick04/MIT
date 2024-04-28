### Maxim Plička (xplick04)
### Akademický rok 2023/24
### Minimální kostra grafu


# Popis použité metody
Projekt se zabývá nelezením minimální kostry grafu.

Po načtení vstupu jsou hrany uloženy do databáze pomocí dynamického predeikátu reprezentujícího hranu. Tyto hrany jsou uloženy v obou směrech, jelikož je na vstupu neorientovaný graf.

Hlavní průchod a nalezení všech cest spočívá v tom, že se vybere první uzel z databáze (pokud je graf souvislý nezáleží, z kterého uzlu se začíná). Z tohoto uzlu se poté postupně zanořuje pomocí dříve definovaných hran. Během zanořovánní se do listu `visited` postupně příkládají prozkoumané uzly, díky čemu se prohledávání nezacyklí. Prohledávání nedělá pouze kroky do hloubky ale i do šířky, díky čemu bude výstup obsahovat všechny možné kostry. Po každém prozkoumaném sousedovi se začne vynořovat a postupně se do seznamu `Solution` přidávají hrany v abecedním pořadí. Prohledávání je spouštěno pomocí `setof`, díky čemu je možné získat všechny možné cesty. 

Cesty jsou na závěr profiltrovány (zanechají se pouze cesty, které obsahují všechny uzly grafu) a poté jsou vypsány na výstup v požadovaném formátu.


# Návod k použití
Projekt se překládá pomocí příkazu `make`.

Po přeložení je projekt možné spustit pomocí `./flp23-proj < TESTFILE` , kde `TESTFILE` je soubor obsahující vstup do programu.

# Omezení
Není prováděna žádná kontrola vstupu, ovšem ignorují se prázdné řádky.