from itertools import islice


WIDTH, HEIGHT = 450, 450

N_CELLS = 9

CELL_SIZE = (WIDTH // N_CELLS, HEIGHT // N_CELLS)

# Convert 1D list to 2D list
def convert_list(lst, var_lst):
	it = iter(lst)
	return [list(islice(it, i)) for i in var_lst]
