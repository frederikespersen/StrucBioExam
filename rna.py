
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from matplotlib import pyplot as plt
plt.rcParams['font.family'] = 'STIXGeneral'

############################################################
################## Scoring parameters ######################
############################################################

base_pairs = {
    'CG': 1.0,
    'GC': 1.0,
    'GU': 1.0,
    'UG': 1.0,
    'AU': 1.0,
    'UA': 1.0}

stacked_base_pairs = {
    'CG': {'AA': 1.0, 'AC': 1.0, 'AG': 1.0, 'CA': 1.0, 'CC': 1.0, 'CU': 1.0, 'GA': 1.0, 'GG': 1.0, 'UC': 1.0, 'UU': 1.0, 'CG': 3.7, 'GC': 4.6, 'GU': 3.4, 'UG': 2.7, 'AU': 3.4, 'UA': 3.4},
    'GC': {'AA': 1.0, 'AC': 1.0, 'AG': 1.0, 'CA': 1.0, 'CC': 1.0, 'CU': 1.0, 'GA': 1.0, 'GG': 1.0, 'UC': 1.0, 'UU': 1.0, 'CG': 4.6, 'GC': 4.7, 'GU': 3.8, 'UG': 2.8, 'AU': 3.5, 'UA': 3.7},
    'GU': {'AA': 1.0, 'AC': 1.0, 'AG': 1.0, 'CA': 1.0, 'CC': 1.0, 'CU': 1.0, 'GA': 1.0, 'GG': 1.0, 'UC': 1.0, 'UU': 1.0, 'CG': 3.4, 'GC': 3.8, 'GU': 0.0, 'UG': 1.8, 'AU': 2.7, 'UA': 2.6},
    'UG': {'AA': 1.0, 'AC': 1.0, 'AG': 1.0, 'CA': 1.0, 'CC': 1.0, 'CU': 1.0, 'GA': 1.0, 'GG': 1.0, 'UC': 1.0, 'UU': 1.0, 'CG': 2.7, 'GC': 2.8, 'GU': 1.8, 'UG': 1.0, 'AU': 1.9, 'UA': 2.3},
    'AU': {'AA': 1.0, 'AC': 1.0, 'AG': 1.0, 'CA': 1.0, 'CC': 1.0, 'CU': 1.0, 'GA': 1.0, 'GG': 1.0, 'UC': 1.0, 'UU': 1.0, 'CG': 3.4, 'GC': 3.5, 'GU': 2.7, 'UG': 1.9, 'AU': 2.4, 'UA': 2.2},
    'UA': {'AA': 1.0, 'AC': 1.0, 'AG': 1.0, 'CA': 1.0, 'CC': 1.0, 'CU': 1.0, 'GA': 1.0, 'GG': 1.0, 'UC': 1.0, 'UU': 1.0, 'CG': 3.4, 'GC': 3.7, 'GU': 2.6, 'UG': 2.3, 'AU': 2.2, 'UA': 2.6}}



############################################################
####################### Nussinov ###########################
############################################################

class Nussinov():

    def __init__(self, rna: str, min_loop_size=3, scoring='standard'):
        """

        A class for executing the Nussinov algorithm.

        --------------------------------------------------------------------------------

        Parameters
        ----------

            `rna`: `str`
                The RNA sequence to run the Nussinov algorithm on
                [May only contain 'A', 'C', 'G', 'U']

            `min_loop_size`: `int`
                The minimum amount of nucleotides required for a loop.

            `scoring`: `str`
                The scoring method to use ['standard': Standard nussinov, 'stacking': Using stacking energies]


        """
        # Checking RNA sequence
        check_rna(rna)
        self.seq = rna.upper()

        # Nussinov algorithm settings
        self.n = len(self.seq)
        self.min_loop_size = min_loop_size
        self.scoring = scoring.lower()

        # Solving
        self.solve()
        self.solve_dot_bracket()

    def __str__(self):
        """

        Printing will yield the sequence and the backtracked dot bracket notation.
        
        """
        return self.seq + "\n" + self.dot_bracket

    def solve(self):
        """
        
        Method that solves the Nussinov algorithm matrix under the given settings.
        
        """
        
        # Initialising scoring matrix
        self.scores = np.zeros(shape=(self.n,self.n))

        # Looping over matrix diagonals
        for offset in range(self.min_loop_size, self.n):

            # Looping over columns 
            for j in range(offset, self.n):

                # Deducing row (i) from diagonal (offset) and column (j)
                i = j - offset
                self.scores[i, j] = self.max_cell_score(i, j)

        # Saving best score as attribute
        self.best_score = round(self.scores[0, -1])


    def solve_dot_bracket(self):
        """
        
        Method that backtracks the dot bracket-notation for an optimal solution (i.e. one of possibly several) to the filled in Nussinov algorithm matrix.
        
        """
        # Initialising dot bracket notation with all dots
        self.dot_bracket = ['.']*self.n

        # Filling in brackets with backtracking
        self.backtrack()

        # Formatting as one string
        self.dot_bracket = ''.join(self.dot_bracket)


    def backtrack(self, i:int=None, j:int=None):
        """
        
        Method that backtracks one optimal solution (i.e. one of possibly several) to the filled in Nussinov algorithm matrix.
        Has option to start backtracking from a specific cell.
        Inserts brackets into an existing attribute `self.dot_bracket` (See `solve_dot_bracket()`)

        Priorites backtrakcing moves in the order:
        1. Left
        2. Down
        3. Diagonal
        4. Bifurcation (lowest k)

        --------------------------------------------------------------------------------

        Parameters
        ----------

            `i`: `int`
                The row index to start backtracking from (Default `None`: Starting from upper right corner)

            `j`: `int`
                The column index to start backtracking from (Default `None`: Starting from upper right corner)
        
        Returns
        -------

            `score`: `float`
                The maximum score for (`i`,`j`)
        
        """
        # Backtracking from upper left corner (best score)
        if (i is None) and (j is None):
            i = 0
            j = self.n - 1
        assert (i is not None) and (j is not None), "Either both i and j must be defined, or neither (full matrix)"
        
        # Continuing backtracking until a 0-score is reached
        while self.scores[i, j] > 0:
            # Checking for possible moves
            
            # Left move
            if self.scores[i, j] == self.left_score(i, j):
                j -= 1
            
            # Down move
            elif self.scores[i, j] == self.down_score(i, j):
                i += 1
            
            # Diagonal move => Base pair
            elif self.scores[i, j] == self.diagonal_score(i, j):
                self.dot_bracket[i] = '('
                self.dot_bracket[j] = ')'
                i += 1
                j -= 1

            # Bifurcation => Backtrack substrings seperately
            elif self.scores[i, j] == max(self.bifurcation_scores(i, j)):
                k = self.bifurcation_scores(i, j, return_k=True)
                self.backtrack(i, k)
                self.backtrack(k+1, j)
                break

            else:
                raise ValueError("Backtracking failed!")


    def max_cell_score(self, i, j):
        """
        
        Method that returns the maximum (`i`,`j`) cell score.

        --------------------------------------------------------------------------------

        Parameters
        ----------

            `i`: `int`
                The row index to evaluate

            `j`: `int`
                The column index to evaluate
        
        Returns
        -------

            `score`: `float`
                The maximum score for (`i`,`j`)
        
        """

        # Choosing the score of the maximum scoring move
        return max(self.left_score(i,j),
                   self.down_score(i, j),
                   self.diagonal_score(i, j),
                   *self.bifurcation_scores(i, j))
    

    def left_score(self, i, j):
        """
        
        Method that returns the (`i`,`j`) cell score if derived from the left cell.

        --------------------------------------------------------------------------------

        Parameters
        ----------

            `i`: `int`
                The row index to evaluate

            `j`: `int`
                The column index to evaluate
        
        Returns
        -------

            `score`: `float`
                The cell score for (`i`,`j`) derived from the left cell
        
        """
        return self.scores[i, j-1]
    

    def down_score(self, i, j):
        """
        
        Method that returns the (`i`,`j`) cell score if derived from the below cell.

        --------------------------------------------------------------------------------

        Parameters
        ----------

            `i`: `int`
                The row index to evaluate

            `j`: `int`
                The column index to evaluate
        
        Returns
        -------

            `score`: `float`
                The cell score for (`i`,`j`) derived from the below cell
        
        """
        return self.scores[i+1,j]
    

    def bifurcation_scores(self, i, j, return_k=False):
        """
        
        Method that returns the (`i`,`j`) cell score if derived from a bifurcation.
        Has a option to return the (lowest) `k` index of the maximum score.

        --------------------------------------------------------------------------------

        Parameters
        ----------

            `i`: `int`
                The row index to evaluate

            `j`: `int`
                The column index to evaluate

            `return_k`: `bool`
                Whether to return the `k` index of the maximum score (`True`) instead of the maximum score (`False`, default)
        
        Returns
        -------

        If `return_k == False:`
            `score`: `float`
                The cell score for (`i`,`j`) derived from the below cell
        
        If `return_k == True:`
            `k`: `int`
                The `k`- (lowest) index from which the maximum score was derived
        
        """
        if return_k:
            k_idx = np.argmax([self.scores[i,k] + self.scores[k+1,j] for k in range(i+1, j-1)])
            return [*range(i+1, j-1)][k_idx]
        else:
            return [self.scores[i,k] + self.scores[k+1,j] for k in range(i+1, j-1)]
        

    def diagonal_score(self, i, j):
        """
        
        Method that returns the (`i`,`j`) cell score if derived from the lower left cell.
        Scoring method depends on `self.scoring` (see `__init__`).

        --------------------------------------------------------------------------------

        Parameters
        ----------

            `i`: `int`
                The row index to evaluate

            `j`: `int`
                The column index to evaluate
        
        Returns
        -------

            `score`: `float`
                The cell score for (`i`,`j`) derived from the lower left cell
        
        """
        if self.scoring == 'standard':
            return self.scores[i+1,j-1] + self.base_pair_score(i, j)
        elif self.scoring == 'stacking':
            return self.scores[i+1,j-1] + self.stacked_base_pairs_score(i, j)


    def base_pair_score(self, i, j):
        """
        
        Method that returns a base pairing scoring according to `base_pairs` dict if contained herein, else 0.
        
        """
        bp = self.seq[i] + self.seq[j]
        try:
            return base_pairs[bp]
        except KeyError:
            return 0
        

    def stacked_base_pairs_score(self, i, j):
        """
        
        Method that returns a base pairing scoring according to `stacked_base_pairs` dict if contained herein, else 0.
        
        """
        bp = self.seq[i] + self.seq[j]
        bp_stacked = self.seq[i+1] + self.seq[j-1]
        try:
            return stacked_base_pairs[bp][bp_stacked]
        except KeyError:
            return 0
        


############################################################
###################### Evaluation ##########################
############################################################

def base_pair_distance(struc1:str, struc2:str):
    """
    
    Takes two RNA structures in dot bracket notation, returns their base pair distance

    --------------------------------------------------------------------------------

    Parameters
    ----------

        `struc1`: `str`
            An RNA secondary structure in dot bracket notation
            [May only contain `.`, `(`, or `)`]

        `struc2`: `str`
            An RNA secondary structure in dot bracket notation
            [May only contain `.`, `(`, or `)`]

    Returns
    -------

        `bp_dist`: `int`
            The base pair distance between the two structures

    """

    # Finding base pairs in both structures
    bp_struc1 = list_base_pairs(struc1)
    bp_struc2 = list_base_pairs(struc2)

    # Finding base pair distance
    bp_dist = len(set(bp_struc1).difference(bp_struc2)) + len(set(bp_struc2).difference(bp_struc1))

    return bp_dist

def list_base_pairs(struc:str):
    """
    
    Takes a RNA structure in dot bracket notation, returns all base pairs as tuples of sequence positions.

    --------------------------------------------------------------------------------

    Parameters
    ----------

        `struc`: `str`
            An RNA secondary structure in dot bracket notation
            [May only contain `.`, `(`, or `)`]

    Returns
    -------

        `bps`: `list[tuple]`
            A list of base pairs as tuples of sequences positions (index 0) (Smallest position first)

    """

    # Checking correct input
    check_dot_bracket(struc)

    # Initiating base pair list
    bps = []

    # Initating container for opened base pair partners (i.e. '(')
    open_bps = []
    
    for i, s in enumerate(struc):

        # Opening base pair
        if s == '(':
            open_bps.append(i)

        # Closing base pair
        if s == ')':
            bps.append((open_bps.pop(), i))
    
    return bps

def check_rna(rna:str):
    """
    
    Takes a RNA sequence, asserts that it is valid notation.
    Raises assertion if not.

    --------------------------------------------------------------------------------

    Parameters
    ----------

        `rna`: `str`
            An RNA sequence
            [May only contain 'A', 'C', 'G', 'U']

    """
    # Assuring correct symbols
    assert all([n in ['A', 'G', 'C', 'U'] for n in set([*rna.upper()])]), "RNA sequence ust only contain nucleotides 'A', 'G', 'C', and 'U'!"

def check_dot_bracket(struc:str):
    """
    
    Takes a dot bracket notation structure, asserts that it is valid notation.
    Raises assertion if not.

    --------------------------------------------------------------------------------

    Parameters
    ----------

        `struc`: `str`
            An RNA secondary structure in dot bracket notation
            [May only contain `.`, `(`, or `)`]

    """
    # Assuring correct symbols
    assert all([s in ['.', '(', ')'] for s in set([*struc])]), "Structures must only contain '.', '(', and ')'!"

    # Checking equal '(' and ')'
    assert sum([1 if s == '(' else -1 if s == ')' else 0 for s in struc]) == 0, "Unequal amount of '(', and ')' in structure!"



############################################################
####################### Test data ##########################
############################################################

rnafold =\
{ 'seq_tgw325_001': { 'energy': -41.6,
                      'seq': 'AGCAUCGCAAUGGCGAGCUCUACGAGACACAUGUUGCGACUGCCGAUAUAUUCUUACUCAGCCAAGCCUCAUGGGACCUUUGAGGCCAUGGGGCCCCGUUGGCUAUGUAGCGGCUCAGCU',
                      'struct': '....((((((((.....(((...))).....))))))))..((((.(((((........(((((((((((((((..(((...))))))))))))....))))))))))).))))......'},
  'seq_tgw325_002': { 'energy': -36.0,
                      'seq': 'GAAAGGUGCGUCCUGGGAUUUCGCGAGGUUUGACAAUUUGGAGAGACUGCAGAGGGGUCACCUAAUCACAGGUCGCGGUGACGGUCACCUAGAAGUGUUCGCUUAAGCUGGGGUCUCGUC',
                      'struct': '....(.((((.((((.((((.....((((..(((..((((.((...)).))))...))))))))))).)))).)))).)(((((...(((((((((....))))...)))))...)))))'},
  'seq_tgw325_003': { 'energy': -27.6,
                      'seq': 'UGGUGCAGUCCUCUCCUUGAACUCUUCGGUCACCCCCGGUAGAAGCUGUUAGAGUCUCGCGUGUCCCGCAGUUUAUCAUACUGUAGACACAUAUGUGUAAAGUAGAAAUUCGUAGGGACU',
                      'struct': '..(((.((..((((........(((((((......)))).))).......)))).)))))..(((((...(..(.((.((((....((((....))))..)))))).)..)...))))).'},
  'seq_tgw325_004': { 'energy': -24.7,
                      'seq': 'CAUUCAGCUACCAUACGACAAGUCCGGAUCGCAGAUUUACGCCGCCAGAGGUGCUUCACGUAUUCAGUGAUGCUGUCCCACGCUAACGAAGUCCUACCAAAGUACAUCGUUAGGUACCGC',
                      'struct': '.........................((((.(((......((((......))))..((((.......))))))).)))).((.(((((((.((.((.....)).)).))))))))).....'},
  'seq_tgw325_005': { 'energy': -24.4,
                      'seq': 'CGAGUGUCCGGUCACAAAGAAAGUCAAUACACCUGUGUUUCGGAUCUAACCAGCCGUGCGAGUUAGUAUUGAUCACACUCGACCGUAAUUAGUACUUAGUCGGCAAAACACGCUACACCA',
                      'struct': '..((.((((((.((((.................))))..))))))))....(((.((((((.(.(((((((((.((........)).))))))))).).)))......))))))......'},
  'seq_tgw325_006': { 'energy': -28.5,
                      'seq': 'AAUACGCACUCCGCCCGCAGGAGGUCACAAAAUCCCCGAGGUGCCUGGUUAUAAUUAUUAGGGACCCAGUUCUUUCACAGACAGCGUCGGAGCCCUAGAUUCCGGUGCCUGCUAUCUUUG',
                      'struct': '.....(((((((.......))))(((..........(..(((.((((((.......)))))).)))..)..........))).(((((((((.......))))))))).)))........'},
  'seq_tgw325_007': { 'energy': -26.2,
                      'seq': 'AGCGUAUUCACCCUUCGAGGGCCUAGCCCUACGUCCCUUCCGUGUGCCUCCCGGCAGAAAGUAGUCGUACUGUUAGAAGUAUACGUUUUUUGGACUCCACAACCGUCUAACACGAACCGC',
                      'struct': '.(((..(((......(((((((...))))).)).......(((((((.((..(((((...(....)...))))).)).)))))))....((((((.........))))))...))).)))'},
  'seq_tgw325_008': { 'energy': -19.9,
                      'seq': 'UAUCACUAUCAGGCACUUAAUACAGUACUGUCUGUACUCACUUCACGGAGUAGGUCGGUUGAGAAUCCAUAGUAACUCUUCCACGCCCUAAGCCACCCAUAGACGCUGCGGCCCUCUGAC',
                      'struct': '........(((((........((((......))))((((........))))..((.((..(((............)))..))))(((...(((...........)))..)))..))))).'},
  'seq_tgw325_009': { 'energy': -35.8,
                      'seq': 'CGACAGGUUAUGUGCUAGAGCCGCGGUCCCGCUCUCGGGCUCCGUCCAAUACAGCGCAAGGCUUUUGAGGCAGUUAUAGCAGGACGCUGUCUUGGUGCAUUGGGUUCGUUGUCUCGGACG',
                      'struct': '..(((.....)))....(((((.(((........))))))))(((((...(((((((((.((..((((((((((...........)))))))))).)).)))....))))))...)))))'},
  'seq_tgw325_010': { 'energy': -27.8,
                      'seq': 'GAUAUGCCAGCAUAAUACCAAGACCGCGUUACCGCCGUACUGUAGCCCCUCGUUAGUAUAGUGUAGCGUCGUUGCAUAUACGGCCAAAGUGCCCACUAAGCGAGCCGGGUUGACCCCUAU',
                      'struct': '..((((....))))....((..((.(((....))).))..))((((((((((((.(((((.((((((...)))))))))))(((......))).....))))))..))))))........'},
  'seq_tgw325_011': { 'energy': -21.4,
                      'seq': 'GGACUAAUAUACAGCAUAAAGAAAGCCGCACCCUUUUCAAGAUCAUGGUACAAACACAAAAGAGACCGGGUUGAGACCUAAAAGACAUCCUAAGGCCGUGUGCUGCUGGUCUCGACGACG',
                      'struct': '.........................(((..(.(((((....................))))).)..)))(((((((((....((((((((...))..)))).))...)))))))))....'},
  'seq_tgw325_012': { 'energy': -21.9,
                      'seq': 'CUAUGGGACUUUAGCGCGCGUUCGCCAUUGAAGUGCAUGUAGCACAGGCAUCACAGAGAGAUCAGUCAUAGAGAAACUACAGACUAAUCAGGAAACGGGAGAAUAGUAACGCGGCCUAAU',
                      'struct': '..........((((..((((((((((......((((.....)))).)))..........(((.((((.(((.....)))..)))).)))................).))))))..)))).'},
  'seq_tgw325_013': { 'energy': -30.6,
                      'seq': 'AUUUGAAUGUCCCGCCUUUGUCAAUGUGACGACAGGAUUAUCUCCAUGCCAUCAUCAGGAUAAUGGAUGUGUAGUUGCCGCCGCGAGAACUGGGGUCAUUAGGGAUGAGCGUGGGCUGAG',
                      'struct': '..............(((((((((...)))))).)))......(((((((..(((((....((((((....((..((((....))))..)).....))))))..)))))))))))).....'},
  'seq_tgw325_014': { 'energy': -23.4,
                      'seq': 'CAUACUAAGGCACAGGAGGCACUGACAAUACUAUUUUCCCGGACUACCUCACUAUUCGGUGGACAACCCAAAGCCAGUAAAGCUUGGCUACGGGAUAUUAAGUUGAGGACUAUAUAGCGG',
                      'struct': '............(((......)))......((((..(((..((((...(((((....)))))....(((..((((((......))))))..)))......))))..)))....))))...'},
  'seq_tgw325_015': { 'energy': -32.6,
                      'seq': 'CCUGUGAGGCGCCGCGUGGAGGUUCUCAGUCGAACAGCAGUCUGGUGCGGGACCUCACAACAGCUACUCUUGAGCGUUCCGCCAAUUAAACUAACAAGCAGUUGGCAAAAGAUCGGUGAC',
                      'struct': '..(((((((..(((((.........((((.(........).)))))))))..)))))))...(((.(....))))....((((.(((...(((((.....))))).....))).))))..'},
  'seq_tgw325_016': { 'energy': -34.0,
                      'seq': 'GUUCCGAUAGGCACGCGGUCUAGCCAGUCACUGUUGUCCUGGUGCCUCAGCGCCCUGUCAUCCGUAAUGAUAUUGGGUCUCGGCCGGUCUGUAGAUAGUCCAGUGCAUGGAAGGUGGCGU',
                      'struct': '........((((((..((.(.(((........)))).))..))))))..(((((...((.(((...(((.(((((((((((((.....))).))))...)))))))))))).)).)))))'},
  'seq_tgw325_017': { 'energy': -23.6,
                      'seq': 'CGGGGUCACAGCUAUCCGUCCUUAUGAAUAUUGUUAAUGAGUGAGUACAUCCCGCGUCCCUCAACCUCGUAAGGAUACUCAGACGUAUCUUAGUCAAUCUUUUUGGUGAUGCAACCCAGG',
                      'struct': '..((((((..((((...((((((((((...(((.......(((.(.....).)))......)))..)))))))))).....(((........)))........))))..))..))))...'},
  'seq_tgw325_018': { 'energy': -32.9,
                      'seq': 'CCCAUGCCCCUCCAAUAGGUCAGUGGCGCGGACUCUUAUCGGGGGCUACGAUUGAAUCGGAUGAUAGCUGGUAUAGCCACGUGUAGAAAGUCACGCGAUGAAGCUCCUGAUUUCGGCACA',
                      'struct': '.((.((((.((((....))..)).)))).)).((...((((((((((.((((...)))).......((((...))))(((((((........))))).)).))))))))))...))....'},
  'seq_tgw325_019': { 'energy': -30.5,
                      'seq': 'AGCCACGUCCGCACUACGUCAAAUAACGCCCCAAGAUGUCAUUUGAUCACAUUGGGUACAUUCCAGCCCUGACUCCAUGGGGCUCGAAGAGGACGCGAGCCUCAGAAGUUCAUCUAUGCU',
                      'struct': '.(((.(((((.(....(((......))).(((((..((((....)).))..)))))....(((.((((((........)))))).)))).))))).).))....................'},
  'seq_tgw325_020': { 'energy': -26.1,
                      'seq': 'GAAUGCAAGCAACGUUAAGCCCUUGCAGGAACGCAGAGGCUCGUAGCAAACACGGAACCCUGUUUAACUUUCCACCCGAAUCGACAGGUCUAUCGUCAUUGUAUCAUACAUUGUAGCAUC',
                      'struct': '..((((..((.(((...((((.((((......)))).))))))).))....((((...((((((.....(((.....)))..))))))....))))((.((((...)))).))..)))).'},
  'seq_tgw325_021': { 'energy': -26.3,
                      'seq': 'CCGGUCGAUUAUGGAUAAGUGCCGACGGUACCUGUCUCUUCAAGUCCGAGAGUAGUACGAGUUCCCCAGAGUAAGUCCGCUGCGGCGAUAAAGUGGAACCUUCCGCAGAAUUUCACAAUC',
                      'struct': '.(((.(......(((((.(((((...))))).)))))......).)))..........((((((..(.(((....((((((..........))))))...))).)..)))))).......'},
  'seq_tgw325_022': { 'energy': -30.7,
                      'seq': 'AAACACCAUAUGCACGUCCAUCGCUAGCCGGCCCGCUCUACUGUUGAUCGUGGUCGGCGAACCCCUAGUCAGACCGUCUGGAACGGUAAACUUACCACGAUGUGUUACAUAAAGCCCCGU',
                      'struct': '...........(((((((........(((((((((.((.......)).)).)))))))......((((.(.....).))))...((((....))))..)))))))...............'},
  'seq_tgw325_023': { 'energy': -27.7,
                      'seq': 'GCGGUCUCUUUUCCUUGCCGAGCGGUAAUCGCUUUCUCAAUGGGCCUAUUAAAGACGAGUCCUUAUUGAUGGAACCAUCGACAGACUUUUAUAGCAAGCAAUAGUUCAGCACCGCUCACG',
                      'struct': '(((((((((.((.(((((.(((((.....)))))..((.((((..(((((((((.......))..)))))))..)))).))............))))).)).))...)).))))).....'},
  'seq_tgw325_024': { 'energy': -43.2,
                      'seq': 'UGCCCAGGGCUGCGUCAUGAGCAGCCGCUGGGGGCACCCUAAUCAGUUGAUUACUGGCCUAGAUUCAUUAUGACAAUCAACGUCAGGCGGCCUUGUCAAUGUCCGAACAAGCCGCAUGCA',
                      'struct': '..(((((((((((.......)))))).)))))(((.(..(((((....)))))..))))...........((((.......)))).(((((.(((((.......).))))))))).....'},
  'seq_tgw325_025': { 'energy': -38.4,
                      'seq': 'CUGCCACGGGAGCCUAAGCUGCUCCAGUACUGGACGUCUGAUGGGAACUACAUUAACAAUGACCGUAAACUCUGGAAGGGUUGGCUUGUGAGGAGGAAACUGAUUCCUCCUCAAAUUCUG',
                      'struct': '.........(((((..((((..((((((((.((.(((.(((((.......)))))...))).)))))....)))))..))))))))).((((((((((.....)))))))))).......'},
  'seq_tgw325_026': { 'energy': -16.2,
                      'seq': 'UCGAGACACAGCCUCCGUGGAACUGGCUUGCUACAGACCUUAUGAGACCAUACCGAAGCCCACUAAGCGACAUAGAACUUAAUCUUGAAAAAUAAUGACGGUAGGCACCAAUUAGUGCAU',
                      'struct': '..........((((((((((..(((........))).)).((((....))))..(..((.......))..)..(((......)))............)))).))))..............'},
  'seq_tgw325_027': { 'energy': -28.2,
                      'seq': 'CUCUACAUUGGACUUGUAGUAUAGGGCUGCGACACUGCUAGAUUGGCUUUAAAGUUGACCUCAAGACUGAGGGUAUAAUAGGUCUACCCGCUUUACCGUUGCUGCGGAGUUCUACAUGAU',
                      'struct': '.....(((.((((((((((((..((...(((.....((((...))))...........(((((....)))))........((....)))))....))..)))))).))))))...)))..'},
  'seq_tgw325_028': { 'energy': -25.2,
                      'seq': 'GGAAUCACCGUAUAACAGGGACUUGUUUUAAAUCAGACGACGUUCUUAUGGGAUAUGCGAUUCGAGAAAUGUUGACAGGUCGGCUUUGGUUCCGUCAGUACGUAGCAAACAUGGCAAUGG',
                      'struct': '(((((((.(((((..(((((((((((((......)))))).)))))..))..)))))((((((((......))))...))))....)))))))((((((.........)).)))).....'},
  'seq_tgw325_029': { 'energy': -38.8,
                      'seq': 'AUGGGUGUAGUCUACAAGCCGCCCAGGGAUACAGGGUCCUUUGGUGCCCAGUGGCAUUAAGCUGAACAUUCCCUGUGAUCUAAGGUGGCGUAUGACGGGAGGCGGUGACCAUGUAUUUUC',
                      'struct': '((((.....((((....((((((..(((.(((((((((.(((((((((....)))))))))..)).....))))))).)))..))))))..........)))).....))))........'},
  'seq_tgw325_030': { 'energy': -27.4,
                      'seq': 'UAUCCAGAUAGAUUUUACUCUUACGAGAGCCGAUUGCCAAAGACGUGUCGAAAGUGUGAAAGGCUCUCUUUAGAGAUAUAUGCAAAUCCUAGAUCGUGCAAAUGAUUCCCGGCUGGAAAC',
                      'struct': '..(((((..........((((...(((((((..(((((...((....))....).))))..)))))))...))))............((..((((((....))))))...)))))))...'},
  'seq_tgw325_031': { 'energy': -27.0,
                      'seq': 'UUUCAGUGGUCAUAAAGCUGAAUAAGGAUCAUCCGCUCUAGCAAGUAAGCUUAGUGUCUUUAACGAUGAUCUGACCUCCGUCCGGCGUCCUUGAUAGGAUUCGCUCCUCCUAAUGCGCAC',
                      'struct': '.((((((.........))))))...((((((((((((..(((......))).))))........))))))))(((....))).((((((((....))))..))))...............'},
  'seq_tgw325_032': { 'energy': -34.5,
                      'seq': 'ACAGCAUAGGGGGGUCACAUUAAAGGCUCCGUCUCUGGACGAAGAAUGAGGCUGCUGGUACACUUGAGCCAAUACUACUAACCCUUAUUGGGUUGUUUAGUUCACAAUAAGUUUACCGGA',
                      'struct': '..........((((((........))))))(((((............)))))..((((((.(((((.......((((.((((((.....))))))..))))......))))).)))))).'},
  'seq_tgw325_033': { 'energy': -30.6,
                      'seq': 'AUGUAGUCUUUUGUGCGAGCCGGGGGUGUUGACGCCAACUAGCCUUAGGUAUCAGAUUACAACUUCCUUAUCAUUAUUGGUCAGGGACAGAACUCAUAUCAUGGGUAAUAUGCGCCAUGC',
                      'struct': '.((((((((.......(((((.(((((((((....))))..))))).))).))))))))))..(((((.((((....)))).)))))............((((((((...))).))))).'},
  'seq_tgw325_034': { 'energy': -32.7,
                      'seq': 'CAGCGCCCUACGAGUAUUAUAGAGGCGCUUUUCAUGAUGGGCUUAUAGCGCGCCUGGAUUAGAGAUCUGAGCUGGGCGUAAAAGUAGAGGCACAAGAAACACGCAACGUGCUAGACCUGA',
                      'struct': '.(((((((((.........))).)))))).........(((((...)))((((((((.((((....)))).)))))))).........(((((..............)))))...))...'},
  'seq_tgw325_035': { 'energy': -31.7,
                      'seq': 'GCUAGCCAAGAUAGGUCAUUAGUUUACCUUUAGUUCAUUAACUGCGGGCGGGAAAGACUUCGCCCAGCGGCCGGGAUGCCAUUUCACGGACUAGCAUCACAUUAACUCUGUGACCGGGUC',
                      'struct': '(((((((......))...(((((..((.....))..)))))(((((((((((......))))))).))))((((((......))).))).))))).(((((.......))))).......'},
  'seq_tgw325_036': { 'energy': -39.7,
                      'seq': 'GGACACGCGACACUCAUGAUCGCCGGGCACUGUGCAGCACAACGGGCAGCGGGUAGGCUGUCUCCAUGUCUCCUGGUGCUCACUCUCACGUCCCUCCAUGUCAUUGUCGCGAGCGUAAAA',
                      'struct': '.....(((((((...(((((....(((.((.(((.((......(((((.((((.((((((....)).)))))))).)))))...)))))))))).....)))))))))))).........'},
  'seq_tgw325_037': { 'energy': -21.7,
                      'seq': 'AAAUUAAAGGACAACAGACCAAUUAAGCUGUAUGAUAAUGUUGUUCACCAAUACAACGUAUAUUGGAUAAGUGAUGCUUGGGAAUGUGUUUGCAAUUCCCGUUUGAAAGAUCAGAAUGAG',
                      'struct': '........((((((((.....((........)).....)))))))).((((((.......)))))).............((((((((....)).)))))).(((((....))))).....'},
  'seq_tgw325_038': { 'energy': -29.2,
                      'seq': 'CCACCGUUCUAUAUCACGGAGCACGGGUCGCCAUUGCAGCGUGUGUGUGCGCGAUAUGAACCAUAGAUAAACCUAAGGCUAUGGUAUGACUAAUUCUGAGACGAUCAACGUAUUUGGAUG',
                      'struct': '.........((((((.((..(((((.((.((....)).)).)))))...)).)))))).(((((((............))))))).........(((((((((.....))).))))))..'},
  'seq_tgw325_039': { 'energy': -26.2,
                      'seq': 'GGGGGGAGCGGUAAGGAGCAUCCAUGAGUCUAUCGGAUAUGAGCCGAGUUGCGAGCUGGAAUGAGCGCAACACGUAAGUAAUAAGAACUUUUGUGGCAGACAAUUUAAAUAUGUUCCAAA',
                      'struct': '..............(((((((...(((((((.((((.......))))((((((..(......)..))))))(((.((((.......)))).)))...))))...)))...)))))))...'},
  'seq_tgw325_040': { 'energy': -23.7,
                      'seq': 'CAAUAAUCCACGGGUAUGGAAGAAUCACGGGUCCGGAUUAUGAUUAGAUCAAGCGUAGAUUGGUAGUAGGUUGAGUGGAGAAACGCAGCUAAAUCCGAUUGGUUUCCUUUUCAGAGGAUA',
                      'struct': '.....((((...((...((((.....((.((((.(((((........(((((.......)))))....(((((.((......)).))))).))))))))).))))))...))...)))).'},
  'seq_tgw325_041': { 'energy': -28.9,
                      'seq': 'GACAGUUGAGGAUGUAAUCAGGCUACUUGUGGAUCUACUAUGGGGCGGAUGCACAGGUCCGUCGGCGUGAUGACACGCUAGAGGCCGAUGAAGUGAAGUUAGAAGCUAUGGGAGAGAAAC',
                      'struct': '........((..(((.((((.((((((((((.((((.((....)).)))).))))))).....))).)))).)))..)).....((.((..(((..........))))).))........'},
  'seq_tgw325_042': { 'energy': -38.2,
                      'seq': 'GGAAUGCGACUCGCGACUACGUCUGGACCGAACGCACUGCGGGCCGACCGAUGUGUCGUUGGGCGUUCUAAGGCUCGCAUCAACUUCGCAUGAACGAUGUCAGUGUUUCAUACGAGCUGG',
                      'struct': '(((((((..((.(((((.(((((.((.(.(..(((...)))..).).)))))))))))).)))))))))..(((((((((..((.(((......))).))..))).......))))))..'},
  'seq_tgw325_043': { 'energy': -23.4,
                      'seq': 'AUUAUGGUUUGACGAAGGGCCAUGAGUCACGUCCUCGAUAGGGCCGAGUAUCCAGGCAUCCUAGUGAAGAUACGUGUCAGCCUAUAAUCCCGUUACACACUUACCCUACCAAACAAGCUC',
                      'struct': '.....((((((..(.((((...(((((...(((.(((.(((((((.........)))..)))).))).)))..((((.(((..........)))))))))))))))).)....)))))).'},
  'seq_tgw325_044': { 'energy': -46.6,
                      'seq': 'UGUCGGACGGAAUUGUCGCGGUAUUUCGCAAGCGAACCUGUGAGUAGACAGACCAUUCGUUCCCUACUGGGACGGGUGGUGCUGCCGCUUCGGUCUGUGACGAGAGCGUUCAGAAUGUGG',
                      'struct': '....(((((...(((((((((....(((.(((((...((((......))))(((((((((.(((....)))))))))))).....)))))))).)))))))))...))))).........'},
  'seq_tgw325_045': { 'energy': -28.3,
                      'seq': 'GUCUGCGCCUUGAUUGGAGCCGAUGUCACUACAUAUCGGUAUACGUGCAUCGAGAUCCUGAACAUUAAGAAGGACGUAAUUACUUAAGAGAAUGGUGUAGCAGCCAGUGGCUCAAAUCGU',
                      'struct': '(.(((((((.........((((((((......)))))))).(((((.(.((..(((.......)))..)).).)))))...............))))))))((((...))))........'},
  'seq_tgw325_046': { 'energy': -24.3,
                      'seq': 'AUGCUCUGUGUGUCGGCGUUAAGAGAGUUCGAAGGAAACUGUGUAUCCGAACUCAAUAAGCUCCCUCUUGAAAGCCGGUCCCGUAACUUAUCAAUGGUCCGGAUAGGCCUCGUCGUAAUG',
                      'struct': '..((.((((.(((((((.((((((((((((...(((.((...)).)))))))))...........))))))..)))))..((((.........))))..)).))))))............'},
  'seq_tgw325_047': { 'energy': -29.3,
                      'seq': 'UGACUCGGAACUUGAUUUGCCGGCCUGAAAUUUGACUAGGUAAAUAGAUGAGAAGGGCCGUACAACAGUUAGUGCAAGUUAUACUAAGGCACAUUGUCUGCGGCUUGCCGGCUUUGAUUC',
                      'struct': '..................((((((......(((.((((......))).).))).((((((((..(((((..((((.(((...)))...))))))))).))))))))))))))........'},
  'seq_tgw325_048': { 'energy': -37.8,
                      'seq': 'UACCCUCUAAGAUGCCUUGCGGGGCACAGGGUGUGAAUACUCGGCACUUAAUUACGAGCAUGAGUCUAAAUUACGUGCCGCUAUACGCUUCCUACCCUGAGGGUAAGUGGACAAGGCAGA',
                      'struct': '..((.((((...((((((.(((((...((((.(((.(((..((((((.(((((..((.......))..))))).)))))).))).))).)))).)))))))))))..))))...))....'},
  'seq_tgw325_049': { 'energy': -25.3,
                      'seq': 'AAGAGUACCUUUCACUUGGUUCAACGGGAAGCCAACGCCCCGGGUAAAAGAUAACUCUAGUACGCGUGUGUACGUCUGUUAAUCUAUCACUUCACCCAUAGUACGUGGGCUCAGUCCAAU',
                      'struct': '..(((.(((........)))....((((..((....))))))(((...(((((((....((((......))))....))).))))...)))...(((((.....))))))))........'},
  'seq_tgw325_050': { 'energy': -28.6,
                      'seq': 'CCAUUUGAUACCUUAAUGCCCGGAAGCGGACCAACUCCGGACCCGCUCAUAACUCGCCCCAAGGCCGCUUCUUUCCAUCUCGCGCUCAGUGCCGCCUGAUUUCCCAGGGCGUAUUCUGGU',
                      'struct': '.........(((...((((((((.(((((.((......))..)))))........(((....)))...........(((..((((.....).)))..)))...)).)))))).....)))'},
  'seq_tgw325_051': { 'energy': -24.2,
                      'seq': 'CAGGGGCGUCCGUACCUAGUCGAGAUAAGGAGUCUUGCCACUGUCUAUGCAUAUCGCUUAAUUCAGUUCAUAAACACUUCUGUGAGUGUUCCUCGGCCUCGAGCCAUAAGAACCGAGCGU',
                      'struct': '...(((((..((((..((((((((((.....))))))..))))..)))).....)))))..(((.((((...(((((((....))))))).((((....))))......)))).)))...'},
  'seq_tgw325_052': { 'energy': -19.7,
                      'seq': 'UAGUUUCACAGACUAUCAAAGAUGCCAAAUCUAGCUGGUCCCGGUUUUUAACAGUUCACACGUUCGCAACAAGAAACCAAGGUUACGAUCCUAUCGAAGCGGUCGUUUCGAAAACACCUG',
                      'struct': '..((((....(((((....((((.....))))...)))))..(((((((......................))))))).(((.......))).(((((((....))))))))))).....'},
  'seq_tgw325_053': { 'energy': -23.0,
                      'seq': 'GAAAGAGUGAUGAUGCUGCUAAACUCAGUCACGUGUACAGGAACUAUGGUAAUUCAUGAAAUGAGGUAGGGAUCGGAGCCCCAGAUUAUAUCAUGUAAUGUGACAGGCUACAUUACAGCG',
                      'struct': '......((..(((((..(((.......(((((((.((((.((..((((((..(((((...)))))...(((........)))..)))))))).))))))))))).)))..)))))..)).'},
  'seq_tgw325_054': { 'energy': -33.3,
                      'seq': 'CGCGCGACGACCCUAGGAGAGGGGACUUAGACGUCUCGUCCUCCACAUGUUCCCUUGAAUCCAUGAUCCGGAAUGAUAGCGGUGCUUAAUAGUUAUCAUUGCGAUCAUCCCUGUCUCCAA',
                      'struct': '...............(((((.(((.....((((...))))........((((....))))..((((((.(.(((((((((.((.....)).))))))))).))))))).))).)))))..'},
  'seq_tgw325_055': { 'energy': -30.8,
                      'seq': 'UCUUGAGCACCUAGGUUGACGAGUCUCAUUCGGAGGGACUGUGAAAUUGACCCUAUUUUCUGUGACUAGUGCGGGCCGCAGACUCAGGACGGUCUCCCACCGGAAAGAGCUUCCCGUCCA',
                      'struct': '.............((.((..(((.(((.(((((.((((((((.........(((....((((((.((......)).))))))...)))))))))))...)))))..))))))..)).)).'},
  'seq_tgw325_056': { 'energy': -42.1,
                      'seq': 'AUGUGCACGUGUUGGCGACCCCACUAGGGGCACAAUUCCUGCACGGGUGCCCUCUCCGUAUGUCUGGAUGGAGAAGAAGAACCGUAGUUUUAGCAUAUUCGGAAGCGUUAGUGGGUACCG',
                      'struct': '..(((((.(.((((..(.((((....))))).)))).).))))).(((((((.((.(((...((((((((.......(((((....)))))....)))))))).)))..)).))))))).'},
  'seq_tgw325_057': { 'energy': -27.1,
                      'seq': 'GACUCGAACUACAUUCCAUUCCGGCAAAUUAUAGGGUAUCGCCACUAGUCGGCUGCGAAUGGACAGGAAACAAACAUUUACUGAGGUCGUAUUUGGUCCAACAAGUAGCUGAUGGAAAGG',
                      'struct': '((((..........(((((((((((..((((..((......))..))))..)))).)))))))(((..((......))..))).))))........((((.((......)).))))....'},
  'seq_tgw325_058': { 'energy': -18.9,
                      'seq': 'CUAGAGGGCGAAAACUCACAACACAAGCGUACAAUGCAGAAUAGUAUAUAAACGGUCCGAAAUACGUCGUAGGGAUUAUUACCGCAUUGUGAAAGUCAGGGAACCAAUGAAACUUUCAGA',
                      'struct': '...(((........))).............(((((((.(((((((...((.(((..........)))..))...)))))).).)))))))((((((((.........)))..)))))...'},
  'seq_tgw325_059': { 'energy': -25.4,
                      'seq': 'UGCCAAACUCAACGUAUUCAAUUUGGGAGCCAUGGCAUGGCCAACCGAGUGAUUGAGGAGGUGGACCAUGUAGAGAGUAACCUGGUUAUAUUUCGCCCAACCAAUACACUUCUCCGGAUC',
                      'struct': '.(((...(((((.....(((.(((((..(((((...)))))...)))))))))))))..)))..........((((((....(((((...........)))))....)).))))......'},
  'seq_tgw325_060': { 'energy': -28.9,
                      'seq': 'UGGUCCGUAAGAAUUCGCACGAUUAUCCCGUGGCGGCAAUGUCUUGCAGAGGCGUUUUAACAGGAGGCCAACCGUUUCAGGAAGACAUGCAAAGUAUUACACAAUUGCGUUUGACUGAAA',
                      'struct': '...((.((((((..(((((((.......))).)))).....)))))).))(((.((((...)))).))).....((((((..((((..((((.((.....))..))))))))..))))))'},
  'seq_tgw325_061': { 'energy': -37.0,
                      'seq': 'AAGAACGGUUCCGGGAACACCGUUUGCUAGGUCGAUCGAAAGGUUGGUGUAGCACAGCGAGGCUAUACUCCAGGGUCGGGGGUGCCCUGUCUGCUUUAACCCAACUUCCACGUGACCUCG',
                      'struct': '..(((((((.........)))))))...((((((......(((((((.((((((..(((.(((....((((......))))..))).))).))))...))))))))).....))))))..'},
  'seq_tgw325_062': { 'energy': -41.9,
                      'seq': 'GGUUCUUAGACUGGGCGGCUUUAGCGCUGGGUAAACCGGUCGUUUGCAGCAAUUGGGCCAGUACGCGCCAUGCUCGCAGAGUGGGCUUGACGUGUAUUAUACAGCAGCGCCGUCUCCUAG',
                      'struct': '..........(((((.(((....(((((((((...(((((.(((...))).))))))))((((((((.((.((((((...)))))).)).))))))))......)))))).))).)))))'},
  'seq_tgw325_063': { 'energy': -23.1,
                      'seq': 'CCUGUCCAUAGUGGUAUCUUUCAACAAUUGCUUGACUAUUUUGACUUGUACGGUUUACCGCUAGUAGUAUGCCGUGCUAGAAACGGUUCAGAUCCCAUAGCAGAUUUCGUGGUCAUAUGC',
                      'struct': '..((.((((...((.((((...........((.((((.((((.....(((((((.(((........))).)))))))..)))).)))).))..........)))).)))))).)).....'},
  'seq_tgw325_064': { 'energy': -24.7,
                      'seq': 'AAUAAGAUUUUGACUCACGGAACUUUGGGAUAAUAACCACCAGAAGGGCGAUCCAGACCUCACUCCGUGUACCUCCAAGCACUAGCCGUUCAAUAGGAUGCGUAGGCGUGUACAAAAUUC',
                      'struct': '.....(((((((((.((((((..(((((...........))))).((((......).)))...)))))).........((.(((..(((((....)))))..)))))..)).))))))).'},
  'seq_tgw325_065': { 'energy': -26.5,
                      'seq': 'GAUAGCGGUUUGUAGAGCCCGGUUAAGAACUGAAAACCCAGCAGACACAGGAGUCGCAGGGUAUCUUAGCGACUUACCGCACUUCCCGCAAUCGAUCUGAACAACCAAGACAAGCACUUA',
                      'struct': '(((.((((......(((..((((.(((..((((..((((.((.(((......))))).))))...))))...)))))))..))).)))).)))...........................'},
  'seq_tgw325_066': { 'energy': -38.7,
                      'seq': 'GGGGUUUCCGAAGUCGGUAACAAUGUUCGGCGGGAGUCUAUGCAUAUGCACUGCUAAACUGCAGUACGCUGGUCAGGACUGAUCCGGCGUGAUUGUGCUCGCGCAGGAUUUAGUAACAGA',
                      'struct': '...(((.(((....))).)))...((((.(((.((((...(((....)))((((......))))(((((((((((....))).)))))))).....)))).))).))))...........'},
  'seq_tgw325_067': { 'energy': -30.4,
                      'seq': 'CGGAACUCGGACGGGGAGACGUGCACUAUGAGAUCCUGGCAGCCGUAUCCCUAAGUAGAACCACACUAGGAGACUAGUCACAAAGGUGCUAGCAGUGGAGGACGCGAUAGUGACCUGAGG',
                      'struct': '.....(((((..(((((.(((.((.(((........)))..))))).)))))........((.((((....(.((((.(((....))))))))))))..)).(((....)))..))))).'},
  'seq_tgw325_068': { 'energy': -30.4,
                      'seq': 'AGCGAGCUGCGGACGCCACUUAUCUGACAUACGCAUAUGUCGGCCUUACUUACUCGUGCACUGGAUCCAGGGAGAGCUAUGUCAUAUCUGUGGUAGAAGAGGCAUAUUGGGUUUAGGCCC',
                      'struct': '...(((..((....))..)))....((((((....))))))(((((.(((((...((((.((....(((.(((.............))).)))....))..))))..)))))..))))).'},
  'seq_tgw325_069': { 'energy': -26.2,
                      'seq': 'ACCCCCCAACGGCAAAACGCUGCCGGCGUCUCUAGCCGACGAACCUCAGUCUGUACAUUAGAGUUUAUUAACGCGUACUUCCUUAAUUCAGGCUCUGUCUUUGGCACAUCCUAGGCGGCG',
                      'struct': '.................(((((((..........(((((.((.....((((((.......((((............))))........))))))...)).)))))........)))))))'},
  'seq_tgw325_070': { 'energy': -28.2,
                      'seq': 'CCGUGCGACAAGUCUCCAGUUCCUGCUCGAGCGGGUGUCAUAAAGGUUCACGAGGGCGCAUGUUGUAGCGUCAUUUUUGGUAGAUAUCUUACGUCUCCUGGGGUCAACCAGCAUAUCUAA',
                      'struct': '..((((.....(.((((((..(((((....))))).......(((((((((.(((((((........)))))....)).)).)).))))).......)))))).).....))))......'},
  'seq_tgw325_071': { 'energy': -20.5,
                      'seq': 'CACAUCGCGAUGAUUUCACUUCGUUAAAUAAUCAUGACUGUACAAAAGCCCAAAACCAUGGACUUGUGCGUUUGCUAAUCUCACUGAUACUCUUUUGGCGAACAAAUCGAAUCUUAUGGG',
                      'struct': '.((((((.(((.((((.((...)).)))).))).))).)))..............(((((((.(((...(((((((((................)))))))))....))).)).))))).'},
  'seq_tgw325_072': { 'energy': -20.8,
                      'seq': 'UUAUAGCUAAAGCGGUCUUAUUGUCUAGCUGUCCUUAGAUAAACCUUCCAACCUAUUGUUACGAUUUUCGAGCAGAGUGCAAAGCGUGUUUAGCCCUAAUUUCAGUAGGAUUGCCGUUGU',
                      'struct': '..........((((((....((((((((......)))))))).........(((((((...........(((((..((.....)).)))))...........)))))))...))))))..'},
  'seq_tgw325_073': { 'energy': -32.3,
                      'seq': 'CCAGCUCCAGAGUUUGUACUUCAAGGCUAGUGUCCGUAGUGGCCCUAGUAGGAUCACAGCCGGAUGCCAGGUCUCUUUGGCAACUUUUUUGGAUGCGACAUGUCUAUCCAGGUCUGCUAU',
                      'struct': '...(((((((((.((((......(((((.(((((((..((((.(((...))).))))...)))))))..))))).....))))...))))))).))(((.((......)).)))......'},
  'seq_tgw325_074': { 'energy': -29.3,
                      'seq': 'CCAUGACCGCUUACAGGCCAUUAAAAGGUUUAUGGGCAAAGCACCUUACGGGGCCCCGUAGGAUUUCGCAAUGGUCACCUAUAGUCGAGAGAUUUCGUCCGUAGAUCCCGAUAGAACAGC',
                      'struct': '...((((((((((.(((((.......))))).))))).......(((((((....)))))))..........))))).((((.(.((((....)))).).))))................'},
  'seq_tgw325_075': { 'energy': -23.3,
                      'seq': 'GCACAAAGGUCCUCCUUAACAGUCCCUGUUCAGCCCAUCCUAAUACUGCCCUACACCGAUUGAGUAAGAAAGAAAAUGAAUUUACUGUCUUCGCUGUUCAUUGAACAGGGGGGGUAAGAC',
                      'struct': '......(((.....((.(((((...))))).)).....)))..((((.(((.....(((...((((((............))))))....)))((((((...))))))))).))))....'},
  'seq_tgw325_076': { 'energy': -26.6,
                      'seq': 'AAUUUUCGAGGACCAAUUUACGCGUAUUCUUGAUGAUCACUAACGACCCCUGAUACGCUAGACACCAAUGGUCUAGGAGUCUUAGGGUCGGCUAUAACAAUGACAAGCAAUCGGGGUAUC',
                      'struct': '..................(((.(..(((((((...........((((((.(((.((.((((((.......))))))..)).))))))))).............)))).)))..).)))..'},
  'seq_tgw325_077': { 'energy': -28.8,
                      'seq': 'CUAAGGGAAGUGAUAUUGCGUGAGGAGUUGGGUCCCGCAAUAACGCUAGAAUCUCUUAACCUAACGAACCAUACCCCGUGUAAGACCAGGCGACCCCACCUCAGACGGUAGAGGCACUAU',
                      'struct': '.(((((((((((.(((((((...(((......)))))))))).))))....)))))))...................((((...((((((.(....).))).....)))....))))...'},
  'seq_tgw325_078': { 'energy': -20.6,
                      'seq': 'GUUCUGCUAACGCGCCACUUAACCCUAAUGAACAAUUUCGGAUGGCCCGGCAUUAUGGAUGUGAUCAGAUAGGCCCGAUACGCUGUGACAAUCGAGCGUAGACAUUAUUACUAGAUAGCA',
                      'struct': '....(((((..............((((.(((.((..((((((((......)))).))))..)).)))..)))).....((((((.(((...)))))))))...............)))))'},
  'seq_tgw325_079': { 'energy': -24.1,
                      'seq': 'CUUCCGUCUGACAGCGAACUCUCAAUUGAAGGAACACUCCUCACCUAAUUAUCAGACCGGGCAAAACCCGGCAUGUUCACACGUUUUAUUAGCCCAAAUGUGGAGUCUUCAUUUCGCCGG',
                      'struct': '...(((..(((.((.(((..(((...(((.(((....)))))).........((..(((((.....)))))..))....(((((((.........))))))))))..))).))))).)))'},
  'seq_tgw325_080': { 'energy': -27.0,
                      'seq': 'AUGCUCCCUAGUUAGGUGACGAGCUACAUAAGAACAUGUAGCAUCCAUGAUGGCGGUAAGCAUAUAUUGUAGUCCGGUUCACUGUGCAGCUCACGACCUCCGGCAAAUGUACUCGCGGAG',
                      'struct': '(((((..((.((((.(((..(((((((((......))))))).)))))..))))))..))))).........((((.......(((((.....((.....)).....)))))...)))).'},
  'seq_tgw325_081': { 'energy': -44.6,
                      'seq': 'GGGCAGCUCCAUAGGAGAAGCGCAUAUAGACUAACAGGUUCUGAUACUCAGGAUCUGCAGCCUUUGGGAAACGGGGCGUGUAAGAUCUUUUAUAUGCCUCGCAUGAAUGUCUAUUGCCCC',
                      'struct': '..((..((((...))))..))(((.((((((...((((((((((...))))))))))...((....))...(((((((((((((....))))))))))))).......)))))))))...'},
  'seq_tgw325_082': { 'energy': -33.2,
                      'seq': 'CAACUGGUUGAAAAGCCUUGACUCGGCAUACAGUCCUCUCUGACAUCCCGGCUAAGAAGAUUCGUCGGUUUGGGACGACAAUACGUCGUCCAUAUUGAAACUGUGUGACUAGCCAUGCUU',
                      'struct': '............((((..((.((.((((((((((.........((..(((((.((.....)).)))))..))(((((((.....)))))))........)))))))).)))).)).))))'},
  'seq_tgw325_083': { 'energy': -23.8,
                      'seq': 'GUCCUUUUGGGUCAAAAUAGAUAGGUCGUUGAUCUCGAUAGAUAGCGAUGAAAUAACAGGGGCAUAUAACUCUUUGCAUGCACUAACUGAGUAUUGCCGUUGGCGCCUGUAAGCUAAUUC',
                      'struct': '(((((((((..(((...........((((((.(((....))))))))))))..))).)))))).........((((((.((.(((((...........))))).)).)))))).......'},
  'seq_tgw325_084': { 'energy': -28.2,
                      'seq': 'UAGUAGGCGCCUUCGGGUGCGGUACCUGAGCCCUGGACGCUUUAAGAAACUUACGUGAGAUGCGACGAAAUGCAUCAUUGUAGAAGACUAUUUCCCACGAGACUAAUUUCUCCUACGACU',
                      'struct': '....(((((((...((((.(((...))).)))).)).)))))................(((((........))))).((((((.(((.((((((....)))).))...))).))))))..'},
  'seq_tgw325_085': { 'energy': -23.4,
                      'seq': 'UAAGUAGGCGAGGCACAACUAGGUCCCGAUACUGGGAGCCAUACCAUGUAGCGACAUUUAAUUCCCAAGAGGCGCGUGACAACCUUAUCCUGUAGCAACAAUGUGCGGUCGUGAGUAACG',
                      'struct': '.......((((.(((((....(((((((....))))).))....(((((.((...................)))))))............(((....))).)))))..))))........'},
  'seq_tgw325_086': { 'energy': -27.4,
                      'seq': 'UCGGAGCGUACAAAUGACAGAAUUCUUACAUUGAGGGCGACCUGCUAUCCCACAUCACCCGACUCAGAGUAUGACCCGGAAAGCUGUCAAGCUCUAUUCAUGGCAGCACGUUACGGAGCA',
                      'struct': '(((..((((.....((((((..(((((((.(((((((.((..((......))..)).)))...)))).)))......))))..)))))).((((((....))).)))))))..)))....'},
  'seq_tgw325_087': { 'energy': -38.8,
                      'seq': 'GAGGCGCACGCGUGCAAGUGUCCUGUGGUAGCAGUACCAAGGCUUGCAGCGGGAUAUGGCGGAUUAAGCCAUAUACACUGGCCGGAGGGGGCUCUGUCAUCGACACCUAUACAUCUUUCA',
                      'struct': '.(((....(((.((((((...(((.(((((....)))))))))))))))))..(((((((.......))))))).....((((......)))).((((...)))))))............'},
  'seq_tgw325_088': { 'energy': -33.5,
                      'seq': 'UUUUAUGGGUGAGCGCUGGUAACUCGUGUUCAAUGCAAAUCCAAAGAUACUGAGCACACGGACAUUACAUUAUGUGGAAUGAACGCUCUCGGUGUCUUCCCCGCCCAGUGGCACGGAGCC',
                      'struct': '.....(((((..(((.(((..((....))))).)))..))))).((((((((((....((..(((..(((...)))..)))..))..))))))))))(((..(((....)))..)))...'},
  'seq_tgw325_089': { 'energy': -23.7,
                      'seq': 'AUUCGAUUAUAGUCUGUGGCUGAAGGGUGAACGGGUCAACUCAGGAAGGAACUAGGCCGCGUCCUGUCGCCGAUAUUUGAAUCUACUUCCCUCGAAUUCGUUCUCGUUAUCAAAAGCAGC',
                      'struct': '.............((((...(((((.(.((((((((....((.(((((......(((.((.....)).)))(((......)))..)))))...)))))))))).).)).)))...)))).'},
  'seq_tgw325_090': { 'energy': -31.8,
                      'seq': 'CUCUAGUCAGCCAGUACUGCGAAGUAAUAAUAAGCCGACAAUGAGGCUGAGCGGCGGCCAUCGAUGCAUUUACCACAUAUGGUGGAUUAGAAAGAGAAACUUUUCUUUCGCGUUGCGAGG',
                      'struct': '(((..(((.((...((((....)))).......)).))).....(((((.....)))))..((((((((((((((....))))))))..((((((((....)))))))))))))).))).'},
  'seq_tgw325_091': { 'energy': -33.6,
                      'seq': 'CAUGCCGGGAGCGGCCGAUAUCUUUCUCUGAUCAUCGCGGACAAUACGAAUGGGCACAACCGGGCGGGUGUACGGGCAGAUCUUAUCGUUAUUGUGUCUGGGCCGUGCGAAGAUUUCGGG',
                      'struct': '....(((....)))((((.((((((.((((.......)))).....((..(((......)))..))...((((((.((((((...........).)))))..)))))))))))).)))).'},
  'seq_tgw325_092': { 'energy': -35.2,
                      'seq': 'AAACUAGCACGUGUGUAGAGGAUCGAAAAACAGUGCCCAAUCGAUAGAUCUACAAUGCUGGUGGACUCUCGGCAACGCUGUGUGUGAAUUGAGGAGUGUAUAGCCACGACCGAGGGUUGU',
                      'struct': '..(((((((....((((((..(((((..............)))))...)))))).))))))).(((((((((...((.((..((((.(((....))).))))..)))).)))))))))..'},
  'seq_tgw325_093': { 'energy': -22.9,
                      'seq': 'AUCUCUCGCAUUCGACUUGAAAUGCCGACGGGGUAUCAUUCCAUACUACUCGAAAGAUCCGGUCUGGCUACUCAAAGAUUAAUGAAUUUCUCCCAACGCGAUCUAGAGCAGCGUCAGUGC',
                      'struct': '.......(((((.(((((((...((((((.((((....(((..........)))..)))).))).)))...))))(((..........)))......((........))...))))))))'},
  'seq_tgw325_094': { 'energy': -26.3,
                      'seq': 'GACGUAUAUAAGGAGCACACUAUAUGAGUGGUUAUACUGAAGAUCGCUUAUGGUCACUAAAGUGAAAAUUGCCGCGUCUUCGAACUAAUCAGAGGAGAUCGUACAUGGUGUCGCAUAACG',
                      'struct': '..............(((((((((((((((((((........))))))))))..((((....))))........((((((((............))))).)))..))))))).))......'},
  'seq_tgw325_095': { 'energy': -31.0,
                      'seq': 'AGGGAGACCGUAGAAUAGUUGACUUGGUUUCGAUGUCGUCUGCCACGAAAGAACGAACACACACGGUAUGUUGGGUGUAUUGACGGGAUUGAGGCGGUGCAACUAGCAACAAUGGAGGUC',
                      'struct': '.....((((......((((((.(.(.((((((((.(((((((((.(((.....((........)).....)))))))....))))).)))))))).).)))))))...........))))'},
  'seq_tgw325_096': { 'energy': -33.6,
                      'seq': 'GGCACUGCAUCACAGUAGAUCUGUGGAUCUAUACGUUCCGAGGCCGAUGACGACUUGGGAUGUAGGUGGGUACCUUGCUGUACAUUCAGAACAAUUUGUCCAGCCCCUCCUUCCCCGCGC',
                      'struct': '.((((((.....))))(((((....))))).(((((((((((..((....)).)))))))))))((.(((......((((.(((............))).))))))).))......))..'},
  'seq_tgw325_097': { 'energy': -16.5,
                      'seq': 'ACACCACUAAGAGUACACAAUGCUAGGCGCCUUUAGCUCGAUGAGGUUACCCACGAACGAGGCAUGGCGGUCUAUGUCGAUUAUCCGUUUCCAACACACCUUGAGUUCAAAUUCACUGAG',
                      'struct': '.....................((((((((((.....((((.((.((....)).))..))))....))).))))).))........................(((((....))))).....'},
  'seq_tgw325_098': { 'energy': -22.9,
                      'seq': 'UCCAAACCUUAUCUGCACUCAAUCCCACCUUUUGGCUUCGUCCUUACGGCCGCGCAGGACCAACUGCCACGCCCCCUAGCGCGUCUAGCUAACCUGCCACUGAAACUAGAGCCAUGAGUU',
                      'struct': '................(((((...........(((((((((.((...(((...((((......))))...)))....)).)))....((......))...........))))))))))).'},
  'seq_tgw325_099': { 'energy': -27.3,
                      'seq': 'UUCGGCCGAAUGGGCUCGGUCAUGAUAAAAUGGCCAAAAGGGCUUCCGAACAUUUCCUAACAUUUGUGAUCACGCAACUCGUUAUUGUACCCCCAAGUCGAAGUGACGUAGCCUCGAUUA',
                      'struct': '...((((.....)))).((((((......))))))....(((((..((..((((((.((((..(((((....)))))...))))(((......)))...)))))).)).)))))......'},
  'seq_tgw325_100': { 'energy': -35.8,
                      'seq': 'CGUCUAUGAAGUCGAACUAUGCGGCCAAUGAUUUGUGAAAAUCCGAGAGACUAUCCGUCAUUCAGAUCGUCUCCAUAGGGCGCUGCCCGAGCUAGGGGCAGCCUGGAACAGUCCCAGUUG',
                      'struct': '..((((((..((((.......))))....(((((....)))))...(((((.(((.(.....).))).)))))))))))..(((((((.......)))))))((((.......))))...'}}



############################################################
######################## M a i n ###########################
############################################################

if __name__ == "__main__":
    
    # Importing results from RNA Fold, and structuring in DataFrame
    strucs = pd.DataFrame(rnafold).transpose()
    strucs.columns = ["RNAFold energy", "Sequence", "RNAFold"]
    strucs = strucs.iloc[:,[1,2,0]]

    # Solving Nussinov using standard base pair scoring
    nussinov_standard = strucs["Sequence"].apply(lambda seq: Nussinov(seq, scoring='standard'))
    strucs["Nussinov standard"] = nussinov_standard.apply(lambda nus: nus.dot_bracket)
    strucs["Nussinov standard score"] = nussinov_standard.apply(lambda nus: nus.best_score)

    # Solving Nussinov using stacking base pair scoring
    nussinov_stacking = strucs["Sequence"].apply(lambda seq: Nussinov(seq, scoring='stacking'))
    strucs["Nussinov stacking"] = nussinov_stacking.apply(lambda nus: nus.dot_bracket)
    strucs["Nussinov stacking score"] = nussinov_stacking.apply(lambda nus: nus.best_score)

    # Computing pairwise base pair distances
    dists = pd.DataFrame(index=strucs.index)
    dists["RNAFold - Nussinov standard"] = strucs.apply(lambda row: base_pair_distance(row["RNAFold"], row["Nussinov standard"]), axis=1)
    dists["RNAFold - Nussinov stacking"] = strucs.apply(lambda row: base_pair_distance(row["RNAFold"], row["Nussinov stacking"]), axis=1)
    dists["Nussinov standard - Nussinov stacking"] = strucs.apply(lambda row: base_pair_distance(row["Nussinov standard"], row["Nussinov stacking"]), axis=1)
    min_dist_plot = dists.values.min() - 5
    max_dist_plot = dists.values.max() + 5

    # Plotting correlation between scores
    def plot_correlation(x_column, y_column):
        plt.plot((min_dist_plot, max_dist_plot), (min_dist_plot, max_dist_plot), linestyle='--', linewidth=1, c='grey', alpha=0.2)
        plt.scatter(dists[x_column], dists[y_column], s=4, c=(dists[x_column] - dists[y_column]) > 0, cmap='coolwarm')
        plt.xlim(min_dist_plot, max_dist_plot)
        plt.ylim(min_dist_plot, max_dist_plot)
        cor, cor_p = pearsonr(dists[x_column], dists[y_column])
        plt.text(min_dist_plot + 5, max_dist_plot - 10, f"Pearson r ≈ {cor:.3f}\n(p ≈ {cor_p:.3e})")
        plt.xlabel(f"({x_column}) base pair distances")
        plt.ylabel(f"({y_column}) base pair distances")
        plt.savefig(f"{x_column} vs. {y_column}.png")
        plt.clf()
    plot_correlation("RNAFold - Nussinov standard", "RNAFold - Nussinov stacking")
    plot_correlation("Nussinov standard - Nussinov stacking", "RNAFold - Nussinov stacking")
    plot_correlation("RNAFold - Nussinov standard", "Nussinov standard - Nussinov stacking")

    # Plot cummulative distribution
    dist_range = np.arange(min_dist_plot, max_dist_plot)
    for col in dists.columns:
        plt.plot(dist_range, (dist_range[:,np.newaxis] >= dists[col].to_numpy()).mean(axis=1), label=col)
    plt.legend()
    plt.xlim(dist_range[0], dist_range[-1])
    plt.ylim(0,1)
    plt.grid()
    plt.xlabel("Base pair distance")
    plt.ylabel("Fraction of structures with at least this distance")
    plt.savefig("Base pair distance cumulative distribution.png")
