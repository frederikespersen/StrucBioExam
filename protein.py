import Bio.PDB as BioPDB

############################################################
################### Utility functions ######################
############################################################

def extract_bb_atoms(atoms):
    """
    
    Takes an iterable object of `Bio.PDB` atoms (list of atoms or .get_atoms() generator), returns a filtered list of only backbone atoms.

    --------------------------------------------------------------------------------

        Parameters
        ----------

            `atoms`: `list` or `generator`
                An iterable object of `Bio.PDB` atoms
        
        Returns
        -------

            `atoms`: `list`
                A list of only backbone `Bio.PDB` atoms
    
    """
    # Filtering to backbone atoms
    return [atom for atom in atoms if atom.get_name() in ['C', 'CA', 'N', 'O']]

def extract_res_range_atoms(atoms, start:int, end:int):
    """
    
    Takes an iterable object of `Bio.PDB` atoms (list of atoms or .get_atoms() generator) and two indices, returns a filtered list of atoms within the specified residue indices.

    --------------------------------------------------------------------------------

        Parameters
        ----------

            `atoms`: `list` or `generator`
                An iterable object of `Bio.PDB` atoms

            `start`: `int`
                The start of a residue range (inclusive)

            `end`: `int`
                The end of a residue range (inclusive)
        
        Returns
        -------

            `atoms`: `list`
                A list of `Bio.PDB` atoms of residues in the `start:end` range (inclusive)
    
    """
    return [atom for atom in atoms if atom.get_parent().id[1] in [*range(start, end+1)]]



############################################################
######################## M a i n ###########################
############################################################

if __name__ == "__main__":
    
    # Parsing structures
    parser = BioPDB.PDBParser()
    pdb_1YCR = parser.get_structure('1YCR', 'rfdiffusion_pdb/1YCR.pdb')[0]
    design_rfdiffusion = parser.get_structure('RFDiffusion', 'rfdiffusion_pdb/design_rfdiffusion.pdb')[0] # Only backbone
    design_proteinmpnn = parser.get_structure('ProteinMPNN', 'rfdiffusion_pdb/design_proteinmpnn.pdb')[0]

    # Final design: Scaffold (1-50) - p53 Helix (51-63) - Scaffold (64-105)

    # Selecting p53 helices backbones from structures (since RFdiffusion only outputs backbone, as no sequence is determined)
    pdb_1YCR_p53helix =                                   extract_bb_atoms(pdb_1YCR['B'].get_atoms())
    design_rfdiffusion_p53helix = extract_bb_atoms(extract_res_range_atoms(design_rfdiffusion['A'].get_atoms(), 51, 63))
    design_proteinmpnn_p53helix = extract_bb_atoms(extract_res_range_atoms(design_proteinmpnn['A'].get_atoms(), 51, 63))

    # Superimposing helices, with pdb_1YCR as reference
    superimposer = BioPDB.Superimposer()
    superimposer.set_atoms(pdb_1YCR_p53helix, design_rfdiffusion_p53helix)
    superimposer.apply(design_rfdiffusion.get_atoms())
    print(superimposer.rms)
    superimposer.set_atoms(pdb_1YCR_p53helix, design_proteinmpnn_p53helix)
    superimposer.apply(design_proteinmpnn.get_atoms())
    print(superimposer.rms)

    # Saving superimposed structures to PDB for visualisation in ChimeraX
    io = BioPDB.PDBIO()
    io.set_structure(design_rfdiffusion['A'])
    io.save("design_rfdiffusion_superimposed.pdb")
    io.set_structure(design_proteinmpnn['A'])
    io.save("design_proteinmpnn_superimposed.pdb")

    # Determining RMSD betweenn RFdiffusion and ProteinMPNN designs
    superimposer.set_atoms(extract_bb_atoms(design_rfdiffusion['A'].get_atoms()), extract_bb_atoms(design_proteinmpnn['A'].get_atoms()))
    superimposer.apply(design_proteinmpnn['A'].get_atoms())
    print(superimposer.rms)

    # Saving superimposed structures to PDB for visualisation in ChimeraX
    io.set_structure(design_proteinmpnn['A'])
    io.save("design_proteinmpnn_superimposed_full.pdb")

