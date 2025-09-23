import os
import jax.numpy as jnp

# TODO: doc string, references
def parse_sk_file(name : str, distances : list):
    """
    Parses a Slater–Koster (SK) parameter file into dictionaries of orbital
    integrals and overlaps at specified distances.

    Parameters:
        name (str): Path to the Slater–Koster parameter file (DFTB format).
        distances (list[float]): List of interatomic distances (in atomic units)
            at which parameters should be extracted.

    Returns:
        dict: A dictionary with the following structure:
            {
                "onsite": dict
                    Onsite energies keyed by orbital type (e.g. {"s": [...], "p": [...]}).
                "hubbard": dict
                    Hubbard U parameters keyed by orbital type.
                "integrals": dict
                    Slater–Koster hopping integrals keyed by interaction type
                    (e.g. "sss", "sps", "ppσ", "ppπ"), with values given as
                    lists over requested distances.
                "overlap": dict
                    Overlap integrals keyed by interaction type, structured
                    the same as `integrals`.
            }

    Notes:
        - Supports both the simple (`s`, `p`, `d`) and extended (`s`, `p`, `d`, `f`)
          Slater–Koster file formats.
        - For homonuclear SK files, onsite energies and Hubbard U values are
          extracted; for heteronuclear files, these are empty.
        - Distances are mapped to tabulated grid points based on the grid
          spacing given in the SK file.
        - If more distances are requested than are tabulated, a warning is raised.

    References:
        - DFTB+: https://pubs.acs.org/doi/10.1021/acs.jpca.5c01146
        - File format specification: https://dftb.org/
    """    
    def clean(line):
        """cleans up unnecessary symbols
        """
        trash = [',', '*']
        table = {ord(i) : None for i in trash}        
        return line.translate(table)

    def process(line):
        """turns line into list of floats
        """
        return list(map(float, clean(line).split()))

    def extract_dicts(values_list, keys):
        make_dict = lambda offset : {key : [x[i + offset] for x in values_list] for i, key in enumerate(keys)}
        return make_dict(0), make_dict(len(keys))
                        
    with open(name, "r") as f:
        # type of format
        first_line = f.readline()
        is_simple_format = "@" not in first_line        

        # keys for overlap and hopping
        offsite_keys = ["dds", "ddp", "ddd", "pds", "pdp", "pps", "ppp", "sds", "sps", "sss"]
        
        # keys for onsite and hubbard
        onsite_keys = ["d", "p", "s"]
        
        # grid line
        line = first_line

        # adjust for extended format
        if not is_simple_format:
            offsite_keys = ["ffs", "ffp", "ffd", "fff", "dfs", "dfp", "dfd", "dds", "ddp", "ddd", "pfs", "pfp", "pds", "pdp", "pps", "ppp", "sfs", "sds", "sps", "sss"]
            onsite_keys = ["f", "d", "p", "s"]
            line = f.readline() 
        
        # grid info
        grid_dist, n_points = process(line)[:2]

        # empty for heteronuclear case
        onsite, hubbard = {}, {}
        
        # homonuclear file contains additonal line with onsite energy and hubbard info
        name_list = os.path.basename(name).split('-')
        homonuclear = (name_list[0] == name_list[1].split('.')[0])
        if homonuclear:
            # skip spin polarization error
            processed_line = process(f.readline())
            processed_line.pop(len(onsite_keys))
            onsite_info = [processed_line]
            onsite, hubbard  = extract_dicts(onsite_info, onsite_keys)
        
        # we need only values at specific distances => get their corresponding line numbers, add 1 to shift past the repulsion line
        line_numbers = (jnp.array(distances[homonuclear:]) / grid_dist).astype(int) + 1

        # extract lines
        offsite_info = [process(line) for i, line in enumerate(f) if i in line_numbers]

        # could be that more distances are required than tabulated
        if any(map(lambda x : len(x) < len(line_numbers), offsite_info)):
            Warning(f"Number of hopping elements required: {len(distances)}. Can only extract {len(line_numbers)}.")
            
        integrals, overlap = extract_dicts(offsite_info, offsite_keys)

    return {"onsite" : onsite, "hubbard" : hubbard, "integrals" : integrals, "overlap" : overlap}
